"""Per-layer boundary-sensitivity scoring for S2Q.

For each quantizable layer of a model, quantize ONLY that layer to `bits` and
measure how much a boundary metric worsens on an iBims subset, paired against the
full-precision baseline. Layers whose quantization most damages boundaries are the
ones S2Q protects. We measure the exact per-layer effect (the oracle) by weight
backup/restore, so no approximation enters the sensitivity itself.

Output: results/sensitivity_<model>_w<bits>.csv with, per layer, the mean paired
increase in DBE-completeness and in the flying-pixel fraction, plus a cheap
weight-quantization-error proxy (||W-Q(W)||_F) for later comparison.
Command: python sensitivity.py --model dav2_large   (also dpt_hybrid, zoedepth)
"""
import argparse
import numpy as np
import pandas as pd
import torch

from common import RESULTS, load_config, seed_everything

CFG = load_config("sensitivity")
from data.loaders import build
from models.registry import load_model, infer_depth
from align import scale_shift_align
import score.boundary as B
from score.aggregate import aggregate_metrics
from score.pointcloud import flying_pixel_fraction
from quant.fakequant import fake_quant_tensor, _QUANTIZABLE


DISP = True  # set from model kind in main (relative->disparity, metric->depth)


def boundary_metrics(raw, s, md, xd):
    a, _, _ = scale_shift_align(raw, s.gt_depth, s.valid_mask, mode="ss",
                               pred_is_disparity=DISP, min_depth=md, max_depth=xd)
    _, com = B.dbe(a, s.gt_edges, s.valid_mask)
    ff = flying_pixel_fraction(a, s.gt_depth, s.gt_edges, s.valid_mask, k=3)["flying_frac"]
    ab = aggregate_metrics(a, s.gt_depth, s.valid_mask, md, xd)["absrel"]
    return com, ff, ab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=CFG["models"])
    ap.add_argument("--bits", type=int, default=CFG["bits"])
    ap.add_argument("--subset", type=int, default=CFG["subset"])
    ap.add_argument("--per_channel", type=int, default=CFG["per_channel"])
    args = ap.parse_args()
    seed_everything(CFG.get("seed", 0))

    global DISP
    from models.registry import MODELS
    DISP = MODELS[args.model].get("kind", "relative") == "relative"
    gen, info = build("ibims")
    md, xd = info["min_depth"], info["max_depth"]
    samples = [s for s in gen() if s.gt_edges is not None and s.K is not None][:args.subset]
    print(f"{len(samples)} images, model={args.model}, bits={args.bits}, disparity={DISP}")

    model, proc, _ = load_model(args.model, dtype=torch.float16)
    # FP16 baseline
    base = {}
    for s in samples:
        raw = infer_depth(model, proc, s.rgb, out_hw=s.gt_depth.shape)
        base[s.id] = boundary_metrics(raw, s, md, xd)   # (com, ff, absrel)

    layers = [(n, m) for n, m in model.named_modules() if isinstance(m, _QUANTIZABLE)]
    print(f"{len(layers)} quantizable layers")

    rows = []
    for i, (name, mod) in enumerate(layers):
        w = mod.weight.data
        orig = w.clone()
        q = fake_quant_tensor(orig, args.bits, per_channel=bool(args.per_channel), ch_axis=0)
        proxy = float((orig.float() - q.float()).norm().item())
        mod.weight.data.copy_(q)
        d_com, d_ff, d_ab = [], [], []
        for s in samples:
            raw = infer_depth(model, proc, s.rgb, out_hw=s.gt_depth.shape)
            com, ff, ab = boundary_metrics(raw, s, md, xd)
            b_com, b_ff, b_ab = base[s.id]
            if np.isfinite(com) and np.isfinite(b_com):
                d_com.append(com - b_com)
            if np.isfinite(ff) and np.isfinite(b_ff):
                d_ff.append(ff - b_ff)
            if np.isfinite(ab) and np.isfinite(b_ab):
                d_ab.append(ab - b_ab)
        mod.weight.data.copy_(orig)                       # restore
        comp = ("encoder" if any(k in name for k in ("backbone", "encoder.layer", "embeddings"))
                else "neck" if any(k in name for k in ("neck.", "reassemble", "fusion"))
                else "head" if "head." in name else "other")
        rows.append({"layer": name, "component": comp, "n_params": int(orig.numel()),
                     "dbe_com_delta": round(float(np.mean(d_com)) if d_com else np.nan, 4),
                     "flying_delta": round(float(np.mean(d_ff)) if d_ff else np.nan, 4),
                     "absrel_delta": round(float(np.mean(d_ab)) if d_ab else np.nan, 5),
                     "wq_error": round(proxy, 4)})
        if i % 25 == 0:
            print(f"  [{i}/{len(layers)}] {name[:50]} dbe_com_delta={rows[-1]['dbe_com_delta']}")
            pd.DataFrame(rows).to_csv(RESULTS / f"sensitivity_{args.model}_w{args.bits}.csv", index=False)
    pd.DataFrame(rows).to_csv(RESULTS / f"sensitivity_{args.model}_w{args.bits}.csv", index=False)
    df = pd.DataFrame(rows)
    print("\nTop-10 most boundary-sensitive layers (by dbe_com_delta):")
    print(df.sort_values("dbe_com_delta", ascending=False).head(10)[["layer","component","dbe_com_delta","flying_delta"]].to_string(index=False))
    print(f"\nwrote results/sensitivity_{args.model}_w{args.bits}.csv")


if __name__ == "__main__":
    main()
