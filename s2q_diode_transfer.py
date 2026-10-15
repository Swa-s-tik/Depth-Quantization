"""Cross-dataset transfer (Table 14): does the S2Q allocation TRANSFER across datasets?
The per-layer sensitivities and the resulting S2Q bit allocation are fit on iBims-1.
Here we take that fixed allocation (~6-bit average, unchanged) and evaluate it on
DIODE indoors, against uniform six-bit and head-protected six-bit. If S2Q still
recovers boundaries on DIODE, the allocation generalizes beyond its calibration set.
Output: results/s2q_diode_transfer.csv.
Command: python s2q_diode_transfer.py
"""
import json
import numpy as np
import torch

from common import CACHE_ROOT, RESULTS, load_config, seed_everything

CFG = load_config("s2q_diode_transfer")
from data.loaders import build
from models.registry import load_model, infer_depth
from align import scale_shift_align
from predict import load_cached_pred
from quant.fakequant import quantize_per_layer_, quantize_model_, _QUANTIZABLE
import score.boundary as B
from stats import cliffs_delta

CACHE = CACHE_ROOT
MODEL = CFG["model"]
SUBSET = CFG["subset"]


def diode_samples():
    gen, info = build("diode", subset=SUBSET)
    return [s for s in gen() if s.gt_edges is not None], info["min_depth"], info["max_depth"]


def boundary_scores(model, proc, samples, fp16, md, xd):
    acc = {"dbe_com": [], "edge_width": []}
    base = {k: [] for k in acc}
    for s in samples:
        raw = infer_depth(model, proc, s.rgb, out_hw=s.gt_depth.shape)
        a, _, _ = scale_shift_align(raw, s.gt_depth, s.valid_mask, mode="ss",
                                    pred_is_disparity=True, min_depth=md, max_depth=xd)
        if s.id not in fp16:
            continue
        vals = {"dbe_com": B.dbe(a, s.gt_edges, s.valid_mask)[1],
                "edge_width": B.edge_sharpness(a, s.gt_edges, s.valid_mask, k=2)[1]}
        for k, v in vals.items():
            if np.isfinite(v) and np.isfinite(fp16[s.id][k]):
                acc[k].append(v); base[k].append(fp16[s.id][k])
    return acc, base


def main():
    seed_everything(CFG.get("seed", 0))
    samples, md, xd = diode_samples()
    print(f"DIODE transfer on {len(samples)} images")
    # FP16 DIODE baseline from the phase6 cache
    fp16 = {}
    for s in samples:
        raw = load_cached_pred(CACHE, "diode", MODEL, "fp16", s.id)
        if raw is None:
            continue
        a, _, _ = scale_shift_align(raw, s.gt_depth, s.valid_mask, mode="ss",
                                    pred_is_disparity=True, min_depth=md, max_depth=xd)
        fp16[s.id] = {"dbe_com": B.dbe(a, s.gt_edges, s.valid_mask)[1],
                      "edge_width": B.edge_sharpness(a, s.gt_edges, s.valid_mask, k=2)[1]}
    samples = [s for s in samples if s.id in fp16]

    alloc = json.load(open(RESULTS / f"{MODEL}_s2q_alloc.json"))
    k6 = min(alloc, key=lambda x: abs(float(x) - CFG["avg_bits"]))   # ~6-bit-average allocation
    s2q_bits = alloc[k6]
    print(f"using S2Q allocation at avg-bit key {k6} ({len(s2q_bits)} layers)")

    rows = []
    for name in ["uniform6", "headprot6", "s2q"]:   # uniform / head-protected at 6 bits
        model, proc, _ = load_model(MODEL, dtype=torch.float16)
        if name == "uniform6":
            quantize_model_(model, 6, per_channel=True)
        elif name == "headprot6":
            quantize_model_(model, 6, per_channel=True, extra_filters=[r"^(?!.*head\.).*"])  # all but head
        else:
            quantize_per_layer_(model, s2q_bits, per_channel=True, default_bits=16)
        acc, base = boundary_scores(model, proc, samples, fp16, md, xd)
        del model; torch.cuda.empty_cache()
        r = {"config": name, "n_img": len(acc["dbe_com"])}
        for k in acc:
            r[f"{k}_cliff"] = round(cliffs_delta(np.array(acc[k]), np.array(base[k])), 3)
        rows.append(r); print(r)
    import pandas as pd
    pd.DataFrame(rows).to_csv(RESULTS / "s2q_diode_transfer.csv", index=False)
    print("wrote results/s2q_diode_transfer.csv")


if __name__ == "__main__":
    main()
