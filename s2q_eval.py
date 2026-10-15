"""Evaluate S2Q allocations against uniform and head-protect at matched budgets.

For each average-bit budget we compare three weight-quantization configurations:
  s2q          : the per-layer allocation from allocate.py (results/<model>_s2q_alloc.json;
                 the boundary-only and aggregate-guided allocations are scored too)
  uniform      : every layer at the same (rounded) bit-width
  head_protect : output head kept at FP16, rest at the budget (Sec. 8 baseline)
On iBims-1 we score boundary metrics (DBE-completeness, boundary-F, flying pixels)
and aggregate AbsRel, all as paired changes from the cached FP16 baseline, and we
record the real weight-memory of each configuration. This is the Pareto data:
memory / avg-bit vs boundary fidelity.

Outputs results/per_image/s2q_pareto_<model>.csv and results/s2q_pareto_<model>.csv.
Commands: python s2q_eval.py --model dav2_large   (also dpt_hybrid, zoedepth; GPU)
          python s2q_eval.py --model dav2_large --from-per-image   (aggregate only, CPU)
"""
import json
import numpy as np
import pandas as pd
import torch

from common import CACHE_ROOT, PER_IMAGE, RESULTS, load_config, seed_everything

CFG = load_config("s2q_eval")
from data.loaders import build
from models.registry import load_model, infer_depth
from align import scale_shift_align
import score.boundary as B
from score.boundary_f import boundary_fscore
from score.aggregate import aggregate_metrics
from score.pointcloud import flying_pixel_fraction
from predict import load_cached_pred
from quant.fakequant import quantize_per_layer_, _QUANTIZABLE
from stats import cliffs_delta, finite_pairs

CACHE = CACHE_ROOT
import argparse
MODEL = "dav2_large"  # overridden in main()


DISP = True  # set from model kind in main


def score_one(raw, s, md, xd):
    a, _, _ = scale_shift_align(raw, s.gt_depth, s.valid_mask, mode="ss",
                               pred_is_disparity=DISP, min_depth=md, max_depth=xd)
    r = {"absrel": aggregate_metrics(a, s.gt_depth, s.valid_mask, md, xd)["absrel"]}
    _, com = B.dbe(a, s.gt_edges, s.valid_mask)
    r["dbe_com"] = com
    r["boundary_f"] = boundary_fscore(a, gt_edges=s.gt_edges, valid_mask=s.valid_mask,
                                      tols=(2,))["boundary_f_t2"]
    if s.K is not None:
        r["flying_frac"] = flying_pixel_fraction(a, s.gt_depth, s.gt_edges, s.valid_mask, k=3)["flying_frac"]
    return r


def all_layer_names(model):
    return [n for n, m in model.named_modules() if isinstance(m, _QUANTIZABLE)]


METS = ["absrel", "dbe_com", "boundary_f", "flying_frac"]


def eval_config(name, bmap, samples, md, xd, fp16):
    """Per-image rows (quantized and FP16 metrics) for one weight configuration."""
    model, proc, _ = load_model(MODEL, dtype=torch.float16)
    _, wbytes = quantize_per_layer_(model, bmap, per_channel=True, default_bits=16)
    nparam = sum(p.numel() for p in model.parameters())
    rows = []
    for s in samples:
        raw = infer_depth(model, proc, s.rgb, out_hw=s.gt_depth.shape)
        q = score_one(raw, s, md, xd)
        rows.append({"config": name, "avg_bits": round(wbytes * 8 / nparam, 3),
                     "weight_MB": round(wbytes / 1e6, 1), "image_id": s.id,
                     **{f"q_{m}": q.get(m, np.nan) for m in METS},
                     **{f"fp16_{m}": fp16[s.id].get(m, np.nan) for m in METS}})
    del model; torch.cuda.empty_cache()
    return rows


def summarize(model):
    df = pd.read_csv(PER_IMAGE / f"s2q_pareto_{model}.csv", float_precision="round_trip")
    rows = []
    for name, g in df.groupby("config", sort=False):
        out = {"config": name, "avg_bits": g.avg_bits.iloc[0], "weight_MB": g.weight_MB.iloc[0]}
        for m in METS:
            qv, fv = finite_pairs(g, m)
            if len(qv):
                out[f"{m}_mean"] = round(float(qv.mean()), 4)
                out[f"{m}_delta"] = round(float((qv - fv).mean()), 4)
                # boundary_f: higher=better, so worse-positive cliff = fp - q
                worse_hi = m != "boundary_f"
                out[f"{m}_cliff"] = round(cliffs_delta(qv, fv) if worse_hi else cliffs_delta(fv, qv), 3)
        rows.append(out)
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / f"s2q_pareto_{model}.csv", index=False)
    return out


def main():
    global MODEL, DISP
    ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True, choices=CFG["models"])
    ap.add_argument("--from-per-image", action="store_true")
    args = ap.parse_args()
    MODEL = args.model
    seed_everything(CFG.get("seed", 0))
    if args.from_per_image:
        return summarize(MODEL)
    from models.registry import MODELS
    DISP = MODELS[MODEL].get("kind", "relative") == "relative"
    gen, info = build("ibims")
    md, xd = info["min_depth"], info["max_depth"]
    # calibration split (same 40 images sensitivity.py used) is HELD OUT of eval,
    # so S2Q is never evaluated on the images its allocation was tuned on.
    CALIB_N = CFG["calib_n"]
    calib_ids = set([s.id for s in gen() if s.gt_edges is not None and s.K is not None][:CALIB_N])
    samples = [s for s in gen() if s.id not in calib_ids]
    fp16 = {}
    for s in samples:
        raw = load_cached_pred(CACHE, "ibims", MODEL, "fp16", s.id)
        if raw is not None:
            fp16[s.id] = score_one(raw, s, md, xd)
    if not fp16:  # no cache (e.g. new model): infer FP16 baseline once
        model, proc, _ = load_model(MODEL, dtype=torch.float16)
        for s in samples:
            raw = infer_depth(model, proc, s.rgb, out_hw=s.gt_depth.shape)
            fp16[s.id] = score_one(raw, s, md, xd)
        del model; torch.cuda.empty_cache()
    samples = [s for s in samples if s.id in fp16]
    print(f"eval on {len(samples)} held-out images (calibration {len(calib_ids)} excluded)")

    model, _, _ = load_model(MODEL, dtype=torch.float16); layers = all_layer_names(model)
    del model; torch.cuda.empty_cache()

    configs = []
    # S2Q (combined) + ablations: boundary-only, aggregate-only
    for tag, fname in [("s2q", f"{MODEL}_s2q_alloc.json"), ("bonly", f"{MODEL}_bonly_alloc.json"),
                       ("aggguided", f"{MODEL}_agg_alloc.json")]:
        configs += [(f"{tag}@{ab}", bmap) for ab, bmap in json.loads((RESULTS / fname).read_text()).items()]
    # uniform + head-protect at integer budgets
    for b in CFG["uniform_bits"]:
        configs += [(f"uniform@{b}", {l: b for l in layers}),
                    (f"headprot@{b}", {l: b for l in layers if "head." not in l})]
    per_img = []
    PER_IMAGE.mkdir(parents=True, exist_ok=True)
    for name, bmap in configs:
        per_img += eval_config(name, bmap, samples, md, xd, fp16)
        pd.DataFrame(per_img).to_csv(PER_IMAGE / f"s2q_pareto_{MODEL}.csv", index=False)
    return summarize(MODEL)


if __name__ == "__main__":
    print(main().to_string(index=False))
