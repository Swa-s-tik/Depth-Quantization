"""Boundary accuracy of real INT8 kernels (Table 22): torchao dynamic per-token
int8 activations + int8 weights (W8A8), against FP16.

For each model we run FP16 and real-INT8 inference on iBims-1, align per image,
and compute the aggregate + boundary + flying-pixel suite. We report the paired
FP16->INT8 effect size (Cliff's delta, positive = worse).
Output results/int8_real.csv (per image) and results/int8_real_summary.csv.
Commands: python predict_score_int8real.py                    (GPU)
          python predict_score_int8real.py --from-per-image   (summary only, CPU)
"""
import sys, gc
import numpy as np
import pandas as pd
import torch

from common import RESULTS, load_config, seed_everything
from data.loaders import build
from models.registry import load_model, infer_depth
from align import scale_shift_align
import score.boundary as B
from score.aggregate import aggregate_metrics
from score.pointcloud import flying_pixel_fraction
from stats import cliffs_delta, wilcoxon_p

CFG = load_config("int8_real")
MODELS = CFG["models"]
WORSE_HIGHER = ["absrel", "dbe_com", "edge_width", "flying_frac", "boundary_absrel"]


def score_one(pred_raw, s, md, xd):
    aligned, _, _ = scale_shift_align(pred_raw, s.gt_depth, s.valid_mask,
                                      mode="ss", pred_is_disparity=True,
                                      min_depth=md, max_depth=xd)
    row = {"absrel": aggregate_metrics(aligned, s.gt_depth, s.valid_mask, md, xd)["absrel"]}
    if s.gt_edges is not None:
        acc, com = B.dbe(aligned, s.gt_edges, s.valid_mask)
        _, width = B.edge_sharpness(aligned, s.gt_edges, s.valid_mask, k=2)
        bm = B.boundary_masked_error(aligned, s.gt_depth, s.gt_edges, s.valid_mask, k=3)
        row.update(dbe_com=com, edge_width=width, boundary_absrel=bm["boundary_absrel"])
        if s.K is not None:
            row["flying_frac"] = flying_pixel_fraction(
                aligned, s.gt_depth, s.gt_edges, s.valid_mask, k=3)["flying_frac"]
    return row


def run_model(name, samples, md, xd):
    """Return list of per-image dicts for fp16 and int8_real."""
    from torchao.quantization import quantize_, Int8DynamicActivationInt8WeightConfig
    out = []
    # FP16 pass
    model, proc, _ = load_model(name, dtype=torch.float16)
    fp16 = {}
    for s in samples:
        raw = infer_depth(model, proc, s.rgb, out_hw=s.gt_depth.shape)
        fp16[s.id] = score_one(raw, s, md, xd)
    del model; gc.collect(); torch.cuda.empty_cache()
    # real INT8 pass
    model, proc, _ = load_model(name, dtype=torch.float16)
    quantize_(model, Int8DynamicActivationInt8WeightConfig())
    for s in samples:
        raw = infer_depth(model, proc, s.rgb, out_hw=s.gt_depth.shape)
        i8 = score_one(raw, s, md, xd)
        rec = {"model": name, "id": s.id}
        for k in set(fp16[s.id]) | set(i8):
            rec[f"fp16_{k}"] = fp16[s.id].get(k, np.nan)
            rec[f"int8_{k}"] = i8.get(k, np.nan)
        out.append(rec)
    del model; gc.collect(); torch.cuda.empty_cache()
    return out


def summarize(df):
    summ = []
    for name in MODELS:
        d_m = df[df.model == name]
        for met in WORSE_HIGHER:
            fc, ic = f"fp16_{met}", f"int8_{met}"
            if fc not in d_m or ic not in d_m:
                continue
            f = d_m[fc].to_numpy(); i = d_m[ic].to_numpy()
            d = i - f
            cliff = cliffs_delta(i, f)             # positive = int8 worse
            summ.append({"model": name, "metric": met,
                         "fp16_mean": round(np.nanmean(f), 4),
                         "int8_mean": round(np.nanmean(i), 4),
                         "mean_delta": round(np.nanmean(d), 4),
                         "cliffs_delta": round(cliff, 3),
                         "wilcoxon_p": wilcoxon_p(d)})
    out = pd.DataFrame(summ)
    out.to_csv(RESULTS / "int8_real_summary.csv", index=False)
    return out


def main():
    gen, info = build("ibims")
    md, xd = info["min_depth"], info["max_depth"]
    samples = list(gen())
    all_rows = []
    for name in MODELS:
        print(f"== {name} ==")
        all_rows += run_model(name, samples, md, xd)
    df = pd.DataFrame(all_rows)
    df.to_csv(RESULTS / "int8_real.csv", index=False)
    return summarize(df)


if __name__ == "__main__":
    seed_everything(CFG.get("seed", 0))
    if "--from-per-image" in sys.argv:
        out = summarize(pd.read_csv(RESULTS / "int8_real.csv", float_precision="round_trip"))
    else:
        out = main()
    print(out.to_string(index=False))
