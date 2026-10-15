"""Generality across quantization schemes (Table 20).

Does the aggregate-vs-boundary dissociation depend on the specific quantization
scheme, or does it hold across schemes? We run DAv2-Large on iBims-1 under four
weight-only schemes at 8/6/4 bits and report the paired FP16->quant effect size
(Cliff's delta, positive = worse) for one aggregate metric (AbsRel) and the
boundary/structural metrics. The FP16 baseline is read from the cache.

Schemes:
  pc_sym  : per-channel symmetric (the paper's primary scheme)
  pt_sym  : per-tensor symmetric
  pc_asym : per-channel asymmetric (unsigned min-max)
  pc_clip : per-channel symmetric with a 0.999 percentile clip
Outputs results/per_image/alt_ptq.csv and results/alt_ptq.csv.
Commands: python alt_ptq_sweep.py                   (GPU)
          python alt_ptq_sweep.py --from-per-image  (aggregate only, CPU)
"""
import gc
import sys

import numpy as np
import pandas as pd
import torch

from common import CACHE_ROOT, PER_IMAGE, RESULTS, load_config, seed_everything
from data.loaders import build
from models.registry import load_model, infer_depth
from align import scale_shift_align
from predict import load_cached_pred
import score.boundary as B
from score.aggregate import aggregate_metrics
from score.pointcloud import flying_pixel_fraction
from quant.fakequant import quantize_model_
from stats import cliffs_delta, finite_pairs, wilcoxon_p

CFG = load_config("alt_ptq")
CACHE = CACHE_ROOT
MODEL = CFG["model"]
BITS = CFG["bits"]
SCHEMES = CFG["schemes"]
METRICS = ["absrel", "dbe_com", "edge_width", "flying_frac"]


def score_one(raw, s, md, xd):
    a, _, _ = scale_shift_align(raw, s.gt_depth, s.valid_mask, mode="ss",
                               pred_is_disparity=True, min_depth=md, max_depth=xd)
    r = {"absrel": aggregate_metrics(a, s.gt_depth, s.valid_mask, md, xd)["absrel"]}
    if s.gt_edges is not None:
        _, com = B.dbe(a, s.gt_edges, s.valid_mask)
        _, w = B.edge_sharpness(a, s.gt_edges, s.valid_mask, k=2)
        r.update(dbe_com=com, edge_width=w)
        if s.K is not None:
            r["flying_frac"] = flying_pixel_fraction(a, s.gt_depth, s.gt_edges, s.valid_mask, k=3)["flying_frac"]
    return r


def main():
    seed_everything(CFG.get("seed", 0))
    gen, info = build("ibims")
    md, xd = info["min_depth"], info["max_depth"]
    samples = list(gen())

    # FP16 baseline from cache
    fp16 = {}
    for s in samples:
        raw = load_cached_pred(CACHE, "ibims", MODEL, "fp16", s.id)
        if raw is not None:
            fp16[s.id] = score_one(raw, s, md, xd)
    ids = list(fp16)
    by_id = {s.id: s for s in samples}
    print(f"FP16 baseline from cache: {len(ids)} images")

    rows = []
    PER_IMAGE.mkdir(parents=True, exist_ok=True)
    for scheme, cfg in SCHEMES.items():
        for bits in BITS:
            model, proc, _ = load_model(MODEL, dtype=torch.float16)
            quantize_model_(model, bits, per_channel=cfg["per_channel"],
                            symmetric=cfg["symmetric"], clip_percentile=cfg["clip_percentile"])
            for i in ids:
                s = by_id[i]
                q = score_one(infer_depth(model, proc, s.rgb, out_hw=s.gt_depth.shape), s, md, xd)
                rows.append({"scheme": scheme, "w_bits": bits, "image_id": i,
                             **{f"q_{m}": q.get(m, np.nan) for m in METRICS},
                             **{f"fp16_{m}": fp16[i].get(m, np.nan) for m in METRICS}})
            del model; gc.collect(); torch.cuda.empty_cache()
            pd.DataFrame(rows).to_csv(PER_IMAGE / "alt_ptq.csv", index=False)
    return summarize()


def summarize():
    df = pd.read_csv(PER_IMAGE / "alt_ptq.csv", float_precision="round_trip")
    rows = []
    for (scheme, bits), g in df.groupby(["scheme", "w_bits"], sort=False):
        for met in METRICS:
            q, f = finite_pairs(g, met)
            if len(q) < 5:
                continue
            rows.append({"scheme": scheme, "w_bits": bits, "metric": met,
                         "fp16_mean": round(float(np.nanmean(f)), 4),
                         "quant_mean": round(float(np.nanmean(q)), 4),
                         "mean_delta": round(float(np.nanmean(q - f)), 5),
                         "cliffs_delta": round(cliffs_delta(q, f), 3),
                         "wilcoxon_p": wilcoxon_p(q - f)})
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "alt_ptq.csv", index=False)
    return out


if __name__ == "__main__":
    print((summarize() if "--from-per-image" in sys.argv else main()).to_string(index=False))
