"""Established contour F-measure across the weight bit-width sweep (Sec. 6.2).

Precision/recall of predicted vs annotated depth edges at a pixel tolerance, so
the six-bit result can be read on an established boundary measure and not only
on the depth-boundary and flying-pixel measures. Scored from the cache.

Outputs results/per_image/boundary_f.csv (model, bit-width, tolerance, image)
and results/boundary_f_sweep_t<tol>.csv per tolerance (FP16 and quantized mean F,
paired delta with bootstrap CI, Cliff's delta with positive = worse, Wilcoxon p).
`--from-per-image` rebuilds the aggregate files without datasets or cache.
Command: python boundary_f_sweep.py
"""
import sys

import numpy as np
import pandas as pd

from common import CACHE_ROOT as CACHE, PER_IMAGE, RESULTS, load_config, seed_everything
from stats import cliffs_delta, wilcoxon_p, bootstrap_ci

CFG = load_config("boundary_f")


def per_image():
    from data.loaders import build
    from align import scale_shift_align
    from predict import load_cached_pred
    from score.boundary_f import boundary_fscore

    gen, info = build(CFG["dataset"])
    md, xd = info["min_depth"], info["max_depth"]
    samples = [s for s in gen() if s.gt_edges is not None]
    tols = tuple(CFG["tolerances"])

    def f_at(raw, s):
        a, _, _ = scale_shift_align(raw, s.gt_depth, s.valid_mask, mode="ss",
                                    pred_is_disparity=True, min_depth=md, max_depth=xd)
        r = boundary_fscore(a, gt_edges=s.gt_edges, valid_mask=s.valid_mask, tols=tols)
        return [r[f"boundary_f_t{t}"] for t in tols]

    rows = []
    for model in CFG["models"]:
        for bits in [16] + CFG["bits"]:
            tag = "fp16" if bits == 16 else f"w{bits}"
            for s in samples:
                raw = load_cached_pred(CACHE, CFG["dataset"], model, tag, s.id)
                if raw is None:
                    continue
                for t, f in zip(tols, f_at(raw, s)):
                    rows.append({"model": model, "w_bits": bits, "tol": t,
                                 "image_id": s.id, "boundary_f": f})
    PER_IMAGE.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(PER_IMAGE / "boundary_f.csv", index=False)


def summarize():
    df = pd.read_csv(PER_IMAGE / "boundary_f.csv", float_precision="round_trip")
    outs = {}
    for tol in CFG["tolerances"]:
        rows = []
        for model in CFG["models"]:
            sub = df[(df.model == model) & (df.tol == tol)]
            base = sub[sub.w_bits == 16].set_index("image_id").boundary_f
            for bits in CFG["bits"]:
                q = sub[sub.w_bits == bits]
                keep = [(fq, base[i]) for i, fq in zip(q.image_id, q.boundary_f)
                        if i in base.index and np.isfinite(base[i]) and np.isfinite(fq)]
                if len(keep) < 5:
                    continue
                f = np.array([k[0] for k in keep]); b = np.array([k[1] for k in keep]); d = f - b
                mean, lo, hi = bootstrap_ci(d, ci=CFG["ci"])
                # boundary-F: higher = better, so worse-positive effect = cliff(fp16, quant)
                rows.append({"model": model, "w_bits": bits,
                             "fp16_F": round(float(b.mean()), 4), "quant_F": round(float(f.mean()), 4),
                             "mean_delta": round(float(d.mean()), 4),
                             "ci_lo": round(lo, 4), "ci_hi": round(hi, 4),
                             "cliffs_delta": round(cliffs_delta(b, f), 3),
                             "wilcoxon_p": wilcoxon_p(d)})
        outs[tol] = pd.DataFrame(rows)
        outs[tol].to_csv(RESULTS / f"boundary_f_sweep_t{tol}.csv", index=False)
    return outs


if __name__ == "__main__":
    seed_everything(CFG.get("seed", 0))
    if "--from-per-image" not in sys.argv:
        per_image()
    for tol, out in summarize().items():
        print(f"tolerance {tol} px\n{out.to_string(index=False)}\n")
