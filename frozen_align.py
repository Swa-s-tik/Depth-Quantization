"""Alignment-protocol robustness (Table 7): is the flat aggregate an artifact of
refitting the scale+shift alignment separately for every precision?

The bit sweep is re-scored two ways on the SAME cached predictions:
  refit  = solve scale+shift on each precision independently (paper protocol)
  frozen = solve scale+shift on FP16 only and apply that transform unchanged to
           every quantized prediction, so a global shift the quantizer injects
           can no longer be absorbed.

Outputs results/per_image/frozen_align.csv (one row per model, bit-width, image)
and results/frozen_align.csv (paired deltas and Cliff's delta per model and
bit-width). `--from-per-image` rebuilds the second file from the first without
datasets or cache.
Command: python frozen_align.py
"""
import sys

import numpy as np
import pandas as pd

from common import CACHE_ROOT as CACHE, PER_IMAGE, RESULTS, load_config, seed_everything
from stats import cliffs_delta

CFG = load_config("frozen_align")
_EPS = 1e-6


def absrel(aligned, gt, valid):
    m = valid & np.isfinite(aligned) & (aligned > 0) & (gt > 0)
    if m.sum() < 10:
        return np.nan
    return float(np.mean(np.abs(aligned[m] - gt[m]) / gt[m]))


def apply_frozen(pred, s, t):
    fit = s * np.asarray(pred, np.float64) + t
    return np.where(fit > _EPS, 1.0 / np.maximum(fit, _EPS), np.inf)


def per_image():
    from data.loaders import build
    from align import scale_shift_align
    from predict import load_cached_pred
    import score.boundary as B

    gen, info = build(CFG["dataset"])
    md, xd = info["min_depth"], info["max_depth"]
    samples = [s for s in gen() if s.gt_edges is not None]
    rows = []
    for model in CFG["models"]:
        # FP16 anchor: refit alignment + its (s,t) per image
        st, fp_ar, fp_dc = {}, {}, {}
        for s in samples:
            raw = load_cached_pred(CACHE, CFG["dataset"], model, "fp16", s.id)
            if raw is None:
                continue
            a, (sc, tt), _ = scale_shift_align(raw, s.gt_depth, s.valid_mask, mode="ss",
                                               pred_is_disparity=True, min_depth=md, max_depth=xd)
            st[s.id] = (sc, tt)
            fp_ar[s.id] = absrel(a, s.gt_depth, s.valid_mask)
            fp_dc[s.id] = B.dbe(a, s.gt_edges, s.valid_mask)[1]
        for bits in CFG["bits"]:
            for s in samples:
                q = load_cached_pred(CACHE, CFG["dataset"], model, f"w{bits}", s.id)
                if q is None or s.id not in st or not np.isfinite(fp_ar.get(s.id, np.nan)):
                    continue
                ar, _, _ = scale_shift_align(q, s.gt_depth, s.valid_mask, mode="ss",
                                             pred_is_disparity=True, min_depth=md, max_depth=xd)
                af = apply_frozen(q, *st[s.id])
                v_ar_r, v_ar_f = absrel(ar, s.gt_depth, s.valid_mask), absrel(af, s.gt_depth, s.valid_mask)
                if not (np.isfinite(v_ar_r) and np.isfinite(v_ar_f)):
                    continue
                rows.append({"model": model, "w_bits": bits, "image_id": s.id,
                             "absrel_fp16": fp_ar[s.id], "absrel_refit": v_ar_r,
                             "absrel_frozen": v_ar_f, "dbe_com_fp16": fp_dc[s.id],
                             "dbe_com_refit": B.dbe(ar, s.gt_edges, s.valid_mask)[1],
                             "dbe_com_frozen": B.dbe(af, s.gt_edges, s.valid_mask)[1]})
    PER_IMAGE.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(PER_IMAGE / "frozen_align.csv", index=False)


def summarize():
    df = pd.read_csv(PER_IMAGE / "frozen_align.csv", float_precision="round_trip")
    rows = []
    for model in CFG["models"]:
        for bits in CFG["bits"]:
            R = df[(df.model == model) & (df.w_bits == bits)]
            if len(R) < 10:
                continue
            b_ar, b_dc = R.absrel_fp16.to_numpy(), R.dbe_com_fp16.to_numpy()
            rows.append({
                "model": model, "w_bits": bits, "n": len(R),
                # aggregate AbsRel: paired delta vs FP16 + effect size, both alignments
                "absrel_refit_delta": round(float((R.absrel_refit.to_numpy() - b_ar).mean()), 5),
                "absrel_frozen_delta": round(float((R.absrel_frozen.to_numpy() - b_ar).mean()), 5),
                "absrel_refit_cliff": round(cliffs_delta(R.absrel_refit.to_numpy(), b_ar), 3),
                "absrel_frozen_cliff": round(cliffs_delta(R.absrel_frozen.to_numpy(), b_ar), 3),
                # boundary DBE-completeness: effect size under both alignments
                "dbe_refit_cliff": round(cliffs_delta(R.dbe_com_refit.to_numpy(), b_dc), 3),
                "dbe_frozen_cliff": round(cliffs_delta(R.dbe_com_frozen.to_numpy(), b_dc), 3),
            })
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "frozen_align.csv", index=False)
    return out


if __name__ == "__main__":
    seed_everything(CFG.get("seed", 0))
    if "--from-per-image" not in sys.argv:
        per_image()
    print(summarize().to_string(index=False))
