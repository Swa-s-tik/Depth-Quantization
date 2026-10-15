"""Robustness of the six-bit boundary effect to the boundary definition (Table 8).

Recomputes the FP16->W6 effect from the cache at band widths k in {1,2,3,5,7}
(band-AbsRel) and at flying-pixel gap factors gamma in {1.5,2,3}, so the finding
cannot be an artifact of one threshold.

Outputs results/per_image/boundary_definition.csv and
results/boundary_sensitivity.csv. `--from-per-image` rebuilds the second file
without datasets or cache.
Command: python boundary_definition.py
"""
import sys

import numpy as np
import pandas as pd

from common import CACHE_ROOT as CACHE, PER_IMAGE, RESULTS, load_config, seed_everything
from stats import cliffs_delta, wilcoxon_p

CFG = load_config("boundary_definition")


def _params():
    return ([("bnd_k", k) for k in CFG["band_ks"]] +
            [("fly_g", g) for g in CFG["gap_factors"]])


def per_image():
    from data.loaders import build
    from align import scale_shift_align
    from predict import load_cached_pred
    import score.boundary as B
    from score.pointcloud import flying_pixel_fraction

    gen, info = build(CFG["dataset"])
    md, xd = info["min_depth"], info["max_depth"]
    samples = [s for s in gen() if s.gt_edges is not None]
    tags = ("fp16", CFG["quant_tag"])

    def aligned(s, model, tag):
        raw = load_cached_pred(CACHE, CFG["dataset"], model, tag, s.id)
        if raw is None:
            return None
        dep, _, _ = scale_shift_align(raw, s.gt_depth, s.valid_mask, mode="ss",
                                      pred_is_disparity=True, min_depth=md, max_depth=xd)
        return dep

    rows = []
    for model in CFG["models"]:
        for s in samples:
            preds = {t: aligned(s, model, t) for t in tags}
            if any(v is None for v in preds.values()):
                continue
            for kind, param in _params():
                if kind == "fly_g" and s.K is None:
                    continue
                v = {}
                for t in tags:
                    if kind == "bnd_k":
                        v[t] = B.boundary_masked_error(preds[t], s.gt_depth, s.gt_edges, s.valid_mask,
                                                       k=param, min_depth=md, max_depth=xd)["boundary_absrel"]
                    else:
                        v[t] = flying_pixel_fraction(preds[t], s.gt_depth, s.gt_edges, s.valid_mask,
                                                     k=3, min_gap_factor=param)["flying_frac"]
                rows.append({"model": model, "definition": kind, "param": float(param),
                             "image_id": s.id, "fp16": v["fp16"], "quant": v[CFG["quant_tag"]]})
    PER_IMAGE.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(PER_IMAGE / "boundary_definition.csv", index=False)


def summarize():
    df = pd.read_csv(PER_IMAGE / "boundary_definition.csv", float_precision="round_trip")
    tag = CFG["quant_tag"]
    rows = []
    for model in CFG["models"]:
        for kind, param in _params():
            g = df[(df.model == model) & (df.definition == kind) & (df.param == float(param))]
            f, q = g.fp16.to_numpy(), g.quant.to_numpy()
            m = np.isfinite(f) & np.isfinite(q)
            if m.sum() < 5:
                continue
            rows.append({"model": model, "definition": kind, "param": float(param),
                         "fp16_mean": round(np.nanmean(f[m]), 4),
                         f"{tag}_mean": round(np.nanmean(q[m]), 4),
                         "cliffs_delta": round(cliffs_delta(q[m], f[m]), 3),
                         "wilcoxon_p": wilcoxon_p(q[m] - f[m])})
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "boundary_sensitivity.csv", index=False)
    return out


if __name__ == "__main__":
    seed_everything(CFG.get("seed", 0))
    if "--from-per-image" not in sys.argv:
        per_image()
    print(summarize().to_string(index=False))
