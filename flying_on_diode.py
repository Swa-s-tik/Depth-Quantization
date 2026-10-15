"""Flying-pixel fraction on DIODE indoors (Table 16, last column).

Algorithm 1 uses only depth values, a pixel window and a metric tolerance, so it
needs no camera intrinsics and is computable on DIODE with the derived edges.
Scored from the phase6_diode_bitsweep cache for DAv2 Base and Large.

Outputs results/per_image/flying_on_diode.csv and results/flying_on_diode.csv.
`--from-per-image` rebuilds the second file without datasets or cache.
Command: python flying_on_diode.py
"""
import sys

import numpy as np
import pandas as pd

from common import CACHE_ROOT as CACHE, PER_IMAGE, RESULTS, load_config, seed_everything
from stats import cliffs_delta

CFG = load_config("flying_on_diode")


def per_image():
    from data.loaders import build
    from align import scale_shift_align
    from predict import load_cached_pred
    from score.pointcloud import flying_pixel_fraction

    gen, info = build(CFG["dataset"], subset=CFG["subset"])
    md, xd = info["min_depth"], info["max_depth"]
    samples = [s for s in gen() if s.gt_edges is not None]
    rows = []
    for model in CFG["models"]:
        def fly(tag, s):
            raw = load_cached_pred(CACHE, CFG["dataset"], model, tag, s.id)
            if raw is None:
                return np.nan
            a, _, _ = scale_shift_align(raw, s.gt_depth, s.valid_mask, mode="ss",
                                        pred_is_disparity=True, min_depth=md, max_depth=xd)
            return flying_pixel_fraction(a, s.gt_depth, s.gt_edges, s.valid_mask, k=3)["flying_frac"]
        for bits in CFG["bits"]:
            for s in samples:
                rows.append({"model": model, "w_bits": bits, "image_id": s.id,
                             "fp16_flying": fly("fp16", s), "quant_flying": fly(f"w{bits}", s)})
    PER_IMAGE.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(PER_IMAGE / "flying_on_diode.csv", index=False)


def summarize():
    df = pd.read_csv(PER_IMAGE / "flying_on_diode.csv", float_precision="round_trip")
    rows = []
    for model in CFG["models"]:
        for bits in CFG["bits"]:
            g = df[(df.model == model) & (df.w_bits == bits)]
            g = g[np.isfinite(g.fp16_flying) & np.isfinite(g.quant_flying)]
            if len(g) < 10:
                continue
            fp, q = g.fp16_flying.to_numpy(), g.quant_flying.to_numpy()
            rows.append({"model": model, "w_bits": bits, "n": len(fp),
                         "fp16_flying": round(float(fp.mean()), 3),
                         "quant_flying": round(float(q.mean()), 3),
                         "flying_cliff": round(cliffs_delta(q, fp), 3)})
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "flying_on_diode.csv", index=False)
    return out


if __name__ == "__main__":
    seed_everything(CFG.get("seed", 0))
    if "--from-per-image" not in sys.argv:
        per_image()
    print(summarize().to_string(index=False))
