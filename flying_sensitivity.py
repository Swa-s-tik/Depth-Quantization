"""Flying-pixel metric validity (Sec. 14.3, Appendix A).

On DAv2-Large / iBims-1, from the cache: the FP16 flying-pixel fraction, the
eligible (n_edge_band) and flying (n_flying) pixel counts per image, and the
robustness of the FP16->W6 effect size to the window half-width k and the noise
floor tau.

Outputs results/per_image/flying_sensitivity.csv (k, tau, image) and
results/flying_sensitivity.csv. `--from-per-image` rebuilds the second file
without datasets or cache.
Command: python flying_sensitivity.py
"""
import sys

import numpy as np
import pandas as pd

from common import CACHE_ROOT as CACHE, PER_IMAGE, RESULTS, load_config, seed_everything
from stats import cliffs_delta

CFG = load_config("flying_sensitivity")


def per_image():
    from data.loaders import build
    from align import scale_shift_align
    from predict import load_cached_pred
    from score.pointcloud import flying_pixel_fraction

    gen, info = build(CFG["dataset"])
    md, xd = info["min_depth"], info["max_depth"]
    samples = [s for s in gen() if s.gt_edges is not None]

    def aligned(tag, s):
        raw = load_cached_pred(CACHE, CFG["dataset"], CFG["model"], tag, s.id)
        if raw is None:
            return None
        a, _, _ = scale_shift_align(raw, s.gt_depth, s.valid_mask, mode="ss",
                                    pred_is_disparity=True, min_depth=md, max_depth=xd)
        return a

    rows = []
    for k in CFG["ks"]:
        for tau in CFG["taus"]:
            t = None if tau == "auto" else tau
            for s in samples:
                af, aq = aligned("fp16", s), aligned(CFG["quant_tag"], s)
                if af is None or aq is None:
                    continue
                rf = flying_pixel_fraction(af, s.gt_depth, s.gt_edges, s.valid_mask, k=k, tau=t)
                rq = flying_pixel_fraction(aq, s.gt_depth, s.gt_edges, s.valid_mask, k=k, tau=t)
                rows.append({"k": k, "tau": tau, "image_id": s.id,
                             "fp16_flying_frac": rf["flying_frac"],
                             "quant_flying_frac": rq["flying_frac"],
                             "fp16_n_eligible": rf["n_edge_band"],
                             "fp16_n_flying": rf["n_flying"]})
    PER_IMAGE.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(PER_IMAGE / "flying_sensitivity.csv", index=False)


def summarize():
    df = pd.read_csv(PER_IMAGE / "flying_sensitivity.csv", float_precision="round_trip",
                     dtype={"tau": str})
    rows = []
    for k in CFG["ks"]:
        for tau in CFG["taus"]:
            g = df[(df.k == k) & (df.tau == str(tau))]
            g = g[np.isfinite(g.fp16_flying_frac) & np.isfinite(g.quant_flying_frac)]
            if len(g) < 10:
                continue
            rows.append({"k": k, "tau": tau, "n_img": len(g),
                         "fp16_flying_frac": round(float(g.fp16_flying_frac.mean()), 3),
                         "mean_eligible_px": int(g.fp16_n_eligible.mean()),
                         "mean_flying_px": int(g.fp16_n_flying.mean()),
                         f"{CFG['quant_tag']}_cliff": round(cliffs_delta(g.quant_flying_frac.to_numpy(),
                                                                         g.fp16_flying_frac.to_numpy()), 3)})
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "flying_sensitivity.csv", index=False)
    return out


if __name__ == "__main__":
    seed_everything(CFG.get("seed", 0))
    if "--from-per-image" not in sys.argv:
        per_image()
    print(summarize().to_string(index=False))
