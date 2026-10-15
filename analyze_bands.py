"""Error vs distance from the nearest boundary (Fig. 3, Sec. 6.1).

From cached iBims-1 predictions (no inference): every pixel gets its Euclidean
distance r to the annotated edge set; AbsRel is computed per image inside each
distance band (lo, hi], and averaged over images per band.

Outputs results/per_image/bands.csv (model, precision, image, band) and
results/bands.csv (mean AbsRel per band). `--from-per-image` rebuilds the second
file without datasets or cache.
Command: python analyze_bands.py
"""
import sys

import numpy as np
import pandas as pd

from common import CACHE_ROOT as CACHE, PER_IMAGE, RESULTS, load_config, seed_everything

CFG = load_config("bands")


def absrel(pred, gt, mask, md, xd):
    m = mask & np.isfinite(pred) & (pred > 0) & np.isfinite(gt) & (gt > md) & (gt < xd)
    if m.sum() < 5:
        return np.nan, 0
    p = np.clip(pred[m], md, xd)
    return float(np.mean(np.abs(p - gt[m]) / gt[m])), int(m.sum())


def per_image():
    from scipy import ndimage
    from data.loaders import build
    from align import scale_shift_align
    from predict import load_cached_pred

    gen, info = build(CFG["dataset"])
    md, xd = info["min_depth"], info["max_depth"]
    bands = CFG["bands"]
    rows = []
    for s in gen():
        if s.gt_edges is None:
            continue
        dist = ndimage.distance_transform_edt(~s.gt_edges.astype(bool))
        for m in CFG["models"]:
            preds = {}
            for p in CFG["precisions"]:
                raw = load_cached_pred(CACHE, CFG["dataset"], m, p, s.id)
                if raw is not None:
                    preds[p], _, _ = scale_shift_align(raw, s.gt_depth, s.valid_mask, mode="ss",
                                                       pred_is_disparity=True,
                                                       min_depth=md, max_depth=xd)
            if len(preds) < len(CFG["precisions"]):
                continue
            for p in CFG["precisions"]:
                for label, lo, hi in bands:
                    ring = s.valid_mask & (dist > lo) & (dist <= hi)
                    e, n = absrel(preds[p], s.gt_depth, ring, md, xd)
                    rows.append({"model": m, "precision": p, "image_id": s.id,
                                 "band": label, "absrel": e, "n_px": n})
    PER_IMAGE.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(PER_IMAGE / "bands.csv", index=False)


def summarize():
    df = pd.read_csv(PER_IMAGE / "bands.csv", float_precision="round_trip")
    labels = [b[0] for b in CFG["bands"]]
    ok = df[np.isfinite(df.absrel) & df.band.isin(labels)]
    g = ok.groupby(["model", "precision", "band"], sort=False).absrel
    out = pd.DataFrame({"mean_absrel": g.mean().round(5), "n_images": g.size()}).reset_index()
    out = out.sort_values(["model", "precision", "band"])
    out.to_csv(RESULTS / "bands.csv", index=False)
    return out


if __name__ == "__main__":
    seed_everything(CFG.get("seed", 0))
    if "--from-per-image" not in sys.argv:
        per_image()
    print(summarize().to_string(index=False))
