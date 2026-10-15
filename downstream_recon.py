"""Downstream task: does quantization-induced boundary damage break 3D reconstruction?

We back-project each aligned depth map into a camera-frame point cloud and compare
it to the ground-truth point cloud (from GT depth + intrinsics). We report, near
object boundaries, the Chamfer distance and a reconstruction F-score at a distance
threshold (fraction of predicted points within tau of a GT point, and vice versa).
This is a geometric downstream quantity distinct from the flying-pixel fraction:
flying pixels count points in the fg/bg gap, whereas reconstruction-F measures
overall cloud accuracy near boundaries. We report paired FP16->quant changes so any
absolute reconstruction error cancels.

The FP16 baseline is read from the cache; quantized models are uniform,
head-protected and the S2Q allocations of results/<model>_s2q_alloc.json.
Output: results/downstream_recon.csv
Command: python downstream_recon.py
"""
import json
import numpy as np
import pandas as pd
import torch
from scipy.spatial import cKDTree
from scipy import ndimage

from common import CACHE_ROOT, RESULTS, load_config, seed_everything

CFG = load_config("downstream_recon")
from data.loaders import build
from models.registry import load_model, infer_depth
from align import scale_shift_align
from score.pointcloud import backproject
from predict import load_cached_pred
from quant.fakequant import quantize_per_layer_, _QUANTIZABLE
from stats import cliffs_delta

CACHE = CACHE_ROOT
MODEL = CFG["model"]
TAU = CFG["tau"]  # reconstruction-F threshold in meters


def recon_metrics(pred_depth, s, md, xd):
    """Chamfer + F-score near boundaries between predicted and GT clouds."""
    a, _, _ = scale_shift_align(pred_depth, s.gt_depth, s.valid_mask, mode="ss",
                                pred_is_disparity=True, min_depth=md, max_depth=xd)
    if s.K is None or s.gt_edges is None:
        return {}
    band = ndimage.binary_dilation(s.gt_edges.astype(bool), iterations=CFG["band_px"]) & s.valid_mask
    m = band & np.isfinite(a) & (a > 0) & np.isfinite(s.gt_depth) & (s.gt_depth > 0)
    if m.sum() < 50:
        return {}
    P = backproject(np.where(m, a, 0.0), s.K).reshape(-1, 3)[m.ravel()]
    G = backproject(np.where(m, s.gt_depth, 0.0), s.K).reshape(-1, 3)[m.ravel()]
    if len(P) < 10 or len(G) < 10:
        return {}
    tp, tg = cKDTree(P), cKDTree(G)
    dpg, _ = tg.query(P)      # pred -> nearest gt
    dgp, _ = tp.query(G)      # gt -> nearest pred
    chamfer = float(dpg.mean() + dgp.mean())
    prec = float((dpg <= TAU).mean()); rec = float((dgp <= TAU).mean())
    f = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return {"chamfer": chamfer, "recon_f": f}


def eval_cfg(name, bmap, samples, md, xd, fp16):
    model, proc, _ = load_model(MODEL, dtype=torch.float16)
    if bmap is not None:
        quantize_per_layer_(model, bmap, per_channel=True, default_bits=16)
    acc = {"chamfer": [], "recon_f": []}
    for s in samples:
        raw = infer_depth(model, proc, s.rgb, out_hw=s.gt_depth.shape)
        r = recon_metrics(raw, s, md, xd)
        if r and s.id in fp16 and fp16[s.id]:
            for k in acc:
                acc[k].append((r[k], fp16[s.id][k]))
    del model; torch.cuda.empty_cache()
    out = {"config": name}
    for k in acc:
        if acc[k]:
            q = np.array([x[0] for x in acc[k]]); fv = np.array([x[1] for x in acc[k]])
            out[f"{k}_delta"] = round(float((q - fv).mean()), 4)
            worse_hi = k == "chamfer"     # chamfer higher=worse; recon_f higher=better
            out[f"{k}_cliff"] = round(cliffs_delta(q, fv) if worse_hi else cliffs_delta(fv, q), 3)
    return out


def main():
    seed_everything(CFG.get("seed", 0))
    gen, info = build("ibims")
    md, xd = info["min_depth"], info["max_depth"]
    calib = set([s.id for s in gen() if s.gt_edges is not None and s.K is not None][:CFG["calib_n"]])
    samples = [s for s in gen() if s.id not in calib]
    fp16 = {}
    for s in samples:
        raw = load_cached_pred(CACHE, "ibims", MODEL, "fp16", s.id)
        fp16[s.id] = recon_metrics(raw, s, md, xd) if raw is not None else {}
    samples = [s for s in samples if fp16.get(s.id)]
    print(f"downstream on {len(samples)} held-out images")

    model, _, _ = load_model(MODEL, dtype=torch.float16)
    layers = [n for n, m in model.named_modules() if isinstance(m, _QUANTIZABLE)]
    del model; torch.cuda.empty_cache()

    rows = []
    for b in CFG["bits"]:
        rows.append(eval_cfg(f"uniform@{b}", {l: b for l in layers}, samples, md, xd, fp16)); print(rows[-1])
        rows.append(eval_cfg(f"headprot@{b}", {l: b for l in layers if "head." not in l}, samples, md, xd, fp16)); print(rows[-1])
        pd.DataFrame(rows).to_csv(RESULTS / "downstream_recon.csv", index=False)
    # S2Q allocations (allocate.py)
    ap = RESULTS / f"{MODEL}_s2q_alloc.json"
    if ap.exists():
        for abk, bmap in json.loads(ap.read_text()).items():
            rows.append(eval_cfg(f"s2q@{abk}", bmap, samples, md, xd, fp16)); print(rows[-1])
    pd.DataFrame(rows).to_csv(RESULTS / "downstream_recon.csv", index=False)
    print("wrote results/downstream_recon.csv")


if __name__ == "__main__":
    main()
