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
Outputs results/per_image/downstream_recon.csv and results/downstream_recon.csv.
Commands: python downstream_recon.py                   (GPU)
          python downstream_recon.py --from-per-image  (aggregate only, CPU)
"""
import json
import sys

import numpy as np
import pandas as pd
import torch
from scipy.spatial import cKDTree
from scipy import ndimage

from common import CACHE_ROOT, PER_IMAGE, RESULTS, load_config, seed_everything
from data.loaders import build
from models.registry import load_model, infer_depth
from align import scale_shift_align
from score.pointcloud import backproject
from predict import load_cached_pred
from quant.fakequant import quantize_per_layer_, _QUANTIZABLE
from stats import cliffs_delta, finite_pairs

CFG = load_config("downstream_recon")

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


METRICS = ["chamfer", "recon_f"]


def eval_cfg(name, bmap, samples, md, xd, fp16):
    model, proc, _ = load_model(MODEL, dtype=torch.float16)
    if bmap is not None:
        quantize_per_layer_(model, bmap, per_channel=True, default_bits=16)
    rows = []
    for s in samples:
        raw = infer_depth(model, proc, s.rgb, out_hw=s.gt_depth.shape)
        r = recon_metrics(raw, s, md, xd)
        if r and s.id in fp16 and fp16[s.id]:
            rows.append({"config": name, "image_id": s.id, **{f"q_{k}": r[k] for k in METRICS},
                         **{f"fp16_{k}": fp16[s.id][k] for k in METRICS}})
    del model; torch.cuda.empty_cache()
    return rows


def summarize():
    df = pd.read_csv(PER_IMAGE / "downstream_recon.csv", float_precision="round_trip")
    rows = []
    for name, g in df.groupby("config", sort=False):
        out = {"config": name}
        for k in METRICS:
            q, fv = finite_pairs(g, k)
            if len(q):
                out[f"{k}_delta"] = round(float((q - fv).mean()), 4)
                worse_hi = k == "chamfer"     # chamfer higher=worse; recon_f higher=better
                out[f"{k}_cliff"] = round(cliffs_delta(q, fv) if worse_hi else cliffs_delta(fv, q), 3)
        rows.append(out)
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "downstream_recon.csv", index=False)
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

    configs = []
    for b in CFG["bits"]:
        configs += [(f"uniform@{b}", {l: b for l in layers}),
                    (f"headprot@{b}", {l: b for l in layers if "head." not in l})]
    # S2Q allocations (allocate.py)
    configs += [(f"s2q@{abk}", bmap) for abk, bmap in
                json.loads((RESULTS / f"{MODEL}_s2q_alloc.json").read_text()).items()]
    per_img = []
    PER_IMAGE.mkdir(parents=True, exist_ok=True)
    for name, bmap in configs:
        per_img += eval_cfg(name, bmap, samples, md, xd, fp16)
        pd.DataFrame(per_img).to_csv(PER_IMAGE / "downstream_recon.csv", index=False)
    return summarize()


if __name__ == "__main__":
    print((summarize() if "--from-per-image" in sys.argv else main()).to_string(index=False))
