"""Cross-dataset transfer (Table 14): does the S2Q allocation TRANSFER across datasets?
The per-layer sensitivities and the resulting S2Q bit allocation are fit on iBims-1.
Here we take that fixed allocation (~6-bit average, unchanged) and evaluate it on
DIODE indoors, against uniform six-bit and head-protected six-bit. If S2Q still
recovers boundaries on DIODE, the allocation generalizes beyond its calibration set.
Outputs results/per_image/s2q_diode_transfer.csv and results/s2q_diode_transfer.csv.
Commands: python s2q_diode_transfer.py                   (GPU)
          python s2q_diode_transfer.py --from-per-image  (aggregate only, CPU)
"""
import json
import sys

import pandas as pd
import torch

from common import CACHE_ROOT, PER_IMAGE, RESULTS, load_config, seed_everything
from data.loaders import build
from models.registry import load_model, infer_depth
from align import scale_shift_align
from predict import load_cached_pred
from quant.fakequant import quantize_per_layer_, quantize_model_
import score.boundary as B
from stats import cliffs_delta, finite_pairs

CFG = load_config("s2q_diode_transfer")
CACHE = CACHE_ROOT
MODEL = CFG["model"]
SUBSET = CFG["subset"]


def diode_samples():
    gen, info = build("diode", subset=SUBSET)
    return [s for s in gen() if s.gt_edges is not None], info["min_depth"], info["max_depth"]


METRICS = ["dbe_com", "edge_width"]


def boundary_scores(model, proc, samples, fp16, md, xd):
    rows = []
    for s in samples:
        raw = infer_depth(model, proc, s.rgb, out_hw=s.gt_depth.shape)
        a, _, _ = scale_shift_align(raw, s.gt_depth, s.valid_mask, mode="ss",
                                    pred_is_disparity=True, min_depth=md, max_depth=xd)
        if s.id not in fp16:
            continue
        vals = {"dbe_com": B.dbe(a, s.gt_edges, s.valid_mask)[1],
                "edge_width": B.edge_sharpness(a, s.gt_edges, s.valid_mask, k=2)[1]}
        rows.append({"image_id": s.id, **{f"q_{k}": v for k, v in vals.items()},
                     **{f"fp16_{k}": fp16[s.id][k] for k in METRICS}})
    return rows


def summarize():
    df = pd.read_csv(PER_IMAGE / "s2q_diode_transfer.csv", float_precision="round_trip")
    rows = []
    for name, g in df.groupby("config", sort=False):
        r = {"config": name, "n_img": len(finite_pairs(g, "dbe_com")[0])}
        for k in METRICS:
            r[f"{k}_cliff"] = round(cliffs_delta(*finite_pairs(g, k)), 3)
        rows.append(r)
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "s2q_diode_transfer.csv", index=False)
    return out


def main():
    seed_everything(CFG.get("seed", 0))
    samples, md, xd = diode_samples()
    print(f"DIODE transfer on {len(samples)} images")
    # FP16 DIODE baseline from the phase6 cache
    fp16 = {}
    for s in samples:
        raw = load_cached_pred(CACHE, "diode", MODEL, "fp16", s.id)
        if raw is None:
            continue
        a, _, _ = scale_shift_align(raw, s.gt_depth, s.valid_mask, mode="ss",
                                    pred_is_disparity=True, min_depth=md, max_depth=xd)
        fp16[s.id] = {"dbe_com": B.dbe(a, s.gt_edges, s.valid_mask)[1],
                      "edge_width": B.edge_sharpness(a, s.gt_edges, s.valid_mask, k=2)[1]}
    samples = [s for s in samples if s.id in fp16]

    alloc = json.load(open(RESULTS / f"{MODEL}_s2q_alloc.json"))
    k6 = min(alloc, key=lambda x: abs(float(x) - CFG["avg_bits"]))   # ~6-bit-average allocation
    s2q_bits = alloc[k6]
    print(f"using S2Q allocation at avg-bit key {k6} ({len(s2q_bits)} layers)")

    per_img = []
    for name in ["uniform6", "headprot6", "s2q"]:   # uniform / head-protected at 6 bits
        model, proc, _ = load_model(MODEL, dtype=torch.float16)
        if name == "uniform6":
            quantize_model_(model, 6, per_channel=True)
        elif name == "headprot6":
            quantize_model_(model, 6, per_channel=True, extra_filters=[r"^(?!.*head\.).*"])  # all but head
        else:
            quantize_per_layer_(model, s2q_bits, per_channel=True, default_bits=16)
        per_img += [{"config": name, **r} for r in boundary_scores(model, proc, samples, fp16, md, xd)]
        del model; torch.cuda.empty_cache()
        PER_IMAGE.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(per_img).to_csv(PER_IMAGE / "s2q_diode_transfer.csv", index=False)
    return summarize()


if __name__ == "__main__":
    print((summarize() if "--from-per-image" in sys.argv else main()).to_string(index=False))
