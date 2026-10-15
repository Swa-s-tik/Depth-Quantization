"""Weight-and-activation regime (Table 19): static activation fake-quantization
on top of W6 weights.

Head protection keeps the head at full-precision WEIGHTS but does nothing for
activations. Each configuration adds per-channel activation fake-quant
(quant/actquant.py) to every Linear/Conv2d input and is scored paired against
FP16 on DAv2-Large, iBims-1 held-out 60 images.
Outputs results/per_image/w4a4_probe.csv and results/w4a4_probe.csv.
Commands: python w4a4_probe.py                   (GPU)
          python w4a4_probe.py --from-per-image  (aggregate only, CPU)
"""
import sys

import numpy as np
import pandas as pd
import torch

from common import CACHE_ROOT, PER_IMAGE, RESULTS, load_config, seed_everything
from data.loaders import build
from models.registry import load_model, infer_depth
from align import scale_shift_align
import score.boundary as B
from score.pointcloud import flying_pixel_fraction
from predict import load_cached_pred
from quant.fakequant import quantize_per_layer_, _QUANTIZABLE
from quant.actquant import add_activation_fakequant, remove_activation_fakequant
from stats import cliffs_delta, finite_pairs

CFG = load_config("wa_probe")
CACHE = CACHE_ROOT; MODEL = CFG["model"]


METRICS = ["dbe_com", "flying_frac", "absrel"]


def score_one(raw, s, md, xd):
    a, _, _ = scale_shift_align(raw, s.gt_depth, s.valid_mask, mode="ss",
                               pred_is_disparity=True, min_depth=md, max_depth=xd)
    _, com = B.dbe(a, s.gt_edges, s.valid_mask)
    ff = flying_pixel_fraction(a, s.gt_depth, s.gt_edges, s.valid_mask, k=3)["flying_frac"] if s.K is not None else np.nan
    from score.aggregate import aggregate_metrics
    return {"dbe_com": com, "flying_frac": ff,
            "absrel": aggregate_metrics(a, s.gt_depth, s.valid_mask, md, xd)["absrel"]}


def run(name, wbits_map, abits, samples, md, xd, fp16):
    model, proc, _ = load_model(MODEL, dtype=torch.float16)
    if wbits_map:
        quantize_per_layer_(model, wbits_map, per_channel=True, default_bits=16)
    handles = add_activation_fakequant(model, abits) if abits < 16 else []
    rows = []
    for s in samples:
        raw = infer_depth(model, proc, s.rgb, out_hw=s.gt_depth.shape)
        q = score_one(raw, s, md, xd)
        rows.append({"config": name, "image_id": s.id, **{f"q_{k}": q[k] for k in METRICS},
                     **{f"fp16_{k}": fp16[s.id][k] for k in METRICS}})
    remove_activation_fakequant(handles); del model; torch.cuda.empty_cache()
    return rows


def summarize():
    df = pd.read_csv(PER_IMAGE / "w4a4_probe.csv", float_precision="round_trip")
    rows = []
    for name, g in df.groupby("config", sort=False):
        out = {"config": name}
        for k in METRICS:
            qv, fv = finite_pairs(g, k)
            if len(qv):
                out[f"{k}_cliff"] = round(cliffs_delta(qv, fv), 3)
        rows.append(out)
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "w4a4_probe.csv", index=False)
    return out


def main():
    seed_everything(CFG.get("seed", 0))
    gen, info = build("ibims"); md, xd = info["min_depth"], info["max_depth"]
    calib = set([s.id for s in gen() if s.gt_edges is not None and s.K is not None][:CFG["calib_n"]])
    samples = [s for s in gen() if s.id not in calib]
    fp16 = {}
    for s in samples:
        raw = load_cached_pred(CACHE, "ibims", MODEL, "fp16", s.id)
        if raw is not None: fp16[s.id] = score_one(raw, s, md, xd)
    samples = [s for s in samples if s.id in fp16]
    print(f"{len(samples)} held-out imgs")
    m, _, _ = load_model(MODEL, dtype=torch.float16)
    layers = [n for n, mod in m.named_modules() if isinstance(mod, _QUANTIZABLE)]
    del m; torch.cuda.empty_cache()
    wmaps = {"headprot6": {l: 6 for l in layers if "head." not in l},   # head weights FP16, rest W6
             "uniform6": {l: 6 for l in layers},
             "none": {}}
    per_img = []
    for c in CFG["configs"]:
        per_img += run(c["name"], wmaps[c["weights"]], c["a_bits"], samples, md, xd, fp16)
        PER_IMAGE.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(per_img).to_csv(PER_IMAGE / "w4a4_probe.csv", index=False)
    return summarize()


if __name__ == "__main__":
    print((summarize() if "--from-per-image" in sys.argv else main()).to_string(index=False))
