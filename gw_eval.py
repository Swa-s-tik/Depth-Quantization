"""Group-wise four-bit weights (Table 18): does group-wise 4-bit quantization (the
deployable GPTQ/AWQ granularity, group size 64/128) sit in the boundary danger window?

Per-tensor/per-channel 4-bit is catastrophic on aggregate too, so the standard
aggregate check catches it. Group-wise 4-bit is far more accurate on aggregate and
is what real 4-bit deployments use, so if the dissociation appears there it is a
deployable-bit-width phenomenon. We fake-quantize each Linear/Conv2d weight
group-wise (per output row, per group of G input elements) at 4 bits and score
aggregate + boundary vs FP16 on iBims-1 (DAv2-Large).
Output: results/gw_eval.csv.  Command: python gw_eval.py
`python gw_eval.py --selftest` runs the core check (no GPU/data).
"""
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from common import CACHE_ROOT, RESULTS, load_config, seed_everything
_QUANTIZABLE = (nn.Linear, nn.Conv2d)


def gw_quant(W, bits, G):
    """Group-wise symmetric fake-quant of 2D weight [out, in]: one scale per
    (output row, group of G input columns). Matches GPTQ/AWQ group granularity."""
    W = W.float()
    out, inn = W.shape
    qmax = 2 ** (bits - 1) - 1
    pad = (-inn) % G
    Wp = F.pad(W, (0, pad)) if pad else W
    Wg = Wp.reshape(out, -1, G)
    scale = Wg.abs().amax(dim=2, keepdim=True).clamp_min(1e-12) / qmax
    q = torch.clamp(torch.round(Wg / scale), -qmax - 1, qmax) * scale
    return q.reshape(out, -1)[:, :inn]


def _selftest():
    torch.manual_seed(0)
    W = torch.randn(16, 260)                       # in not a multiple of G -> tests padding
    q128 = gw_quant(W, 4, 128); q64 = gw_quant(W, 4, 64)
    assert q128.shape == W.shape and q64.shape == W.shape
    # finer groups -> lower error than coarser, and both beat per-tensor 4-bit
    e128 = (q128 - W).abs().mean(); e64 = (q64 - W).abs().mean()
    qmax = 7; s = W.abs().amax() / qmax
    ept = (torch.clamp(torch.round(W / s), -8, 7) * s - W).abs().mean()
    assert e64 < e128 < ept, (e64.item(), e128.item(), ept.item())
    print(f"gw selftest OK  err g64={e64:.4f} < g128={e128:.4f} < per-tensor={ept:.4f}")


if __name__ == "__main__" and "--selftest" in sys.argv:
    _selftest(); sys.exit(0)

import pandas as pd
from data.loaders import build
from models.registry import load_model, infer_depth
from align import scale_shift_align
from predict import load_cached_pred
import score.boundary as B
from score.pointcloud import flying_pixel_fraction
from stats import cliffs_delta

CFG = load_config("gw_eval")
CACHE = CACHE_ROOT
MODEL = CFG["model"]


@torch.no_grad()
def quantize_gw_(model, bits, G):
    n = 0
    for _, mod in model.named_modules():
        if isinstance(mod, _QUANTIZABLE):
            W = mod.weight.data
            Wq = gw_quant(W.reshape(W.shape[0], -1).cpu(), bits, G).reshape(W.shape).to(W.device, W.dtype)
            mod.weight.data.copy_(Wq); n += 1
    return n


def score(model, proc, samples, fp16, md, xd):
    acc = {"absrel": [], "dbe_com": [], "edge_width": [], "flying": []}
    base = {k: [] for k in acc}
    for s in samples:
        raw = infer_depth(model, proc, s.rgb, out_hw=s.gt_depth.shape)
        a, _, _ = scale_shift_align(raw, s.gt_depth, s.valid_mask, mode="ss",
                                    pred_is_disparity=True, min_depth=md, max_depth=xd)
        f = fp16.get(s.id)
        if f is None:
            continue
        m = s.valid_mask & np.isfinite(a) & (a > 0) & (s.gt_depth > 0)
        if m.sum() < 10:
            continue
        vals = {"absrel": float(np.mean(np.abs(a[m] - s.gt_depth[m]) / s.gt_depth[m])),
                "dbe_com": B.dbe(a, s.gt_edges, s.valid_mask)[1],
                "edge_width": B.edge_sharpness(a, s.gt_edges, s.valid_mask, k=2)[1],
                "flying": flying_pixel_fraction(a, s.gt_depth, s.gt_edges, s.valid_mask, k=3)["flying_frac"]}
        for k, v in vals.items():
            if np.isfinite(v) and np.isfinite(f[k]):
                acc[k].append(v); base[k].append(f[k])
    return acc, base


def fp16_metrics(samples, md, xd):
    out = {}
    for s in samples:
        raw = load_cached_pred(CACHE, "ibims", MODEL, "fp16", s.id)
        if raw is None:
            continue
        a, _, _ = scale_shift_align(raw, s.gt_depth, s.valid_mask, mode="ss",
                                    pred_is_disparity=True, min_depth=md, max_depth=xd)
        m = s.valid_mask & np.isfinite(a) & (a > 0) & (s.gt_depth > 0)
        if m.sum() < 10:
            continue
        out[s.id] = {"absrel": float(np.mean(np.abs(a[m] - s.gt_depth[m]) / s.gt_depth[m])),
                     "dbe_com": B.dbe(a, s.gt_edges, s.valid_mask)[1],
                     "edge_width": B.edge_sharpness(a, s.gt_edges, s.valid_mask, k=2)[1],
                     "flying": flying_pixel_fraction(a, s.gt_depth, s.gt_edges, s.valid_mask, k=3)["flying_frac"]}
    return out


def main():
    seed_everything(CFG.get("seed", 0))
    gen, info = build("ibims")
    md, xd = info["min_depth"], info["max_depth"]
    samples = [s for s in gen() if s.gt_edges is not None and s.K is not None]
    print(f"gw eval on {len(samples)} images")
    fp16 = fp16_metrics(samples, md, xd)
    rows = []
    for bits, G in CFG["schemes"]:
        model, proc, _ = load_model(MODEL, dtype=torch.float16)
        nq = quantize_gw_(model, bits, G)
        acc, base = score(model, proc, samples, fp16, md, xd)
        del model; torch.cuda.empty_cache()
        r = {"scheme": f"gw{bits}_g{G}", "n_layers": nq, "n_img": len(acc["absrel"])}
        for k in acc:
            r[f"{k}_cliff"] = round(cliffs_delta(np.array(acc[k]), np.array(base[k])), 3)
        rows.append(r); print(r)
        pd.DataFrame(rows).to_csv(RESULTS / "gw_eval.csv", index=False)
    print("wrote results/gw_eval.csv")


if __name__ == "__main__":
    main()
