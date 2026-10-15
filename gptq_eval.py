"""Round-to-nearest vs GPTQ (Table 17): does the six-bit boundary damage survive
an error-minimizing PTQ, or is it an artifact of naive round-to-nearest (RTN)?

GPTQ (Frantar et al., 2023) quantizes each layer to minimize the layer's OUTPUT
reconstruction error on calibration activations, using the Hessian H = X X^T and
column-wise error compensation. We use the SAME per-output-channel symmetric grid
as the RTN sweep, so GPTQ and RTN differ ONLY in the rounding, not the grid or the
bit-width. If the boundary effect at six bits is as large under GPTQ as under RTN,
the phenomenon is a property of low-bit weights and not of naive rounding, and it
shows that a reconstruction objective (bulk output MSE) does not protect boundaries.

GPTQ is run at W8/W6/W4 on DAv2-Large (16 calibration images, evaluated on the
remaining 84): results/per_image/gptq_eval.csv and results/gptq_eval.csv. The RTN
rows are scored on the same 84 images from the phase3_bitsweep prediction cache
(CPU only, `--rtn`): results/per_image/gptq_rtn.csv and results/gptq_rtn.csv.
Commands: python gptq_eval.py            (GPTQ, GPU)
          python gptq_eval.py --rtn      (RTN arm from the cache, CPU)
          add --from-per-image to either to rebuild the aggregate only (CPU)
          python gptq_eval.py --selftest (core-quantizer check, no GPU/data)
"""
import sys
import numpy as np
import torch
import torch.nn as nn

from common import CACHE_ROOT, PER_IMAGE, RESULTS, load_config, seed_everything

_QUANTIZABLE = (nn.Linear, nn.Conv2d)


def _row_scale(W, bits):
    """Per-output-row symmetric scale, matching quant/fakequant per-channel grid."""
    qmax = 2 ** (bits - 1) - 1
    amax = W.abs().amax(dim=1, keepdim=True).clamp_min(1e-12)
    return amax / qmax, qmax


def gptq_layer(W, H, bits, damp=0.01, blocksize=128):
    """GPTQ-quantize a 2D weight [out, in] given input Hessian H [in, in].
    Fixed per-row symmetric scale from the original W (weight-only PTQ)."""
    W = W.clone().float()
    rows, cols = W.shape
    scale, qmax = _row_scale(W, bits)

    sc = scale.squeeze(1)                          # [rows]

    def q(col):                                    # col: [rows] -> quantized [rows]
        return torch.clamp(torch.round(col / sc), -qmax - 1, qmax) * sc

    H = H.clone().float()
    dead = torch.diag(H) == 0
    H[dead, dead] = 1.0
    H[:, dead] = 0; H[dead, :] = 0
    d = damp * torch.mean(torch.diag(H))
    H[range(cols), range(cols)] += d
    # inverse via Cholesky (upper) — standard GPTQ
    L = torch.linalg.cholesky(H)
    Hinv = torch.cholesky_inverse(L)
    Hinv = torch.linalg.cholesky(Hinv, upper=True)   # Hinv now upper-tri factor

    Q = torch.zeros_like(W)
    for i1 in range(0, cols, blocksize):
        i2 = min(i1 + blocksize, cols)
        cnt = i2 - i1
        W1 = W[:, i1:i2].clone()
        Q1 = torch.zeros_like(W1)
        Err1 = torch.zeros_like(W1)
        Hinv1 = Hinv[i1:i2, i1:i2]
        for i in range(cnt):
            w = W1[:, i]
            dinv = Hinv1[i, i]
            qcol = q(w)
            Q1[:, i] = qcol
            err = (w - qcol) / dinv
            W1[:, i:] -= err.unsqueeze(1) * Hinv1[i, i:].unsqueeze(0)
            Err1[:, i] = err
        Q[:, i1:i2] = Q1
        W[:, i2:] -= Err1 @ Hinv[i1:i2, i2:]
    return Q.to(W.dtype)


def _selftest():
    torch.manual_seed(0)
    out_f, in_f, n = 32, 64, 512
    W = torch.randn(out_f, in_f)
    X = torch.randn(n, in_f)                       # calibration inputs
    H = X.t() @ X
    bits = 4
    # RTN at the same grid
    scale, qmax = _row_scale(W, bits)
    Wrtn = torch.clamp(torch.round(W / scale), -qmax - 1, qmax) * scale
    Wgptq = gptq_layer(W, H, bits)
    # GPTQ must reduce OUTPUT reconstruction error vs RTN on the calibration set
    e_rtn = ((X @ Wrtn.t() - X @ W.t()) ** 2).mean().item()
    e_gptq = ((X @ Wgptq.t() - X @ W.t()) ** 2).mean().item()
    assert e_gptq < e_rtn, f"GPTQ {e_gptq:.4f} !< RTN {e_rtn:.4f}"
    # 8-bit near-lossless
    W8 = gptq_layer(W, H, 8)
    assert ((X @ W8.t() - X @ W.t()) ** 2).mean().item() < e_gptq
    print(f"gptq selftest OK  out-MSE  gptq={e_gptq:.4f} < rtn={e_rtn:.4f}")


if __name__ == "__main__" and "--selftest" in sys.argv:
    _selftest(); sys.exit(0)


# ---- full evaluation (GPU + iBims) --------------------------------------------
import pandas as pd
from data.loaders import build
from models.registry import load_model, infer_depth
from align import scale_shift_align
from predict import load_cached_pred
import score.boundary as B
from score.pointcloud import flying_pixel_fraction
from stats import cliffs_delta, finite_pairs

CFG = load_config("gptq_eval")
CACHE = CACHE_ROOT
MODEL = CFG["model"]
N_CALIB = CFG["n_calib"]
METRICS = ["absrel", "dbe_com", "edge_width", "flying"]


def _unfold_input(mod, x):
    """Layer input -> 2D [samples, in_features] for Hessian accumulation."""
    if isinstance(mod, nn.Linear):
        return x.reshape(-1, x.shape[-1])
    # Conv2d: im2col patches
    up = nn.functional.unfold(x, kernel_size=mod.kernel_size, dilation=mod.dilation,
                              padding=mod.padding, stride=mod.stride)   # [N, C*kh*kw, L]
    return up.transpose(1, 2).reshape(-1, up.shape[1])


def collect_hessians(model, proc, calib):
    H, nsamp = {}, {}
    hooks = []

    def mk(name, mod):
        def hook(m, inp):
            x = _unfold_input(mod, inp[0].detach().float())
            h = (x.t() @ x).cpu()
            if name in H:
                H[name] += h; nsamp[name] += x.shape[0]
            else:
                H[name] = h; nsamp[name] = x.shape[0]
        return hook

    named = [(n, m) for n, m in model.named_modules() if isinstance(m, _QUANTIZABLE)]
    for n, m in named:
        hooks.append(m.register_forward_pre_hook(mk(n, m)))
    with torch.no_grad():
        for s in calib:
            infer_depth(model, proc, s.rgb, out_hw=s.gt_depth.shape)
    for h in hooks:
        h.remove()
    return H


@torch.no_grad()
def gptq_quantize_model_(model, H, bits):
    n = 0
    for name, mod in model.named_modules():
        if not isinstance(mod, _QUANTIZABLE) or name not in H:
            continue
        W = mod.weight.data
        W2d = W.reshape(W.shape[0], -1)
        Wq = gptq_layer(W2d.cpu(), H[name], bits).reshape(W.shape).to(W.device, W.dtype)
        mod.weight.data.copy_(Wq)
        n += 1
    return n


def score(model, proc, samples, fp16, md, xd):
    """Per-image GPTQ and FP16 metrics (images with a valid FP16 baseline)."""
    rows = []
    for s in samples:
        raw = infer_depth(model, proc, s.rgb, out_hw=s.gt_depth.shape)
        a, _, _ = scale_shift_align(raw, s.gt_depth, s.valid_mask, mode="ss",
                                    pred_is_disparity=True, min_depth=md, max_depth=xd)
        f = fp16.get(s.id)
        if f is None or not np.all(np.isfinite([f["absrel"]])):
            continue
        m = s.valid_mask & np.isfinite(a) & (a > 0) & (s.gt_depth > 0)
        if m.sum() < 10:
            continue
        ar = float(np.mean(np.abs(a[m] - s.gt_depth[m]) / s.gt_depth[m]))
        _, com = B.dbe(a, s.gt_edges, s.valid_mask)
        _, ew = B.edge_sharpness(a, s.gt_edges, s.valid_mask, k=2)
        ff = flying_pixel_fraction(a, s.gt_depth, s.gt_edges, s.valid_mask, k=3)["flying_frac"]
        rows.append({"image_id": s.id, **{f"fp16_{k}": f[k] for k in METRICS},
                     "gptq_absrel": ar, "gptq_dbe_com": com, "gptq_edge_width": ew, "gptq_flying": ff})
    return rows


def fp16_metrics(samples, md, xd, tag="fp16"):
    """Per-image metrics of a cached prediction (FP16 baseline, or an RTN tag)."""
    out = {}
    for s in samples:
        raw = load_cached_pred(CACHE, "ibims", MODEL, tag, s.id)
        if raw is None:
            continue
        a, _, _ = scale_shift_align(raw, s.gt_depth, s.valid_mask, mode="ss",
                                    pred_is_disparity=True, min_depth=md, max_depth=xd)
        m = s.valid_mask & np.isfinite(a) & (a > 0) & (s.gt_depth > 0)
        if m.sum() < 10:
            continue
        _, com = B.dbe(a, s.gt_edges, s.valid_mask)
        _, ew = B.edge_sharpness(a, s.gt_edges, s.valid_mask, k=2)
        out[s.id] = {
            "absrel": float(np.mean(np.abs(a[m] - s.gt_depth[m]) / s.gt_depth[m])),
            "dbe_com": com, "edge_width": ew,
            "flying": flying_pixel_fraction(a, s.gt_depth, s.gt_edges, s.valid_mask, k=3)["flying_frac"],
        }
    return out


def _splits():
    gen, info = build("ibims")
    md, xd = info["min_depth"], info["max_depth"]
    samples = [s for s in gen() if s.gt_edges is not None and s.K is not None]
    return samples[:N_CALIB], samples[N_CALIB:], md, xd


def _effects(acc, base, row):
    for k in METRICS:                              # all four metrics: higher = worse
        if len(acc[k]) >= 10:
            row[f"{k}_cliff"] = round(cliffs_delta(np.array(acc[k]), np.array(base[k])), 3)
            row[f"{k}_mean"] = round(float(np.mean(acc[k])), 4)
    return row


def rtn_per_image():
    """RTN arm of Table 17: the cached round-to-nearest sweep, same 84 images."""
    _, evalset, md, xd = _splits()
    fp16 = fp16_metrics(evalset, md, xd)
    rows = []
    for bits in CFG["bits"]:
        q = fp16_metrics(evalset, md, xd, tag=f"w{bits}")
        for s in evalset:
            if s.id in q and s.id in fp16:
                rows.append({"w_bits": bits, "image_id": s.id,
                             **{f"fp16_{k}": fp16[s.id][k] for k in METRICS},
                             **{f"rtn_{k}": q[s.id][k] for k in METRICS}})
    PER_IMAGE.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(PER_IMAGE / "gptq_rtn.csv", index=False)


def summary(method):
    """Aggregate of results/per_image/gptq_<method>.csv (method: rtn or eval = GPTQ)."""
    q = "rtn" if method == "rtn" else "gptq"
    df = pd.read_csv(PER_IMAGE / f"gptq_{method}.csv", float_precision="round_trip")
    rows = []
    for bits in CFG["bits"]:
        g = df[df.w_bits == bits]
        acc, base = {}, {}
        for k in METRICS:
            acc[k], base[k] = map(list, finite_pairs(g, k, q=q))
        head = ({"method": "rtn", "w_bits": bits, "n_img": len(g)} if q == "rtn" else
                {"method": "gptq", "w_bits": bits, "n_layers": int(g.n_layers.iloc[0])})
        rows.append(_effects(acc, base, head))
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / f"gptq_{method}.csv", index=False)
    return out


def main():
    calib, evalset, md, xd = _splits()
    print(f"GPTQ eval: {len(calib)} calib, {len(evalset)} eval images")
    fp16 = fp16_metrics(evalset, md, xd)

    # collect Hessians ONCE on the fp16 model (bit-independent)
    model, proc, _ = load_model(MODEL, dtype=torch.float16)
    H = collect_hessians(model, proc, calib)
    del model; torch.cuda.empty_cache()
    print(f"collected Hessians for {len(H)} layers")

    per_img = []
    PER_IMAGE.mkdir(parents=True, exist_ok=True)
    for bits in CFG["bits"]:
        model, proc, _ = load_model(MODEL, dtype=torch.float16)
        nq = gptq_quantize_model_(model, H, bits)
        per_img += [{"w_bits": bits, "n_layers": nq, **r} for r in score(model, proc, evalset, fp16, md, xd)]
        del model; torch.cuda.empty_cache()
        pd.DataFrame(per_img).to_csv(PER_IMAGE / "gptq_eval.csv", index=False)


if __name__ == "__main__":
    seed_everything(CFG.get("seed", 0))
    method = "rtn" if "--rtn" in sys.argv else "eval"
    if "--from-per-image" not in sys.argv:
        rtn_per_image() if method == "rtn" else main()
    print(summary(method).to_string(index=False))
