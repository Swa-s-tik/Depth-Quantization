"""Alignment of affine-invariant depth predictions to metric GT.

Relative/affine-invariant models (MiDaS, DPT, Depth Anything relative) predict a
DISPARITY-like map (increasing with proximity), defined only up to an unknown
scale+shift. Metrics require aligning to GT first. We fit in *inverse-depth
(disparity) space* — the MiDaS/DPT standard — then invert back to depth.

The SAME alignment is applied to FP16 and every quantized variant, so any
per-image (s,t) idiosyncrasy is part of the measured paired delta and any bug
here cancels in the FP16-vs-quant comparison.
"""
import numpy as np

_EPS = 1e-6


def _valid(gt_depth, valid_mask, min_depth, max_depth):
    m = np.isfinite(gt_depth) & (gt_depth > min_depth) & (gt_depth < max_depth)
    if valid_mask is not None:
        m &= valid_mask.astype(bool)
    return m


def scale_shift_align(pred, gt_depth, valid_mask=None, mode="ss",
                      pred_is_disparity=True, min_depth=1e-3, max_depth=1e3):
    """Align `pred` to `gt_depth`. Returns (aligned_depth, (s, t), residual_rms).

    pred_is_disparity: True if `pred` is disparity-like (DAv2/MiDaS/DPT relative).
      We then fit s,t against GT disparity (1/gt_depth) and invert.
    mode: 'ss'   = least-squares scale+shift (primary protocol)
          'scale'= median-ratio scale-only, t=0 (sensitivity alternative)
    aligned_depth is float64, positive; invalid/degenerate pixels -> +inf.
    """
    pred = np.asarray(pred, dtype=np.float64)
    gt_depth = np.asarray(gt_depth, dtype=np.float64)
    m = _valid(gt_depth, valid_mask, min_depth, max_depth)
    if m.sum() < 10:
        return np.full_like(pred, np.inf), (np.nan, np.nan), np.nan

    target = 1.0 / gt_depth[m] if pred_is_disparity else gt_depth[m]
    x = pred[m]

    # degenerate predictions (constant / NaN / inf, e.g. from an over-quantized
    # layer) make the least-squares fit ill-posed; return an invalid alignment
    # gracefully so scoring yields NaN rather than crashing the whole run.
    if not np.all(np.isfinite(x)) or float(np.std(x)) < _EPS:
        return np.full_like(pred, np.inf), (np.nan, np.nan), np.nan

    if mode == "ss":
        A = np.stack([x, np.ones_like(x)], axis=1)
        try:
            (s, t), *_ = np.linalg.lstsq(A, target, rcond=None)
        except np.linalg.LinAlgError:
            return np.full_like(pred, np.inf), (np.nan, np.nan), np.nan
    elif mode == "scale":
        s = float(np.median(target) / (np.median(x) + _EPS))
        t = 0.0
    else:
        raise ValueError(f"unknown mode {mode}")

    fit = s * pred + t
    resid = float(np.sqrt(np.mean((s * x + t - target) ** 2)))
    if pred_is_disparity:
        aligned = np.where(fit > _EPS, 1.0 / np.maximum(fit, _EPS), np.inf)
    else:
        aligned = np.where(fit > 0, fit, np.inf)
    return aligned, (float(s), float(t)), resid


def _selftest():
    rng = np.random.default_rng(0)
    gt = rng.uniform(0.5, 8.0, size=(64, 80))          # metric depth (m)
    valid = np.ones_like(gt, bool)
    valid[rng.random(gt.shape) < 0.1] = False           # some holes
    s0, t0 = 2.7, 0.13
    pred_disp = (1.0 / gt - t0) / s0                     # exact affine of gt disparity
    aligned, (s, t), resid = scale_shift_align(pred_disp, gt, valid, mode="ss")
    assert abs(s - s0) < 1e-6 and abs(t - t0) < 1e-6, (s, t)
    assert resid < 1e-8, resid
    assert np.allclose(aligned[valid], gt[valid], atol=1e-6), np.abs(aligned[valid]-gt[valid]).max()
    # scale-only recovers when shift truly zero
    pred2 = (1.0 / gt) / s0
    a2, (s2, t2), _ = scale_shift_align(pred2, gt, valid, mode="scale")
    assert abs(s2 - s0) < 1e-3 and t2 == 0.0, (s2, t2)  # _EPS zero-guard => ~1e-5 drift
    print("align selftest OK  s=%.6f t=%.6f resid=%.2e" % (s, t, resid))


if __name__ == "__main__":
    _selftest()
