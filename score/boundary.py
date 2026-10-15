"""Boundary-fidelity metrics.

Three quantities, all reported together (the sharpness confound cuts both ways:
a blurrier map can *improve* some pixelwise scores while destroying the physical
edge, so error is only trustworthy beside sharpness):

  1. Depth Boundary Error (DBE) — accuracy eps_acc + completeness eps_comp,
     Koch et al. 2018 (iBims-1), following the official iBims-1 evaluation code
     (`compute_depth_boundary_error` in evaluate_ibims_error_metrics.py).
  2. Boundary-region masked AbsRel — AbsRel on pixels within k px of a GT
     discontinuity, reported beside interior AbsRel (the dissociation IS
     boundary-delta >> interior-delta).
  3. Edge sharpness / width proxy — relative depth gradient across the GT edge
     band; larger gradient = sharper. width_proxy = 1/sharpness.

Edges default to a cv2-free gradient-threshold (same routine for GT and pred, so
they are comparable); Canny is available if cv2 is present. Report threshold /
k sensitivity — a conclusion that holds for only one edge definition is reported
as edge-definition-dependent, not as a result.
"""
import numpy as np
from scipy import ndimage
from skimage import feature

_EPS = 1e-6


def _ibims_dbe(edges, pred, max_dist_thr=10.0):
    """Depth Boundary Error exactly as in the official iBims-1 evaluation code
    (Koch et al. 2018): Canny (sigma=sqrt(2), thresholds 0.1/0.2) on the 0-1
    normalized prediction, then truncated chamfer distances. `pred` uses 0 for
    invalid pixels. Same operations in the same order, so the result is
    bit-identical to the official function. Returns (dbe_acc, dbe_com)."""
    if np.sum(edges) == 0:
        return np.nan, np.nan
    pred_normalized = pred.copy().astype("f")
    pred_normalized[pred_normalized == 0] = np.nan
    pred_normalized = pred_normalized - np.nanmin(pred_normalized)
    pred_normalized = pred_normalized / np.nanmax(pred_normalized)
    edges_est = feature.canny(pred_normalized, sigma=np.sqrt(2),
                              low_threshold=0.1, high_threshold=0.2)
    D_gt = ndimage.distance_transform_edt(1 - edges)
    D_est = ndimage.distance_transform_edt(1 - edges_est)
    E_fin_est_filt = edges_est * (D_gt < max_dist_thr)
    if np.sum(E_fin_est_filt) == 0:      # no predicted edge near any GT edge
        return max_dist_thr, max_dist_thr
    dbe_acc = np.nansum(D_gt * E_fin_est_filt) / np.nansum(E_fin_est_filt)
    dbe_com = np.nansum(D_est * edges) / np.nansum(edges)
    return dbe_acc, dbe_com


def _sobel_mag(x):
    gx = ndimage.sobel(x, axis=1, mode="nearest")
    gy = ndimage.sobel(x, axis=0, mode="nearest")
    return np.hypot(gx, gy)


def depth_discontinuities(depth, valid_mask=None, percentile=98.0, relative=True):
    """Binary edge map from depth: |grad| (relative to depth, scale-invariant)
    thresholded at `percentile` over valid pixels."""
    d = np.asarray(depth, float)
    m = np.isfinite(d) & (d > 0)
    if valid_mask is not None:
        m &= valid_mask.astype(bool)
    g = _sobel_mag(np.where(m, d, 0.0))
    if relative:
        g = g / np.maximum(d, _EPS)          # relative gradient: unitless, scale-free
    g = np.where(m, g, 0.0)
    if m.sum() < 10:
        return np.zeros_like(d, bool)
    thr = np.percentile(g[m], percentile)
    return (g > thr) & m


def predicted_edges(pred_depth, valid_mask=None, percentile=98.0, method="grad",
                    canny_lo=50, canny_hi=150):
    if method == "canny":
        try:
            import cv2
            d = np.asarray(pred_depth, float)
            dn = np.clip((d - d.min()) / (np.ptp(d) + _EPS) * 255, 0, 255).astype(np.uint8)
            e = cv2.Canny(dn, canny_lo, canny_hi) > 0
            if valid_mask is not None:
                e &= valid_mask.astype(bool)
            return e
        except Exception:
            pass
    return depth_discontinuities(pred_depth, valid_mask, percentile)


def dbe(pred_depth, gt_edges, valid_mask=None):
    """Canonical Depth Boundary Error (official iBims-1 protocol, see
    `_ibims_dbe`: canny sigma=sqrt(2), low=0.1, high=0.2 on 0-1 normalized
    depth; chamfer acc + completeness, 10px truncation). Returns (dbe_acc,
    dbe_com). Lower is better. Invalid pixels are set to 0 before scoring."""
    pred = np.asarray(pred_depth, dtype=np.float64).copy()
    bad = ~np.isfinite(pred)
    if valid_mask is not None:
        bad |= ~valid_mask.astype(bool)
    pred[bad] = 0.0
    return _ibims_dbe(np.asarray(gt_edges).astype(float), pred)


def dbe_grad(pred_depth, gt_edges, valid_mask=None, th=10.0, pred_percentile=98.0):
    """Sensitivity variant: gradient-threshold predicted edges instead of canny.
    Used only to test whether the DBE conclusion depends on edge-extraction method."""
    gt_edges = gt_edges.astype(bool)
    est = predicted_edges(pred_depth, valid_mask, pred_percentile, method="grad")
    if gt_edges.sum() == 0:
        return np.nan, np.nan
    dt_to_gt = ndimage.distance_transform_edt(~gt_edges)
    dt_to_est = ndimage.distance_transform_edt(~est) if est.any() else np.full_like(dt_to_gt, th)
    filt = est & (dt_to_gt < th)
    eps_acc = float(dt_to_gt[filt].mean()) if filt.any() else th
    eps_comp = float(dt_to_est[gt_edges].mean())
    return eps_acc, eps_comp


def boundary_masked_error(pred_depth, gt_depth, gt_edges, valid_mask=None, k=3,
                          min_depth=1e-3, max_depth=10.0):
    """AbsRel on the k-px boundary band vs the interior. Returns dict."""
    pred = np.asarray(pred_depth, float); gt = np.asarray(gt_depth, float)
    m = np.isfinite(gt) & (gt > min_depth) & (gt < max_depth) & np.isfinite(pred) & (pred > 0)
    if valid_mask is not None:
        m &= valid_mask.astype(bool)
    band = ndimage.binary_dilation(gt_edges.astype(bool), iterations=int(k))
    bnd = m & band
    inr = m & ~band

    def _absrel(mask):
        if mask.sum() < 5:
            return np.nan, int(mask.sum())
        p = np.clip(pred[mask], min_depth, max_depth); g = gt[mask]
        return float(np.mean(np.abs(p - g) / g)), int(mask.sum())

    b_absrel, n_b = _absrel(bnd)
    i_absrel, n_i = _absrel(inr)
    return {"boundary_absrel": b_absrel, "interior_absrel": i_absrel,
            "n_boundary": n_b, "n_interior": n_i, "k": k}


def edge_sharpness(pred_depth, gt_edges, valid_mask=None, k=2):
    """Mean RELATIVE depth gradient over the k-px GT-edge band. Higher = sharper
    transition. Returns (sharpness, width_proxy=1/sharpness)."""
    pred = np.asarray(pred_depth, float)
    m = np.isfinite(pred) & (pred > 0)
    if valid_mask is not None:
        m &= valid_mask.astype(bool)
    band = ndimage.binary_dilation(gt_edges.astype(bool), iterations=int(k)) & m
    if band.sum() < 5:
        return np.nan, np.nan
    g = _sobel_mag(np.where(m, pred, 0.0)) / np.maximum(pred, _EPS)
    sharp = float(np.mean(g[band]))
    return sharp, float(1.0 / (sharp + _EPS))


def _selftest():
    # Synthetic step edge: fg=1m (left half), bg=3m (right half)
    H, W = 64, 64
    gt = np.ones((H, W)); gt[:, W // 2:] = 3.0
    valid = np.ones((H, W), bool)
    gt_edges = depth_discontinuities(gt, valid, percentile=98)
    assert gt_edges.sum() > 0, "no GT edge found on a step"

    sharp_pred = gt.copy()                              # perfect sharp step
    blur_pred = ndimage.gaussian_filter(gt, sigma=2.5)  # smeared ramp

    s_sharp, w_sharp = edge_sharpness(sharp_pred, gt_edges)
    s_blur, w_blur = edge_sharpness(blur_pred, gt_edges)
    assert s_sharp > s_blur, (s_sharp, s_blur)          # sharp map has higher gradient
    assert w_blur > w_sharp, (w_blur, w_sharp)          # blur => wider transition

    bm_sharp = boundary_masked_error(sharp_pred, gt, gt_edges, valid, k=3)
    bm_blur = boundary_masked_error(blur_pred, gt, gt_edges, valid, k=3)
    assert bm_blur["boundary_absrel"] > bm_sharp["boundary_absrel"], (bm_blur, bm_sharp)
    assert bm_sharp["boundary_absrel"] < 1e-9           # perfect step, zero boundary error

    acc_s, comp_s = dbe(sharp_pred, gt_edges, valid)     # canonical (iBims canny)
    acc_b, comp_b = dbe(blur_pred, gt_edges, valid)
    assert np.isfinite(acc_s) and np.isfinite(comp_s), (acc_s, comp_s)
    assert comp_s <= comp_b + 1e-6                        # sharp recovers edge >= as well
    ga_s, gc_s = dbe_grad(sharp_pred, gt_edges, valid)    # sensitivity variant runs
    assert np.isfinite(ga_s)
    print("boundary selftest OK  sharp/blur sharpness=%.3f/%.3f  bnd_absrel=%.3f/%.3f  dbe_comp=%.2f/%.2f"
          % (s_sharp, s_blur, bm_sharp["boundary_absrel"], bm_blur["boundary_absrel"], comp_s, comp_b))


if __name__ == "__main__":
    _selftest()
