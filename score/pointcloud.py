"""Downstream consequence — depth -> point cloud, quantify FLYING PIXELS.

A flying pixel is a back-projected point that lands in the empty 3D gap between
a foreground and background surface at a depth discontinuity (Sabov & Krueger
2008; the artifact discriminative depth models create by averaging fg/bg depth
at edges). Operational criterion: near a GT discontinuity where the local depth
gap (Z_bg - Z_fg) exceeds 2*tau, a predicted depth is "flying" if it sits
strictly inside (Z_fg + tau, Z_bg - tau) — i.e. on neither surface.

This converts the boundary-metric argument into a 3D-reconstruction cost.
"""
import numpy as np
from scipy import ndimage

_EPS = 1e-6


def backproject(depth, K):
    """depth HxW + intrinsics 3x3 -> points (H,W,3) in camera frame (X right, Y down, Z fwd)."""
    H, W = depth.shape
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    X = (u - cx) * depth / fx
    Y = (v - cy) * depth / fy
    return np.stack([X, Y, depth], axis=-1)


def flying_pixel_fraction(pred_depth, gt_depth, gt_edges, valid_mask=None, k=3,
                          tau=None, min_gap_factor=2.0):
    """Fraction of near-edge valid pixels whose predicted depth falls in the empty
    fg/bg gap. Returns dict(flying_frac, n_edge_band, n_flying, tau).

    tau defaults to a GT-noise floor = max(2cm, 1% of local fg depth). Only pixels
    at a *real* discontinuity (local gap > min_gap_factor*tau) are eligible, so
    smooth regions never count.
    """
    pred = np.asarray(pred_depth, float)
    gt = np.asarray(gt_depth, float)
    m = np.isfinite(pred) & (pred > 0) & np.isfinite(gt) & (gt > 0)
    if valid_mask is not None:
        m &= valid_mask.astype(bool)

    band = ndimage.binary_dilation(np.asarray(gt_edges).astype(bool), iterations=int(k)) & m
    if band.sum() < 5:
        return {"flying_frac": np.nan, "n_edge_band": int(band.sum()), "n_flying": 0, "tau": np.nan}

    win = 2 * int(k) + 1
    gt_for_minmax = np.where(m, gt, np.nan)
    z_fg = ndimage.minimum_filter(np.where(m, gt, np.inf), size=win)   # local nearest surface
    z_bg = ndimage.maximum_filter(np.where(m, gt, -np.inf), size=win)  # local farthest surface

    if tau is None:
        tau_map = np.maximum(0.02, 0.01 * np.maximum(z_fg, _EPS))       # 2cm or 1% of fg
    else:
        tau_map = np.full_like(gt, float(tau))

    real_disc = band & ((z_bg - z_fg) > min_gap_factor * tau_map) & np.isfinite(z_fg) & np.isfinite(z_bg)
    n_band = int(real_disc.sum())
    if n_band < 5:
        return {"flying_frac": np.nan, "n_edge_band": n_band, "n_flying": 0,
                "tau": float(np.nanmedian(tau_map))}
    flying = real_disc & (pred > z_fg + tau_map) & (pred < z_bg - tau_map)
    return {"flying_frac": float(flying.sum() / n_band), "n_edge_band": n_band,
            "n_flying": int(flying.sum()), "tau": float(np.nanmedian(tau_map))}


def _selftest():
    from score.boundary import depth_discontinuities
    H, W = 48, 48
    gt = np.ones((H, W)); gt[:, W // 2:] = 3.0          # fg 1m | bg 3m step
    valid = np.ones((H, W), bool)
    edges = depth_discontinuities(gt, valid, percentile=97)

    sharp = gt.copy()                                    # lands on surfaces -> ~0 flying
    fs = flying_pixel_fraction(sharp, gt, edges, valid, k=3)
    assert fs["flying_frac"] < 0.05, fs

    ramp = gt.copy()                                     # smear: middle cols get in-gap 2.0m
    ramp[:, W // 2 - 1] = 1.7; ramp[:, W // 2] = 2.0; ramp[:, W // 2 + 1] = 2.4
    fr = flying_pixel_fraction(ramp, gt, edges, valid, k=3)
    assert fr["flying_frac"] > fs["flying_frac"] and fr["n_flying"] > 0, (fr, fs)

    # backprojection sanity: principal point maps to (0,0,Z)
    K = np.array([[500., 0, W / 2], [0, 500., H / 2], [0, 0, 1.]])
    pts = backproject(gt, K)
    assert pts.shape == (H, W, 3) and abs(pts[H // 2, W // 2, 0]) < 1e-6
    print(f"pointcloud selftest OK  sharp_flying={fs['flying_frac']:.3f} "
          f"ramp_flying={fr['flying_frac']:.3f} (n={fr['n_flying']}/{fr['n_edge_band']})")


if __name__ == "__main__":
    _selftest()
