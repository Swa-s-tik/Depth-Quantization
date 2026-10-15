"""Qualitative figures (Figs. 1 and 4) from cached predictions (no inference).

Two figures, both from iBims-1 cached preds for DAv2-Large:
  1. qualitative_depth.png  - RGB / GT / FP16 / uniform-W6 / head-protected-W6
     depth maps, plus a zoomed boundary crop row.
  2. qualitative_flying.png - a depth profile across a boundary (step vs smear)
     and a top-down back-projected point scatter showing flying pixels in the gap.

The illustrated image is chosen deterministically as the iBims image whose
flying-pixel fraction increases most from FP16 to uniform-W6, i.e. the image
that most exemplifies the effect. Needs iBims-1 and the prediction cache of
phase3_bitsweep and phase4_mitigation. Run: `python figures_qualitative.py`.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from common import CACHE_ROOT as CACHE, FIGURES as FIG, seed_everything
from data.loaders import build
from align import scale_shift_align
from predict import load_cached_pred
from score.pointcloud import backproject, flying_pixel_fraction

MODEL = "dav2_large"
TAGS = {"FP16": "fp16", "uniform W6": "w6", "head-protected W6": "mix_e6_n6"}
CMAP = "turbo"


def load_samples():
    gen, info = build("ibims")
    md, xd = info["min_depth"], info["max_depth"]
    return list(gen()), md, xd


def aligned_depth(sample, tag, md, xd):
    raw = load_cached_pred(CACHE, "ibims", MODEL, tag, sample.id)
    if raw is None:
        return None
    dep, _, _ = scale_shift_align(raw, sample.gt_depth, sample.valid_mask,
                                  mode="ss", pred_is_disparity=True,
                                  min_depth=md, max_depth=xd)
    return dep


def pick_image(samples, md, xd):
    """Deterministically pick the image with the largest FP16 -> W6 rise in
    flying-pixel fraction (the clearest instance of the effect)."""
    best, best_gain = None, -1.0
    for s in samples:
        if s.gt_edges is None or s.K is None:
            continue
        fp16 = aligned_depth(s, "fp16", md, xd)
        w6 = aligned_depth(s, "w6", md, xd)
        if fp16 is None or w6 is None:
            continue
        f0 = flying_pixel_fraction(fp16, s.gt_depth, s.gt_edges, s.valid_mask, k=3)
        f1 = flying_pixel_fraction(w6, s.gt_depth, s.gt_edges, s.valid_mask, k=3)
        g0, g1 = f0["flying_frac"], f1["flying_frac"]
        if not (np.isfinite(g0) and np.isfinite(g1)):
            continue
        gain = float(g1) - float(g0)
        if gain > best_gain:
            best, best_gain = s, gain
    return best, best_gain


def _disp(dep, mask):
    """Depth -> displayable array (meters), invalid set to NaN."""
    out = np.array(dep, dtype=np.float64)
    out[~np.isfinite(out)] = np.nan
    out[~mask] = np.nan
    return out


def _zoom_box(gt_edges, mask, h, w, frac=0.28):
    """Pick a square crop centered on the densest boundary region."""
    e = gt_edges & mask
    ys, xs = np.nonzero(e)
    if len(ys) == 0:
        return h // 4, w // 4, min(h, w) // 3
    # densest edge cluster via the median of edge coords is robust and det.
    cy, cx = int(np.median(ys)), int(np.median(xs))
    side = int(round(frac * min(h, w)))
    y0 = int(np.clip(cy - side // 2, 0, h - side))
    x0 = int(np.clip(cx - side // 2, 0, w - side))
    return y0, x0, side


def fig_depth(sample, md, xd):
    h, w = sample.gt_depth.shape
    mask = sample.valid_mask
    gtd = _disp(sample.gt_depth, mask)
    vmin = np.nanpercentile(gtd, 2)
    vmax = np.nanpercentile(gtd, 98)
    preds = {name: _disp(aligned_depth(sample, tag, md, xd), mask)
             for name, tag in TAGS.items()}
    y0, x0, side = _zoom_box(sample.gt_edges, mask, h, w)

    panels = [("RGB", None), ("Ground truth", gtd)] + \
             [(name, preds[name]) for name in TAGS]
    fig, axes = plt.subplots(2, len(panels), figsize=(3.0 * len(panels), 6.0),
                             gridspec_kw={"height_ratios": [1.0, 0.9]})
    for j, (title, arr) in enumerate(panels):
        ax = axes[0, j]
        if arr is None:
            ax.imshow(np.asarray(sample.rgb))
        else:
            ax.imshow(arr, cmap=CMAP, vmin=vmin, vmax=vmax)
            ax.add_patch(Rectangle((x0, y0), side, side, fill=False,
                                   edgecolor="white", lw=1.5))
        ax.set_title(title, fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
        # zoom row
        axz = axes[1, j]
        if arr is None:
            axz.imshow(np.asarray(sample.rgb)[y0:y0 + side, x0:x0 + side])
        else:
            axz.imshow(arr[y0:y0 + side, x0:x0 + side], cmap=CMAP,
                       vmin=vmin, vmax=vmax)
        axz.set_xticks([]); axz.set_yticks([])
        if j == 0:
            axz.set_ylabel("boundary zoom", fontsize=10)
    fig.suptitle(f"iBims-1 image '{sample.id}', DAv2-Large. "
                 f"Depth (meters); white box is the zoom region.", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = FIG / "qualitative_depth.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", metadata={"Software": None})
    plt.close(fig)
    return out


def fig_flying(sample, md, xd):
    h, w = sample.gt_depth.shape
    mask = sample.valid_mask
    y0, x0, side = _zoom_box(sample.gt_edges, mask, h, w)
    fp16 = aligned_depth(sample, "fp16", md, xd)
    w6 = aligned_depth(sample, "w6", md, xd)
    gt = np.array(sample.gt_depth, float)

    # scanline: the row inside the zoom with the largest GT depth jump.
    rows = range(y0, y0 + side)
    best_y, best_jump = y0 + side // 2, -1.0
    for y in rows:
        seg = gt[y, x0:x0 + side]
        mseg = mask[y, x0:x0 + side]
        if mseg.sum() < side // 2:
            continue
        j = np.nanmax(seg[mseg]) - np.nanmin(seg[mseg])
        if j > best_jump:
            best_jump, best_y = j, y
    xs = np.arange(x0, x0 + side)

    def prof(arr):
        v = arr[best_y, x0:x0 + side].astype(float)
        v[~np.isfinite(v)] = np.nan
        v[~mask[best_y, x0:x0 + side]] = np.nan
        return v

    fig, (axp, axs) = plt.subplots(1, 2, figsize=(11, 4.2))
    axp.plot(xs, prof(gt), color="k", lw=2.2, label="ground truth")
    axp.plot(xs, prof(fp16), color="#1f77b4", lw=1.8, label="FP16")
    axp.plot(xs, prof(w6), color="#d62728", lw=1.8, label="uniform W6")
    axp.set_xlabel("image column (px)")
    axp.set_ylabel("depth (m)")
    axp.set_title(f"Depth profile across a boundary (row {best_y})")
    axp.legend(fontsize=9, frameon=False)

    # top-down back-projection of a band around the scanline
    band = slice(max(best_y - 3, 0), min(best_y + 4, h))
    cols = slice(x0, x0 + side)
    K = sample.K

    def band_pts(arr):
        d = np.array(arr, float)
        d[~np.isfinite(d)] = np.nan
        full = np.full_like(gt, np.nan)
        full[band, cols] = d[band, cols]
        full[~mask] = np.nan
        pts = backproject(np.nan_to_num(full, nan=0.0), K)
        keep = np.isfinite(full).ravel() & (full.ravel() > 0)
        P = pts.reshape(-1, 3)[keep]
        return P

    Pg, Pf, Pw = band_pts(gt), band_pts(fp16), band_pts(w6)
    axs.scatter(Pg[:, 0], Pg[:, 2], s=6, c="k", label="ground truth", alpha=0.5)
    axs.scatter(Pf[:, 0], Pf[:, 2], s=6, c="#1f77b4", label="FP16", alpha=0.6)
    axs.scatter(Pw[:, 0], Pw[:, 2], s=6, c="#d62728", label="uniform W6", alpha=0.6)
    axs.set_xlabel("X (m, camera frame)")
    axs.set_ylabel("Z depth (m)")
    axs.set_title("Top-down point cloud near the edge; W6 bridges the gap")
    axs.legend(fontsize=9, frameon=False)
    fig.suptitle(f"iBims-1 image '{sample.id}', DAv2-Large. Flying pixels appear "
                 f"where W6 smears the depth step.", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = FIG / "qualitative_flying.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", metadata={"Software": None})
    plt.close(fig)
    return out


def main():
    seed_everything()
    FIG.mkdir(parents=True, exist_ok=True)
    samples, md, xd = load_samples()
    sample, gain = pick_image(samples, md, xd)
    assert sample is not None, "no usable iBims image with cached fp16+w6 preds"
    print(f"selected '{sample.id}' (FP16->W6 flying gain {gain:+.3f})")
    o1 = fig_depth(sample, md, xd)
    o2 = fig_flying(sample, md, xd)
    for o in (o1, o2):
        assert o.exists() and o.stat().st_size > 20_000, f"bad figure {o}"
        print("wrote", o, o.stat().st_size, "bytes")


if __name__ == "__main__":
    main()
