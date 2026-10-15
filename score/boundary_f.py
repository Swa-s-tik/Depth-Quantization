"""Established boundary F-score for depth, beside the custom flying-pixel metric.

This is the standard contour/boundary F-measure (as used for DAVIS masks and by
recent sharp-depth papers such as Depth Pro): extract depth-discontinuity edges
from the prediction and the ground truth, then match them within a pixel
tolerance. Precision is the fraction of predicted edge pixels near a GT edge,
recall the fraction of GT edge pixels near a predicted edge, and F = 2PR/(P+R).

We reuse the edge extractor already used for the Depth Boundary Error so the two
metrics are consistent. Reported at tolerances t in {1,2,3} px.
"""
import numpy as np
from scipy import ndimage

from .boundary import depth_discontinuities, predicted_edges


def _match(edges_a, edges_b, tol):
    """Fraction of edges_a pixels within `tol` px of any edges_b pixel."""
    a = np.asarray(edges_a, bool)
    if a.sum() == 0:
        return np.nan
    # distance from every pixel to nearest edges_b pixel
    dist_to_b = ndimage.distance_transform_edt(~np.asarray(edges_b, bool))
    return float((dist_to_b[a] <= tol).mean())


def boundary_fscore(pred_depth, gt_depth=None, gt_edges=None, valid_mask=None,
                    tols=(1, 2, 3), percentile=98.0):
    """Boundary F-score at each tolerance. GT edges come from `gt_edges` if given,
    else from the GT depth discontinuities. Returns dict f_t / precision_t / recall_t."""
    pred_e = predicted_edges(pred_depth, valid_mask, percentile=percentile, method="grad")
    if gt_edges is not None:
        gt_e = np.asarray(gt_edges, bool)
    else:
        gt_e = depth_discontinuities(gt_depth, valid_mask, percentile=percentile)
    if valid_mask is not None:
        m = valid_mask.astype(bool)
        pred_e = pred_e & m
        gt_e = gt_e & m
    out = {}
    for t in tols:
        p = _match(pred_e, gt_e, t)     # precision: pred near gt
        r = _match(gt_e, pred_e, t)     # recall: gt near pred
        if not (np.isfinite(p) and np.isfinite(r)):
            f = np.nan                  # one side has no edges at all
        elif (p + r) == 0:
            f = 0.0                     # edges exist but nothing matches: worst F
        else:
            f = 2 * p * r / (p + r)
        out[f"boundary_f_t{t}"] = f
        out[f"boundary_prec_t{t}"] = p
        out[f"boundary_rec_t{t}"] = r
    out["n_gt_edge"] = int(gt_e.sum())
    out["n_pred_edge"] = int(pred_e.sum())
    return out


def _selftest():
    H, W = 64, 64
    gt = np.ones((H, W)); gt[:, W // 2:] = 3.0          # a vertical depth step
    valid = np.ones((H, W), bool)
    # perfect prediction -> F ~ 1
    perfect = boundary_fscore(gt.copy(), gt_depth=gt, valid_mask=valid)
    assert perfect["boundary_f_t2"] > 0.9, perfect
    # shifted boundary (5 px) -> low F at tol 1
    shifted = gt.copy() * 0 + 1.0; shifted[:, W // 2 + 5:] = 3.0
    sh = boundary_fscore(shifted, gt_depth=gt, valid_mask=valid, tols=(1,))
    assert sh["boundary_f_t1"] < perfect["boundary_f_t1"], (sh, perfect)
    # smeared/blurred boundary -> recall drops (pred edge weak/misplaced)
    print("boundary_f selftest OK  perfect_f2=%.3f shifted_f1=%.3f" %
          (perfect["boundary_f_t2"], sh["boundary_f_t1"]))


if __name__ == "__main__":
    _selftest()
