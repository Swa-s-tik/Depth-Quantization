"""Standard aggregate depth metrics, computed on aligned DEPTH over valid pixels.

These are the metrics the hypothesis says are dominated by easy pixels: AbsRel,
RMSE, RMSE_log, log10, and threshold accuracies delta1/2/3. Depth-range clip
(min/max) follows the dataset convention (NYU 1e-3..10, KITTI 1e-3..80).
"""
import numpy as np

_EPS = 1e-6


def aggregate_metrics(pred_depth, gt_depth, valid_mask=None,
                      min_depth=1e-3, max_depth=10.0):
    """Return dict of aggregate metrics over valid, in-range pixels.

    pred_depth: aligned metric depth (same units as gt). Non-finite preds are
    dropped from the valid set (an unalignable image contributes no pixels).
    """
    pred = np.asarray(pred_depth, dtype=np.float64)
    gt = np.asarray(gt_depth, dtype=np.float64)
    m = np.isfinite(gt) & (gt > min_depth) & (gt < max_depth) & np.isfinite(pred) & (pred > 0)
    if valid_mask is not None:
        m &= valid_mask.astype(bool)
    n = int(m.sum())
    if n < 10:
        return {k: np.nan for k in
                ("absrel", "rmse", "rmse_log", "log10", "d1", "d2", "d3", "sqrel", "n")} | {"n": n}

    p = np.clip(pred[m], min_depth, max_depth)
    g = gt[m]
    absrel = float(np.mean(np.abs(p - g) / g))
    sqrel = float(np.mean((p - g) ** 2 / g))
    rmse = float(np.sqrt(np.mean((p - g) ** 2)))
    rmse_log = float(np.sqrt(np.mean((np.log(p) - np.log(g)) ** 2)))
    log10 = float(np.mean(np.abs(np.log10(p) - np.log10(g))))
    ratio = np.maximum(p / g, g / p)
    d1 = float(np.mean(ratio < 1.25))
    d2 = float(np.mean(ratio < 1.25 ** 2))
    d3 = float(np.mean(ratio < 1.25 ** 3))
    return {"absrel": absrel, "sqrel": sqrel, "rmse": rmse, "rmse_log": rmse_log,
            "log10": log10, "d1": d1, "d2": d2, "d3": d3, "n": n}


def _selftest():
    rng = np.random.default_rng(1)
    gt = rng.uniform(0.5, 7.0, size=(48, 64))   # *1.3 stays < 10m clip
    perfect = aggregate_metrics(gt.copy(), gt)
    assert perfect["absrel"] < 1e-9 and perfect["d1"] == 1.0, perfect
    assert perfect["rmse"] < 1e-9, perfect
    # 10% multiplicative error -> absrel ~0.1, still all within delta1 (1.10<1.25)
    noisy = gt * 1.10
    mm = aggregate_metrics(noisy, gt)
    assert abs(mm["absrel"] - 0.10) < 1e-9, mm
    assert mm["d1"] == 1.0, mm
    # 30% error breaks delta1 (1.30>1.25)
    mm2 = aggregate_metrics(gt * 1.30, gt)
    assert mm2["d1"] == 0.0, mm2
    print("aggregate selftest OK", {k: round(v, 4) for k, v in mm.items() if k != "n"})


if __name__ == "__main__":
    _selftest()
