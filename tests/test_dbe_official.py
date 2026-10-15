"""Optional: check that score/boundary.py's Depth Boundary Error is bit-identical
to the official iBims-1 implementation. Set IBIMS_EVAL_SCRIPT to the path of
evaluate_ibims_error_metrics.py from the iBims-1 evaluation_scripts.zip
(needs scikit-learn, which that script imports). Skips when unset.
Command: IBIMS_EVAL_SCRIPT=/path/to/evaluate_ibims_error_metrics.py python tests/test_dbe_official.py
"""
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from score.boundary import dbe, depth_discontinuities  # noqa: E402

if __name__ == "__main__":
    path = os.environ.get("IBIMS_EVAL_SCRIPT")
    if not path:
        print("SKIP (IBIMS_EVAL_SCRIPT not set)")
        sys.exit(0)
    spec = importlib.util.spec_from_file_location("ibims_eval", path)
    official = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(official)
    rng = np.random.default_rng(0)
    n = 0
    for trial in range(20):
        gt = ndimage.gaussian_filter(rng.uniform(0.5, 8.0, (120, 160)), 6)
        gt[:, 60 + trial:] += 2.0
        valid = rng.random(gt.shape) > 0.05
        edges = depth_discontinuities(gt, valid, percentile=97)
        for sigma in (0.0, 1.0, 3.0):
            pred = ndimage.gaussian_filter(gt, sigma) * rng.uniform(0.98, 1.02, gt.shape)
            p0 = pred.copy(); p0[~valid] = 0.0
            ref = official.compute_depth_boundary_error(edges.astype(float), p0)
            got = dbe(pred, edges, valid)
            assert all((np.isnan(a) and np.isnan(b)) or a == b for a, b in zip(ref, got)), (ref, got)
            n += 1
    print(f"PASS ({n} cases bit-identical)")
