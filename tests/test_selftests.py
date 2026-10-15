"""Built-in self-tests of the toolkit: alignment, metrics, statistics, fake
quantization, activation quantization, GPTQ and group-wise cores. Synthetic
inputs only (no GPU, datasets, cache or checkpoints).
Command: python tests/test_selftests.py
"""
import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

MODULES = ["align", "score.aggregate", "stats", "score.boundary", "score.boundary_f",
           "score.pointcloud", "quant.fakequant", "quant.actquant", "gptq_eval", "gw_eval"]

if __name__ == "__main__":
    from common import seed_everything
    seed_everything()
    for name in MODULES:
        importlib.import_module(name)._selftest()
    print(f"PASS ({len(MODULES)} self-tests)")
