"""Repository paths, the global random seed and experiment configs.

Paths default to directories inside the repository and can be moved with
environment variables:
  DQ_DATA     dataset root (default: ./datasets)
  DQ_CACHE    prediction cache root (default: ./cache)
  DQ_RESULTS  results directory (default: ./results)
  DQ_FIGURES  figure directory (default: ./figures)
"""
import os
import random
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent
DATA_ROOT = Path(os.environ.get("DQ_DATA", ROOT / "datasets"))
CACHE_ROOT = Path(os.environ.get("DQ_CACHE", ROOT / "cache"))
RESULTS = Path(os.environ.get("DQ_RESULTS", ROOT / "results"))
PER_IMAGE = RESULTS / "per_image"
TABLES = RESULTS / "tables"
FIGURES = Path(os.environ.get("DQ_FIGURES", ROOT / "figures"))

SEED = 0


def seed_everything(seed=SEED):
    """One global seed for Python, NumPy and PyTorch (all devices)."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def load_config(name):
    """Load configs/<name>.yaml (or a path to a .yaml file)."""
    p = Path(name)
    if p.suffix != ".yaml":
        p = ROOT / "configs" / f"{name}.yaml"
    return yaml.safe_load(p.read_text())
