"""Dataset loaders — yield a common Sample so predict.py / scorers are dataset-agnostic.

A Sample carries everything a metric could need: rgb (PIL), gt_depth (meters,
float32 HxW), valid_mask (bool), K (3x3 intrinsics or None), gt_edges (bool or
None — provided annotation preferred over derived), and free-form meta.

We cache RAW model output (not aligned) so alignment protocol can be varied at
score time. Datasets are registered by name; a builder returns (iterator, info).
"""
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
from PIL import Image

from common import DATA_ROOT as DATASETS_ROOT


@dataclass
class Sample:
    id: str
    rgb: object                      # PIL.Image
    gt_depth: np.ndarray             # float32 HxW, meters
    valid_mask: np.ndarray           # bool HxW
    K: np.ndarray = None             # 3x3 intrinsics or None
    gt_edges: np.ndarray = None      # bool HxW provided edge annotation, or None
    meta: dict = field(default_factory=dict)

    @property
    def hw(self):
        return self.gt_depth.shape[:2]


# ---- dataset builders (registered) --------------------------------------
_BUILDERS = {}


def register(name):
    def deco(fn):
        _BUILDERS[name] = fn
        return fn
    return deco


def build(name, **kw):
    if name not in _BUILDERS:
        raise KeyError(f"unknown dataset '{name}'; have {list(_BUILDERS)}")
    return _BUILDERS[name](**kw)


@register("ibims")
def _ibims(root=None, subset=None):
    """iBims-1: 100 indoor images, laser GT, provided edge annotations + intrinsics.
    Expects the extracted per-image .mat files anywhere under <DQ_DATA>/ibims/."""
    from scipy.io import loadmat
    root = Path(root or DATASETS_ROOT / "ibims")
    mats = sorted(root.rglob("*.mat"))
    # drop any eval-script .mat; keep per-image data files
    mats = [m for m in mats if m.stat().st_size > 100_000]
    if subset:
        mats = mats[:subset]
    info = {"name": "ibims", "n": len(mats), "domain": "indoor",
            "min_depth": 1e-3, "max_depth": 10.0, "pred_is_disparity": True}

    def gen():
        for m in mats:
            d = loadmat(str(m), squeeze_me=True, struct_as_record=False)
            data = d["data"]
            rgb = np.asarray(data.rgb)
            depth = np.asarray(data.depth, dtype=np.float32)
            # iBims invalid/transparent masks -> valid where depth>0 and not flagged
            # NOTE: iBims mask_invalid / mask_transp are 1=VALID (used by the
            # official eval via multiplication, despite the "invalid" name).
            valid = np.isfinite(depth) & (depth > 0)
            for mk in ("mask_invalid", "mask_transp"):
                if hasattr(data, mk):
                    valid &= np.asarray(getattr(data, mk)).astype(bool)
            edges = np.asarray(data.edges).astype(bool) if hasattr(data, "edges") else None
            K = None
            if hasattr(data, "calib"):
                calib = np.asarray(data.calib, dtype=np.float64)
                if calib.shape == (3, 3):
                    K = calib.T          # iBims stores K transposed (cx@[2,0], cy@[2,1])
            yield Sample(id=m.stem, rgb=Image.fromarray(rgb.astype(np.uint8)),
                         gt_depth=depth, valid_mask=valid, K=K, gt_edges=edges,
                         meta={"path": str(m)})
    return gen, info


@register("sintel")
def _sintel(root=None, subset=None):
    """MPI Sintel clean + depth (synthetic, sharp GT). Not used in the paper.
    Expects <DQ_DATA>/sintel/training/{clean/<scene>/*.png, depth/<scene>/*.dpt}."""
    root = Path(root or DATASETS_ROOT / "sintel")
    clean = root / "training" / "clean"
    depthd = root / "training" / "depth"
    frames = []
    if clean.exists():
        for scene in sorted(clean.iterdir()):
            for png in sorted(scene.glob("*.png")):
                dpt = depthd / scene.name / (png.stem + ".dpt")
                if dpt.exists():
                    frames.append((png, dpt, scene.name))
    if subset:
        frames = frames[::max(1, len(frames) // subset)][:subset]
    info = {"name": "sintel", "n": len(frames), "domain": "outdoor-synthetic",
            "min_depth": 1e-3, "max_depth": 72.0, "pred_is_disparity": True}

    def _read_dpt(p):
        with open(p, "rb") as f:
            assert np.frombuffer(f.read(4), np.float32)[0] == 202021.25, "bad .dpt magic"
            w = int(np.frombuffer(f.read(4), np.int32)[0])
            h = int(np.frombuffer(f.read(4), np.int32)[0])
            return np.frombuffer(f.read(w * h * 4), np.float32).reshape(h, w).copy()

    def gen():
        for png, dpt, scene in frames:
            depth = _read_dpt(dpt)
            valid = np.isfinite(depth) & (depth > 0) & (depth < info["max_depth"])
            yield Sample(id=f"{scene}_{png.stem}", rgb=Image.open(png).convert("RGB"),
                         gt_depth=depth, valid_mask=valid, K=None, gt_edges=None,
                         meta={"scene": scene})
    return gen, info


@register("nyu")
def _nyu(root=None, subset=None):
    """NYUv2 official 654-image test split (HDF5 files: depth in meters + rgb),
    read from <DQ_DATA>/nyu/extracted/val/official/*.h5.
    Eigen crop + 0-10m cap for aggregate metrics. Fixed NYU intrinsics."""
    import h5py
    root = Path(root or DATASETS_ROOT / "nyu" / "extracted" / "val" / "official")
    h5s = sorted(root.glob("*.h5"))
    if subset:
        h5s = h5s[:subset]
    info = {"name": "nyu", "n": len(h5s), "domain": "indoor",
            "min_depth": 1e-3, "max_depth": 10.0, "pred_is_disparity": True}
    K = np.array([[5.1885790e2, 0, 3.2558245e2],
                  [0, 5.1946961e2, 2.5373617e2], [0, 0, 1.0]])

    def gen():
        for p in h5s:
            with h5py.File(p, "r") as f:
                depth = f["depth"][:].astype(np.float32)
                rgb = np.transpose(f["rgb"][:], (1, 2, 0)).astype(np.uint8)
            H, W = depth.shape
            crop = np.zeros((H, W), bool)
            crop[45:471, 41:601] = True                 # standard NYU Eigen crop
            valid = np.isfinite(depth) & (depth > 1e-3) & (depth < 10.0) & crop
            yield Sample(id=p.stem, rgb=Image.fromarray(rgb), gt_depth=depth,
                         valid_mask=valid, K=K, gt_edges=None, meta={})
    return gen, info


@register("diode")
def _diode(root=None, subset=None, domain="indoors"):
    """DIODE val (FARO laser GT, metric meters). Second sharp-GT indoor dataset,
    independent of iBims (different scenes/camera). No edge annotation, so GT edges
    are derived from depth discontinuity via the Depth-Pro adjacent-ratio rule
    (a pixel is a boundary if its 3x3 max/min depth ratio exceeds 1.05); this is the
    standard boundary definition for datasets without edge labels. No per-image
    intrinsics shipped, so K is None (boundary metrics do not use K)."""
    from scipy import ndimage
    root = Path(root or DATASETS_ROOT / "diode" / "val")
    scans = sorted((root / domain).rglob("*.png")) if (root / domain).exists() else []
    # keep RGB pngs that have a matching depth .npy
    frames = [(p, p.with_name(p.stem + "_depth.npy"),
               p.with_name(p.stem + "_depth_mask.npy")) for p in scans]
    frames = [(p, d, m) for p, d, m in frames if d.exists() and m.exists()]
    if subset:
        frames = frames[::max(1, len(frames) // subset)][:subset]
    xd = 50.0 if domain == "indoors" else 300.0
    info = {"name": "diode", "n": len(frames), "domain": domain,
            "min_depth": 1e-3, "max_depth": xd, "pred_is_disparity": True}

    def _edges(depth, valid, ratio=1.05):
        d = np.where(valid & (depth > 0), depth, np.nan)
        dmax = ndimage.maximum_filter(np.nan_to_num(d, nan=0.0), size=3)
        dmin = ndimage.minimum_filter(np.nan_to_num(d, nan=1e9), size=3)
        allvalid = ndimage.minimum_filter(valid.astype(np.uint8), size=3) > 0
        return (dmax / np.maximum(dmin, 1e-6) > ratio) & valid & allvalid

    def gen():
        for png, dnpy, mnpy in frames:
            depth = np.load(dnpy).astype(np.float32).squeeze()
            mask = np.load(mnpy).astype(np.float32).squeeze()
            valid = (mask > 0) & np.isfinite(depth) & (depth > info["min_depth"]) & (depth < xd)
            yield Sample(id=png.stem, rgb=Image.open(png).convert("RGB"),
                         gt_depth=depth, valid_mask=valid, K=None,
                         gt_edges=_edges(depth, valid), meta={"domain": domain})
    return gen, info


if __name__ == "__main__":
    # smoke: list what's available
    for name in _BUILDERS:
        try:
            gen, info = build(name)
            print(name, "->", info)
        except Exception as e:
            print(name, "-> unavailable:", type(e).__name__, str(e)[:80])
