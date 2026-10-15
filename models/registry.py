"""Model zoo — one code path (AutoModelForDepthEstimation) for the whole study.

`kind` decides the eval protocol:
  relative -> affine-invariant disparity-like output; align with scale-shift.
  metric   -> metric depth in meters; NO alignment (used for the clean gate).
Predictions are generated once and cached to disk by predict.py; every scorer
reads the cache. We log sum(p.numel()) at load — never trust a README count.
"""
import numpy as np
import torch
from transformers import AutoModelForDepthEstimation, AutoImageProcessor

MODELS = {
    # relative (primary scientific arm) — DINOv2 backbone + DPT head
    "dav2_small": dict(repo="depth-anything/Depth-Anything-V2-Small-hf", family="dav2", backbone="dinov2-s", kind="relative"),
    "dav2_base":  dict(repo="depth-anything/Depth-Anything-V2-Base-hf",  family="dav2", backbone="dinov2-b", kind="relative"),
    "dav2_large": dict(repo="depth-anything/Depth-Anything-V2-Large-hf", family="dav2", backbone="dinov2-l", kind="relative"),
    # metric fine-tuned (GATE anchor — no alignment confound)
    "dav2_metric_indoor_small": dict(repo="depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf", family="dav2", backbone="dinov2-s", kind="metric", domain="indoor"),
    "dav2_metric_indoor_base":  dict(repo="depth-anything/Depth-Anything-V2-Metric-Indoor-Base-hf",  family="dav2", backbone="dinov2-b", kind="metric", domain="indoor"),
    "dav2_metric_indoor_large": dict(repo="depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf", family="dav2", backbone="dinov2-l", kind="metric", domain="indoor"),
    "dav2_metric_outdoor_small": dict(repo="depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf", family="dav2", backbone="dinov2-s", kind="metric", domain="outdoor"),
    "dav2_metric_outdoor_base":  dict(repo="depth-anything/Depth-Anything-V2-Metric-Outdoor-Base-hf",  family="dav2", backbone="dinov2-b", kind="metric", domain="outdoor"),
    "dav2_metric_outdoor_large": dict(repo="depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf", family="dav2", backbone="dinov2-l", kind="metric", domain="outdoor"),
    # control family (architecture axis)
    "dpt_large":  dict(repo="Intel/dpt-large",        family="dpt", backbone="vit-l", kind="relative"),
    "dpt_hybrid": dict(repo="Intel/dpt-hybrid-midas", family="dpt", backbone="resnet-vit", kind="relative"),
    "zoedepth":   dict(repo="Intel/zoedepth-nyu-kitti", family="zoe", backbone="beit-l", kind="metric"),
}


def load_model(name, dtype=torch.float16, device="cuda"):
    spec = MODELS[name]
    proc = AutoImageProcessor.from_pretrained(spec["repo"])
    model = AutoModelForDepthEstimation.from_pretrained(spec["repo"], torch_dtype=dtype)
    model.eval().to(device)
    nparam = sum(p.numel() for p in model.parameters())
    return model, proc, {"name": name, **spec, "nparam": int(nparam),
                         "mb_fp16": round(nparam * 2 / 1e6, 1)}


@torch.inference_mode()
def infer_depth(model, proc, pil_image, out_hw=None, device="cuda"):
    """Return raw model depth map (numpy float32, HxW) resized to out_hw (H,W).

    For 'relative' models this is disparity-like (higher = closer). For 'metric'
    models it is depth in meters. Resizing uses bilinear at model dtype.
    """
    inputs = proc(images=pil_image, return_tensors="pt").to(device)
    inputs = {k: (v.half() if torch.is_floating_point(v) else v) if model.dtype == torch.float16
              else v for k, v in inputs.items()}
    out = model(**inputs)
    pred = out.predicted_depth                       # (1, h, w) or (1,1,h,w)
    if pred.dim() == 3:
        pred = pred.unsqueeze(1)
    if out_hw is not None:
        pred = torch.nn.functional.interpolate(pred.float(), size=out_hw,
                                               mode="bilinear", align_corners=False)
    return pred.squeeze().float().cpu().numpy()


if __name__ == "__main__":
    # smoke: load smallest model, run on a random image, sanity-check output
    from PIL import Image
    m, p, meta = load_model("dav2_small")
    print("loaded", meta)
    img = Image.fromarray((np.random.rand(240, 320, 3) * 255).astype(np.uint8))
    d = infer_depth(m, p, img, out_hw=(240, 320))
    print("depth shape", d.shape, "range", round(float(d.min()), 3), round(float(d.max()), 3),
          "finite", bool(np.isfinite(d).all()))
    assert d.shape == (240, 320) and np.isfinite(d).all() and d.max() > d.min()
    print("registry smoke OK")
