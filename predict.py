"""Prediction cache — generate depth maps ONCE per (dataset, model, precision),
score many times. Caches RAW model output (disparity for relative, metric depth
for metric models) at GT resolution, so alignment + every metric is a cheap
re-read. Resumable: existing .npz are skipped.
"""
import gc
from pathlib import Path
import numpy as np
import torch

from common import CACHE_ROOT
from models.registry import load_model, infer_depth
from quant.fakequant import quantize_model_, quantize_mixed_
import data.loaders as loaders


def precision_tag(spec):
    """spec -> short cache tag. spec keys: kind(fp16|wq), w_bits, per_channel,
    clip, component."""
    if spec.get("kind", "fp16") == "fp16":
        return "fp16"
    if spec.get("kind") == "mixed":
        bm = spec["bits"]
        return "mix_" + "_".join(f"{c[0]}{bm[c]}" for c in sorted(bm))
    parts = [f"w{spec['w_bits']}"]
    comp = spec.get("component", "all")
    if comp != "all":
        parts.append(comp)
    if not spec.get("per_channel", True):
        parts.append("pt")
    if spec.get("clip") is not None:
        parts.append(f"clip{int(spec['clip']*1000)}")
    return "_".join(parts)


def _apply_precision(model, spec):
    kind = spec.get("kind", "fp16")
    if kind == "fp16":
        return 0
    if kind == "mixed":
        n, _ = quantize_mixed_(model, spec["bits"],
                               per_channel=spec.get("per_channel", True),
                               clip_percentile=spec.get("clip"))
        return n
    n, _ = quantize_model_(model, spec["w_bits"],
                           per_channel=spec.get("per_channel", True),
                           clip_percentile=spec.get("clip"),
                           component=spec.get("component", "all"))
    return n


def run_predictions(model_name, dataset_name, spec, subset=None,
                    cache_root=CACHE_ROOT, device="cuda", overwrite=False,
                    dataset_kwargs=None):
    """Generate + cache predictions. Returns (cache_dir, n_written, n_skipped, model_meta)."""
    tag = precision_tag(spec)
    gen, info = loaders.build(dataset_name, subset=subset, **(dataset_kwargs or {}))
    out_dir = Path(cache_root) / dataset_name / model_name / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    # decide which samples still need computing (resume)
    samples = list(gen())
    todo = [s for s in samples if overwrite or not (out_dir / f"{s.id}.npz").exists()]
    if not todo:
        _, meta = load_meta_only(model_name)
        return out_dir, 0, len(samples), meta

    model, proc, meta = load_model(model_name, device=device)
    nq = _apply_precision(model, spec)
    meta["n_quantized_layers"] = nq
    written = 0
    try:
        for s in todo:
            pred = infer_depth(model, proc, s.rgb, out_hw=s.hw, device=device)
            np.savez_compressed(out_dir / f"{s.id}.npz",
                                pred=pred.astype(np.float16),
                                kind=info["pred_is_disparity"])
            written += 1
            del pred
        torch.cuda.synchronize()
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()
    return out_dir, written, len(samples) - written, meta


def load_meta_only(model_name):
    """Lightweight meta without GPU load (for resumed runs)."""
    from models.registry import MODELS
    return None, {"name": model_name, **MODELS[model_name]}


def load_cached_pred(cache_root, dataset_name, model_name, tag, image_id):
    p = Path(cache_root) / dataset_name / model_name / tag / f"{image_id}.npz"
    if not p.exists():
        return None
    return np.load(p)["pred"].astype(np.float32)


if __name__ == "__main__":
    import sys
    # smoke: predict FP16 on a tiny subset of whatever dataset is given
    ds = sys.argv[1] if len(sys.argv) > 1 else "ibims"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    d, w, sk, meta = run_predictions("dav2_small", ds, {"kind": "fp16"}, subset=n)
    print(f"cached {w} (skipped {sk}) -> {d}  model={meta['name']} nparam={meta['nparam']/1e6:.1f}M")
    ex = list(d.glob("*.npz"))
    if ex:
        pr = np.load(ex[0])["pred"]
        print("example pred", ex[0].name, "shape", pr.shape, "range",
              round(float(pr.min()), 2), round(float(pr.max()), 2))
