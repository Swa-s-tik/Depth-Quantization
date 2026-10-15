"""Real deployment measurements: FLOPs, model size, and measured GPU latency for
FP32 / FP16 / real-INT8 (torchao) per model. Latency uses CUDA events with warmup
and many iterations at batch 1. Any precision whose kernels do not run on the GPU
is recorded as NaN. Output results/deploy.csv.

Run: python bench_deploy.py   (needs the GPU environment with torchao + thop)
"""
import gc, argparse
import numpy as np
import pandas as pd
import torch

from common import RESULTS, load_config, seed_everything

CFG = load_config("deploy")
from data.loaders import build
from models.registry import load_model, MODELS

MODELS_BENCH = CFG["models"]
BYTES = {"fp32": 4, "fp16": 2, "int8": 1, "int4": 0.5}
N_WARMUP, N_ITER = CFG["n_warmup"], CFG["n_iter"]


def one_image_inputs(proc, device, dtype):
    gen, _ = build("ibims")
    s = next(iter(gen()))
    inp = proc(images=s.rgb, return_tensors="pt").to(device)
    x = inp["pixel_values"]
    x = x.to(dtype) if torch.is_floating_point(x) else x
    return x


def time_forward(model, x):
    """Median ms per forward over N_ITER, CUDA-event timed."""
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        for _ in range(N_WARMUP):
            model(x)
        torch.cuda.synchronize()
        ts = []
        for _ in range(N_ITER):
            a = torch.cuda.Event(enable_timing=True); b = torch.cuda.Event(enable_timing=True)
            a.record(); model(x); b.record()
            torch.cuda.synchronize()
            ts.append(a.elapsed_time(b))
    peak = torch.cuda.max_memory_allocated() / 1e6
    return float(np.median(ts)), float(peak)


def flops_macs(name):
    """thop MACs + params at the model's native input resolution, FP32 on GPU."""
    try:
        from thop import profile
        model, proc, info = load_model(name, dtype=torch.float32)
        x = one_image_inputs(proc, "cuda", torch.float32)
        macs, params = profile(model, inputs=(x,), verbose=False)
        res = tuple(x.shape[-2:])
        del model; gc.collect(); torch.cuda.empty_cache()
        return macs, params, res
    except Exception as e:
        print(f"  FLOPs FAIL {name}: {type(e).__name__} {str(e)[:80]}")
        return np.nan, np.nan, None


def bench_precision(name, prec):
    """Return dict(ms, peak_mb) for a precision, or NaNs on failure."""
    try:
        base_dtype = torch.float32 if prec in ("fp32",) else torch.float16
        model, proc, info = load_model(name, dtype=base_dtype)
        x = one_image_inputs(proc, "cuda", base_dtype)
        if prec in ("int8", "int4"):
            from torchao.quantization import (quantize_, Int8DynamicActivationInt8WeightConfig,
                                              Int4WeightOnlyConfig)
            cfg = Int8DynamicActivationInt8WeightConfig() if prec == "int8" else Int4WeightOnlyConfig()
            quantize_(model, cfg)
        ms, peak = time_forward(model, x)
        del model; gc.collect(); torch.cuda.empty_cache()
        return {"ms": round(ms, 3), "fps": round(1000.0 / ms, 1), "peak_mb": round(peak, 1)}
    except Exception as e:
        print(f"  {prec} FAIL {name}: {type(e).__name__} {str(e)[:100]}")
        return {"ms": np.nan, "fps": np.nan, "peak_mb": np.nan}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=MODELS_BENCH)
    ap.add_argument("--precisions", nargs="*", default=CFG["precisions"])
    args = ap.parse_args()
    seed_everything(CFG.get("seed", 0))
    rows = []
    for name in args.models:
        print(f"== {name} ==")
        macs, params, res = flops_macs(name)
        nparam = params if np.isfinite(params) else sum(
            p.numel() for p in load_model(name, dtype=torch.float16)[0].parameters())
        for prec in args.precisions:
            r = bench_precision(name, prec)
            size_mb = round(nparam * BYTES.get(prec, np.nan) / 1e6, 1)
            rows.append({"model": name, "precision": prec,
                         "params_M": round(nparam / 1e6, 2),
                         "gmacs": round(macs / 1e9, 2) if np.isfinite(macs) else np.nan,
                         "input_hw": f"{res[0]}x{res[1]}" if res else "",
                         "size_mb": size_mb, **r})
            print(f"  {prec}: {r}  size {size_mb} MB")
        pd.DataFrame(rows).to_csv(RESULTS / "deploy.csv", index=False)
    df = pd.DataFrame(rows)
    print("\n", df.to_string(index=False))
    print("\nwrote results/deploy.csv")


if __name__ == "__main__":
    main()
