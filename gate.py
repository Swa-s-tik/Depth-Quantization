"""Harness check: FP16 depth on NYUv2 against published Depth Anything V2 numbers.

Uses the RELATIVE (affine-invariant) models with least-squares scale-shift
alignment, the protocol of the Depth Anything V2 zero-shot relative-depth table
(arXiv:2406.09414, Table 2). Alignment is applied identically to every
precision, so it cannot confound an FP16-vs-quant delta. The metric-depth
checkpoints carry a resolution/scale offset and are not used as the anchor.

Tolerance: AbsRel +/-10% relative, delta1 +/-1.5 points. Models without a
published reference (DPT) are scored and reported without a pass/fail flag.

Run: python gate.py configs/gate_nyu.yaml   -> results/gate_nyu.csv
"""
import sys
import numpy as np
import pandas as pd

from predict import run_predictions
from evaluate import evaluate_run

PUBLISHED = {
    "nyu": {
        "dav2_small": {"absrel": 0.053, "d1": 0.973},
        "dav2_base":  {"absrel": 0.049, "d1": 0.976},
        "dav2_large": {"absrel": 0.045, "d1": 0.979},
    },
}
TOL_ABSREL_REL = 0.10
TOL_D1_ABS = 0.015


def per_image(dataset, models, subset=None):
    """FP16 predictions (cached once) scored per image, all models stacked."""
    rows = []
    for m in models:
        run_predictions(m, dataset, {"kind": "fp16"}, subset=subset)
        r, _ = evaluate_run(dataset, m, {"kind": "fp16"}, align_mode="ss",
                            subset=subset, compute_boundary=False)
        rows += r
    return pd.DataFrame(rows)


def gate(per_img, dataset, models):
    rows = []
    for m in models:
        df = per_img[per_img.model == m]
        absrel, d1 = float(df.absrel.mean()), float(df.d1.mean())
        pub = PUBLISHED[dataset].get(m, {"absrel": np.nan, "d1": np.nan})
        rel_err = abs(absrel - pub["absrel"]) / pub["absrel"]
        d1_err = abs(d1 - pub["d1"])
        ok = rel_err <= TOL_ABSREL_REL and d1_err <= TOL_D1_ABS if np.isfinite(rel_err) else np.nan
        rows.append({"model": m, "n": len(df), "absrel": round(absrel, 4),
                     "pub_absrel": pub["absrel"], "absrel_rel_err": round(rel_err, 3),
                     "d1": round(d1, 4), "pub_d1": pub["d1"], "d1_err": round(d1_err, 4),
                     "PASS": ok})
        print(f"  {m}: AbsRel {absrel:.4f} (pub {pub['absrel']}, {rel_err*100:.1f}% off)  "
              f"d1 {d1:.4f} (pub {pub['d1']})  -> {ok}", flush=True)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    from common import PER_IMAGE, RESULTS, load_config, seed_everything
    args = [a for a in sys.argv[1:] if a != "--from-per-image"]
    if len(args) != 1:
        sys.exit("usage: python gate.py configs/gate_nyu.yaml [--from-per-image]")
    cfg = load_config(args[0])
    seed_everything(cfg.get("seed", 0))
    pi = PER_IMAGE / f"{cfg['name']}.csv"
    if "--from-per-image" in sys.argv:
        per_img = pd.read_csv(pi, float_precision="round_trip")
    else:
        print(f"GATE {cfg['dataset']} subset={cfg.get('subset')} models={cfg['models']}")
        per_img = per_image(cfg["dataset"], cfg["models"], subset=cfg.get("subset"))
        PER_IMAGE.mkdir(parents=True, exist_ok=True)
        per_img.to_csv(pi, index=False)
    res = gate(per_img, cfg["dataset"], cfg["models"])
    res.to_csv(RESULTS / f"{cfg['name']}.csv", index=False)
    print("\n" + res.to_string(index=False))
