"""Equivalence of the aggregate metrics at the six-bit window (Tables 2 and 4),
from the shipped per-image bit-sweep table (no data or cache needed).

For DAv2 Base and Large at W6: the paired delta of every aggregate metric, its
90% bootstrap CI, the equivalence margin eps (5% of the FP16 mean) and the TOST
verdict (equivalent iff the CI lies inside [-eps, eps]); then the verdict at
margins of 3/5/7/10% and the smallest margin (in % of the FP16 mean) at which the
metric passes, max(|ci_lo|, |ci_hi|) / |FP16 mean|.

Outputs results/equivalence.csv and results/equivalence_margins.csv.
Command: python analyze_equivalence.py
"""
import numpy as np
import pandas as pd

from common import RESULTS as RES, load_config, seed_everything
from stats import tost_equivalence

CFG = load_config("equivalence")


def equivalence_table():
    raw = pd.read_csv(RES / f"{CFG['sweep']}.csv")
    rows, margin_rows = [], []
    for model in CFG["models"]:
        sub = raw[raw.model == model]
        fp = sub[sub.precision == "fp16"].set_index("image_id")
        q = sub[sub.w_bits == CFG["w_bits"]].set_index("image_id")
        ids = fp.index.intersection(q.index)
        for m in CFG["metrics"]:
            if m not in fp.columns:
                continue
            fv = fp.loc[ids, m].to_numpy(); qv = q.loc[ids, m].to_numpy()
            deltas = qv - fv
            mu = abs(np.nanmean(fv))
            t = tost_equivalence(deltas, CFG["margin"] * mu, ci=CFG["ci"])
            rows.append({"model": model, "metric": m,
                         "fp16_mean": round(float(np.nanmean(fv)), 4),
                         "delta": round(t["mean"], 5),
                         "ci_lo": round(t["lo"], 5), "ci_hi": round(t["hi"], 5),
                         "eps": round(t["eps"], 5),
                         "equivalent": t["equivalent"]})
            r = {"model": model, "metric": m}
            for pct in CFG["margin_sweep_pct"]:
                eps = pct / 100 * mu
                r[f"equivalent_at_{pct}pct"] = bool(t["lo"] >= -eps and t["hi"] <= eps)
            r["passes_at_pct"] = round(100 * max(abs(t["lo"]), abs(t["hi"])) / mu, 3)
            margin_rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(RES / "equivalence.csv", index=False)
    mg = pd.DataFrame(margin_rows)
    mg.to_csv(RES / "equivalence_margins.csv", index=False)
    return df, mg


if __name__ == "__main__":
    seed_everything(CFG.get("seed", 0))
    eq, mg = equivalence_table()
    print(eq.to_string(index=False))
    print("\n" + mg.to_string(index=False))
