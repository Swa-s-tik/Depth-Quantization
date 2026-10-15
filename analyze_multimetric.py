"""Multi-metric dissociation (Table 3) and Dissociation Index (Table 6), from the
per-image bit-sweep table results/phase3_bitsweep.csv (no data or cache needed).

(a) FP16->quant Cliff's delta for EVERY conventional aggregate metric (AbsRel,
    SqRel, RMSE, RMSE-log, d1, d2, d3) beside the boundary metrics, per bit-width,
    oriented so that positive = quantized worse.
(b) Dissociation Index DI = (flying-pixel Cliff's delta) - max_m |aggregate
    Cliff's delta_m|, one aggregate-vs-structure score per bit-width.

Outputs results/multimetric.csv and results/dissociation_index.csv.
Command: python analyze_multimetric.py
"""
import numpy as np
import pandas as pd

from common import RESULTS as RES, load_config, seed_everything
from stats import cliffs_delta

CFG = load_config("multimetric")
AGG = ["absrel", "sqrel", "rmse", "rmse_log", "d1", "d2", "d3"]
BND = ["dbe_com", "edge_width", "flying_frac", "boundary_absrel_k3"]
WORSE_LOWER = {"d1", "d2", "d3"}  # higher is better -> worse means it dropped


def cliff_worse(q, f, metric):
    """Cliff's delta oriented so positive = quantized worse."""
    q = np.asarray(q, float); f = np.asarray(f, float)
    d = cliffs_delta(q, f)
    return -d if metric in WORSE_LOWER else d


# ---------- multi-metric effect sizes + dissociation index ----------
def multimetric_and_di():
    df = pd.read_csv(RES / f"{CFG['sweep']}.csv")
    rows, di_rows = [], []
    for model in CFG["models"]:
        sub = df[df.model == model]
        fp = sub[sub.precision == "fp16"].set_index("image_id")
        for bits in CFG["bits"]:
            q = sub[sub.w_bits == bits].set_index("image_id")
            if len(q) == 0:
                continue
            ids = fp.index.intersection(q.index)
            rec = {"model": model, "w_bits": bits}
            eff = {}
            for m in AGG + BND:
                if m not in fp.columns:
                    continue
                fv = fp.loc[ids, m].to_numpy(); qv = q.loc[ids, m].to_numpy()
                c = cliff_worse(qv, fv, m)
                rec[f"{m}_cliff"] = round(c, 3)
                rec[f"{m}_delta"] = round(float(np.nanmean(qv - fv)), 4)
                eff[m] = c
            rows.append(rec)
            # dissociation index: boundary (flying/dbe) minus worst aggregate
            agg_max = max(abs(eff.get(a, 0)) for a in AGG if a in eff)
            bnd_val = eff.get("flying_frac", 0)
            di_rows.append({"model": model, "w_bits": bits,
                            "boundary_cliff_flying": round(eff.get("flying_frac", np.nan), 3),
                            "boundary_cliff_dbe": round(eff.get("dbe_com", np.nan), 3),
                            "max_aggregate_cliff": round(agg_max, 3),
                            "dissociation_index": round(bnd_val - agg_max, 3)})
    pd.DataFrame(rows).to_csv(RES / "multimetric.csv", index=False)
    pd.DataFrame(di_rows).to_csv(RES / "dissociation_index.csv", index=False)
    return pd.DataFrame(rows), pd.DataFrame(di_rows)


if __name__ == "__main__":
    seed_everything(CFG.get("seed", 0))
    mm, di = multimetric_and_di()
    cols = ["model", "w_bits"] + [f"{m}_cliff" for m in ["absrel", "rmse", "d1", "d2", "d3", "dbe_com", "flying_frac"]]
    print(mm[cols].to_string(index=False))
    print("\n" + di.to_string(index=False))
