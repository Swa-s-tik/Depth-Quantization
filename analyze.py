"""Paired analysis of a per-image sweep table (results/<sweep>.csv).

Pairs each quantized run against its FP16 anchor per image, computes bootstrap
CIs + Wilcoxon + Cliff's delta (BH-corrected across the metrics of each
(model, precision)), TOST equivalence on aggregate AbsRel, and a dissociation
verdict. Writes <sweep>_summary.csv and <sweep>_summary_verdicts.csv.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

from stats import summarize_delta, bh_fdr, tost_equivalence, paired_delta

# metric -> True if a HIGHER value means WORSE quality
WORSE_IF_HIGHER = {
    "absrel": True, "rmse": True, "d1": False,
    "dbe_acc": True, "dbe_com": True, "edge_width": True, "edge_sharpness": False,
    "boundary_absrel_k3": True, "interior_absrel_k3": True,
    "boundary_absrel_k2": True, "boundary_absrel_k5": True,
    "flying_frac": True,
}
AGG = ["absrel", "d1", "rmse"]
BND = ["dbe_acc", "dbe_com", "boundary_absrel_k3", "edge_width", "edge_sharpness", "flying_frac"]
CLIFF_MED = 0.33            # medium effect threshold (Vargha-Delaney/Cliff)


def _pair(df_fp, df_q, metric):
    m = pd.merge(df_fp[["image_id", metric]], df_q[["image_id", metric]],
                 on="image_id", suffixes=("_fp", "_q")).dropna()
    return m[f"{metric}_q"].values, m[f"{metric}_fp"].values


def analyze(csv_path, out_summary=None):
    df = pd.read_csv(csv_path)
    rows = []
    verdicts = []
    for (dataset, model), g in df.groupby(["dataset", "model"]):
        fp = g[g.precision == "fp16"]
        if fp.empty:
            continue
        for tag, gq in g[g.precision != "fp16"].groupby("precision"):
            wbits = int(gq.w_bits.iloc[0])
            # collect paired summaries for this (model, tag)
            summ = {}
            pvals = {}
            for metric in AGG + BND:
                if metric not in g.columns:
                    continue
                q, f = _pair(fp, gq, metric)
                if len(q) < 5:
                    continue
                s = summarize_delta(q, f, seed=0)
                # direction: is the change in the WORSE direction?
                worse_dir = (s["mean_delta"] > 0) == WORSE_IF_HIGHER.get(metric, True)
                s["worse"] = bool(worse_dir)
                summ[metric] = s
                pvals[metric] = s["wilcoxon_p"]
                rows.append({"dataset": dataset, "model": model, "precision": tag,
                             "w_bits": wbits, "metric": metric, "n": s["n"],
                             "mean_delta": s["mean_delta"], "ci_lo": s["ci_lo"],
                             "ci_hi": s["ci_hi"], "wilcoxon_p": s["wilcoxon_p"],
                             "cliffs_delta": s["cliffs_delta"], "cohens_d": s["cohens_d"],
                             "worse": s["worse"]})
            if not summ:
                continue
            # BH-correct across all metrics tested for this (model,tag)
            mkeys = list(pvals.keys())
            padj, _ = bh_fdr([pvals[k] for k in mkeys])
            padj = dict(zip(mkeys, padj))
            for r in rows:
                if r["model"] == model and r["precision"] == tag and r["metric"] in padj:
                    r["p_adj"] = padj[r["metric"]]

            # --- decision inputs ---
            # aggregate negligible? TOST on absrel, eps = 5% relative of fp16 mean absrel
            fp_absrel = fp["absrel"].mean()
            eps = 0.05 * fp_absrel
            d_absrel = paired_delta(*_pair(fp, gq, "absrel"))
            tost = tost_equivalence(d_absrel, eps, seed=0)
            agg_equivalent = tost["equivalent"]
            # boundary significantly worse? any boundary metric: p_adj<.05 & |cliff|>=.33 & worse
            bnd_hits = [k for k in BND if k in summ and padj.get(k, 1) < 0.05
                        and abs(summ[k]["cliffs_delta"]) >= CLIFF_MED and summ[k]["worse"]]
            bnd_significant = len(bnd_hits) > 0
            verdicts.append({"dataset": dataset, "model": model, "precision": tag,
                             "w_bits": wbits, "agg_equivalent": agg_equivalent,
                             "tost_ci": (round(tost["lo"], 5), round(tost["hi"], 5)),
                             "eps": round(eps, 5), "bnd_significant": bnd_significant,
                             "bnd_hits": bnd_hits})

    summ_df = pd.DataFrame(rows)
    vdf = pd.DataFrame(verdicts)
    if out_summary:
        summ_df.to_csv(out_summary, index=False)
        vdf.to_csv(str(out_summary).replace(".csv", "_verdicts.csv"), index=False)

    # ---- overall verdict ----
    verdict = _overall_verdict(vdf)
    return summ_df, vdf, verdict


def sensitivity_gap(summary_df, cliff_thresh=0.15):
    """Onset bit-width per (model, metric): the HIGHEST bit-width (least
    compression) at which the metric is significantly WORSE than FP16
    (p_adj<0.05, |Cliff's|>=thresh, worse direction). A higher onset = the metric
    detects damage EARLIER = more sensitive. The boundary-minus-aggregate onset
    difference (in bits) is the sensitivity gap that motivates the PIVOT thesis.
    """
    s = summary_df.copy()
    if "p_adj" not in s.columns:
        s["p_adj"] = s["wilcoxon_p"]
    hit = s[(s.worse) & (s.p_adj < 0.05) & (s.cliffs_delta.abs() >= cliff_thresh)]
    rows = []
    for (model, metric), g in s.groupby(["model", "metric"]):
        h = hit[(hit.model == model) & (hit.metric == metric)]
        onset = int(h.w_bits.max()) if len(h) else -1     # -1 = never significant
        rows.append({"model": model, "metric": metric, "onset_bit": onset,
                     "family": "aggregate" if metric in AGG else "boundary"})
    ot = pd.DataFrame(rows)
    # gap per model = max boundary onset - max aggregate onset
    gaps = []
    for model, g in ot.groupby("model"):
        agg_on = g[g.family == "aggregate"].onset_bit.max()
        bnd_on = g[g.family == "boundary"].onset_bit.max()
        gaps.append({"model": model, "aggregate_onset": agg_on,
                     "boundary_onset": bnd_on, "gap_bits": bnd_on - agg_on})
    return ot.sort_values(["model", "family", "metric"]), pd.DataFrame(gaps)


def _overall_verdict(vdf):
    if vdf.empty:
        return "INSUFFICIENT DATA"
    # dissociation anywhere: aggregate equivalent AND boundary significantly worse
    diss = vdf[(vdf.agg_equivalent) & (vdf.bnd_significant)]
    coupled = vdf[(~vdf.agg_equivalent) & (vdf.bnd_significant)]
    any_bnd = vdf.bnd_significant.any()
    if len(diss) > 0:
        return ("GO — dissociation: aggregate equivalent (TOST) while boundary "
                f"significantly degraded at: "
                + "; ".join(f"{r.model}/{r.precision}({r.bnd_hits})" for r in diss.itertuples()))
    if not any_bnd:
        return ("STOP-AND-REPORT — boundary fidelity preserved at all tested bit-widths; "
                "extend to W3/W2 + a second lever to find the breaking point.")
    if len(coupled) > 0:
        return ("PIVOT — coupled: boundary degrades but aggregate also moves; reframe as "
                "sensitivity-gap (boundary metrics as early-warning). Quantify bit-width lead.")
    return "MIXED — inspect verdicts table"


if __name__ == "__main__":
    from common import seed_everything
    seed_everything()
    if len(sys.argv) != 2:
        sys.exit("usage: python analyze.py results/<sweep>.csv")
    csv = sys.argv[1]
    out = str(csv).replace(".csv", "_summary.csv")
    s, v, verdict = analyze(csv, out_summary=out)
    pd.set_option("display.width", 200, "display.max_columns", 30)
    print("=== per-metric paired summary ==="); print(s.to_string(index=False))
    print("\n=== decision inputs per (model,precision) ==="); print(v.to_string(index=False))
    if s["w_bits"].nunique() > 1:
        ot, gaps = sensitivity_gap(s)
        print("\n=== onset bit-width per metric (higher = detects damage earlier) ===")
        print(ot.to_string(index=False))
        print("\n=== SENSITIVITY GAP (boundary_onset - aggregate_onset, in bits) ===")
        print(gaps.to_string(index=False))
    print("\n=== VERDICT ===\n" + verdict)
