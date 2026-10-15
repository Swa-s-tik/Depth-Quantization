"""Every table of the paper, plus the numbers quoted in the text, from the shipped
results (no datasets, cache or GPU needed).

Each table is written to results/tables/table_<NN>_<name>.csv with the values
formatted exactly as printed in the paper; the in-text numbers go to
results/tables/in_text_numbers.json together with their source file and the value
printed in the paper. No metric is recomputed here: the inputs are the per-image
tables and the aggregate results under results/.
Command: python make_tables.py
"""
import json

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from common import RESULTS as R, TABLES as T, seed_everything
from stats import cliffs_delta, cliffs_delta_ci

NICE = {"dav2_small": "DAv2-small", "dav2_base": "DAv2-base", "dav2_large": "DAv2-large"}


def f2(x):
    return f"{x:.2f}"


def f3(x):
    return f"{x:.3f}"


def f4(x):
    return f"{x:.4f}"


def _summary(name):
    return pd.read_csv(R / f"{name}_summary.csv")


def _cliff(s, model, bits, metric, precision=None):
    sel = (s.model == model) & (s.metric == metric)
    sel &= (s.precision == precision) if precision else (s.w_bits == bits)
    return float(s[sel].cliffs_delta.iloc[0])


def _closest(df, method, target):
    """Row of a Pareto table whose config is <method>@<avg bits> closest to target."""
    sub = df[df.config.str.startswith(method + "@")]
    k = (sub.config.str.split("@").str[1].astype(float) - target).abs().idxmin()
    return sub.loc[k]


# ---------------------------------------------------------------- tables
def table_02_tost_margin():
    m = pd.read_csv(R / "equivalence_margins.csv")
    names = {"absrel": "AbsRel", "rmse": "RMSE", "d1": "delta1"}
    rows = []
    for _, r in m[m.metric.isin(names)].iterrows():
        rows.append({"model": r.model.replace("dav2_", "").capitalize(), "metric": names[r.metric],
                     **{f"eps={p}%": "yes" if r[f"equivalent_at_{p}pct"] else "no" for p in (3, 5, 7, 10)},
                     "passes at eps >=": f"{r.passes_at_pct:.1f}%"})
    return pd.DataFrame(rows)


def table_03_multimetric():
    mm = pd.read_csv(R / "multimetric.csv")
    cols = ["absrel", "rmse", "d1", "d2", "d3", "dbe_com", "flying_frac"]
    heads = ["AbsRel", "RMSE", "delta1", "delta2", "delta3", "DBE-c", "flying"]
    rows = []
    for model in ["dav2_base", "dav2_large"]:
        for _, r in mm[mm.model == model].iterrows():
            rows.append({"model": NICE[model], "W": int(r.w_bits),
                         **{h: f2(r[f"{c}_cliff"]) for h, c in zip(heads, cols)}})
    return pd.DataFrame(rows)


def table_04_equivalence():
    eq = pd.read_csv(R / "equivalence.csv")
    names = {"absrel": "AbsRel", "rmse": "RMSE", "d1": "delta1"}
    rows = []
    for model in ["dav2_base", "dav2_large"]:
        for m, nm in names.items():
            r = eq[(eq.model == model) & (eq.metric == m)].iloc[0]
            rows.append({"model": NICE[model], "metric": nm, "delta": f4(r.delta),
                         "90% CI": f"[{r.ci_lo:.4f}, {r.ci_hi:.4f}]", "eps": f4(r.eps),
                         "TOST equiv.": "yes" if r.equivalent else "no"})
    return pd.DataFrame(rows)


def table_05_dissociation():
    s = _summary("phase3_bitsweep")
    rows = []
    for model in ["dav2_base", "dav2_large"]:
        for b in [8, 7, 6, 5, 4, 3]:
            a = s[(s.model == model) & (s.w_bits == b) & (s.metric == "absrel")].iloc[0]
            rows.append({"model": NICE[model], "W": b, "AbsRel delta": f4(a.mean_delta),
                         "DBE-comp": f2(_cliff(s, model, b, "dbe_com")),
                         "flying": f2(_cliff(s, model, b, "flying_frac")),
                         "edge-w": f2(_cliff(s, model, b, "edge_width"))})
    return pd.DataFrame(rows)


def table_06_dissociation_index():
    di = pd.read_csv(R / "dissociation_index.csv")
    di = di[di.model.isin(["dav2_base", "dav2_large"])]
    return pd.DataFrame([{"model": NICE[r.model], "W": int(r.w_bits),
                          "boundary (flying)": f2(r.boundary_cliff_flying),
                          "max aggregate": f2(r.max_aggregate_cliff),
                          "DI": f2(r.dissociation_index)} for _, r in di.iterrows()])


def table_07_frozen_alignment():
    fa = pd.read_csv(R / "frozen_align.csv")
    fa = fa[fa.model == "dav2_large"].set_index("w_bits")
    return pd.DataFrame([{"bit-width": f"W{b}",
                          "AbsRel refit": f2(fa.loc[b, "absrel_refit_cliff"]),
                          "AbsRel frozen": f2(fa.loc[b, "absrel_frozen_cliff"]),
                          "DBE-comp refit": f2(fa.loc[b, "dbe_refit_cliff"]),
                          "DBE-comp frozen": f2(fa.loc[b, "dbe_frozen_cliff"])} for b in (6, 4)])


def _p(p):
    if p < 1e-9:
        return "<1e-9"
    return f"{p:.2f}" if p >= 0.01 else f"{p:.1e}"


def table_08_boundary_definition():
    bs = pd.read_csv(R / "boundary_sensitivity.csv")
    lab = {"bnd_k": "band-AbsRel, k px", "fly_g": "flying, gap factor"}
    return pd.DataFrame([{"model": NICE[r.model], "boundary definition": lab[r.definition],
                          "param": f"{r.param:g}", "FP16->W6 cliff": f2(r.cliffs_delta),
                          "p": _p(r.wilcoxon_p)} for _, r in bs.iterrows()])


def table_09_downstream():
    d = pd.read_csv(R / "downstream_recon.csv").set_index("config")
    rows = [("uniform W6", "uniform@6"), ("head-protected W6", "headprot@6")]
    for k in [c for c in d.index if c.startswith("s2q@") and 4.5 < float(c[4:]) < 6.5]:
        rows.append((f"S2Q ({float(k[4:]):.1f}-bit avg)", k))
    return pd.DataFrame([{"config": n, "Chamfer": f3(d.loc[k, "chamfer_cliff"]),
                          "reconstruction-F": f3(d.loc[k, "recon_f_cliff"])} for n, k in rows])


def table_10_memory():
    mem = pd.read_csv(R / "mitigation_memory.csv")
    return pd.DataFrame([{"model": NICE[r.model], "config": r.config,
                          "head (M)": f3(r.head_params_M), "memory (MB)": f"{r.MB:.1f}",
                          "vs W6": "-" if r.config == "FP16" else f"{r.overhead_vs_W6_pct:.2f}%"}
                         for _, r in mem.iterrows()])


def table_11_mitigation():
    m = pd.read_csv(R / "phase4_mitigation.csv").groupby(["model", "precision"])[
        ["absrel", "dbe_com", "flying_frac"]].mean()
    names = {"fp16": "FP16", "w6": "uniform W6", "mix_e6_n6": "head-protected W6 (iso-mem)",
             "w4": "uniform W4", "mix_e4_n4": "head-protected W4 (iso-mem)"}
    return pd.DataFrame([{"model": NICE[model], "config": names[p],
                          "AbsRel": f4(m.loc[(model, p), "absrel"]),
                          "DBE-comp": f2(m.loc[(model, p), "dbe_com"]),
                          "flying": f3(m.loc[(model, p), "flying_frac"])}
                         for model in ["dav2_base", "dav2_large"] for p in names])


def table_12_components():
    s = _summary("phase4_mechanism")
    rows = []
    for model, bits, lab in [("dav2_large", 6, "Large, 6-bit"), ("dav2_base", 4, "Base, 4-bit")]:
        for comp in ["encoder", "neck", "head"]:
            tag = f"w{bits}_{comp}"
            rows.append({"setting": lab, "component": comp,
                         **{h: f2(_cliff(s, model, None, m, precision=tag)) for h, m in
                            [("AbsRel", "absrel"), ("DBE-comp", "dbe_com"),
                             ("edge-w", "edge_width"), ("flying", "flying_frac")]}})
    return pd.DataFrame(rows)


def table_13_s2q_architectures():
    rows = []
    for model, name in [("dav2_large", "Depth Anything V2 Large"), ("dpt_hybrid", "DPT-Hybrid"),
                        ("zoedepth", "ZoeDepth")]:
        d = pd.read_csv(R / f"s2q_pareto_{model}.csv")
        rows.append({"model": name,
                     **{h: f3(_closest(d, m, 6.0).dbe_com_cliff) for h, m in
                        [("uniform", "uniform"), ("head-protect", "headprot"),
                         ("aggregate-guided", "aggguided"), ("S2Q", "s2q")]}})
    return pd.DataFrame(rows)


def table_14_s2q_transfer():
    d = pd.read_csv(R / "s2q_diode_transfer.csv").set_index("config")
    names = {"uniform6": "uniform", "headprot6": "head-protected", "s2q": "S2Q (allocation fit on iBims-1)"}
    return pd.DataFrame([{"config (six-bit average)": n, "DBE-comp": f3(d.loc[k, "dbe_com_cliff"]),
                          "edge-width": f3(d.loc[k, "edge_width_cliff"])} for k, n in names.items()])


def table_15_s2q_matched():
    d = pd.read_csv(R / "s2q_pareto_dav2_large.csv")
    rows = []
    for m, n in [("uniform", "uniform"), ("aggguided", "aggregate-guided"),
                 ("headprot", "head-protection"), ("s2q", "S2Q (ours)")]:
        r = _closest(d, m, 6.0)
        rows.append({"method": n, "AbsRel delta": f4(r.absrel_delta),
                     "DBE-comp": f3(r.dbe_com_cliff), "flying": f3(r.flying_frac_cliff)})
    return pd.DataFrame(rows)


def table_16_cross_dataset():
    ib, di = _summary("phase3_bitsweep"), _summary("phase6_diode_bitsweep")
    fd = pd.read_csv(R / "flying_on_diode.csv")
    rows = []
    for model, name in [("dav2_base", "DAv2-Base"), ("dav2_large", "DAv2-Large")]:
        for ds, s in [("iBims-1", ib), ("DIODE", di)]:
            fly = (_cliff(s, model, 6, "flying_frac") if ds == "iBims-1" else
                   float(fd[(fd.model == model) & (fd.w_bits == 6)].flying_cliff.iloc[0]))
            rows.append({"model": name, "dataset": ds,
                         **{h: f3(_cliff(s, model, 6, m)) for h, m in
                            [("AbsRel", "absrel"), ("RMSE", "rmse"), ("DBE-comp", "dbe_com"),
                             ("edge-width", "edge_width")]},
                         "flying": f3(fly)})
    return pd.DataFrame(rows)


def table_17_gptq():
    g = pd.concat([pd.read_csv(R / "gptq_rtn.csv"), pd.read_csv(R / "gptq_eval.csv")])
    rows = []
    for b in (8, 6, 4):
        for meth in ("rtn", "gptq"):
            r = g[(g.method == meth) & (g.w_bits == b)].iloc[0]
            rows.append({"bit-width": f"W{b}", "method": meth.upper(),
                         **{h: f3(r[f"{m}_cliff"]) for h, m in
                            [("AbsRel", "absrel"), ("DBE-comp", "dbe_com"),
                             ("edge-width", "edge_width"), ("flying", "flying")]}})
    return pd.DataFrame(rows)


def table_18_groupwise():
    s, gw = _summary("phase3_bitsweep"), pd.read_csv(R / "gw_eval.csv").set_index("scheme")
    mets = [("AbsRel", "absrel"), ("DBE-comp", "dbe_com"), ("edge-width", "edge_width"),
            ("flying", "flying_frac")]
    rows = [{"scheme": n, **{h: f3(_cliff(s, "dav2_large", b, m)) for h, m in mets}}
            for n, b in [("per-channel W6 (danger window)", 6), ("per-channel W4", 4)]]
    for k, n in [("gw4_g128", "group-wise W4, G=128"), ("gw4_g64", "group-wise W4, G=64")]:
        rows.append({"scheme": n, **{h: f3(gw.loc[k, f"{m.replace('_frac', '')}_cliff"])
                                     for h, m in mets}})
    return pd.DataFrame(rows)


def table_19_weight_activation():
    w = pd.read_csv(R / "w4a4_probe.csv").set_index("config")
    names = [("W6 uniform, A16", "weight-only W6 (uniform)"),
             ("W6 headprot, A8", "W6 head-protected + 8-bit activations"),
             ("W6 uniform, A4", "W6 uniform + 4-bit activations")]
    return pd.DataFrame([{"configuration": n, "AbsRel": f3(w.loc[k, "absrel_cliff"]),
                          "DBE-comp": f3(w.loc[k, "dbe_com_cliff"]),
                          "flying": f3(w.loc[k, "flying_frac_cliff"])} for k, n in names])


def table_20_schemes():
    a = pd.read_csv(R / "alt_ptq.csv")
    names = {"pc_sym": "per-channel symmetric", "pt_sym": "per-tensor symmetric",
             "pc_asym": "per-channel asymmetric"}
    rows = []
    for k, n in names.items():
        for b in (8, 6, 4):
            g = a[(a.scheme == k) & (a.w_bits == b)].set_index("metric").cliffs_delta
            rows.append({"scheme": n, "W": b, "AbsRel": f2(g["absrel"]), "DBE-c": f2(g["dbe_com"]),
                         "edge-w": f2(g["edge_width"]), "flying": f2(g["flying_frac"])})
    return pd.DataFrame(rows)


def table_21_deployment():
    d = pd.read_csv(R / "deploy.csv")
    names = {"dav2_small": "DAv2-Small", "dav2_base": "DAv2-Base", "dav2_large": "DAv2-Large",
             "dpt_large": "DPT-Large", "dpt_hybrid": "DPT-Hybrid"}
    return pd.DataFrame([{"model": names[r.model], "precision": r.precision.upper(),
                          "size (MB)": f"{r.size_mb:.1f}", "GMACs": f"{r.gmacs:.1f}",
                          "latency (ms)": f"{r.ms:.1f}", "FPS": f"{r.fps:.1f}"}
                         for _, r in d.iterrows()])


def table_22_int8():
    s = pd.read_csv(R / "int8_real_summary.csv")
    fmt = {"absrel": ("AbsRel", f4), "dbe_com": ("DBE-completeness", f2),
           "edge_width": ("edge width", f3), "flying_frac": ("flying-pixel fraction", f3)}
    rows = []
    for model, name in [("dav2_large", "DAv2-Large"), ("dav2_base", "DAv2-Base")]:
        for m, (n, f) in fmt.items():
            r = s[(s.model == model) & (s.metric == m)].iloc[0]
            rows.append({"model": name, "metric": n, "FP16": f(r.fp16_mean),
                         "real INT8": f(r.int8_mean), "Cliff's delta": f"{r.cliffs_delta:+.3f}"})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- in-text numbers
def _pairs(sweep, model, bits, metric):
    df = pd.read_csv(R / f"{sweep}.csv")
    sub = df[df.model == model]
    fp = sub[sub.precision == "fp16"].set_index("image_id")[metric]
    q = sub[sub.w_bits == bits].set_index("image_id")[metric]
    ids = fp.index.intersection(q.index)
    f, qq = fp.loc[ids].to_numpy(), q.loc[ids].to_numpy()
    m = np.isfinite(f) & np.isfinite(qq)
    return qq[m], f[m]


def cliff_ci(q, f, ci):
    lo, hi = cliffs_delta_ci(q, f, ci=ci)
    return f"[{lo:.2f}, {hi:.2f}]"


def in_text_numbers():
    out = {}

    def put(key, value, source, paper, where):
        out[key] = {"value": value, "paper": paper, "where": where, "source": source}

    gate = pd.read_csv(R / "gate_nyu.csv").set_index("model")
    for m, p in [("dav2_small", "0.062"), ("dav2_base", "0.057"), ("dav2_large", "0.055")]:
        put(f"nyu_fp16_absrel/{m}", f3(gate.loc[m, "absrel"]), "results/gate_nyu.csv", p, "Sec. 4.3")
    put("nyu_offset_vs_published_pct",
        f"{100 * gate.absrel_rel_err.min():.0f} to {100 * gate.absrel_rel_err.max():.0f}",
        "results/gate_nyu.csv", "16 to 22", "Sec. 4.3")

    s = _summary("phase3_bitsweep")
    put("abstract/max_flying_cliff", f2(_cliff(s, "dav2_large", 6, "flying_frac")),
        "results/phase3_bitsweep_summary.csv", "0.76", "Abstract")
    mm = pd.read_csv(R / "multimetric.csv")
    r6 = mm[(mm.model == "dav2_large") & (mm.w_bits == 6)].iloc[0]
    put("abstract/max_abs_aggregate_cliff_large_w6",
        f2(max(abs(r6[f"{a}_cliff"]) for a in ["absrel", "sqrel", "rmse", "rmse_log", "d1", "d2", "d3"])),
        "results/multimetric.csv", "0.09", "Abstract")
    put("sec13/padj_max_boundary_w6",
        f"{s[(s.w_bits == 6) & s.model.isin(['dav2_base', 'dav2_large']) & s.metric.isin(['dbe_com', 'edge_width', 'flying_frac'])].p_adj.max():.1e}",
        "results/phase3_bitsweep_summary.csv", "< 1e-8", "Sec. 13")

    # Sec. 13 / Sec. 5.2 confidence intervals of Cliff's delta (paired bootstrap, Sec. 4.2)
    for metric, paper in [("flying_frac", "[0.69, 0.83]"), ("dbe_com", "[0.28, 0.43]"),
                          ("edge_width", "[0.28, 0.38]"), ("absrel", "[0.05, 0.10]")]:
        q, f = _pairs("phase3_bitsweep", "dav2_large", 6, metric)
        put(f"sec13/large_w6_{metric}_cliff_90ci", cliff_ci(q, f, 90),
            "results/phase3_bitsweep.csv", paper, "Sec. 13")
    for bits, metric, paper in [(7, "flying_frac", "[0.32, 0.48]"), (6, "flying_frac", "[0.22, 0.36]"),
                                (7, "dbe_com", "[0.10, 0.22]"), (6, "dbe_com", "[0.07, 0.19]")]:
        q, f = _pairs("phase3_bitsweep", "dav2_base", bits, metric)
        put(f"sec5.2/base_w{bits}_{metric}_cliff_95ci", cliff_ci(q, f, 95),
            "results/phase3_bitsweep.csv", paper, "Sec. 5.2")

    # Sec. 13: per-image change in band AbsRel over N_3 vs AbsRel over the remaining pixels
    sweep = pd.read_csv(R / "phase3_bitsweep.csv")
    for model, paper in [("dav2_large", "0.15 [-0.07, 0.36], n=85"), ("dav2_base", "0.59 [0.41, 0.73], n=85")]:
        sub = sweep[sweep.model == model]
        fp = sub[sub.precision == "fp16"].set_index("image_id")
        q = sub[sub.w_bits == 6].set_index("image_id")
        g = (q.interior_absrel_k3 - fp.interior_absrel_k3).to_numpy()
        b = (q.boundary_absrel_k3 - fp.boundary_absrel_k3).to_numpy()
        keep = np.isfinite(g) & np.isfinite(b)
        g, b = g[keep], b[keep]
        rng = np.random.default_rng(0)
        idx = rng.integers(0, len(g), size=(10000, len(g)))
        lo, hi = np.nanpercentile([spearmanr(g[i], b[i])[0] for i in idx], [2.5, 97.5])
        put(f"sec13/boundary_band_vs_rest_spearman/{model}",
            f"{spearmanr(g, b)[0]:.2f} [{lo:.2f}, {hi:.2f}], n={len(g)}",
            "results/phase3_bitsweep.csv", paper, "Sec. 13")

    bands = pd.read_csv(R / "bands.csv").set_index(["model", "precision", "band"]).mean_absrel
    put("sec6.1/large_fp16_interior_absrel", f3(bands[("dav2_large", "fp16", "interior")]),
        "results/bands.csv", "0.031", "Sec. 6.1")
    put("sec6.1/large_fp16_0-1px_absrel", f3(bands[("dav2_large", "fp16", "0-1")]),
        "results/bands.csv", "0.138", "Sec. 6.1")

    pf = pd.read_csv(R / "per_image" / "boundary_f.csv", float_precision="round_trip")
    pf = pf[pf.model == "dav2_large"]
    for tol, bits, paper, fmt in [(2, 6, "0.006", f3), (1, 6, "0.018", f3), (1, 4, "0.035", f3), (1, 3, "0.99", f2)]:
        g = pf[pf.tol == tol]
        base = g[g.w_bits == 16].set_index("image_id").boundary_f
        q = g[g.w_bits == bits].set_index("image_id").boundary_f
        both = base.index.intersection(q.index)
        b, f = base[both].to_numpy(), q[both].to_numpy()
        keep = np.isfinite(b) & np.isfinite(f)
        p = pd.read_csv(R / f"boundary_f_sweep_t{tol}.csv").set_index(["model", "w_bits"]).wilcoxon_p[("dav2_large", bits)]
        put(f"sec6.2/contour_f_t{tol}_large_w{bits}_cliff", fmt(cliffs_delta(b[keep], f[keep])) + f" (p={p:.1e})",
            "results/per_image/boundary_f.csv", paper, "Sec. 6.2")

    dpt = _summary("phase3_dpt")
    for m, pa, pf in [("dpt_large", "0.005", "0.12"), ("dpt_hybrid", "0.013", "0.13")]:
        put(f"sec11.1/{m}_w6_absrel_cliff", f3(_cliff(dpt, m, 6, "absrel")),
            "results/phase3_dpt_summary.csv", pa, "Sec. 11.1")
        put(f"sec11.1/{m}_w6_flying_cliff", f2(_cliff(dpt, m, 6, "flying_frac")),
            "results/phase3_dpt_summary.csv", pf, "Sec. 11.1")

    nyu = _summary("phase3_nyu_bitsweep")
    put("sec11.2/nyu_bitsweep_images", f"{int(nyu.n.max())} of 654 (first in file-name order)",
        "results/phase3_nyu_bitsweep_summary.csv", "first 200 of the 654 test images", "Sec. 3.4, 11.2")
    for m, p in [("dav2_base", "0.0003"), ("dav2_small", "0.001")]:
        v = nyu[(nyu.model == m) & (nyu.w_bits == 6) & (nyu.metric == "absrel")].mean_delta.iloc[0]
        put(f"sec11.2/nyu_{m}_w6_absrel_delta", f"{v:.4f}", "results/phase3_nyu_bitsweep_summary.csv",
            p, "Sec. 11.2")
    dio = _summary("phase6_diode_bitsweep")
    for m, met, p in [("dav2_large", "dbe_com", "p < 1e-12"), ("dav2_large", "edge_width", "p < 1e-13"),
                      ("dav2_base", "dbe_com", "p < 1e-7")]:
        r = dio[(dio.model == m) & (dio.w_bits == 6) & (dio.metric == met)].iloc[0]
        put(f"sec11.2/diode_{m}_w6_{met}_padj", f"{r.p_adj:.1e}",
            "results/phase6_diode_bitsweep_summary.csv", p, "Sec. 11.2")

    sens = pd.read_csv(R / "sensitivity_dav2_large_w4.csv")
    top = sens.sort_values("dbe_com_delta", ascending=False).iloc[0]
    put("sec10.1/top_boundary_layer_large", f"{top.layer}, mean DBE-completeness increase {top.dbe_com_delta:.2f} px",
        "results/sensitivity_dav2_large_w4.csv", "penultimate head convolution, 0.92 pixels", "Sec. 10.1")
    for m, p, pdrop, ptop10 in [("dav2_large", "-0.68, -0.16", "-0.26", "3"), ("dpt_hybrid", "-0.93, -0.02", "0.08", "2")]:
        sm = pd.read_csv(R / f"sensitivity_{m}_w4.csv")
        put(f"sec10.1/boundary_vs_aggregate_sensitivity_pearson_spearman/{m}",
            f"{pearsonr(sm.dbe_com_delta, sm.absrel_delta)[0]:.2f}, "
            f"{spearmanr(sm.dbe_com_delta, sm.absrel_delta)[0]:.2f}",
            f"results/sensitivity_{m}_w4.csv", p, "Sec. 10.1")
        top_agg = sm.nlargest(1, "absrel_delta").layer.iloc[0]
        rest = sm[sm.layer != top_agg]
        put(f"sec10.1/pearson_without_top_aggregate_layer/{m}",
            f"{pearsonr(rest.dbe_com_delta, rest.absrel_delta)[0]:.2f} (without {top_agg})",
            f"results/sensitivity_{m}_w4.csv", pdrop, "Sec. 10.1")
        shared = [len(set(sm.nlargest(k, "dbe_com_delta").layer) & set(sm.nlargest(k, "absrel_delta").layer))
                  for k in (5, 10)]
        put(f"sec10.1/shared_layers_top5_top10/{m}", f"{shared[0]}, {shared[1]}",
            f"results/sensitivity_{m}_w4.csv", f"0, {ptop10}", "Sec. 10.1")
    z = pd.read_csv(R / "sensitivity_zoedepth_w4.csv")
    pos = z.assign(p=z.dbe_com_delta.clip(lower=0)).groupby("component").p.sum()
    put("sec10.4/zoedepth_boundary_sensitivity_outside_head_pct",
        f"{100 * (1 - pos['head'] / pos.sum()):.0f}", "results/sensitivity_zoedepth_w4.csv", "93", "Sec. 10.4")
    d = pd.read_csv(R / "s2q_pareto_dav2_large.csv")
    s5, hp = _closest(d, "s2q", 5.0), _closest(d, "headprot", 6.0)
    put("sec10.3/s2q_5bit_vs_headprot_w6_MB",
        f"{s5.weight_MB:.0f} vs {hp.weight_MB:.0f} ({100 * (1 - s5.weight_MB / hp.weight_MB):.0f}% less)",
        "results/s2q_pareto_dav2_large.csv", "207 vs 249 (17 percent)", "Sec. 10.3")

    mem = pd.read_csv(R / "mitigation_memory.csv")
    for c, p in [("encoder", "89 to 91"), ("neck", "9 to 11"), ("head", "roughly zero")]:
        put(f"sec9/{c}_share_of_quantizable_weights_pct",
            f"{mem[f'{c}_share_pct'].min():.1f} to {mem[f'{c}_share_pct'].max():.1f}",
            "results/mitigation_memory.csv", p, "Sec. 9")

    fs = pd.read_csv(R / "flying_sensitivity.csv", dtype={"tau": str})
    auto = fs[fs.tau == "auto"].set_index("k")
    put("sec14.3/flying_cliff_range_over_k", f"{auto.w6_cliff.min():.2f} to {auto.w6_cliff.max():.2f}",
        "results/flying_sensitivity.csv", "0.70 to 0.76", "Sec. 14.3")
    put("sec14.3/flying_cliff_range_over_k_and_tau", f"{fs.w6_cliff.min():.2f} to {fs.w6_cliff.max():.2f}",
        "results/flying_sensitivity.csv", "0.66 to 0.76", "Sec. 14.3")
    put("sec14.3/large_fp16_flying_frac_k3", f2(auto.loc[3, "fp16_flying_frac"]),
        "results/flying_sensitivity.csv", "0.62", "Sec. 3.8, 14.3")
    for k, p in [(3, "12,000"), (2, "8,500")]:
        put(f"appA/mean_eligible_px_k{k}", f"{auto.loc[k, 'mean_eligible_px']:,}",
            "results/flying_sensitivity.csv", p, "Appendix A")
    put("sec14.3/mean_flying_px_k3", f"{auto.loc[3, 'mean_flying_px']:,}",
        "results/flying_sensitivity.csv", "7,700", "Sec. 14.3")
    return out


TABLES = {
    "02_tost_margin": table_02_tost_margin, "03_multimetric": table_03_multimetric,
    "04_equivalence": table_04_equivalence, "05_dissociation": table_05_dissociation,
    "06_dissociation_index": table_06_dissociation_index,
    "07_frozen_alignment": table_07_frozen_alignment,
    "08_boundary_definition": table_08_boundary_definition, "09_downstream": table_09_downstream,
    "10_memory": table_10_memory, "11_head_protection": table_11_mitigation,
    "12_components": table_12_components, "13_s2q_architectures": table_13_s2q_architectures,
    "14_s2q_transfer": table_14_s2q_transfer, "15_s2q_matched": table_15_s2q_matched,
    "16_cross_dataset": table_16_cross_dataset, "17_gptq": table_17_gptq,
    "18_groupwise": table_18_groupwise, "19_weight_activation": table_19_weight_activation,
    "20_schemes": table_20_schemes, "21_deployment": table_21_deployment, "22_int8": table_22_int8,
}


if __name__ == "__main__":
    seed_everything()
    T.mkdir(parents=True, exist_ok=True)
    for name, fn in TABLES.items():
        df = fn()
        df.to_csv(T / f"table_{name}.csv", index=False)
        print(f"Table {name}\n{df.to_string(index=False)}\n")
    nums = in_text_numbers()
    (T / "in_text_numbers.json").write_text(json.dumps(nums, indent=1) + "\n")
    for k, v in nums.items():
        print(f"{k:60s} {v['value']:45s} paper: {v['paper']}")
