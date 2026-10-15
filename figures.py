"""Paper figures from the shipped results (no datasets, cache or GPU needed).

  Fig. 2, 9  dissociation_<model>.png    effect size vs bit-width (sweep summaries)
  Fig. 3     error_vs_distance.png       AbsRel per distance band (results/bands.csv)
  Fig. 5     mitigation_w6.png           DBE-completeness, head protection at W6
  Fig. 6     mechanism_<model>_w<b>.png  component ablation
  Fig. 7     sensitivity_scatter.png     per-layer boundary vs aggregate sensitivity
  Fig. 8     s2q_pareto.png              boundary fidelity vs average bit-width
Figs. 1 and 4 need the dataset and cache: figures_qualitative.py.
Command: python figures.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import FIGURES as FIGDIR, RESULTS

SAVE = dict(bbox_inches="tight", metadata={"Software": None}, pil_kwargs={"compress_level": 9})

# colorblind-safe (Wong): aggregate = blues/greys, boundary = orange/red/vermillion
C = {"absrel": "#0072B2", "d1": "#56B4E9", "rmse": "#999999",
     "dbe_com": "#D55E00", "dbe_acc": "#E69F00", "boundary_absrel_k3": "#CC79A7",
     "edge_width": "#9400D3", "flying_frac": "#000000"}
LABEL = {"absrel": "AbsRel (agg)", "d1": "δ1 (agg)", "rmse": "RMSE (agg)",
         "dbe_com": "DBE completeness", "dbe_acc": "DBE accuracy",
         "boundary_absrel_k3": "Boundary AbsRel", "edge_width": "Edge width",
         "flying_frac": "Flying pixels (3D)"}
AGG = ["absrel", "d1", "rmse"]
BND = ["dbe_com", "dbe_acc", "boundary_absrel_k3", "edge_width", "flying_frac"]


def _bits(precision):
    if precision == "fp16":
        return 16
    import re
    m = re.match(r"w(\d+)", str(precision))
    return int(m.group(1)) if m else np.nan


def dissociation_plot(summary_csv, model, out=None):
    """Effect-size (signed Cliff's δ, worse-direction positive) vs bit-width, from
    the analyze summary CSV. Bounded [-1,1] axis so the low-bit blur explosion
    can't hijack the plot; the medium-effect line (0.33) marks 'real' damage. The
    dissociation is visible as boundary lines rising above 0.33 while aggregate
    stays near 0."""
    s = pd.read_csv(summary_csv)
    s = s[s.model == model].copy()
    if s.empty:
        raise ValueError(f"no rows for {model}")
    s["signed"] = s.cliffs_delta.abs() * s.worse.map({True: 1, False: -1})
    metrics = [m for m in AGG + BND if m in s.metric.unique()]

    fig, ax = plt.subplots(figsize=(6.6, 4.3))
    for m in metrics:
        d = s[s.metric == m].sort_values("w_bits", ascending=False)
        if d.empty:
            continue
        ax.plot(d.w_bits, d.signed, color=C.get(m, "#333"), label=LABEL.get(m, m),
                linestyle="-" if m in BND else "--",
                lw=2.4 if m in BND else 1.8, marker="o", ms=6 if m in BND else 5)
    ax.axhline(0, color="k", lw=0.8, alpha=0.5)
    ax.axhline(0.33, color="k", lw=0.8, ls=":", alpha=0.6)
    ax.text(s.w_bits.min(), 0.35, "medium effect (0.33)", fontsize=10, alpha=0.7)
    ax.set_xlabel("weight bit-width", fontsize=13)
    ax.set_ylabel("Cliff's δ vs FP16  (↑ = worse; sign = direction)", fontsize=12)
    ax.set_title(f"Sensitivity dissociation — {model}", fontsize=13)
    ax.tick_params(labelsize=11)
    ax.set_ylim(-1, 1)
    ax.invert_xaxis()
    ax.legend(fontsize=11, ncol=2, frameon=False, loc="upper left")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    out = out or (FIGDIR / f"dissociation_{model}.png")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, **SAVE)
    plt.close(fig)
    return out, s


def mechanism_plot(summary_csv, model, bits, out=None,
                   metrics=("flying_frac", "dbe_com", "edge_width", "absrel")):
    """Grouped bars: Cliff's δ (worse-positive) per component (encoder/neck/head/all)
    for the chosen bit-width, showing WHERE quantization damage originates."""
    s = pd.read_csv(summary_csv)
    s = s[s.model == model].copy()
    s["signed"] = s.cliffs_delta.abs() * s.worse.map({True: 1, False: -1})
    comps = [("encoder", f"w{bits}_encoder"), ("neck", f"w{bits}_neck"),
             ("head", f"w{bits}_head"), ("all", f"w{bits}")]
    comps = [(c, p) for c, p in comps if p in s.precision.unique()]
    metrics = [m for m in metrics if m in s.metric.unique()]
    import numpy as _np
    x = _np.arange(len(comps)); w = 0.8 / max(1, len(metrics))
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    for i, m in enumerate(metrics):
        vals = [float(s[(s.precision == p) & (s.metric == m)].signed.mean()) for _, p in comps]
        ax.bar(x + i * w, vals, w, label=LABEL.get(m, m), color=C.get(m, "#333"))
    ax.axhline(0, color="k", lw=0.8); ax.axhline(0.33, color="k", ls=":", lw=0.8, alpha=0.6)
    ax.set_xticks(x + w * (len(metrics) - 1) / 2)
    ax.set_xticklabels([c for c, _ in comps])
    ax.set_ylabel("Cliff's δ vs FP16 (↑ = worse)")
    ax.set_title(f"Where damage originates — {model}, W{bits} by component")
    ax.legend(fontsize=8, frameon=False, ncol=2)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    out = out or (FIGDIR / f"mechanism_{model}_w{bits}.png")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, **SAVE)
    plt.close(fig)
    return out


def bands_plot(models=("dav2_large", "dav2_base")):
    """Fig. 3: AbsRel per distance band from the nearest GT boundary."""
    df = pd.read_csv(RESULTS / "bands.csv")
    xlab = ["0-1", "1-3", "3-5", "5-10", "interior"]
    colors = {"fp16": "#1f77b4", "w6": "#d62728", "w4": "#7f7f7f"}
    names = {"fp16": "FP16", "w6": "uniform W6", "w4": "uniform W4"}
    fig, axes = plt.subplots(1, len(models), figsize=(11, 4.2), sharey=True)
    for ax, m in zip(axes, models):
        for p in names:
            y = [df[(df.model == m) & (df.precision == p) & (df.band == b)].mean_absrel.mean() for b in xlab]
            ax.plot(xlab, y, marker="o", color=colors[p], label=names[p])
        ax.set_title(m.replace("dav2_", "DAv2-"))
        ax.set_xlabel("distance from boundary (px)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("AbsRel")
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle("Error concentrates at boundaries. AbsRel rises sharply near a "
                 "depth discontinuity and is flat in the interior.", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = FIGDIR / "error_vs_distance.png"
    fig.savefig(out, dpi=150, **SAVE)
    plt.close(fig)
    return out


def mitigation_plot(models=("dav2_base", "dav2_large")):
    """Fig. 5: DBE-completeness at six bits, uniform vs head-protected, FP16 dashed."""
    m = pd.read_csv(RESULTS / "phase4_mitigation.csv").groupby(["model", "precision"]).dbe_com.mean()
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    x = np.arange(len(models)); w = 0.36
    ax.bar(x - w / 2, [m[(k, "w6")] for k in models], w, color="#E69F00", label="uniform W6")
    ax.bar(x + w / 2, [m[(k, "mix_e6_n6")] for k in models], w, color="#009E73",
           label="head-protected W6")
    for i, k in enumerate(models):
        ax.hlines(m[(k, "fp16")], i - 0.45, i + 0.45, colors="k", linestyles="--",
                  label="FP16" if i == 0 else None)
    ax.set_xticks(x)
    ax.set_xticklabels([k.replace("dav2_", "DAv2-") for k in models])
    ax.set_ylabel("DBE-completeness (px, lower is better)")
    ax.set_title("Head protection recovers boundary completeness at six bits")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    out = FIGDIR / "mitigation_w6.png"
    fig.savefig(out, dpi=200, **SAVE)
    plt.close(fig)
    return out


def sensitivity_scatter(model="dav2_large"):
    """Fig. 7: per-layer boundary vs aggregate sensitivity (one layer at 4 bits)."""
    s = pd.read_csv(RESULTS / f"sensitivity_{model}_w4.csv")
    colors = {"encoder": "#0072B2", "neck": "#999999", "head": "#D55E00"}
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for comp, g in s.groupby("component"):
        ax.scatter(g.absrel_delta, g.dbe_com_delta, s=18, color=colors.get(comp, "#333"), label=comp)
    for _, r in pd.concat([s.nlargest(1, "dbe_com_delta"), s.nlargest(1, "absrel_delta")]).iterrows():
        ax.annotate(r.layer, (r.absrel_delta, r.dbe_com_delta), fontsize=8,
                    xytext=(5, 5), textcoords="offset points")
    ax.axhline(0, color="k", lw=0.6); ax.axvline(0, color="k", lw=0.6)
    ax.set_xlabel("aggregate sensitivity (mean AbsRel increase)")
    ax.set_ylabel("boundary sensitivity (mean DBE-completeness increase, px)")
    ax.set_title(f"Per-layer sensitivity, {model.replace('dav2_', 'DAv2-')}, one layer at 4 bits")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    out = FIGDIR / "sensitivity_scatter.png"
    fig.savefig(out, dpi=200, **SAVE)
    plt.close(fig)
    return out


def pareto_plot(model="dav2_large"):
    """Fig. 8: boundary effect size vs average weight bit-width per allocation method."""
    d = pd.read_csv(RESULTS / f"s2q_pareto_{model}.csv")
    d["method"] = d.config.str.split("@").str[0]
    styles = {"s2q": ("#d62728", "o", "S2Q (boundary)"),
              "aggguided": ("#1f77b4", "s", "aggregate-guided"),
              "uniform": ("#7f7f7f", "^", "uniform"),
              "headprot": ("#2ca02c", "D", "head-protect")}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    for ax, met, lab in [(axes[0], "flying_frac_cliff", "flying-pixel effect size"),
                         (axes[1], "dbe_com_cliff", "DBE-completeness effect size")]:
        for m, (c, mk, name) in styles.items():
            sub = d[d.method == m].sort_values("avg_bits")
            ax.plot(sub.avg_bits, sub[met], marker=mk, color=c, label=name)
        ax.set_xlabel("average weight bit-width")
        ax.set_ylabel(lab + " vs FP16 (lower better)")
        ax.grid(alpha=0.3)
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle("S2Q allocates bits by boundary sensitivity; boundary fidelity at "
                 "matched average bit-width.", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = FIGDIR / "s2q_pareto.png"
    fig.savefig(out, dpi=150, **SAVE)
    plt.close(fig)
    return out


if __name__ == "__main__":
    from common import seed_everything
    seed_everything()
    FIGDIR.mkdir(parents=True, exist_ok=True)
    outs = [dissociation_plot(RESULTS / "phase3_bitsweep_summary.csv", m)[0]
            for m in ("dav2_large", "dav2_small")]
    outs += [dissociation_plot(RESULTS / "phase3_dpt_summary.csv", m)[0]
             for m in ("dpt_large", "dpt_hybrid")]
    outs += [mechanism_plot(RESULTS / "phase4_mechanism_summary.csv", "dav2_large", 6),
             mechanism_plot(RESULTS / "phase4_mechanism_summary.csv", "dav2_base", 4),
             bands_plot(), mitigation_plot(), sensitivity_scatter(), pareto_plot()]
    for o in outs:
        print("wrote", o)
