"""Paired statistics for per-image metric deltas.

Unit of analysis = per-image delta  d_i = metric(quant, img_i) - metric(fp16, img_i).
Depth-error distributions are heavy-tailed, so: Wilcoxon signed-rank (not t-test),
rank-based effect size (Cliff's delta) reported beside Cohen's d, bootstrap CIs,
Benjamini-Hochberg FDR across the grid, and TOST for the "aggregate is negligibly
changed" equivalence claim (absence of significance != evidence of absence).
"""
import numpy as np
from scipy import stats


def paired_delta(quant_vals, fp16_vals):
    q = np.asarray(quant_vals, float)
    f = np.asarray(fp16_vals, float)
    m = np.isfinite(q) & np.isfinite(f)
    return q[m] - f[m]


def bootstrap_ci(x, n_boot=10000, ci=95, seed=0, statistic=np.mean):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(n_boot, x.size))
    boot = statistic(x[idx], axis=1)
    lo, hi = np.percentile(boot, [(100 - ci) / 2, 100 - (100 - ci) / 2])
    return float(statistic(x)), float(lo), float(hi)


def wilcoxon_p(deltas):
    d = np.asarray(deltas, float)
    d = d[np.isfinite(d)]
    if d.size < 5 or np.allclose(d, 0):
        return 1.0
    try:
        return float(stats.wilcoxon(d, zero_method="wilcox", alternative="two-sided").pvalue)
    except ValueError:
        return 1.0


def cliffs_delta(a, b):
    """Rank-based effect size in [-1,1]. a,b are unpaired samples (or paired
    treated as two groups). |0.11|,|0.28|,|0.43| ~ small/medium/large (Vargha-Delaney)."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        return np.nan
    # P(a>b) - P(a<b) via rank trick, O(n log n)
    all_v = np.concatenate([a, b])
    ranks = stats.rankdata(all_v)
    ra = ranks[:a.size].sum()
    u = ra - a.size * (a.size + 1) / 2.0          # Mann-Whitney U for a
    return float(2.0 * u / (a.size * b.size) - 1.0)


def cohens_d_paired(deltas):
    d = np.asarray(deltas, float); d = d[np.isfinite(d)]
    sd = d.std(ddof=1)
    return float(d.mean() / sd) if sd > 0 else 0.0


def bh_fdr(pvals, alpha=0.05):
    """Benjamini-Hochberg. Returns (adjusted_pvals, reject_mask)."""
    p = np.asarray(pvals, float)
    n = p.size
    order = np.argsort(p)
    ranked = p[order]
    adj = ranked * n / (np.arange(n) + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]     # enforce monotonicity
    adj = np.clip(adj, 0, 1)
    out = np.empty_like(adj); out[order] = adj
    return out, out < alpha


def tost_equivalence(deltas, eps, ci=90, n_boot=10000, seed=0):
    """Two-one-sided-test equivalence via bootstrap CI. Equivalent (negligible
    change) iff the `ci`% CI of the mean delta lies fully within [-eps, +eps].
    Returns dict(mean, lo, hi, eps, equivalent)."""
    mean, lo, hi = bootstrap_ci(deltas, n_boot=n_boot, ci=ci, seed=seed)
    equiv = bool(np.isfinite(lo) and lo >= -eps and hi <= eps)
    return {"mean": mean, "lo": lo, "hi": hi, "eps": eps, "equivalent": equiv}


def summarize_delta(quant_vals, fp16_vals, eps_equiv=None, seed=0):
    """One-stop paired summary for a metric across images."""
    d = paired_delta(quant_vals, fp16_vals)
    mean, lo, hi = bootstrap_ci(d, seed=seed)
    out = {"n": int(d.size), "mean_delta": mean, "ci_lo": lo, "ci_hi": hi,
           "wilcoxon_p": wilcoxon_p(d),
           "cliffs_delta": cliffs_delta(np.asarray(quant_vals, float),
                                        np.asarray(fp16_vals, float)),
           "cohens_d": cohens_d_paired(d)}
    if eps_equiv is not None:
        out["tost"] = tost_equivalence(d, eps_equiv, seed=seed)
    return out


def _selftest():
    rng = np.random.default_rng(2)
    fp16 = rng.uniform(0.05, 0.15, size=100)
    # null: quant == fp16 -> ~0 delta, high p, ~0 effect, equivalent
    s0 = summarize_delta(fp16.copy(), fp16, eps_equiv=0.01)
    assert abs(s0["mean_delta"]) < 1e-9 and s0["wilcoxon_p"] == 1.0, s0
    assert abs(s0["cliffs_delta"]) < 1e-9, s0
    assert s0["tost"]["equivalent"], s0
    # real shift: quant worse by +0.05 -> significant, positive effect, NOT equivalent
    quant = fp16 + 0.05 + rng.normal(0, 0.005, size=100)
    s1 = summarize_delta(quant, fp16, eps_equiv=0.01)
    assert s1["wilcoxon_p"] < 1e-3 and s1["mean_delta"] > 0.04, s1
    assert s1["cliffs_delta"] > 0.3, s1
    assert not s1["tost"]["equivalent"], s1
    # BH monotonic + bounded
    adj, rej = bh_fdr([0.001, 0.02, 0.5, 0.9])
    assert adj[0] <= adj[1] <= adj[2] <= adj[3] and adj.max() <= 1.0, adj
    print("stats selftest OK  null_p=%.2f shift_p=%.1e cliffs=%.2f" %
          (s0["wilcoxon_p"], s1["wilcoxon_p"], s1["cliffs_delta"]))


if __name__ == "__main__":
    _selftest()
