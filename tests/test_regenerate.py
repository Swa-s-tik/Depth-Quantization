"""Regenerate every derived result from the shipped per-image tables and compare
it byte for byte with the shipped file (no GPU, datasets, cache or checkpoints).

Copies results/ to a temporary directory, deletes every derived file there,
re-runs the analysis, table and figure scripts against the copy (DQ_RESULTS,
DQ_FIGURES) and compares. Figures are compared too but reported separately,
since PNG bytes depend on the matplotlib/freetype build.
Command: python tests/test_regenerate.py
"""
import filecmp
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SWEEPS = ["phase3_bitsweep", "phase3_dpt", "phase3_nyu_bitsweep", "phase4_mechanism",
          "phase6_diode_bitsweep"]
DERIVED = ([f"{s}_summary.csv" for s in SWEEPS] + [f"{s}_summary_verdicts.csv" for s in SWEEPS] +
           ["multimetric.csv", "dissociation_index.csv", "equivalence.csv", "equivalence_margins.csv",
            "frozen_align.csv", "boundary_f_sweep_t1.csv", "boundary_f_sweep_t2.csv",
            "flying_sensitivity.csv", "flying_on_diode.csv", "bands.csv", "boundary_sensitivity.csv",
            "gptq_rtn.csv", "int8_real_summary.csv", "gate_nyu.csv", "gate_nyu_dpt.csv",
            "gptq_eval.csv", "gw_eval.csv", "s2q_diode_transfer.csv", "alt_ptq.csv", "downstream_recon.csv"] +
           [f"s2q_pareto_{m}.csv" for m in ("dav2_large", "dpt_hybrid", "zoedepth")])
STEPS = ([["analyze.py", f"RESULTS/{s}.csv"] for s in SWEEPS] +
         [["analyze_multimetric.py"], ["analyze_equivalence.py"],
          ["frozen_align.py", "--from-per-image"], ["boundary_f_sweep.py", "--from-per-image"],
          ["flying_sensitivity.py", "--from-per-image"], ["flying_on_diode.py", "--from-per-image"],
          ["analyze_bands.py", "--from-per-image"], ["boundary_definition.py", "--from-per-image"],
          ["gptq_eval.py", "--rtn", "--from-per-image"],
          ["predict_score_int8real.py", "--from-per-image"],
          ["gate.py", "configs/gate_nyu.yaml", "--from-per-image"],
          ["gate.py", "configs/gate_nyu_dpt.yaml", "--from-per-image"],
          ["gptq_eval.py", "--from-per-image"], ["gw_eval.py", "--from-per-image"],
          ["s2q_diode_transfer.py", "--from-per-image"], ["alt_ptq_sweep.py", "--from-per-image"],
          ["downstream_recon.py", "--from-per-image"]] +
         [["s2q_eval.py", "--model", m, "--from-per-image"] for m in ("dav2_large", "dpt_hybrid", "zoedepth")] +
         [["make_tables.py"], ["figures.py"]])


def main():
    with tempfile.TemporaryDirectory() as tmp:
        res, fig = Path(tmp) / "results", Path(tmp) / "figures"
        shutil.copytree(ROOT / "results", res)
        for f in DERIVED:
            (res / f).unlink()
        shutil.rmtree(res / "tables")
        env = dict(os.environ, DQ_RESULTS=str(res), DQ_FIGURES=str(fig))
        for step in STEPS:
            cmd = [sys.executable] + [a.replace("RESULTS", str(res)) for a in step]
            r = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)
            if r.returncode:
                sys.exit(f"FAIL: {' '.join(step)}\n{r.stderr[-2000:]}")
        expected = DERIVED + [f"tables/{p.name}" for p in sorted((ROOT / "results" / "tables").iterdir())]
        bad = [f for f in expected if not filecmp.cmp(res / f, ROOT / "results" / f, shallow=False)]
        figs = sorted(p.name for p in (ROOT / "figures").glob("*.png"))
        same_figs = [f for f in figs if filecmp.cmp(fig / f, ROOT / "figures" / f, shallow=False)]
        print(f"results: {len(expected) - len(bad)}/{len(expected)} regenerated files identical")
        print(f"figures: {len(same_figs)}/{len(figs)} identical PNG bytes (informational)")
        if bad:
            sys.exit("FAIL: differ: " + ", ".join(bad))
        print("PASS")


if __name__ == "__main__":
    main()
