"""Sweep runner — config (yaml) -> predict (cache) -> evaluate -> results.csv.

Deterministic, resumable: predictions are cached, evaluation re-reads them. One
row per (model, dataset, precision, image). No DB, no framework — pandas reads
the CSV downstream.
"""
import sys
import time

from common import RESULTS, load_config, seed_everything
from predict import run_predictions, precision_tag
from evaluate import evaluate_run, write_rows


def run_config(cfg_path):
    cfg = load_config(cfg_path)
    seed_everything(cfg.get("seed", 0))
    name = cfg["name"]
    dataset = cfg["dataset"]
    subset = cfg.get("subset")
    models = cfg["models"]
    specs = cfg["specs"]
    align = cfg.get("align", "ss")
    bks = tuple(cfg.get("boundary_ks", [3]))
    compute_boundary = cfg.get("boundary", True)
    out_csv = RESULTS / f"{name}.csv"
    if out_csv.exists() and cfg.get("fresh", True):
        out_csv.unlink()

    log = []
    for model in models:
        for spec in specs:
            tag = precision_tag(spec)
            t0 = time.time()
            cdir, w, sk, meta = run_predictions(model, dataset, spec, subset=subset)
            rows, info = evaluate_run(dataset, model, spec, align_mode=align,
                                      subset=subset, compute_boundary=compute_boundary,
                                      boundary_ks=bks)
            write_rows(rows, out_csv, append=True)
            msg = (f"[{model}/{tag}] pred(+{w}/skip{sk}) scored {len(rows)} imgs "
                   f"in {time.time()-t0:.1f}s  nq={meta.get('n_quantized_layers','-')}")
            print(msg, flush=True)
            log.append(msg)
    (RESULTS / f"{name}.log").write_text("\n".join(log))
    print(f"\nDONE -> {out_csv}")
    return out_csv


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python run_sweep.py configs/<sweep>.yaml")
    run_config(sys.argv[1])
