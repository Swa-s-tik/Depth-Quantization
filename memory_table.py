"""Weight memory of FP16, uniform W6 and head-protected W6 (Table 10).

Counts the quantizable (Linear/Conv2d) weights of each disjoint component and
prices them at the configuration's bit-width, so the near-equal-memory claim of
head protection is stated with numbers. Parameter counts depend only on the
architecture, so the model is built from its config (only config.json is
downloaded, no weights). Also reports each component's share of the
quantizable weights (Sec. 9).

Output: results/mitigation_memory.csv
Command: python memory_table.py
"""
import pandas as pd
from transformers import AutoConfig, AutoModelForDepthEstimation

from common import RESULTS, load_config, seed_everything
from models.registry import MODELS
from quant.fakequant import component_param_counts

CFG = load_config("memory")


def memory_table():
    rows = []
    for model in CFG["models"]:
        m = AutoModelForDepthEstimation.from_config(AutoConfig.from_pretrained(MODELS[model]["repo"]))
        parts = component_param_counts(m)          # dict: encoder/neck/head param counts
        enc, neck, head = parts["encoder"], parts["neck"], parts["head"]
        total = enc + neck + head
        cfgs = {
            "FP16": total * 2,
            "uniform W6": total * 6 / 8,
            "head-protected W6": (enc + neck) * 6 / 8 + head * 2,
        }
        base = cfgs["uniform W6"]
        for name, b in cfgs.items():
            rows.append({"model": model, "config": name,
                         "params_M": round(total / 1e6, 2),
                         "head_params_M": round(head / 1e6, 3),
                         "MB": round(b / 1e6, 2),
                         "overhead_vs_W6_pct": round(100 * (b - base) / base, 3),
                         **{f"{c}_share_pct": round(100 * parts[c] / total, 2)
                            for c in ("encoder", "neck", "head")}})
        del m
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "mitigation_memory.csv", index=False)
    return df


if __name__ == "__main__":
    seed_everything(CFG.get("seed", 0))
    print(memory_table().to_string(index=False))
