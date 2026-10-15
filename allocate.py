"""S2Q bit allocator.

Given per-layer boundary sensitivity s_l (from sensitivity.py, measured as the
boundary-metric increase when layer l alone is quantized to 4 bits) and the real
per-layer weight-quantization-error curve e_l(b), allocate a bit-width b_l per
layer to minimize predicted boundary damage under an average-bit budget.

Damage model: damage_l(b) = max(s_l,0) * e_l(b)/e_l(4), i.e. the measured 4-bit
sensitivity scaled by how much more (or less) error bit-width b injects than 4 bit.
Allocation: Lagrangian sweep. For multiplier lambda, each layer independently picks
b_l = argmin_b [ damage_l(b) + lambda * n_l * b ]; sweeping lambda traces the
Pareto front of (avg bits, predicted damage). We emit the allocation whose weighted
average bit-width matches each target budget.

Output: results/<model>_{s2q,bonly,agg}_alloc.json  { "<avg_bits>": {layer: bits, ...}, ... }
(S2Q = combined boundary+aggregate signal; bonly = boundary-only ablation;
agg = aggregate-guided baseline). Layers absent from a map stay at 16 bits.
Command: python allocate.py --model dav2_large   (also dpt_hybrid, zoedepth)
"""
import json, argparse
import numpy as np
import pandas as pd
import torch

from common import RESULTS, load_config, seed_everything

CFG = load_config("allocate")
from models.registry import load_model
from quant.fakequant import fake_quant_tensor, _QUANTIZABLE

# candidate per-layer bit-widths. 2-bit weight-only is catastrophic per layer (and
# not deployable), so the allocator works in the survivable 4-16 bit range; the
# sensitivity damage model is calibrated at 4 bit and does not capture the 2/3-bit
# cliff, so we exclude those rather than let the allocator create dead layers.
CAND = CFG["candidates"]


def quant_error_curve(model):
    """e_l(b) = ||W - Q_b(W)||_F per quantizable layer, for each candidate b."""
    out = {}
    for name, mod in model.named_modules():
        if not isinstance(mod, _QUANTIZABLE):
            continue
        w = mod.weight.data
        e = {}
        for b in CAND:
            q = fake_quant_tensor(w, b, per_channel=True, ch_axis=0)
            e[b] = float((w.float() - q.float()).norm().item())
        out[name] = {"n": int(w.numel()), "err": e}
    return out


def _norm_sens(s, layers, col):
    v = np.array([max(float(s.loc[l, col]) if l in s.index else 0.0, 0.0) for l in layers])
    return v / (v.max() + 1e-12)


def allocate(sens_csv, model_name, signal="dbe_com_delta", outfile="s2q_alloc.json",
             targets=tuple(CFG["targets"]), curves=None):
    s = pd.read_csv(sens_csv).set_index("layer")
    if curves is None:
        model, _, _ = load_model(model_name, dtype=torch.float32, device="cpu")
        curves = quant_error_curve(model)
    layers = list(curves)
    n = np.array([curves[l]["n"] for l in layers], float)
    # sensitivity per layer, min-max normalized. `signal` is a column name, or
    # "combined" = protect a layer critical on EITHER the boundary or aggregate axis
    # (they are anti-correlated, so the union matters).
    if signal == "combined":
        sens = np.maximum(_norm_sens(s, layers, "dbe_com_delta"),
                          _norm_sens(s, layers, "absrel_delta"))
    else:
        sens = _norm_sens(s, layers, signal)
    e4 = np.array([max(curves[l]["err"][4], 1e-9) for l in layers])
    # damage_l(b) table
    dmg = {b: sens * (np.array([curves[l]["err"][b] for l in layers]) / e4) for b in CAND}

    def alloc_for_lambda(lam):
        bits = np.zeros(len(layers), int)
        for i in range(len(layers)):
            costs = [dmg[b][i] + lam * n[i] * b for b in CAND]
            bits[i] = CAND[int(np.argmin(costs))]
        return bits

    def avg_bits(bits):
        return float((bits * n).sum() / n.sum())

    out = {}
    for tgt in targets:
        # binary search lambda for target avg bits (higher lambda -> fewer bits)
        lo, hi = 1e-12, 1e3
        for _ in range(60):
            mid = np.sqrt(lo * hi)
            ab = avg_bits(alloc_for_lambda(mid))
            if ab > tgt:
                lo = mid            # need more penalty -> fewer bits
            else:
                hi = mid
        bits = alloc_for_lambda(np.sqrt(lo * hi))
        ab = avg_bits(bits)
        bmap = {layers[i]: int(bits[i]) for i in range(len(layers)) if bits[i] < 16}
        out[f"{ab:.2f}"] = bmap
        prot = [(layers[i], int(bits[i])) for i in range(len(layers)) if bits[i] >= 8]
        print(f"  [{signal}] target {tgt}b -> avg {ab:.2f}b, {len(bmap)} quantized, "
              f"{len(prot)} protected(>=8b)")
    (RESULTS / outfile).write_text(json.dumps(out, indent=1))
    print(f"wrote results/{outfile}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=CFG["models"])
    ap.add_argument("--sens", default=None)
    a = ap.parse_args()
    seed_everything(CFG.get("seed", 0))
    sens = a.sens or RESULTS / f"sensitivity_{a.model}_w{CFG['sensitivity_bits']}.csv"
    # compute the quant-error curves once, reuse for both allocations
    model, _, _ = load_model(a.model, dtype=torch.float32, device="cpu")
    curves = quant_error_curve(model)
    pre = a.model
    print("== S2Q (ours): combined boundary+aggregate sensitivity ==")
    allocate(sens, a.model, "combined", f"{pre}_s2q_alloc.json", curves=curves)
    print("== ablation: boundary-only signal ==")
    allocate(sens, a.model, "dbe_com_delta", f"{pre}_bonly_alloc.json", curves=curves)
    print("== baseline: aggregate-only signal ==")
    allocate(sens, a.model, "absrel_delta", f"{pre}_agg_alloc.json", curves=curves)
