"""Weight-only fake quantization (quant-dequant) for accuracy studies.

Simulates reduced-precision WEIGHTS in FP: w -> dequant(quant(w)). This measures
ONLY the accuracy effect of low-bit weights; it saves neither memory nor compute
at runtime (weights are FP16 again before the matmul). Real memory/latency claims
come from bench_deploy.py and predict_score_int8real.py. Per-output-channel
symmetric min-max by default, with an optional percentile clip for outlier-heavy
channels.

Component name-filters (encoder / neck / head) drive the component ablation.
"""
import re
import torch
import torch.nn as nn

_QUANTIZABLE = (nn.Linear, nn.Conv2d)


@torch.no_grad()
def fake_quant_tensor(w, bits, per_channel=True, ch_axis=0,
                      symmetric=True, clip_percentile=None):
    """Return a quant-dequant copy of weight tensor `w`. bits>=16 -> exact no-op."""
    if bits >= 16:
        return w.clone()
    if symmetric:
        qmax = 2 ** (bits - 1) - 1
        qmin = -(2 ** (bits - 1))
    else:                                   # unsigned asymmetric
        qmax = 2 ** bits - 1
        qmin = 0

    if per_channel:
        dims = [d for d in range(w.dim()) if d != ch_axis]
        if clip_percentile is not None:
            flat = w.abs().flatten(start_dim=1) if ch_axis == 0 else \
                   w.abs().transpose(0, ch_axis).flatten(start_dim=1)
            amax = torch.quantile(flat.float(), clip_percentile, dim=1)
        else:
            amax = w.abs().amax(dim=dims)
        shape = [1] * w.dim(); shape[ch_axis] = -1
        amax = amax.reshape(shape).clamp_min(1e-12)
    else:
        amax = w.abs().amax().clamp_min(1e-12) if clip_percentile is None \
            else torch.quantile(w.abs().float().flatten(), clip_percentile).clamp_min(1e-12)

    if symmetric:
        scale = amax / qmax
        zp = 0.0
    else:
        wmin = w.amin(dim=dims, keepdim=True) if per_channel else w.min()
        scale = (amax - wmin).clamp_min(1e-12) / qmax
    # The 1e-12 floor underflows to 0 in float16, so an all-zero channel (or tensor)
    # would give 0/0 = NaN. A unit scale maps such a channel to exact zeros and leaves
    # every channel with a positive scale bit-identical.
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    if not symmetric:
        zp = torch.round(-wmin / scale)

    q = torch.clamp(torch.round(w / scale + (zp if not symmetric else 0.0)), qmin, qmax)
    return (q - (zp if not symmetric else 0.0)) * scale


def _match(name, filters):
    if filters is None:
        return True
    return any(re.search(f, name) for f in filters)


# DISJOINT component partition for DAv2 (transformers DepthAnything) + DPT.
# Verified on DAv2-Small: encoder 73 + neck 29 + head 3 = 105 (all quantizable).
#   encoder = ViT/DINOv2 backbone; neck = DPT reassemble+fusion+upsample decoder;
#   head    = final output convs.
COMPONENT_FILTERS = {
    "all": None,
    "encoder": [r"backbone", r"encoder\.layer", r"embeddings"],
    "neck":    [r"neck\.", r"reassemble", r"fusion"],
    "head":    [r"head\."],
}


@torch.no_grad()
def quantize_model_(model, w_bits, per_channel=True, clip_percentile=None,
                    symmetric=True, component="all", extra_filters=None, verbose=False):
    """In-place fake-quantize matching Linear/Conv2d weights. Returns count + names.
    `component` selects a COMPONENT_FILTERS entry; `extra_filters` adds regexes.
    `symmetric=False` uses unsigned asymmetric (min-max) quantization."""
    filters = COMPONENT_FILTERS.get(component, None)
    if extra_filters:
        filters = (filters or []) + list(extra_filters)
    touched = []
    for name, mod in model.named_modules():
        if isinstance(mod, _QUANTIZABLE) and _match(name, filters):
            mod.weight.data.copy_(
                fake_quant_tensor(mod.weight.data, w_bits, per_channel=per_channel,
                                  ch_axis=0, symmetric=symmetric,
                                  clip_percentile=clip_percentile))
            touched.append(name)
    if verbose:
        print(f"quantized {len(touched)} layers @ {w_bits}b comp={component}")
    return len(touched), touched


@torch.no_grad()
def quantize_per_layer_(model, bits_by_layer, per_channel=True, symmetric=True,
                        default_bits=16):
    """Fake-quantize each Linear/Conv2d to the bit-width given in `bits_by_layer`
    (dict: layer_name -> bits). Layers absent from the dict get `default_bits`
    (>=16 -> untouched). Drives the S2Q mixed-precision allocation. Returns
    (count_quantized, total_weight_bytes)."""
    n, total_bytes = 0, 0.0
    for name, mod in model.named_modules():
        if not isinstance(mod, _QUANTIZABLE):
            continue
        bits = bits_by_layer.get(name, default_bits)
        total_bytes += mod.weight.numel() * bits / 8.0
        if bits < 16:
            mod.weight.data.copy_(
                fake_quant_tensor(mod.weight.data, bits, per_channel=per_channel,
                                  ch_axis=0, symmetric=symmetric))
            n += 1
    return n, total_bytes


@torch.no_grad()
def quantize_mixed_(model, bits_by_component, per_channel=True, clip_percentile=None):
    """Apply different bit-widths to disjoint components (encoder/neck/head).
    Components not listed stay FP16. Returns (total_quantized, per-comp counts)."""
    total, per = 0, {}
    for comp, bits in bits_by_component.items():
        n, _ = quantize_model_(model, bits, per_channel=per_channel,
                               clip_percentile=clip_percentile, component=comp)
        per[comp] = n
        total += n
    return total, per


def component_param_counts(model):
    """Param count of quantizable weights per disjoint component — for iso-memory
    budgeting in the mitigation arm."""
    counts = {c: 0 for c in ("encoder", "neck", "head")}
    for name, mod in model.named_modules():
        if isinstance(mod, _QUANTIZABLE):
            for comp in ("encoder", "neck", "head"):
                if _match(name, COMPONENT_FILTERS[comp]):
                    counts[comp] += mod.weight.numel()
                    break
    return counts


def model_weight_bytes(model, bits_map=None, default_bits=16):
    """Sum params*bits/8 over quantizable weights. bits_map: {layer_name: bits}.
    Used to enforce ISO-MEMORY in the mitigation arm."""
    total = 0.0
    for name, mod in model.named_modules():
        if isinstance(mod, _QUANTIZABLE):
            b = default_bits
            if bits_map:
                for k, v in bits_map.items():
                    if re.search(k, name):
                        b = v; break
            total += mod.weight.numel() * b / 8.0
    return total


def _selftest():
    torch.manual_seed(0)
    w = torch.randn(16, 8, 3, 3)
    # bits>=16 exact no-op
    assert torch.allclose(fake_quant_tensor(w, 32), w), "no-op failed"
    # 8-bit per-channel: bounded error, correct scale broadcast
    q8 = fake_quant_tensor(w, 8, per_channel=True)
    assert q8.shape == w.shape
    err8 = (q8 - w).abs().max().item()
    q4 = fake_quant_tensor(w, 4, per_channel=True)
    err4 = (q4 - w).abs().max().item()
    assert err4 > err8, (err4, err8)          # fewer bits -> larger error
    # per-channel beats per-tensor on an outlier-heavy channel
    w2 = torch.randn(4, 32); w2[0] *= 50.0
    e_pc = (fake_quant_tensor(w2, 4, per_channel=True) - w2).abs().mean().item()
    e_pt = (fake_quant_tensor(w2, 4, per_channel=False) - w2).abs().mean().item()
    assert e_pc < e_pt, (e_pc, e_pt)
    # float16 with an all-zero channel stays finite (zero channel -> exact zeros)
    z = torch.randn(4, 8, dtype=torch.float16); z[1] = 0
    for pc in (True, False):
        zq = fake_quant_tensor(z, 8, per_channel=pc)
        assert torch.isfinite(zq).all() and (zq[1] == 0).all(), pc
    # model-level: quantize a tiny net, count layers, no-op at 32b
    net = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Conv2d(3, 4, 3))
    ref = net[0].weight.data.clone()
    n, names = quantize_model_(net, 32)
    assert n == 2 and torch.allclose(net[0].weight.data, ref), (n, names)
    quantize_model_(net, 4)
    assert not torch.allclose(net[0].weight.data, ref)
    print(f"fakequant selftest OK  err8={err8:.4f} err4={err4:.4f} pc<pt={e_pc:.3f}<{e_pt:.3f}")


if __name__ == "__main__":
    _selftest()
