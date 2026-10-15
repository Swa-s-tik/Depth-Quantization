"""Activation fake-quantization via forward-pre-hooks, to enable W/A quantization
(e.g. W8A8, W4A4) alongside the weight-only fake-quant in fakequant.py.

Each target Linear/Conv2d gets a pre-hook that quant-dequants its input tensor to
`a_bits`. The scale is computed from each input at every forward pass, with one of
two granularities:
  per_tensor  : one scale per input tensor (the paper's main setting, Table 19)
  per_channel : one scale per feature channel (dim 1 for conv NCHW, last dim for
                linear/token tensors); a sensitivity check
Reversible: keep the handles and call remove_activation_fakequant.
"""
import torch
import torch.nn as nn

from .fakequant import fake_quant_tensor, _QUANTIZABLE, _match, COMPONENT_FILTERS


GRANULARITIES = ("per_tensor", "per_channel")


def _act_hook(a_bits, symmetric, granularity):
    def hook(module, inputs):
        if not inputs:
            return inputs
        x = inputs[0]
        if not torch.is_floating_point(x):
            return inputs
        if granularity == "per_channel" and x.dim() >= 2:
            cax = 1 if x.dim() == 4 else x.dim() - 1
            xq = fake_quant_tensor(x, a_bits, per_channel=True, ch_axis=cax, symmetric=symmetric)
        else:
            xq = fake_quant_tensor(x, a_bits, per_channel=False, symmetric=symmetric)
        return (xq,) + inputs[1:]
    return hook


def add_activation_fakequant(model, a_bits, symmetric=True, granularity="per_tensor",
                             component="all", extra_filters=None):
    """Register activation fake-quant pre-hooks. a_bits>=16 is a no-op. Returns the
    list of hook handles (call .remove() or remove_activation_fakequant to undo)."""
    if granularity not in GRANULARITIES:
        raise ValueError(f"granularity must be one of {GRANULARITIES}, got {granularity!r}")
    if a_bits >= 16:
        return []
    filters = COMPONENT_FILTERS.get(component, None)
    if extra_filters:
        filters = (filters or []) + list(extra_filters)
    handles = []
    for name, mod in model.named_modules():
        if isinstance(mod, _QUANTIZABLE) and _match(name, filters):
            handles.append(mod.register_forward_pre_hook(_act_hook(a_bits, symmetric, granularity)))
    return handles


def remove_activation_fakequant(handles):
    for h in handles:
        h.remove()


def _selftest():
    torch.manual_seed(0)
    lin = nn.Linear(64, 64)
    x = torch.randn(8, 64)
    y0 = lin(x)
    h8 = add_activation_fakequant(lin, 8)
    y8 = lin(x)
    remove_activation_fakequant(h8)
    h4 = add_activation_fakequant(lin, 4)
    y4 = lin(x)
    remove_activation_fakequant(h4)
    e8 = (y8 - y0).abs().mean().item()
    e4 = (y4 - y0).abs().mean().item()
    assert e4 > e8 > 0, (e8, e4)
    # a_bits>=16 is a no-op
    h16 = add_activation_fakequant(lin, 16)
    assert len(h16) == 0
    y16 = lin(x); remove_activation_fakequant(h16)
    assert torch.allclose(y16, y0)
    # per-tensor equals one tensor-wide symmetric quant-dequant of the input
    h = add_activation_fakequant(lin, 8, granularity="per_tensor")
    yt = lin(x); remove_activation_fakequant(h)
    assert torch.equal(yt, lin(fake_quant_tensor(x, 8, per_channel=False))), "per_tensor mismatch"
    # float16 activations with an all-zero channel stay finite at both granularities
    conv = nn.Conv2d(4, 4, 3, padding=1).half()
    xc = torch.randn(1, 4, 8, 8, dtype=torch.float16); xc[:, 2] = 0
    for g in GRANULARITIES:
        h = add_activation_fakequant(conv, 8, granularity=g)
        yc = conv(xc); remove_activation_fakequant(h)
        assert torch.isfinite(yc).all(), g
    print(f"actquant selftest OK  errA8={e8:.4f} < errA4={e4:.4f}; A16 no-op; fp16 zero channel finite")


if __name__ == "__main__":
    _selftest()
