"""Contextual activation sparsity on SwiGLU FFNs.

A LLaMA-style MLP computes, per token x:

    a_i   = act_fn(gate_proj(x)_i) * up_proj(x)_i        # neuron i's activation
    out   = sum_i  a_i * down_proj[:, i]                 # its output contribution

There are no exact zeros (SiLU != ReLU), so "sparsity" means *skipping* the
neurons that contribute least for this token. We wrap each MLP so we can:
  * record the per-neuron activation magnitudes (characterization), and
  * zero a per-token subset of neurons before down_proj (the sparsity itself).

Swapping the shared `ctrl["masker"]` changes the method/sparsity without
touching weights or reloading the model. Maskers act per token (last dim),
so the kept set is fully input-dependent — contextual, not static.
"""
import torch
import torch.nn as nn


def get_decoder_layers(model):
    base = model.model
    if hasattr(base, "layers"):
        return base.layers
    if hasattr(base, "decoder") and hasattr(base.decoder, "layers"):
        return base.decoder.layers
    raise AttributeError("could not locate decoder layers")


class SparseMLP(nn.Module):
    def __init__(self, mlp, ctrl, idx):
        super().__init__()
        self.mlp = mlp
        self.ctrl = ctrl          # shared {"masker": fn|None, "recorder": fn|None}
        self.idx = idx
        # ||down_proj[:, i]||_2 per neuron i (weight is [hidden, intermediate]).
        # Under weight-only quantization (bnb int8/4bit) the stored .weight is
        # packed integers, not a float matrix — column norms are unavailable
        # without a dequant. oracle_gate (rank by |a|) doesn't need them, so we
        # leave col_norm None there; only contribution-aware paths require it.
        w = mlp.down_proj.weight
        col_norm = w.detach().float().norm(dim=0) if w.is_floating_point() else None
        self.register_buffer("col_norm", col_norm, persistent=False)

    def forward(self, x):
        m = self.mlp
        a = m.act_fn(m.gate_proj(x)) * m.up_proj(x)     # [..., intermediate]
        rec = self.ctrl.get("recorder")
        if rec is not None:
            rec(self, a)
        msk = self.ctrl.get("masker")
        if msk is not None:
            a = msk(a, self.col_norm)
        return m.down_proj(a)


def install_sparse_mlps(model):
    """Replace every decoder block's `.mlp` with a SparseMLP. Returns the shared
    ctrl dict (mutate ctrl["masker"] / ctrl["recorder"] to change behaviour)."""
    ctrl = {"masker": None, "recorder": None}
    wrappers = []
    for i, layer in enumerate(get_decoder_layers(model)):
        if isinstance(layer.mlp, SparseMLP):
            wrappers.append(layer.mlp)
            continue
        sp = SparseMLP(layer.mlp, ctrl, i)
        layer.mlp = sp
        wrappers.append(sp)
    return ctrl, wrappers


def _drop_smallest(a, score, n_drop):
    if n_drop <= 0:
        return a
    idx = torch.topk(score, n_drop, dim=-1, largest=False).indices
    return a.scatter(-1, idx, 0.0)


def make_oracle_masker(sparsity, contribution_aware):
    """Per-token top-k: keep the (1-sparsity) neurons with the largest true
    importance, zero the rest. This is the *ceiling* — it needs every neuron's
    activation to rank them, so it gives no speedup; it bounds what perfect
    detection could achieve.
      contribution_aware=False -> rank by |a_i|              (gate*up magnitude)
      contribution_aware=True  -> rank by |a_i| * ||down_i|| (true output norm)
    """
    def masker(a, col_norm):
        n_drop = int(round(sparsity * a.shape[-1]))
        score = a.abs().float()
        if contribution_aware:
            score = score * col_norm
        return _drop_smallest(a, score, n_drop)
    return masker


def make_random_masker(sparsity, generator=None):
    """Lower bound: drop a random per-token subset."""
    def masker(a, col_norm):
        n_drop = int(round(sparsity * a.shape[-1]))
        if n_drop <= 0:
            return a
        score = torch.rand(a.shape, device=a.device, generator=generator)
        return _drop_smallest(a, score, n_drop)
    return masker


def build_masker(method, sparsity, device, seed=0):
    if method == "random":
        g = torch.Generator(device=device)
        g.manual_seed(seed)
        return make_random_masker(sparsity, g)
    if method == "oracle_gate":
        return make_oracle_masker(sparsity, contribution_aware=False)
    if method == "oracle_contrib":
        return make_oracle_masker(sparsity, contribution_aware=True)
    raise ValueError(f"unknown method '{method}'")


METHODS = ["random", "oracle_gate", "oracle_contrib"]


class MassRecorder:
    """Per layer, the mean over tokens of the sorted-descending cumulative
    contribution fraction. curve[L][r] = average share of a token's total
    |contribution| captured by its top-(r+1) neurons. Tells you the intrinsic
    headroom: if 90% of mass sits in the top 20% of neurons, ~80% of neurons
    are skippable in principle."""

    def __init__(self, contribution_aware=True):
        self.contribution_aware = contribution_aware
        self.curves = {}
        self.counts = {}

    def __call__(self, wrapper, a):
        score = a.abs().float()
        if self.contribution_aware:
            score = score * wrapper.col_norm
        s = score.reshape(-1, score.shape[-1])           # [tokens, intermediate]
        sorted_desc, _ = torch.sort(s, dim=-1, descending=True)
        cum = sorted_desc.cumsum(-1)
        frac = (cum / cum[:, -1:].clamp_min(1e-9)).mean(0)  # [intermediate]
        i, n = wrapper.idx, s.shape[0]
        if i not in self.curves:
            self.curves[i] = torch.zeros_like(frac)
            self.counts[i] = 0
        c = self.counts[i]
        self.curves[i] = (self.curves[i] * c + frac * n) / (c + n)
        self.counts[i] = c + n
