"""Intra-expert activation sparsity for Qwen3-MoE (expr2 / Q1).

Each *active* expert is itself a SwiGLU FFN, per routed token x:

    a   = act(gate_up(x).gate) * gate_up(x).up     # [routed_tokens, moe_inter]
    out = down_proj @ a                            # expert's contribution

expr1 asked this of a dense FFN; here we ask it one level deeper, *inside each
selected expert*. The router's pick of 8/128 experts is a separate, coarser
sparsity — this measures how concentrated the contribution is *within* an
expert that was already chosen.

transformers stores Qwen3-MoE experts as batched 3D params and runs them in one
`Qwen3MoeExperts.forward` loop, dispatched at runtime via
`config._experts_implementation`. We register a custom "recording" impl that
mirrors that loop and taps `a` per active expert into a recorder, then flip the
flag. Because the dispatcher runs inside the module's own forward, this rides
within accelerate's weight-onload window — device_map CPU/disk offload is fine.
"""
import torch
import torch.nn.functional as F
from transformers.integrations.moe import ALL_EXPERTS_FUNCTIONS

from src.actsparse import get_decoder_layers

_IMPL = "recording"


class ExpertMassRecorder:
    """Per layer, the mean over (token, active-expert) rows of the sorted-
    descending cumulative contribution-mass curve over an expert's neurons.

    curve[L][r] = average share of a routed token's total |contribution| (within
    the expert it was routed to) captured by that expert's top-(r+1) neurons.
    Saturates early => most of an active expert's output rides on a few neurons,
    i.e. lots of skippable neurons *inside* the expert. Directly comparable to
    expr1's dense headroom curve (same construction, x-axis is moe_inter)."""

    def __init__(self, contribution_aware=True):
        self.contribution_aware = contribution_aware
        self.curves = {}
        self.counts = {}

    def add(self, layer_idx, a, col_norm):
        score = a.abs().float()
        if self.contribution_aware:
            score = score * col_norm                       # |a_i| * ||down_i||
        s = score.reshape(-1, score.shape[-1])             # [rows, moe_inter]
        sorted_desc, _ = torch.sort(s, dim=-1, descending=True)
        cum = sorted_desc.cumsum(-1)
        frac = (cum / cum[:, -1:].clamp_min(1e-9)).mean(0)  # [moe_inter]
        i, n = layer_idx, s.shape[0]
        if i not in self.curves:
            self.curves[i] = torch.zeros_like(frac)
            self.counts[i] = 0
        c = self.counts[i]
        self.curves[i] = (self.curves[i] * c + frac * n) / (c + n)
        self.counts[i] = c + n


def _recording_experts_forward(self, hidden_states, top_k_index, top_k_weights):
    """Faithful mirror of Qwen3MoeExperts.forward, with a recorder tap on `a`.
    Output is identical to upstream, so PPL/masking can reuse this later."""
    ctrl = getattr(self, "_ctrl", None)
    rec = ctrl.get("recorder") if ctrl else None
    final_hidden_states = torch.zeros_like(hidden_states)
    with torch.no_grad():
        expert_mask = F.one_hot(top_k_index, num_classes=self.num_experts).permute(2, 1, 0)
        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
    for expert_idx in expert_hit:
        expert_idx = expert_idx[0]
        if expert_idx == self.num_experts:
            continue
        top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
        current_state = hidden_states[token_idx]
        gate, up = F.linear(current_state, self.gate_up_proj[expert_idx]).chunk(2, dim=-1)
        a = self.act_fn(gate) * up
        if rec is not None:
            col_norm = self.down_proj[expert_idx].detach().float().norm(dim=0)  # [moe_inter]
            rec.add(self._layer_idx, a.detach(), col_norm)
        current_hidden_states = F.linear(a, self.down_proj[expert_idx])
        current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
        final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))
    return final_hidden_states


ALL_EXPERTS_FUNCTIONS[_IMPL] = _recording_experts_forward


def install_expert_recorder(model, recorder):
    """Route every MoE block's experts through `_recording_experts_forward` and
    attach the shared ctrl. Returns (ctrl, restore): mutate ctrl['recorder'] to
    change behaviour, call restore() to put the original impl back."""
    ctrl = {"recorder": recorder}
    prev = getattr(model.config, "_experts_implementation", "eager")
    blocks = []
    for i, layer in enumerate(get_decoder_layers(model)):
        experts = getattr(layer.mlp, "experts", None)
        if experts is None:                 # dense block (if decoder_sparse_step>1)
            continue
        experts._layer_idx = i
        experts._ctrl = ctrl
        experts.config._experts_implementation = _IMPL
        blocks.append(experts)
    model.config._experts_implementation = _IMPL

    def restore():
        for e in blocks:
            e.config._experts_implementation = prev
        model.config._experts_implementation = prev
    return ctrl, restore
