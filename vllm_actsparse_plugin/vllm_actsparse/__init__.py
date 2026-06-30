"""Inject the contextual activation-sparsity masker into vLLM's Gemma-4 FFN.

vLLM serves Gemma-4-12B with a dense `Gemma4MLP` whose forward is

    gate_up, _ = self.gate_up_proj(x)     # [tokens, 2*I], order [gate | up]
    a = self.act_fn(gate_up)              # GeluAndMul -> [tokens, I] = gelu(gate)*up
    x, _ = self.down_proj(a)              # [tokens, hidden]

This is the exact spot `src.actsparse.SparseMLP` masks in the HF path: we drop a
per-token subset of the I intermediate neurons before `down_proj`. We monkeypatch
`Gemma4MLP.forward` (one patch covers all 48 layers, every TP rank, regardless of
the multimodal `language_model` nesting) and reuse `src.actsparse.build_masker`
verbatim so the kept-set rule is identical to the HF baseline.

Controlled entirely by env, one fixed config per server process (the agent engine
is single-sparsity; the user simulator runs a separate, unpatched vLLM):
    ACTSPARSE_SPARSITY  per-token FFN drop fraction; <=0 (or unset) => no patch, dense
    ACTSPARSE_METHOD    random | oracle_gate | oracle_gateonly | oracle_contrib
    ACTSPARSE_SEED      seed for the `random` method (default 0)

Requirements: TP=1 (the intermediate dim is column-parallel sharded, so top-k under
TP>1 would be shard-local, not global), and `src` importable (set PYTHONPATH to the
repo root). oracle_contrib needs down_proj column norms, which are only faithful in
bf16; under fp8 the weight is quantized (norms ranking-preserved only up to the
per-tensor scale), so prefer oracle_gate/oracle_gateonly/random for local fp8 dev.
"""
import os

_PATCHED = False


def _coerce_sparsity():
    raw = os.environ.get("ACTSPARSE_SPARSITY", "") or "0"
    try:
        return float(raw)
    except ValueError:
        return 0.0


def register():
    """vllm.general_plugins entry point. Patches Gemma4MLP.forward iff sparsity > 0."""
    global _PATCHED
    if _PATCHED:
        return
    sparsity = _coerce_sparsity()
    if sparsity <= 0.0:
        # Dense run == stock vLLM. Leaving the model untouched keeps the dense
        # condition numerically identical to an un-plugged engine.
        return
    method = os.environ.get("ACTSPARSE_METHOD", "oracle_gate")
    seed = int(os.environ.get("ACTSPARSE_SEED", "0"))
    _install(method, sparsity, seed)
    _PATCHED = True


def _install(method, sparsity, seed):
    import torch
    import torch.nn.functional as F
    from vllm.model_executor.models.gemma4 import Gemma4MLP
    from vllm.logger import init_logger

    # Single source of truth for the kept-set rule -- the same factory the HF
    # SparseMLP uses. Requires the repo root on PYTHONPATH.
    from src.actsparse import build_masker, METHODS

    logger = init_logger("vllm_actsparse")
    if method not in METHODS:
        raise ValueError(f"ACTSPARSE_METHOD={method!r} not in {METHODS}")

    need_gate = method == "oracle_gateonly"   # masker ranks by |gelu(gate)| alone
    need_colnorm = method == "oracle_contrib"  # masker ranks by |a| * ||down_col||

    # Lazily-built so we know the activation device (for the `random` generator);
    # one masker, shared by every layer, fixed for the life of the process.
    state = {"masker": None, "tp_checked": False}

    def _maybe_warn_tp():
        if state["tp_checked"]:
            return
        state["tp_checked"] = True
        try:
            from vllm.distributed import get_tensor_model_parallel_world_size
            tp = get_tensor_model_parallel_world_size()
        except Exception:
            tp = 1
        if tp != 1:
            logger.warning(
                "actsparse: tensor_parallel_size=%d -- per-token top-k is "
                "SHARD-LOCAL, not global; kept set differs from the TP=1/HF "
                "baseline. Run the masked agent at TP=1.", tp)

    def patched_forward(self, x):
        gate_up, _ = self.gate_up_proj(x)
        a = self.act_fn(gate_up)
        _maybe_warn_tp()
        if state["masker"] is None:
            state["masker"] = build_masker(method, sparsity, a.device, seed)
        col_norm = None
        if need_colnorm:
            col_norm = getattr(self, "_actsparse_colnorm", None)
            if col_norm is None:
                w = self.down_proj.weight
                col_norm = w.detach().float().norm(dim=0) if w.is_floating_point() else None
                self._actsparse_colnorm = col_norm
        gate = None
        if need_gate:
            # gate_up is [gate | up]; gelu(gate) is the gate-only ranking signal.
            gate = F.gelu(gate_up[..., : a.shape[-1]], approximate="tanh")
        a = state["masker"](a, col_norm, gate)
        if not state.get("probed"):
            state["probed"] = True
            import sys
            z = (a == 0).float().mean().item()
            print(f"[actsparse] first forward: zeroed_frac={z:.3f} I={a.shape[-1]} "
                  f"(target sparsity={sparsity})", file=sys.stderr, flush=True)
        x, _ = self.down_proj(a)
        return x

    Gemma4MLP.forward = patched_forward
    import sys
    print(f"[actsparse] patched Gemma4MLP.forward method={method} "
          f"sparsity={sparsity} seed={seed}", file=sys.stderr, flush=True)
    logger.info(
        "actsparse: patched Gemma4MLP.forward (method=%s sparsity=%.3f seed=%d)",
        method, sparsity, seed)
