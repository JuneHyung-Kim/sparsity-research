"""Inject the contextual activation-sparsity masker into a vLLM gated-MLP FFN.

vLLM serves both Gemma-4 (`Gemma4MLP`, GeGLU) and Qwen3 (`Qwen3MLP`, which is
`from .qwen2 import Qwen2MLP as Qwen3MLP`, SwiGLU) with the SAME forward shape:

    gate_up, _ = self.gate_up_proj(x)     # [tokens, 2*I], order [gate | up]
    a = self.act_fn(gate_up)              # {Gelu,Silu}AndMul -> [tokens, I] = act(gate)*up
    x, _ = self.down_proj(a)              # [tokens, hidden]

This is the exact spot `src.actsparse.SparseMLP` masks in the HF path: we drop a
per-token subset of the I intermediate neurons before `down_proj`. We monkeypatch
`<Model>MLP.forward` (one patch covers all layers, every TP rank, regardless of any
multimodal `language_model` nesting) and reuse `src.actsparse.build_masker` verbatim
so the kept-set rule is identical to the HF baseline. Every known MLP class that is
importable gets patched; only the loaded model's class is ever called, so patching
the others is harmless (no need to know the served model at register time).

Controlled entirely by env, one fixed config per server process:
    ACTSPARSE_SPARSITY  per-token FFN drop fraction; <=0 (or unset) => no patch, dense
    ACTSPARSE_METHOD    random | oracle_gate | oracle_gateonly | oracle_contrib
    ACTSPARSE_SEED      seed for the `random` method (default 0)

Requirements: TP=1 (the intermediate dim is column-parallel sharded, so top-k under
TP>1 would be shard-local, not global), and `src` importable (set PYTHONPATH to the
repo root). oracle_contrib needs down_proj column norms, faithful only in bf16.
oracle_gateonly ranks by |act(gate)| alone -- the act differs by family (gelu-tanh
for Gemma-4, silu for Qwen3); oracle_gate/random/contrib don't use it.
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
    """vllm.general_plugins entry point. Patches the gated-MLP forward iff sparsity > 0."""
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


def _maybe_warn_tp(state, logger):
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
            "actsparse: tensor_parallel_size=%d -- per-token top-k is SHARD-LOCAL, "
            "not global; kept set differs from the TP=1/HF baseline. Run at TP=1.", tp)


def _install(method, sparsity, seed):
    import importlib
    import sys
    import torch.nn.functional as F
    from vllm.logger import init_logger

    # Single source of truth for the kept-set rule -- the same factory the HF
    # SparseMLP uses. Requires the repo root on PYTHONPATH.
    from src.actsparse import build_masker, METHODS

    logger = init_logger("vllm_actsparse")
    if method not in METHODS:
        raise ValueError(f"ACTSPARSE_METHOD={method!r} not in {METHODS}")

    need_gate = method == "oracle_gateonly"    # ranks by |act(gate)| alone
    need_colnorm = method == "oracle_contrib"  # ranks by |a| * ||down_col||

    # (import_path, class, act on the gate half for the gate-only signal).
    # vLLM's Qwen3MLP is `from .qwen2 import Qwen2MLP as Qwen3MLP`, so patching
    # Qwen2MLP covers Qwen2/Qwen3.
    targets = [
        ("vllm.model_executor.models.gemma4", "Gemma4MLP",
         lambda g: F.gelu(g, approximate="tanh")),
        ("vllm.model_executor.models.qwen2", "Qwen2MLP", F.silu),
    ]

    def _make_forward(gate_act):
        # One masker per patched class, lazily built (needs the activation device),
        # fixed for the life of the process.
        state = {"masker": None, "tp_checked": False, "probed": False}

        def patched_forward(self, x):
            gate_up, _ = self.gate_up_proj(x)
            a = self.act_fn(gate_up)
            _maybe_warn_tp(state, logger)
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
                # gate_up is [gate | up]; act(gate) is the gate-only ranking signal.
                gate = gate_act(gate_up[..., : a.shape[-1]])
            a = state["masker"](a, col_norm, gate)
            if not state["probed"]:
                state["probed"] = True
                z = (a == 0).float().mean().item()
                print(f"[actsparse] first forward: zeroed_frac={z:.3f} I={a.shape[-1]} "
                      f"(target sparsity={sparsity})", file=sys.stderr, flush=True)
            x, _ = self.down_proj(a)
            return x

        return patched_forward

    patched = []
    for mod, cls, gate_act in targets:
        try:
            MLP = getattr(importlib.import_module(mod), cls)
        except (ImportError, AttributeError):
            continue
        MLP.forward = _make_forward(gate_act)
        patched.append(cls)
        print(f"[actsparse] patched {cls}.forward method={method} "
              f"sparsity={sparsity} seed={seed}", file=sys.stderr, flush=True)

    if not patched:
        raise RuntimeError(
            "actsparse: no known gated-MLP class to patch (need Gemma4MLP or Qwen2MLP)")
    logger.info("actsparse: patched %s (method=%s sparsity=%.3f seed=%d)",
                patched, method, sparsity, seed)
