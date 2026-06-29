"""Thin wrapper around the tau2 CLI.

tau2 computes a $ cost for every LLM call via litellm.completion_cost(). For our
self-hosted endpoint the model id (e.g. "Qwen3-8B-agent") isn't in litellm's price
map, so each call logs a noisy `ERROR ... This model isn't mapped yet`. The cost is
irrelevant here (local GPU), so we pre-register our local model ids at zero cost
before delegating to tau2's normal CLI entry point. Purely cosmetic: it silences
the per-call cost ERROR; it does not change generation or scoring.

Model ids come from $TAU2_LOCAL_MODELS (comma-separated), set by run.sh.
"""
import os
import sys

import litellm

_ZERO = {"input_cost_per_token": 0.0, "output_cost_per_token": 0.0,
         "litellm_provider": "openai", "mode": "chat"}
_names = [n.strip() for n in os.environ.get("TAU2_LOCAL_MODELS", "").split(",") if n.strip()]
# register both the bare id and the openai/ -prefixed form (litellm may look up either)
litellm.register_model({n: _ZERO for name in _names for n in (name, f"openai/{name}")})

from tau2.cli import main

if __name__ == "__main__":
    sys.exit(main())
