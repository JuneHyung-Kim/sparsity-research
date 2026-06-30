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

# --- tolerant tool-call argument parsing -------------------------------------
# tau2.utils.llm_utils.get_response does `json.loads(tool_call.function.arguments)`
# (llm_utils.py:440). vLLM's native gemma4 tool parser serializes a no-argument
# tool call with an EMPTY arguments string ("") rather than "{}", so json.loads("")
# raises "Expecting value: line 1 column 1" -> the whole task is retried 4x and
# dropped as an infrastructure_error (≈25% of retail tasks in a dense smoke). The
# parse is deterministic in the seed, so retries never recover. We swap llm_utils's
# `json` for a proxy whose .loads() treats empty/invalid argument strings as {} (an
# empty tool call IS a no-arg call), and log each occurrence so the trigger stays
# visible. Everything else delegates to the real json module. Applies to BOTH the
# dense baseline and the masked sweep (parser behaviour is masker-independent).
import json as _json
import tau2.utils.llm_utils as _llm


class _TolerantJson:
    def __getattr__(self, name):
        return getattr(_json, name)

    def loads(self, s, *a, **k):
        if isinstance(s, (str, bytes)) and (not s or not s.strip()):
            print("[tau2_cli] tool-call arguments empty -> {}", file=sys.stderr, flush=True)
            return {}
        try:
            return _json.loads(s, *a, **k)
        except _json.JSONDecodeError:
            print(f"[tau2_cli] tool-call arguments not JSON ({s!r:.80}) -> {{}}",
                  file=sys.stderr, flush=True)
            return {}


_llm.json = _TolerantJson()
# -----------------------------------------------------------------------------

# --- force the NL-assertion reward judge to emit parseable JSON ---------------
# The judge (tau2 config DEFAULT_LLM_NL_ASSERTIONS, which we alias onto the dense
# Gemma engine) is prompted for a JSON object but NOT given a response_format, then
# its content is json.loads'd (evaluator_nl_assertions.py:127). gpt-4.1 complies;
# Gemma wraps the JSON in a ```json ... ``` markdown fence, so json.loads fails at
# char 0 -> the task is retried 4x and dropped as infrastructure_error (it hit the
# ~35% of retail tasks carrying nl_assertions). We inject an OpenAI structured-output
# response_format so vLLM constrains the judge to a bare JSON object of the exact
# shape the evaluator indexes (results[].expectedOutcome/reasoning/metExpectation) --
# verified to flow through litellm -> vLLM guided decoding. The judge is the fixed
# dense engine across all sparsity points, so this affects dense and sweep equally.
import tau2.config as _cfg

_NL_JUDGE_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "nl_assertions",
        "schema": {
            "type": "object",
            "properties": {
                "results": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "expectedOutcome": {"type": "string"},
                            "reasoning": {"type": "string"},
                            "metExpectation": {"type": "boolean"},
                        },
                        "required": ["expectedOutcome", "reasoning", "metExpectation"],
                    },
                }
            },
            "required": ["results"],
        },
    },
}
# Mutate the shared args dict in place so the binding evaluator_nl_assertions already
# imported sees it.
_cfg.DEFAULT_LLM_NL_ASSERTIONS_ARGS["response_format"] = _NL_JUDGE_SCHEMA
# -----------------------------------------------------------------------------

from tau2.cli import main

if __name__ == "__main__":
    sys.exit(main())
