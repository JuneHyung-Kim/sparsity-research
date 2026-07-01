"""BFCL prompt-mode handler for google/gemma-4-12B-it, served by vLLM.

Gemma-4 is not in BFCL's `MODEL_CONFIG_MAPPING`; `_bfcl_cli.py` registers it and
points it here. This mirrors the official `GemmaHandler` (prompt mode: functions
are injected into the system prompt as text and the reply is parsed by BFCL's
default AST decoder), but adapts three things that differ in Gemma-4:

  1. Turn delimiters. Gemma-4's chat template uses `<|turn>{role}\n...<turn|>\n`
     (role `model` for the assistant), NOT Gemma-3's `<start_of_turn>`.
  2. Reasoning channel. Gemma-4 emits its chain-of-thought inside a
     `<|channel>thought\n...\n<channel|>` block. We toggle it via the same
     mechanism as the model's own template (`enable_thinking`) -- OFF appends an
     already-closed empty thought channel to the generation prompt to suppress
     reasoning; ON leaves the channel open and injects `<|think|>` in the system
     turn -- and strip the emitted block before AST parsing (same logic as the
     template's `strip_thinking` macro).
  3. Context length. Gemma-4's top-level config has `max_position_embeddings=None`
     (the real value lives under `text_config`), so the base handler leaves
     `max_context_length=None` and its token-budget math would crash. We set it.

Controlled by env (set per run by run_vllm.sh):
    BFCL_THINK    1 => enable Gemma-4 reasoning (default 0). Policy: single_turn
                  OFF, multi_turn ON.
    BFCL_MAX_CTX  context length used for the completion token budget (default
                  16384; keep == vLLM --max-model-len).
"""
import os
from typing import Any

from bfcl_eval.model_handler.local_inference.base_oss_handler import OSSHandler
from bfcl_eval.model_handler.utils import (
    combine_consecutive_user_prompts,
    system_prompt_pre_processing_chat_model,
)
from overrides import override


def _strip_thinking(text: str) -> str:
    """Remove `<|channel>thought...<channel|>` blocks, mirroring the Gemma-4
    chat template's `strip_thinking` macro: split on the closing tag, and for any
    part that opened a channel keep only the text before the opener."""
    result = []
    for part in text.split("<channel|>"):
        if "<|channel>" in part:
            result.append(part.split("<|channel>")[0])
        else:
            result.append(part)
    return "".join(result).strip()


class Gemma4Handler(OSSHandler):
    def __init__(
        self,
        model_name,
        temperature,
        registry_name,
        is_fc_model,
        dtype="bfloat16",
        **kwargs,
    ) -> None:
        super().__init__(model_name, temperature, registry_name, is_fc_model, **kwargs)
        self._enable_thinking = os.getenv("BFCL_THINK", "0").strip().lower() not in (
            "0", "", "false", "no",
        )
        self._ctx_len = int(os.getenv("BFCL_MAX_CTX", "16384"))
        # Gemma-4 delimits reasoning (and its native tool DSL) with special tokens;
        # keep them in the returned text so we can strip the thought channel
        # ourselves. The base handler forwards this to vLLM via extra_body.
        self.skip_special_tokens = False

    @override
    def _pre_query_processing_prompting(self, test_entry: dict) -> dict:
        # Same as GemmaHandler: fold the function declarations into the system
        # prompt and merge back-to-back user turns. Role remap (assistant->model)
        # happens in _format_prompt, so no _substitute_prompt_role pass here.
        functions: list = test_entry["function"]
        test_entry_id: str = test_entry["id"]

        test_entry["question"][0] = system_prompt_pre_processing_chat_model(
            test_entry["question"][0], functions, test_entry_id
        )
        for round_idx in range(len(test_entry["question"])):
            test_entry["question"][round_idx] = combine_consecutive_user_prompts(
                test_entry["question"][round_idx]
            )

        return {"message": [], "function": functions}

    @override
    def _format_prompt(self, messages, function):
        bos = self.tokenizer.bos_token or "<bos>"
        formatted_prompt = bos

        msgs = list(messages)
        system_content = ""
        if msgs and msgs[0]["role"] == "system":
            system_content = (msgs[0]["content"] or "").strip()
            msgs = msgs[1:]

        # System turn carries the (function-injected) system prompt, and -- when
        # thinking is on -- the `<|think|>` opener the template puts there.
        if system_content or self._enable_thinking:
            formatted_prompt += "<|turn>system\n"
            if self._enable_thinking:
                formatted_prompt += "<|think|>\n"
            if system_content:
                formatted_prompt += system_content
            formatted_prompt += "<turn|>\n"

        for message in msgs:
            role = message["role"]
            if role in ("assistant", "model"):
                role = "model"
            elif role == "tool":
                # Prompt mode has no native tool role; surface tool results to the
                # model as user input (BFCL fills {role:tool, name, content}).
                role = "user"
            content = message["content"]
            if not isinstance(content, str):
                content = "" if content is None else str(content)
            formatted_prompt += f"<|turn>{role}\n{content.strip()}<turn|>\n"

        formatted_prompt += "<|turn>model\n"
        if not self._enable_thinking:
            # Empty, already-closed thought channel == suppress reasoning.
            formatted_prompt += "<|channel>thought\n<channel|>"

        return formatted_prompt

    @override
    def _query_prompting(self, inference_data: dict):
        # Gemma-4's top-level config has no usable max_position_embeddings, so the
        # base handler leaves this None; set it before the token-budget math runs.
        if not getattr(self, "max_context_length", None):
            self.max_context_length = self._ctx_len
        return super()._query_prompting(inference_data)

    @override
    def _parse_query_response_prompting(self, api_response: Any) -> dict:
        raw = api_response.choices[0].text
        cleaned = _strip_thinking(raw)
        return {
            "model_responses": cleaned,
            "reasoning_content": raw if cleaned != raw else "",
            "input_token": api_response.usage.prompt_tokens,
            "output_token": api_response.usage.completion_tokens,
        }
