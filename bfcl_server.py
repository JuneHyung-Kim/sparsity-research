"""Minimal OpenAI-compatible /v1/completions server for BFCL, with the
contextual activation-sparsity masker installed on the FFNs.

BFCL's OSS handler (with --skip-server-setup) renders the full chat prompt
itself and POSTs a raw `prompt` string to `/v1/completions`, then reads
`choices[0].text` + `usage`. So this server is a pure text-completion endpoint:
it does NOT apply a chat template and does NOT parse tool calls — it just
generates and returns Qwen's raw text (incl. <tool_call>...). BFCL does the
templating and parsing on both ends.

The point: weights load once, `install_sparse_mlps` wraps every block's `.mlp`,
and `ctrl["masker"]` is fixed at startup (--method/--sparsity). Running this at
sparsity=0 gives the dense BFCL baseline; re-running at sparsity>0 with an oracle
masker measures how per-token FFN sparsity degrades function-calling — the BFCL
analogue of the PPL-vs-sparsity sweep.
"""
import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.actsparse import build_masker, install_sparse_mlps

# ---- globals set in main() ----
MODEL = None
TOK = None
CTRL = None
GEN_LOCK = threading.Lock()   # serialize GPU generation (BFCL is sequential by default anyway)
ARGS = None
EMPTY_THINK = "<think>\n\n</think>\n\n"   # Qwen3 non-thinking trigger


@torch.no_grad()
def generate(prompt: str, max_tokens: int, temperature: float):
    # Disable Qwen3 "thinking" by closing an empty think block right after the
    # assistant tag (BFCL's prompt ends with "<|im_start|>assistant\n" and does
    # not inject this itself). Keeps generations short and parse-clean.
    if not ARGS.think and prompt.endswith("assistant\n"):
        prompt = prompt + EMPTY_THINK

    # The prompt is already fully templated (has <|im_start|> etc.), so do NOT
    # add special tokens again.
    enc = TOK(prompt, return_tensors="pt", add_special_tokens=False)
    input_ids = enc.input_ids.to(MODEL.device)
    attn = enc.attention_mask.to(MODEL.device)
    n_in = int(input_ids.shape[1])

    do_sample = temperature is not None and temperature > 1e-4
    cap = min(int(max_tokens) if max_tokens else ARGS.max_new, ARGS.max_new)
    with GEN_LOCK:
        out = MODEL.generate(
            input_ids,
            attention_mask=attn,
            max_new_tokens=max(1, cap),
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            top_p=0.8 if do_sample else None,
            pad_token_id=TOK.pad_token_id or TOK.eos_token_id,
        )
    gen_ids = out[0, n_in:]
    text = TOK.decode(gen_ids, skip_special_tokens=False)
    # Trim a trailing EOS/im_end marker if the tokenizer left it in the string.
    for stop in ("<|im_end|>", "<|endoftext|>"):
        idx = text.find(stop)
        if idx != -1:
            text = text[:idx]
    return text, n_in, int(gen_ids.shape[0])


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):   # silence per-request logging
        pass

    def _json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        # BFCL readiness probe: GET {base_url}/models must return 200.
        if self.path.rstrip("/").endswith("/models"):
            self._json(200, {"object": "list",
                             "data": [{"id": ARGS.served_name, "object": "model"}]})
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):
        if not self.path.rstrip("/").endswith("/completions"):
            self._json(404, {"error": "not found"})
            return
        n = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(n) or b"{}")
        prompt = req.get("prompt", "")
        if isinstance(prompt, list):              # batch -> take first (BFCL sends a str)
            prompt = prompt[0] if prompt else ""
        max_tokens = req.get("max_tokens", ARGS.max_new)
        temperature = req.get("temperature", 0.0)
        try:
            text, n_in, n_out = generate(prompt, max_tokens, temperature)
        except Exception as e:                     # surface errors instead of hanging BFCL
            self._json(500, {"error": str(e)})
            return
        self._json(200, {
            "id": "cmpl-bfcl",
            "object": "text_completion",
            "created": int(time.time()),
            "model": req.get("model", ARGS.served_name),
            "choices": [{"index": 0, "text": text, "logprobs": None,
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": n_in, "completion_tokens": n_out,
                      "total_tokens": n_in + n_out},
        })


def main():
    global MODEL, TOK, CTRL, ARGS
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--served-name", default="Qwen/Qwen3-8B-FC")
    ap.add_argument("--port", type=int, default=1053)
    ap.add_argument("--method", default="oracle_gate",
                    help="masker method (only used when --sparsity > 0)")
    ap.add_argument("--sparsity", type=float, default=0.0,
                    help="per-token FFN neuron drop fraction; 0 = dense baseline")
    ap.add_argument("--max-new", type=int, default=1024,
                    help="hard cap on new tokens (bounds smoke-test time)")
    ap.add_argument("--think", action="store_true",
                    help="allow Qwen3 thinking (default: disabled for speed)")
    ARGS = ap.parse_args()

    print(f"[server] loading {ARGS.model} (bf16) ...", flush=True)
    TOK = AutoTokenizer.from_pretrained(ARGS.model)
    MODEL = AutoModelForCausalLM.from_pretrained(
        ARGS.model, torch_dtype=torch.bfloat16, device_map="cuda")
    MODEL.eval()

    CTRL, _ = install_sparse_mlps(MODEL)
    # sanity: confirm the wrap took on Qwen's SwiGLU MLP
    sample = MODEL.model.layers[0].mlp
    assert hasattr(sample.mlp, "gate_proj") and hasattr(sample.mlp, "act_fn"), \
        "MLP does not look like a SwiGLU gate/up/down FFN"
    if ARGS.sparsity > 0:
        CTRL["masker"] = build_masker(ARGS.method, ARGS.sparsity, MODEL.device)
        print(f"[server] masker ON: method={ARGS.method} sparsity={ARGS.sparsity}", flush=True)
    else:
        print("[server] masker OFF (dense baseline)", flush=True)

    srv = ThreadingHTTPServer(("0.0.0.0", ARGS.port), Handler)
    print(f"[server] ready on :{ARGS.port} (think={'on' if ARGS.think else 'off'})", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
