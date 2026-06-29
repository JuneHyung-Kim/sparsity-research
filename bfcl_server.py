"""Minimal OpenAI-compatible /v1/completions server for BFCL, with the
contextual activation-sparsity masker installed on the FFNs.

BFCL's OSS handler (with --skip-server-setup) renders the full chat prompt
itself and POSTs a raw `prompt` string to `/v1/completions`, then reads
`choices[0].text` + `usage`. So this server is a pure text-completion endpoint:
it does NOT apply a chat template and does NOT parse tool calls.

The point: weights load once, `install_sparse_mlps` wraps every block's `.mlp`,
and `ctrl["masker"]` is fixed at startup (--method/--sparsity). Running this at
sparsity=0 gives the dense BFCL baseline; re-running at sparsity>0 with an oracle
masker measures how per-token FFN sparsity degrades function-calling.

Throughput: BFCL is embarrassingly parallel across test cases, but batch-1
`generate()` leaves the GPU ~95% idle (8B decode is memory-bandwidth-bound).
This server therefore **micro-batches**: handler threads drop their request on a
queue and block; one worker thread drains up to --batch requests (within a short
--batch-wait window), left-pads them, and runs a single batched `generate()`.
The masker acts per token (last dim), so batching needs no masker change and the
result is faithful to the batch-1 path (pad positions are attention-masked).
Run BFCL with a matching --num-threads so enough requests are in flight.
"""
import argparse
import json
import queue
import re
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
ARGS = None
REQ_Q = "queue.Queue of _Req"   # set in main()
EMPTY_THINK = "<think>\n\n</think>\n\n"   # Qwen3 non-thinking trigger
STOPS = ("<|im_end|>", "<|endoftext|>")
# A leading <think>...</think> block. BFCL's OSS handler appends the raw
# completion to the chat history every step (up to 20 per turn), so if we leave
# the reasoning in, multi_turn context explodes and the GPU OOMs. Qwen's own
# multi-turn convention drops prior-turn reasoning; we strip it here so history
# carries only the answer/tool_call. The model still reasons during generation.
THINK_RE = re.compile(r"^\s*<think>.*?</think>\s*", re.S)


class _Req:
    __slots__ = ("prompt", "max_tokens", "temperature",
                 "ev", "text", "n_in", "n_out", "err")

    def __init__(self, prompt, max_tokens, temperature):
        self.prompt = prompt
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.ev = threading.Event()
        self.text = None
        self.n_in = 0
        self.n_out = 0
        self.err = None


def _prep(prompt):
    # Disable Qwen3 "thinking" by closing an empty think block right after the
    # assistant tag (BFCL's prompt ends with "<|im_start|>assistant\n").
    if not ARGS.think and prompt.endswith("assistant\n"):
        return prompt + EMPTY_THINK
    return prompt


@torch.no_grad()
def _run_group(reqs):
    """Generate for one same-config group of requests as a single batch."""
    prompts = [_prep(r.prompt) for r in reqs]
    # left-padded batch so generated tokens align at a common offset (n_in).
    enc = TOK(prompts, return_tensors="pt", add_special_tokens=False, padding=True)
    input_ids = enc.input_ids.to(MODEL.device)
    attn = enc.attention_mask.to(MODEL.device)
    n_in = int(input_ids.shape[1])

    temp = reqs[0].temperature
    do_sample = temp is not None and temp > 1e-4
    # one max_new_tokens for the batch: take the largest request cap (others stop
    # early at EOS), bounded by the hard --max-new.
    want = max((int(r.max_tokens) if r.max_tokens else ARGS.max_new) for r in reqs)
    cap = min(want, ARGS.max_new)

    out = MODEL.generate(
        input_ids,
        attention_mask=attn,
        max_new_tokens=max(1, cap),
        do_sample=do_sample,
        temperature=temp if do_sample else None,
        top_p=0.8 if do_sample else None,
        pad_token_id=TOK.pad_token_id,
    )
    gen = out[:, n_in:]                         # [B, new] — common offset (left pad)
    pad_id = TOK.pad_token_id
    for i, r in enumerate(reqs):
        ids = gen[i]
        text = TOK.decode(ids, skip_special_tokens=False)
        for stop in STOPS:                     # trim trailing EOS/im_end marker
            j = text.find(stop)
            if j != -1:
                text = text[:j]
        if ARGS.think:                         # drop reasoning so it can't pile
            text = THINK_RE.sub("", text, count=1)   # up in multi_turn history
        r.text = text
        r.n_in = int(attn[i].sum())            # this row's real prompt tokens
        r.n_out = int((ids != pad_id).sum())   # generated tokens (approx)


def _worker():
    """Single GPU worker: drain a batch, group by sampling config, generate."""
    while True:
        first = REQ_Q.get()
        batch = [first]
        deadline = time.time() + ARGS.batch_wait / 1000.0
        while len(batch) < ARGS.batch:
            remaining = deadline - time.time()
            if remaining <= 0:
                break
            try:
                batch.append(REQ_Q.get(timeout=remaining))
            except queue.Empty:
                break
        # group by (do_sample, temperature) so one generate() config fits all.
        groups = {}
        for r in batch:
            key = round(r.temperature, 4) if (r.temperature and r.temperature > 1e-4) else 0.0
            groups.setdefault(key, []).append(r)
        for g in groups.values():
            try:
                _run_group(g)
            except Exception as e:             # surface to each waiting handler
                for r in g:
                    r.err = str(e)
            finally:
                for r in g:
                    r.ev.set()


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
        r = _Req(prompt, req.get("max_tokens", ARGS.max_new),
                 req.get("temperature", 0.0))
        REQ_Q.put(r)                              # hand off to the batching worker
        r.ev.wait()
        if r.err is not None:
            self._json(500, {"error": r.err})
            return
        self._json(200, {
            "id": "cmpl-bfcl",
            "object": "text_completion",
            "created": int(time.time()),
            "model": req.get("model", ARGS.served_name),
            "choices": [{"index": 0, "text": r.text, "logprobs": None,
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": r.n_in, "completion_tokens": r.n_out,
                      "total_tokens": r.n_in + r.n_out},
        })


def main():
    global MODEL, TOK, CTRL, ARGS, REQ_Q
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--served-name", default="Qwen/Qwen3-8B-FC")
    ap.add_argument("--port", type=int, default=1053)
    ap.add_argument("--method", default="oracle_gate",
                    help="masker method (only used when --sparsity > 0)")
    ap.add_argument("--sparsity", type=float, default=0.0,
                    help="per-token FFN neuron drop fraction; 0 = dense baseline")
    ap.add_argument("--max-new", type=int, default=1024,
                    help="hard cap on new tokens")
    ap.add_argument("--batch", type=int, default=24,
                    help="max requests per batched generate() (lower if OOM)")
    ap.add_argument("--batch-wait", type=float, default=10.0,
                    help="ms to wait accumulating a batch before generating")
    ap.add_argument("--think", action="store_true",
                    help="allow Qwen3 thinking (default: disabled for speed)")
    ARGS = ap.parse_args()
    REQ_Q = queue.Queue()

    print(f"[server] loading {ARGS.model} (bf16) ...", flush=True)
    TOK = AutoTokenizer.from_pretrained(ARGS.model)
    # batched decoding needs left padding (generated tokens share a right offset).
    TOK.padding_side = "left"
    if TOK.pad_token_id is None:
        TOK.pad_token = TOK.eos_token
    MODEL = AutoModelForCausalLM.from_pretrained(
        ARGS.model, dtype=torch.bfloat16, device_map="cuda")
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

    threading.Thread(target=_worker, daemon=True).start()
    srv = ThreadingHTTPServer(("0.0.0.0", ARGS.port), Handler)
    print(f"[server] ready on :{ARGS.port} "
          f"(batch={ARGS.batch}, wait={ARGS.batch_wait}ms, think={'on' if ARGS.think else 'off'})",
          flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
