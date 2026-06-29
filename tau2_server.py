"""OpenAI-compatible /v1/chat/completions server for tau2-bench, with the
contextual activation-sparsity masker installed on the FFNs.

Unlike BFCL (which rendered the prompt itself and POSTed a raw `/v1/completions`
string), tau2-bench talks the OpenAI *chat + tools* contract through LiteLLM:
it sends `messages` + `tools` (function schemas) and reads back
`choices[0].message.tool_calls`. So this server must honour that contract:

  * apply Qwen3's chat template with `tools=` (the template renders OpenAI
    function schemas into <tools> and tool results into <tool_response>),
  * generate (with the masker in the FFN path),
  * parse Qwen3's `<tool_call>{json}</tool_call>` blocks back into OpenAI
    `tool_calls`, and strip any leading <think> so multi-turn history stays lean.

tau2 needs TWO model roles: the **agent** (the policy we evaluate) and the
**user simulator** (part of the environment). The masker belongs only on the
agent. We therefore load ONE Qwen3-8B and route by the request's `model` name:
  --served-name      -> agent,    masker = build_masker(--method, --sparsity)
  --user-served-name -> user-sim, masker = None (dense)
The user simulator is the same dense weights held fixed across a sweep, so the
ONLY thing that varies between sparsity points is the agent's masker. The single
GPU worker sets ctrl["masker"] per same-role batch before generate(), so a batch
is always homogeneous in masker state and the result is faithful to batch-1.
"""
import argparse
import json
import queue
import re
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.actsparse import build_masker, install_sparse_mlps

# ---- globals set in main() ----
MODEL = None
TOK = None
CTRL = None
ARGS = None
REQ_Q = None
AGENT_MASKER = None              # built once at startup (None when --sparsity == 0)
STOPS = ("<|im_end|>", "<|endoftext|>")
# Strip a leading <think>...</think> from the generated text: when thinking is on
# the model reasons, but we keep only the answer/tool_call in the returned content
# so tau2's re-sent multi-turn history doesn't accumulate the chain-of-thought.
THINK_RE = re.compile(r"^\s*<think>.*?</think>\s*", re.S)
TOOLCALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)


class _Req:
    __slots__ = ("messages", "tools", "max_tokens", "temperature", "role",
                 "ev", "content", "tool_calls", "n_in", "n_out", "err")

    def __init__(self, messages, tools, max_tokens, temperature, role):
        self.messages = messages
        self.tools = tools
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.role = role                 # "agent" (maskable) or "user" (dense)
        self.ev = threading.Event()
        self.content = None
        self.tool_calls = None
        self.n_in = 0
        self.n_out = 0
        self.err = None


def _render(req):
    """OpenAI messages(+tools) -> a single prompt string via Qwen3's chat
    template. enable_thinking is controlled at template time (not by appending an
    empty <think>), which is the supported Qwen3 knob."""
    return TOK.apply_chat_template(
        req.messages,
        tools=req.tools or None,
        add_generation_prompt=True,
        tokenize=False,
        enable_thinking=ARGS.think,
    )


def _parse_completion(text):
    """Qwen3 completion -> (content, tool_calls) in OpenAI shape.
    tool_calls[i].function.arguments is a JSON *string* (litellm/tau2 json.loads
    it). content is the non-tool-call text, or None when the turn is pure calls."""
    text = THINK_RE.sub("", text, count=1)
    for stop in STOPS:
        j = text.find(stop)
        if j != -1:
            text = text[:j]
    tool_calls = []
    for m in TOOLCALL_RE.finditer(text):
        try:
            obj = json.loads(m.group(1))
        except json.JSONDecodeError:
            continue
        tool_calls.append({
            "id": "call_" + uuid.uuid4().hex[:24],
            "type": "function",
            "function": {"name": obj.get("name", ""),
                         "arguments": json.dumps(obj.get("arguments", {}))},
        })
    content = TOOLCALL_RE.sub("", text).strip()
    return (content or None), (tool_calls or None)


@torch.no_grad()
def _run_group(reqs):
    """Generate one same-(role, sampling) batch. The role fixes the masker."""
    CTRL["masker"] = AGENT_MASKER if reqs[0].role == "agent" else None
    prompts = [_render(r) for r in reqs]
    enc = TOK(prompts, return_tensors="pt", add_special_tokens=False, padding=True)
    input_ids = enc.input_ids.to(MODEL.device)
    attn = enc.attention_mask.to(MODEL.device)
    n_in = int(input_ids.shape[1])

    temp = reqs[0].temperature
    do_sample = temp is not None and temp > 1e-4
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
    gen = out[:, n_in:]                          # [B, new] — common offset (left pad)
    pad_id = TOK.pad_token_id
    for i, r in enumerate(reqs):
        ids = gen[i]
        text = TOK.decode(ids, skip_special_tokens=False)
        r.content, r.tool_calls = _parse_completion(text)
        r.n_in = int(attn[i].sum())
        r.n_out = int((ids != pad_id).sum())


def _worker():
    """Single GPU worker: drain a batch, split by (role, sampling), generate."""
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
        groups = {}
        for r in batch:
            t = round(r.temperature, 4) if (r.temperature and r.temperature > 1e-4) else 0.0
            groups.setdefault((r.role, t), []).append(r)
        for g in groups.values():
            try:
                _run_group(g)
            except Exception as e:
                for r in g:
                    r.err = str(e)
            finally:
                for r in g:
                    r.ev.set()


def _model_role(name):
    """Map a request's `model` field to a role. LiteLLM strips the 'openai/'
    provider prefix, so we see the bare served name."""
    name = (name or "").split("/")[-1]
    if name == ARGS.user_served_name.split("/")[-1]:
        return "user"
    return "agent"


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.rstrip("/").endswith("/models"):
            self._json(200, {"object": "list", "data": [
                {"id": ARGS.served_name, "object": "model"},
                {"id": ARGS.user_served_name, "object": "model"}]})
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):
        if not self.path.rstrip("/").endswith("/chat/completions"):
            self._json(404, {"error": "not found"})
            return
        n = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(n) or b"{}")
        r = _Req(
            messages=req.get("messages", []),
            tools=req.get("tools"),
            max_tokens=req.get("max_tokens") or req.get("max_completion_tokens"),
            temperature=req.get("temperature", 0.0),
            role=_model_role(req.get("model")),
        )
        REQ_Q.put(r)
        r.ev.wait()
        if r.err is not None:
            self._json(500, {"error": r.err})
            return
        msg = {"role": "assistant", "content": r.content}
        if r.tool_calls:
            msg["tool_calls"] = r.tool_calls
        self._json(200, {
            "id": "chatcmpl-" + uuid.uuid4().hex[:24],
            "object": "chat.completion",
            "created": int(time.time()),
            "model": req.get("model", ARGS.served_name),
            "choices": [{"index": 0, "message": msg,
                         "finish_reason": "tool_calls" if r.tool_calls else "stop"}],
            "usage": {"prompt_tokens": r.n_in, "completion_tokens": r.n_out,
                      "total_tokens": r.n_in + r.n_out},
        })


def main():
    global MODEL, TOK, CTRL, ARGS, REQ_Q, AGENT_MASKER
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--served-name", default="Qwen3-8B-agent",
                    help="model id for the AGENT role (masker applied)")
    ap.add_argument("--user-served-name", default="Qwen3-8B-user",
                    help="model id for the USER-SIMULATOR role (always dense)")
    ap.add_argument("--port", type=int, default=1055)
    ap.add_argument("--method", default="oracle_gate",
                    help="masker method (only used when --sparsity > 0)")
    ap.add_argument("--sparsity", type=float, default=0.0,
                    help="per-token FFN neuron drop fraction on the AGENT; 0 = dense")
    ap.add_argument("--max-new", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--batch-wait", type=float, default=15.0,
                    help="ms to accumulate a batch before generating")
    ap.add_argument("--think", action="store_true",
                    help="allow Qwen3 thinking (default: off for speed)")
    ARGS = ap.parse_args()
    REQ_Q = queue.Queue()

    print(f"[tau2-server] loading {ARGS.model} (bf16) ...", flush=True)
    TOK = AutoTokenizer.from_pretrained(ARGS.model)
    TOK.padding_side = "left"            # batched decode -> generated tokens share a right offset
    if TOK.pad_token_id is None:
        TOK.pad_token = TOK.eos_token
    MODEL = AutoModelForCausalLM.from_pretrained(
        ARGS.model, dtype=torch.bfloat16, device_map="cuda")
    MODEL.eval()

    CTRL, _ = install_sparse_mlps(MODEL)
    sample = MODEL.model.layers[0].mlp
    assert hasattr(sample.mlp, "gate_proj") and hasattr(sample.mlp, "act_fn"), \
        "MLP does not look like a SwiGLU gate/up/down FFN"
    if ARGS.sparsity > 0:
        AGENT_MASKER = build_masker(ARGS.method, ARGS.sparsity, MODEL.device)
        print(f"[tau2-server] agent masker ON: method={ARGS.method} "
              f"sparsity={ARGS.sparsity}; user-sim dense", flush=True)
    else:
        print("[tau2-server] agent masker OFF (dense baseline)", flush=True)

    threading.Thread(target=_worker, daemon=True).start()
    srv = ThreadingHTTPServer(("0.0.0.0", ARGS.port), Handler)
    print(f"[tau2-server] ready on :{ARGS.port} "
          f"(agent={ARGS.served_name}, user={ARGS.user_served_name}, "
          f"batch={ARGS.batch}, think={'on' if ARGS.think else 'off'})", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
