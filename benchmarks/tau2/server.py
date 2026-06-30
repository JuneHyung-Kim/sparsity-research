"""OpenAI-compatible /v1/chat/completions server for tau2-bench, with the
contextual activation-sparsity masker installed on the FFNs.

Unlike BFCL (which rendered the prompt itself and POSTed a raw `/v1/completions`
string), tau2-bench talks the OpenAI *chat + tools* contract through LiteLLM:
it sends `messages` + `tools` (function schemas) and reads back
`choices[0].message.tool_calls`. So this server must honour that contract:

  * apply Gemma 4's chat template with `tools=` (it renders the function schemas
    and OpenAI role:"tool" results natively into Gemma's tool DSL),
  * generate (with the masker in the FFN path),
  * parse Gemma's `<|tool_call>call:NAME{args}<tool_call|>` blocks back into
    OpenAI `tool_calls`, and strip the `<|channel>thought` reasoning channel so
    the multi-turn history stays lean.

tau2 needs TWO model roles: the **agent** (the policy we evaluate) and the
**user simulator** (part of the environment). The masker belongs only on the
agent. We therefore load ONE Gemma-4-12B and route by the request's `model` name:
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
from transformers import (AutoTokenizer, BitsAndBytesConfig,
                          Gemma4UnifiedForConditionalGeneration)

from src.actsparse import build_masker, install_sparse_mlps, get_decoder_layers

# ---- globals set in main() ----
MODEL = None
TOK = None
CTRL = None
ARGS = None
REQ_Q = None
AGENT_MASKER = None              # built once at startup (None when --sparsity == 0)
STOP_IDS = None                  # set in main(): [<eos>, <turn|>, <|tool_response>]
# Gemma 4 ends an agent turn with <turn|>; right after a tool call it instead emits
# <|tool_response> (waiting for the result). Either terminates the assistant's text.
STOPS = ("<turn|>", "<|tool_response>", "<eos>")
# Gemma 4 reasons inside a <|channel>thought ... <channel|> channel; strip it so the
# multi-turn history we hand back to tau2 carries only the answer/tool_calls.
CHANNEL_RE = re.compile(r"<\|channel>.*?<channel\|>", re.S)
# A tool call is <|tool_call>call:NAME{args}<tool_call|>; the captured group is
# NAME{...} where args is Gemma's mini-DSL (strings <|"|>..<|"|>, true/false,
# bare numbers, [..] arrays, {..} objects).
TOOLCALL_RE = re.compile(r"<\|tool_call>call:(.*?)<tool_call\|>", re.S)


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
    """OpenAI messages(+tools) -> a single prompt string via Gemma 4's chat
    template. The template renders tool schemas and OpenAI role:"tool" results
    natively; enable_thinking toggles the <|think|> / <|channel>thought reasoning
    channel at template time."""
    return TOK.apply_chat_template(
        req.messages,
        tools=req.tools or None,
        add_generation_prompt=True,
        tokenize=False,
        enable_thinking=ARGS.think,
    )


_QUOTE = '<|"|>'                                  # Gemma's string delimiter


def _parse_value(s, i):
    """Parse one Gemma-DSL value from s at index i -> (python value, next index).
    Strings are <|"|>..<|"|>, bools true/false, objects {k:v,..}, arrays [v,..],
    everything else a bare scalar (int/float, else raw string)."""
    n = len(s)
    if s.startswith(_QUOTE, i):
        j = s.find(_QUOTE, i + len(_QUOTE))
        if j == -1:
            return s[i + len(_QUOTE):], n
        return s[i + len(_QUOTE):j], j + len(_QUOTE)
    c = s[i]
    if c == '{':
        return _parse_obj(s, i)
    if c == '[':
        arr = []
        i += 1
        while i < n and s[i] != ']':
            v, i = _parse_value(s, i)
            arr.append(v)
            if i < n and s[i] == ',':
                i += 1
        return arr, (i + 1 if i < n else n)
    j = i
    while j < n and s[j] not in ',}]':
        j += 1
    tok = s[i:j].strip()
    if tok == 'true':
        v = True
    elif tok == 'false':
        v = False
    elif tok in ('null', 'none', ''):
        v = None
    else:
        try:
            v = int(tok)
        except ValueError:
            try:
                v = float(tok)
            except ValueError:
                v = tok
    return v, j


def _parse_obj(s, i):
    """s[i] == '{'; parse a {bareKey:value,..} object -> (dict, next index)."""
    obj = {}
    n = len(s)
    i += 1                                        # past '{'
    while i < n and s[i] != '}':
        k = i
        while i < n and s[i] != ':':
            i += 1
        key = s[k:i].strip()
        i += 1                                    # past ':'
        if i >= n:
            break
        val, i = _parse_value(s, i)
        obj[key] = val
        if i < n and s[i] == ',':
            i += 1
    return obj, (i + 1 if i < n else n)


def _parse_completion(text):
    """Gemma 4 completion -> (content, tool_calls) in OpenAI shape.
    tool_calls[i].function.arguments is a JSON *string* (litellm/tau2 json.loads
    it). content is the non-tool-call text, or None when the turn is pure calls."""
    text = CHANNEL_RE.sub("", text)               # drop the reasoning channel
    for stop in STOPS:
        j = text.find(stop)
        if j != -1:
            text = text[:j]
    tool_calls = []
    for m in TOOLCALL_RE.finditer(text):
        inner = m.group(1)                        # NAME{...}
        b = inner.find('{')
        name = (inner if b == -1 else inner[:b]).strip()
        try:
            args, _ = _parse_obj(inner[b:], 0) if b != -1 else ({}, 0)
        except Exception:
            args = {}
        tool_calls.append({
            "id": "call_" + uuid.uuid4().hex[:24],
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)},
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
        eos_token_id=STOP_IDS,                       # stop at turn / tool-response / eos
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
    global MODEL, TOK, CTRL, ARGS, REQ_Q, AGENT_MASKER, STOP_IDS
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-4-12B-it")
    ap.add_argument("--served-name", default="gemma-4-12b-agent",
                    help="model id for the AGENT role (masker applied)")
    ap.add_argument("--user-served-name", default="gemma-4-12b-user",
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
                    help="enable Gemma 4 reasoning channel (default: off for speed)")
    ap.add_argument("--load-4bit", action="store_true",
                    help="bitsandbytes nf4 load for a 24GB dev GPU; omit for bf16 (Vulcan)")
    ARGS = ap.parse_args()
    REQ_Q = queue.Queue()

    print(f"[tau2-server] loading {ARGS.model} "
          f"({'nf4' if ARGS.load_4bit else 'bf16'}) ...", flush=True)
    TOK = AutoTokenizer.from_pretrained(ARGS.model)
    TOK.padding_side = "left"            # batched decode -> generated tokens share a right offset
    if TOK.pad_token_id is None:
        TOK.pad_token = TOK.eos_token
    # Halt generation at the turn boundary (end-of-turn / tool-response / eos)
    # instead of always running to --max-new.
    STOP_IDS = [i for i in (TOK.eos_token_id,
                            TOK.convert_tokens_to_ids("<turn|>"),
                            TOK.convert_tokens_to_ids("<|tool_response>"))
                if i is not None and i >= 0]
    load_kwargs = dict(dtype=torch.bfloat16, device_map="auto")
    if ARGS.load_4bit:
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16)
    MODEL = Gemma4UnifiedForConditionalGeneration.from_pretrained(ARGS.model, **load_kwargs)
    MODEL.eval()

    CTRL, _ = install_sparse_mlps(MODEL)
    sample = get_decoder_layers(MODEL)[0].mlp        # SparseMLP after install
    assert hasattr(sample.mlp, "gate_proj") and hasattr(sample.mlp, "act_fn"), \
        "MLP does not look like a gate/up/down (Sw/Ge)GLU FFN"
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
