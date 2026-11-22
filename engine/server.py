"""Nuemac: a local LLM server for memory-tight Apple Silicon Macs, with continuous batching.

The engine (this file) is model-agnostic: chunked prefill that computes logits
for the last position only, pipelined decode, multi-turn cache reuse (snapshot
at the end of each prompt), and llama-server + OpenAI compatible HTTP APIs.
A model family (engine/families/<name>/) supplies the model-specific parts:
loading and patching, the KV-cache layout and kernels, stop tokens and the
output parser. Requests run concurrently: they join and leave one decode batch between steps (Engine).

API: /health, /completion (llama-server style, with "timings"), /v1/models and
/v1/chat/completions (streaming, reasoning_content, tool calls).
"""
import argparse
import json
import logging
import os
import queue
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mlx.core as mx  # noqa: E402
from mlx_lm.sample_utils import make_sampler  # noqa: E402

from families import ContextFull, for_model  # noqa: E402

__version__ = "0.5.0"
log = logging.getLogger("nuemac")

BANDS = (4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288)


def band_for(n, ctx):
    """The cache length a sequence gets: the first band holding n keys, capped at ctx (few distinct shapes, so the
    compiled decode step compiles a handful of times)."""
    for b in BANDS:
        if n <= b:
            return min(b, ctx)
    return ctx


class Request:
    """One generation: filled by an HTTP thread, run by the engine thread, read back through `events`."""

    def __init__(self, ids, max_tokens, sampler, ignore_eos=False, cache_prompt=True, on_token=None):
        self.ids, self.max_tokens, self.sampler = list(ids), max_tokens, sampler
        self.ignore_eos, self.cache_prompt, self.on_token = ignore_eos, cache_prompt, on_token
        self.events = queue.Queue()      # ("done", out, stats) or ("error", exc)
        self.cancelled = False
        # engine-side state
        self.cache = None
        self.tokens = []                 # tokens whose K/V are in self.cache
        self.snap = None                 # (n_tokens, family state) at the end of the prompt
        self.todo = []                   # prompt tokens still to prefill
        self.reused = 0
        self.y = None                    # next token to emit (lazy)
        self.out = []
        self.max_len = 0
        self.t1 = 0.0
        self.t_prompt = 0.0

    def cancel(self):
        self.cancelled = True


class Engine:
    """Continuous batching: requests join and leave a running decode batch between steps.
    - Each sequence has its own caches; a batched step runs the projections, experts and head for all rows at once
      and each row's attention against its own cache (family.batch_cache). One sequence alone uses the compiled
      decode step.
    - Prompts prefill in chunks between decode steps, taking turns so decoding users keep getting tokens.
    - Admission: at most `slots` sequences, and their caches (sized to prompt + max_tokens, in BANDS) must fit
      the memory budget. Finished sequences' caches stay for prompt reuse while memory allows (`keep`)."""

    def __init__(self, model_path, ctx, prefill_chunk, options=None):
        options = options or {}
        info = mx.device_info()
        self.limit = int(info["max_recommended_working_set_size"])
        mx.set_wired_limit(self.limit)
        mx.set_cache_limit(512 * 2**20)
        self.family = for_model(model_path)
        log.info("nuemac %s, family %s", __version__, self.family.NAME)
        self.model_path = model_path
        self.model, self.tok = self.family.load(model_path, options)
        self.decoder = None
        if options.get("compiled_decode", True) and hasattr(self.family, "make_decoder"):
            self.decoder = self.family.make_decoder(self.model, options)
        self.dec_seq = None              # the sequence the compiled decoder holds
        self.ctx = ctx
        self.chunk = prefill_chunk
        self.slots = max(1, int(options.get("slots", 4)))
        if not hasattr(self.family, "batch_cache"):
            self.slots = 1
        self.keep = max(0, int(options.get("keep_caches", 2)))
        weights = mx.get_active_memory()
        self.budget = self.limit - weights - int(float(options.get("reserve_gb", 1.5)) * 2**30)
        self.stop_ids = self.family.stop_ids(self.tok)
        self.inbox = queue.Queue()
        self.waiting, self.prefilling, self.decoding = [], [], []
        self.idle = []                   # finished sequences kept for prompt reuse, oldest first
        self.decode_credit = 0.0         # decode time owed before the next prefill chunk
        log.info("loaded %s: %.2f GiB weights, ctx %d, %d slots, cache budget %.1f GiB, stop ids %s", model_path,
                 weights / 2**30, ctx, self.slots, self.budget / 2**30, sorted(self.stop_ids))

    # -- memory -------------------------------------------------------------------------------------------------
    def _bytes(self, max_len):
        if hasattr(self.family, "cache_bytes"):
            return self.family.cache_bytes(self.model, max_len)
        return 0

    def _in_use(self):
        return sum(self._bytes(r.max_len) for r in self.prefilling + self.decoding + self.idle)

    # -- admission ----------------------------------------------------------------------------------------------
    @staticmethod
    def _reuse_len(req, r):
        """(tokens reusable from idle sequence r, whether its prompt snapshot must be restored)."""
        n = len(r.tokens)
        if n and len(req.ids) > n and req.ids[:n] == r.tokens:
            return n, False
        if r.snap is not None:
            m = r.snap[0]
            if m and len(req.ids) > m and req.ids[:m] == r.tokens[:m]:
                return m, True
        return 0, False

    def _peek_reusable(self, req):
        best, best_n = None, 0
        for r in self.idle:
            n, _ = self._reuse_len(req, r)
            if n > best_n:
                best, best_n = r, n
        return best

    def _claim(self, req, r):
        """Move idle sequence r's cache into req. Returns the reused token count."""
        n, use_snap = self._reuse_len(req, r)
        self.idle.remove(r)
        req.cache, req.tokens, req.snap, req.max_len = r.cache, r.tokens, r.snap, r.max_len
        r.cache = None
        if use_snap:
            self.family.restore(req.cache, req.snap[1])
            req.tokens = req.tokens[:n]
        return n

    def _admit(self):
        while self.waiting and len(self.prefilling) + len(self.decoding) < self.slots:
            req = self.waiting[0]
            if req.cancelled:
                self.waiting.pop(0)
                req.events.put(("done", [], self._stats(req, "stop")))
                continue
            if len(req.ids) + req.max_tokens > self.ctx:
                req.max_tokens = self.ctx - len(req.ids)
                if req.max_tokens < 1:
                    self.waiting.pop(0)
                    req.events.put(("error", ContextFull(f"prompt is {len(req.ids)} tokens; context is {self.ctx}")))
                    continue
            need = band_for(len(req.ids) + req.max_tokens, self.ctx)
            # memory first: a new cache (or a reused one grown to `need`) must fit beside the running ones
            cand = self._peek_reusable(req) if req.cache_prompt else None
            new_bytes = self._bytes(need) - (self._bytes(cand.max_len) if cand is not None else 0)
            while self._in_use() + new_bytes > self.budget and [r for r in self.idle if r is not cand]:
                self.idle.remove([r for r in self.idle if r is not cand][0])
                mx.clear_cache()
            if self._in_use() + new_bytes > self.budget and (self.prefilling or self.decoding):
                return                              # wait for a running sequence to finish
            self.waiting.pop(0)
            reused = self._claim(req, cand) if cand is not None else 0
            if req.cache is not None and req.max_len < need:
                self.family.fit_cache(req.cache, need)
                req.max_len = need
            if req.cache is None:
                req.cache = self.family.make_cache(self.model, need)
                req.tokens, req.snap, req.max_len = [], None, need
            req.reused = reused
            req.todo = req.ids[reused:]
            self.prefilling.append(req)

    # -- prefill --------------------------------------------------------------------------------------------------
    def _prefill_chunk(self, req):
        t = time.time()
        part = req.todo[: self.chunk]
        h = self.model.model(mx.array(part)[None], req.cache)
        logits = self.model.logits_from_hidden(h[:, -1:, :])[:, -1, :]
        mx.eval(logits)
        del h
        mx.clear_cache()                      # prefill workspaces are large; don't let the pool keep them
        req.tokens.extend(part)
        req.todo = req.todo[len(part):]
        dt = time.time() - t
        req.t_prompt += dt
        self.decode_credit = dt               # decoding sequences get as much time before the next chunk
        if req.cancelled:
            self.prefilling.remove(req)
            self._finish(req, "stop")
            return
        if not req.todo:
            req.snap = (len(req.tokens), self.family.snapshot(req.cache))
            req.y = req.sampler(logits)
            mx.eval(req.y)
            req.t1 = time.time()
            self.prefilling.remove(req)
            self.decoding.append(req)
            mx.clear_cache()

    # -- decode ---------------------------------------------------------------------------------------------------
    def _use_decoder(self, req):
        """Hand the compiled decoder to req (or to nobody)."""
        if self.decoder is None or self.dec_seq is req:
            return
        if self.dec_seq is not None:
            self.decoder.end()
            self.dec_seq = None
        if req is not None:
            self.decoder.begin(req.cache)
            self.dec_seq = req

    def _decode_step(self):
        t = time.time()
        # a sequence about to emit its last allowed token is not stepped (its cache has no room past it)
        last = [r for r in self.decoding if len(r.out) + 1 >= r.max_tokens]
        step = [r for r in self.decoding if r not in last]
        nxts = []
        if step:
            if len(step) == 1 and self.decoder is not None:
                self._use_decoder(step[0])
                logits = self.decoder.step(step[0].y.reshape(1, 1))
            else:
                self._use_decoder(None)
                ys = mx.concatenate([r.y.reshape(1, 1) for r in step], axis=0)
                cache = step[0].cache if len(step) == 1 else self.family.batch_cache([r.cache for r in step])
                logits = self.model(ys, cache=cache)[:, -1, :]
            nxts = [r.sampler(logits[i:i + 1]) for i, r in enumerate(step)]
            mx.async_eval(nxts)               # queue the next step before reading this one's tokens
        for r, nxt in zip(step, nxts):
            tok = r.y.item()
            r.tokens.append(tok)              # the queued step fed it: the cache holds it
            r.out.append(tok)
            stop = (tok in self.stop_ids and not r.ignore_eos) or r.cancelled
            if not stop and r.on_token is not None and r.on_token(tok) is False:
                stop = True
            if stop:
                self.decoding.remove(r)
                self._finish(r, "stop")
            else:
                r.y = nxt
        for r in last:
            tok = r.y.item()
            r.out.append(tok)
            if r.on_token is not None:
                r.on_token(tok)
            self.decoding.remove(r)
            self._finish(r, "stop" if tok in self.stop_ids and not r.ignore_eos else "length")
        self.decode_credit -= time.time() - t

    # -- finish ---------------------------------------------------------------------------------------------------
    def _stats(self, req, finish):
        n = len(req.out)
        t_gen = time.time() - req.t1 if req.t1 else 0.0
        return {"prompt_n": len(req.ids) - req.reused, "cache_n": req.reused, "prompt_ms": req.t_prompt * 1000,
                "prompt_per_second": (len(req.ids) - req.reused) / req.t_prompt if req.t_prompt > 0 else 0.0,
                "predicted_n": n, "predicted_ms": t_gen * 1000,
                "predicted_per_second": (n - 1) / t_gen if n > 1 and t_gen > 0 else 0.0, "finish": finish}

    def _finish(self, req, finish):
        if self.dec_seq is req:
            self._use_decoder(None)
        req.on_token = None
        if req.cache_prompt and req.cache is not None and self.keep:
            self.idle.append(req)
            while len(self.idle) > self.keep:
                self.idle.pop(0)
        else:
            req.cache = None
        req.events.put(("done", req.out, self._stats(req, finish)))
        mx.clear_cache()

    # -- the loop -------------------------------------------------------------------------------------------------
    def submit(self, req):
        self.inbox.put(req)

    def run_forever(self):
        while True:
            busy = bool(self.prefilling or self.decoding)
            try:
                if not busy:
                    self.waiting.append(self.inbox.get(timeout=None if not self.waiting else 0.05))
                while True:
                    self.waiting.append(self.inbox.get_nowait())
            except queue.Empty:
                pass
            try:
                self._admit()
                if self.prefilling and (not self.decoding or self.decode_credit <= 0):
                    self._prefill_chunk(self.prefilling[0])
                elif self.decoding:
                    self._decode_step()
            except Exception as e:  # noqa: BLE001  fail the requests in flight, keep serving
                log.exception("engine step failed")
                if self.dec_seq is not None:
                    try:
                        self.decoder.end()
                    except Exception:  # noqa: BLE001
                        pass
                    self.dec_seq = None
                for r in self.prefilling + self.decoding:
                    r.cache = None
                    r.events.put(("error", e))
                self.prefilling, self.decoding = [], []
                mx.clear_cache()


def sampler_from(body):
    temp = float(body.get("temperature", 0.0) if body.get("temperature") is not None else 0.0)
    return make_sampler(temp=temp, top_p=float(body.get("top_p", 1.0) or 1.0),
                        min_p=float(body.get("min_p", 0.0) or 0.0),
                        top_k=int(body.get("top_k", 0) or 0))


class ThinkRouter:
    """Splits parser output with <think>...</think> into reasoning and content."""

    def __init__(self):
        self.in_think = False
        self.buf = ""

    def feed(self, text, final=False):
        self.buf += text
        reasoning, content = "", ""
        while self.buf:
            marker = "</think>" if self.in_think else "<think>"
            i = self.buf.find(marker)
            if i >= 0:
                piece, self.buf = self.buf[:i], self.buf[i + len(marker):]
                if self.in_think:
                    reasoning += piece
                else:
                    content += piece
                self.in_think = not self.in_think
                continue
            keep = 0
            if not final:
                for k in range(min(len(marker) - 1, len(self.buf)), 0, -1):
                    if marker.startswith(self.buf[-k:]):
                        keep = k
                        break
            piece = self.buf[: len(self.buf) - keep]
            self.buf = self.buf[len(self.buf) - keep:]
            if self.in_think:
                reasoning += piece
            else:
                content += piece
            break
        return reasoning, content


ENGINE = None
WORKER = None
MODEL_ID = "local"


class EngineThread(threading.Thread):
    """Owns every MLX call: MLX streams belong to the thread that made them, so the model loads here and the
    scheduler runs here. HTTP threads only submit requests and read their events."""

    def __init__(self, *engine_args):
        super().__init__(daemon=True, name="engine")
        self.engine_args = engine_args
        self.ready = threading.Event()
        self.error = None

    def run(self):
        global ENGINE
        try:
            ENGINE = Engine(*self.engine_args)
        except BaseException as e:  # noqa: BLE001
            self.error = e
            self.ready.set()
            return
        self.ready.set()
        ENGINE.run_forever()

    def generate(self, req):
        """Submit and wait. Returns (tokens, stats)."""
        ENGINE.submit(req)
        kind, *rest = req.events.get()
        if kind == "error":
            raise rest[0]
        return rest[0], rest[1]


def timings(stats):
    return {k: stats[k] for k in ("prompt_n", "prompt_ms", "prompt_per_second",
                                  "predicted_n", "predicted_ms", "predicted_per_second",
                                  "cache_n")}


def canonical_messages(msgs):
    """OpenAI clients send an assistant tool call's arguments as a JSON string; chat templates such as
    Inkling's want the parsed object (and raise otherwise). Parse them; text that isn't JSON is kept under "raw"."""
    out = []
    for m in msgs:
        calls = m.get("tool_calls") if isinstance(m, dict) else None
        if calls:
            m = dict(m)
            fixed = []
            for c in calls:
                fn = (c or {}).get("function") or {}
                args = fn.get("arguments")
                if isinstance(args, str):
                    try:
                        args = json.loads(args) if args.strip() else {}
                    except ValueError:
                        args = {"raw": args}
                    if not isinstance(args, dict):
                        args = {"value": args}
                    c = dict(c, function=dict(fn, arguments=args))
                fixed.append(c)
            m["tool_calls"] = fixed
        out.append(m)
    return out


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "nuemac/" + __version__

    def log_message(self, fmt, *args):
        log.info("%s %s", self.address_string(), fmt % args)

    def _json(self, code, obj):
        self._started = True
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(n) or b"{}")

    def do_GET(self):
        if self.path.rstrip("/") in ("/health", "/v1/health"):
            return self._json(200, {"status": "ok"})
        if self.path.rstrip("/") in ("/v1/models", "/models"):
            return self._json(200, {"object": "list", "data": [
                {"id": MODEL_ID, "object": "model", "owned_by": "local",
                 "meta": {"n_ctx": ENGINE.ctx}}]})
        return self._json(404, {"error": {"message": "not found"}})

    def do_POST(self):
        try:
            body = self._body()
        except Exception as e:  # noqa: BLE001
            return self._json(400, {"error": {"message": f"bad JSON: {e}"}})
        path = self.path.rstrip("/")
        try:
            if path in ("/completion", "/completions"):
                return self.completion(body)
            if path in ("/v1/chat/completions", "/chat/completions"):
                return self.chat(body)
        except ContextFull as e:
            return self._json(400, {"error": {"message": str(e), "type": "exceed_context_size_error"}})
        except Exception as e:  # noqa: BLE001  answer with an error instead of dropping the connection
            log.exception("request failed")
            if not getattr(self, "_started", False):
                return self._json(500, {"error": {"message": f"{type(e).__name__}: {e}", "type": "server_error"}})
            return None
        return self._json(404, {"error": {"message": "not found"}})

    # llama-server style raw completion (used by the bench)
    def completion(self, body):
        prompt = body.get("prompt", "")
        ids = ENGINE.tok.encode(prompt) if isinstance(prompt, str) else list(prompt)
        n = int(body.get("n_predict", body.get("max_tokens", 128)))
        out, st = WORKER.generate(Request(ids, max(1, n), sampler_from(body),
                                          ignore_eos=bool(body.get("ignore_eos", False)),
                                          cache_prompt=bool(body.get("cache_prompt", True))))
        text = ENGINE.tok.decode(out)
        return self._json(200, {"content": text, "tokens_predicted": len(out),
                                "tokens_evaluated": len(ids), "model": MODEL_ID,
                                "stop": True, "timings": timings(st)})

    def _prompt_ids(self, body):
        kwargs = dict(body.get("chat_template_kwargs") or {})
        if body.get("reasoning_effort") and "reasoning_effort" not in kwargs:
            kwargs["reasoning_effort"] = body["reasoning_effort"]
        msgs = canonical_messages(body.get("messages") or [])
        tools = body.get("tools") or None
        text = ENGINE.tok.apply_chat_template(msgs, tools=tools, add_generation_prompt=True,
                                              tokenize=False, **kwargs)
        return ENGINE.tok.encode(text, add_special_tokens=False)

    def chat(self, body):
        ids = self._prompt_ids(body)
        max_tokens = int(body.get("max_tokens") or body.get("max_completion_tokens") or 4096)
        stream = bool(body.get("stream", False))
        parser = ENGINE.family.output_parser(ENGINE.tok, ENGINE.model_path)
        router = ThinkRouter()
        cid = "chatcmpl-" + uuid.uuid4().hex[:24]
        created = int(time.time())
        parts = {"reasoning": [], "content": []}
        state = {"broken": False}

        def send_chunk(delta, finish=None):
            obj = {"id": cid, "object": "chat.completion.chunk", "created": created,
                   "model": MODEL_ID,
                   "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
            try:
                self.wfile.write(f"data: {json.dumps(obj)}\n\n".encode())
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                state["broken"] = True

        req = None

        def emit(reasoning, content, direct=False):
            if reasoning:
                parts["reasoning"].append(reasoning)
            if content:
                parts["content"].append(content)
            if stream and (reasoning or content):
                d = {}
                if reasoning:
                    d["reasoning_content"] = reasoning
                if content:
                    d["content"] = content
                if direct:
                    send_chunk(d)
                else:
                    req.events.put(("delta", d))     # engine thread: queue it, the HTTP thread writes

        def on_token(tok):                          # runs on the engine thread
            r = parser.process_token(tok)
            if r.visible_text:
                emit(*router.feed(r.visible_text))
            if r.is_stop or state["broken"]:
                return False
            return True

        if stream:
            self._started = True
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True
            send_chunk({"role": "assistant", "content": ""})
        req = Request(ids, max(1, max_tokens), sampler_from(body),
                      cache_prompt=bool(body.get("cache_prompt", True)), on_token=on_token)
        ENGINE.submit(req)
        st = None
        while st is None:
            kind, *rest = req.events.get()
            if kind == "delta":
                if not state["broken"]:
                    send_chunk(rest[0])
                    if state["broken"]:
                        req.cancel()                # client went away: stop at the next step
            elif kind == "done":
                st = rest[1]
            else:
                e = rest[0]
                if not stream:
                    raise e
                send_chunk({"content": f"[error: {e}]"}, "length")
                return
        fin = parser.finalize()
        if fin.visible_text:
            emit(*router.feed(fin.visible_text), direct=True)
        emit(*router.feed("", final=True), direct=True)
        tool_calls = [{"id": "call_" + uuid.uuid4().hex[:24], "type": "function",
                       "function": {"name": t["name"], "arguments": t["arguments"]}}
                      for t in fin.tool_calls]
        finish = "tool_calls" if tool_calls else ("length" if st["finish"] == "length" else "stop")
        usage = {"prompt_tokens": len(ids), "completion_tokens": st["predicted_n"],
                 "total_tokens": len(ids) + st["predicted_n"],
                 "prompt_tokens_details": {"cached_tokens": st["cache_n"]}}
        if stream:
            if tool_calls:
                send_chunk({"tool_calls": [dict(tc, index=i) for i, tc in enumerate(tool_calls)]})
            obj = {"id": cid, "object": "chat.completion.chunk", "created": created,
                   "model": MODEL_ID, "choices": [{"index": 0, "delta": {}, "finish_reason": finish}],
                   "usage": usage, "timings": timings(st)}
            try:
                self.wfile.write(f"data: {json.dumps(obj)}\n\ndata: [DONE]\n\n".encode())
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            return
        msg = {"role": "assistant", "content": "".join(parts["content"]).strip()}
        reasoning = "".join(parts["reasoning"]).strip()
        if reasoning:
            msg["reasoning_content"] = reasoning
        if tool_calls:
            msg["tool_calls"] = tool_calls
        return self._json(200, {"id": cid, "object": "chat.completion", "created": created,
                                "model": MODEL_ID,
                                "choices": [{"index": 0, "message": msg, "finish_reason": finish}],
                                "usage": usage, "timings": timings(st)})


def main():
    global ENGINE, MODEL_ID
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True, help="MLX model directory")
    ap.add_argument("--alias", default=MODEL_ID)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--ctx", type=int, default=98304)
    ap.add_argument("--prefill-chunk", type=int, default=256)
    ap.add_argument("--expect-omlx", default="", help="oMLX version the model's recipe was tested with")
    ap.add_argument("--slots", type=int, default=4,
                    help="most sequences decoding at once (continuous batching); 1 = one request at a time")
    ap.add_argument("--keep-caches", type=int, default=2,
                    help="finished conversations whose caches stay for prompt reuse (memory permitting)")
    ap.add_argument("--no-compiled-decode", action="store_true",
                    help="decode eagerly (one MLX graph built per token) instead of the compiled decode step")
    ap.add_argument("--attn-bits", type=int, default=8, choices=(4, 5, 6, 8),
                    help="inkling2: requantize the 8-bit attention projections")
    ap.add_argument("--head-bits", type=int, default=6, choices=(4, 5, 6, 8),
                    help="inkling2: requantize the 8-bit output head (6: near-lossless, faster)")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    global WORKER
    MODEL_ID = a.alias
    WORKER = EngineThread(a.model, a.ctx, a.prefill_chunk,
                          {"expect_omlx": a.expect_omlx, "head_bits": a.head_bits, "attn_bits": a.attn_bits,
                           "compiled_decode": not a.no_compiled_decode, "slots": a.slots,
                           "keep_caches": a.keep_caches})
    WORKER.start()
    WORKER.ready.wait()
    if WORKER.error is not None:
        raise SystemExit(f"model failed to load: {WORKER.error}")
    srv = ThreadingHTTPServer((a.host, a.port), Handler)
    log.info("listening on http://%s:%d (model id %s)", a.host, a.port, MODEL_ID)
    srv.serve_forever()


if __name__ == "__main__":
    main()
