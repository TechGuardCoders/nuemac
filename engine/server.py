"""Nuemac: a single-stream local LLM server for memory-tight Apple Silicon Macs.

The engine (this file) is model-agnostic: chunked prefill that computes logits
for the last position only, pipelined decode, multi-turn cache reuse (snapshot
at the end of each prompt), and llama-server + OpenAI compatible HTTP APIs.
A model family (engine/families/<name>/) supplies the model-specific parts:
loading and patching, the KV-cache layout and kernels, stop tokens and the
output parser. One request runs at a time.

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

__version__ = "0.4.0"
log = logging.getLogger("nuemac")

class Engine:
    def __init__(self, model_path, ctx, prefill_chunk, options=None):
        info = mx.device_info()
        mx.set_wired_limit(int(info["max_recommended_working_set_size"]))
        mx.set_cache_limit(512 * 2**20)
        self.family = for_model(model_path)
        log.info("nuemac %s, family %s", __version__, self.family.NAME)
        self.model_path = model_path
        self.model, self.tok = self.family.load(model_path, options or {})
        # Compiled single-token decode, when the family has one (inkling2: fastpath.py).
        self.decoder = None
        if (options or {}).get("compiled_decode", True) and hasattr(self.family, "make_decoder"):
            self.decoder = self.family.make_decoder(self.model, options or {})
        self.ctx = ctx
        self.chunk = prefill_chunk
        self.cache = None
        self.tokens = []   # tokens whose K/V are in self.cache, in order
        self.snap = None   # (n_tokens, family state) at the end of the last prompt
        self.stop_ids = self.family.stop_ids(self.tok)
        log.info("loaded %s: %.2f GiB weights, ctx %d, stop ids %s", model_path,
                 mx.get_active_memory() / 2**30, ctx, sorted(self.stop_ids))

    # -- cache --------------------------------------------------------------
    def _new_cache(self):
        self.cache = None
        self.tokens = []
        self.snap = None
        mx.clear_cache()
        self.cache = self.family.make_cache(self.model, self.ctx)

    def _take_snapshot(self):
        self.snap = (len(self.tokens), self.family.snapshot(self.cache))

    def _restore_snapshot(self):
        n, st = self.snap
        self.family.restore(self.cache, st)
        self.tokens = self.tokens[:n]

    def _reuse(self, ids, allow):
        """Return how many leading prompt tokens are already in the cache."""
        if not allow or self.cache is None:
            self._new_cache()
            return 0
        n = len(self.tokens)
        if n and len(ids) > n and ids[:n] == self.tokens:
            return n
        if self.snap is not None:
            m = self.snap[0]
            if m and len(ids) > m and ids[:m] == self.tokens[:m]:
                self._restore_snapshot()
                return m
        self._new_cache()
        return 0

    # -- generation -----------------------------------------------------------
    def generate(self, ids, max_tokens, sampler, ignore_eos=False, cache_prompt=True,
                 on_token=None, should_stop=None):
        """Yields token ids. Fills stats dict on self.last_stats."""
        if len(ids) + max_tokens > self.ctx:
            max_tokens = self.ctx - len(ids)
            if max_tokens < 1:
                raise ContextFull(f"prompt is {len(ids)} tokens; context is {self.ctx}")
        reused = self._reuse(ids, cache_prompt)
        todo = ids[reused:]
        t0 = time.time()
        logits = None
        for s in range(0, len(todo), self.chunk):
            x = mx.array(todo[s : s + self.chunk])[None]
            h = self.model.model(x, self.cache)
            logits = self.model.logits_from_hidden(h[:, -1:, :])[:, -1, :]
            mx.eval(logits)
            del h
            mx.clear_cache()
            self.tokens.extend(todo[s : s + self.chunk])
            if should_stop and should_stop():
                break
        t_prompt = time.time() - t0
        self._take_snapshot()
        stats = {"prompt_n": len(todo), "cache_n": reused,
                 "prompt_ms": t_prompt * 1000,
                 "prompt_per_second": len(todo) / t_prompt if t_prompt > 0 else 0.0}
        out = []
        t1 = time.time()
        y = sampler(logits)
        mx.eval(y)
        finish = "length"
        dec = self.decoder
        if dec is not None:
            dec.begin(self.cache)
            step = lambda y: dec.step(y.reshape(1, 1))  # noqa: E731
        else:
            step = lambda y: self.model(y.reshape(1, 1), cache=self.cache)[:, -1, :]  # noqa: E731
        try:
            for i in range(max_tokens):
                # Queue the next step before waiting for this token, so the GPU never idles while Python builds
                # the next graph (waiting first cost ~2.8 ms per token on an M4 Pro). If this token turns out to
                # stop the reply, the queued step is wasted once; the cache then holds the token, so it is
                # recorded in self.tokens either way and prompt reuse stays exact.
                if i + 1 < max_tokens:
                    nxt = sampler(step(y))
                    mx.async_eval(nxt)
                else:
                    nxt = None
                tok = y.item()
                if nxt is not None:
                    self.tokens.append(tok)
                out.append(tok)
                if on_token is not None and on_token(tok) is False:
                    finish = "stop"
                    if nxt is not None:
                        mx.eval(nxt)
                    break
                if tok in self.stop_ids and not ignore_eos:
                    finish = "stop"
                    break
                if should_stop and should_stop():
                    finish = "stop"
                    if nxt is not None:
                        mx.eval(nxt)
                    break
                if nxt is None:
                    break
                y = nxt
        finally:
            if dec is not None:
                dec.end()
        t_gen = time.time() - t1
        n = len(out)
        stats.update({"predicted_n": n, "predicted_ms": t_gen * 1000,
                      "predicted_per_second": (n - 1) / t_gen if n > 1 and t_gen > 0 else 0.0,
                      "finish": finish})
        self.last_stats = stats
        mx.clear_cache()
        return out


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
    """Owns every MLX call: MLX streams belong to the thread that made them,
    so the model loads here and requests run here, one at a time."""

    def __init__(self, *engine_args):
        super().__init__(daemon=True, name="engine")
        self.engine_args = engine_args
        self.jobs = queue.Queue()
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
        while True:
            fn, box, done = self.jobs.get()
            try:
                box["result"] = fn()
            except BaseException as e:  # noqa: BLE001
                log.exception("request failed")
                box["error"] = e
            done.set()

    def call(self, fn):
        box, done = {}, threading.Event()
        self.jobs.put((fn, box, done))
        done.wait()
        if "error" in box:
            raise box["error"]
        return box["result"]


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
        out, st = WORKER.call(lambda: (
            ENGINE.generate(ids, max(1, n), sampler_from(body),
                            ignore_eos=bool(body.get("ignore_eos", False)),
                            cache_prompt=bool(body.get("cache_prompt", True))),
            ENGINE.last_stats))
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

        def emit(reasoning, content):
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
                send_chunk(d)

        def on_token(tok):
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
        try:
            # on_token runs on the engine thread; it only decodes and writes.
            st = WORKER.call(lambda: (
                ENGINE.generate(ids, max(1, max_tokens), sampler_from(body),
                                cache_prompt=bool(body.get("cache_prompt", True)),
                                on_token=on_token, should_stop=lambda: state["broken"]),
                ENGINE.last_stats)[1])
        except ContextFull as e:
            if not stream:
                raise
            send_chunk({"content": f"[error: {e}]"}, "length")
            return
        fin = parser.finalize()
        if fin.visible_text:
            emit(*router.feed(fin.visible_text))
        emit(*router.feed("", final=True))
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
                           "compiled_decode": not a.no_compiled_decode})
    WORKER.start()
    WORKER.ready.wait()
    if WORKER.error is not None:
        raise SystemExit(f"model failed to load: {WORKER.error}")
    srv = ThreadingHTTPServer((a.host, a.port), Handler)
    log.info("listening on http://%s:%d (model id %s)", a.host, a.port, MODEL_ID)
    srv.serve_forever()


if __name__ == "__main__":
    main()
