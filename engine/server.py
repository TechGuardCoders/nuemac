"""Nuemac: a local LLM server for memory-tight Apple Silicon Macs, with continuous batching.

The engine (this file) is model-agnostic: chunked prefill that computes logits
for the last position only, pipelined decode, multi-turn cache reuse (snapshot
at the end of each prompt), and llama-server + OpenAI compatible HTTP APIs.
A model family (engine/families/<name>/) supplies the model-specific parts:
loading and patching, the KV-cache layout and kernels, stop tokens and the
output parser. Requests run concurrently: they join and leave one decode batch between steps (Engine).

API: /health, /completion (llama-server style, with "timings"), /v1/models and
/v1/chat/completions (streaming, reasoning_content, tool calls, images when the build has a vision tower).
"""
import argparse
import base64
import hashlib
import json
import logging
import os
import queue
import sys
import threading
import time
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# MLX ends a Metal command buffer after 40 ops (or 40 MB of bound arrays) on M4 / M4 Pro. A decode step is thousands of
# small kernels, so the default commits dozens of buffers per token. 200 ops measured best on an M4 Pro: decode +2%,
# 4 requests at once +5%; 1000 ops was 4% slower. The MB cap stays at MLX's default: it also bounds how much temporary
# memory one buffer holds, and lifting it made prompt reading slower and ran a 16 GB Mac out of GPU memory.
# Set before MLX initializes the device; an explicit environment value still wins.
os.environ.setdefault("MLX_MAX_OPS_PER_BUFFER", "200")

import mlx.core as mx  # noqa: E402
from mlx_lm.sample_utils import make_sampler  # noqa: E402

from families import ContextFull, for_model  # noqa: E402

__version__ = "0.7.3"
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

    def __init__(self, ids, max_tokens, sampler, ignore_eos=False, cache_prompt=True, on_token=None, images=None):
        self.ids, self.max_tokens, self.sampler = list(ids), max_tokens, sampler
        self.ignore_eos, self.cache_prompt, self.on_token = ignore_eos, cache_prompt, on_token
        self.images = images or {}       # image id (negative token stand-in) -> PromptImage
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
        self.want = 0                    # prompt + max_tokens: the most keys this reply can need
        self.paused = False              # waiting for memory to grow its cache
        self.t1 = 0.0
        self.t_prompt = 0.0
        self.drafter = None              # CopyDrafter while this request decodes alone

    def cancel(self):
        self.cancelled = True


class PromptImage:
    """One image in a prompt. Its image tokens are written in the request's ids as one negative stand-in id (made from
    the image's bytes and its place among the prompt's images), so prompt reuse can only match the same image at the
    same place; prefill swaps the stand-ins for the image token and its embeddings for the image's features."""

    def __init__(self, pixels, grid, start):
        self.pixels, self.grid, self.start = pixels, grid, start
        self.count = grid[1] * grid[2]
        self.feats = None


def image_stand_in(digest, k):
    return -1 - (int(digest[:15], 16) ^ (k * 0x9E3779B1)) % (1 << 60)


class CopyDrafter:
    """Drafts from the context: when the reply is repeating an earlier stretch of the prompt or reply (a rewritten
    file, an edit, quoted code), propose the tokens that followed that stretch. The lookup uses only tokens already
    known, so a step that finds nothing costs no GPU sync; a draft is checked in one forward pass.
    - Candidates: up to CANDS earlier positions of the last NGRAM tokens; the one with the longest run of equal tokens
      before it wins, and drafting needs at least MIN_RUN equal tokens (verbatim copying, not a common phrase).
    - The pending next token is checked along with the draft (the draft starts with the token after it)."""

    NGRAM = 4
    CANDS = 8
    MIN_RUN = 12              # equal tokens before the match point needed to draft (8 cost code 5%: common lines)
    MIN, START, MAX = 3, 4, 15
    COOLDOWN = 8              # normal tokens after a draft whose first token missed
    SCAN = 64                 # how far back a candidate's run is compared

    def __init__(self):
        self.seq = []
        self.index = {}       # n-gram -> recent positions just past it, newest last
        self.k = self.START
        self.wait = 0
        self.stats = [0, 0, 0]   # checks, drafted, accepted
        self.secs = [0.0, 0.0, 0.0, 0]   # switch out of the compiled decoder, check pass, re-run, re-runs

    def extend(self, toks):
        n, seq, idx = self.NGRAM, self.seq, self.index
        for t in toks:
            seq.append(t)
            if len(seq) > n:
                key = tuple(seq[-n - 1:-1])
                lst = idx.get(key)
                if lst is None:
                    idx[key] = [len(seq) - 1]
                else:
                    lst.append(len(seq) - 1)
                    if len(lst) > self.CANDS:
                        del lst[0]

    def _run(self, p):
        """Equal tokens immediately before position p and before the end of seq."""
        seq, n, i = self.seq, len(self.seq), 0
        while i < self.SCAN and p - 1 - i >= 0 and seq[p - 1 - i] == seq[n - 1 - i] and p - 1 - i < n - 1 - i:
            i += 1
        return i

    def propose(self, limit):
        """Draft for the tokens after the pending one: [guess of the pending token's successor, ...]."""
        if self.wait > 0:
            self.wait -= 1
            return []
        if len(self.seq) < self.MIN_RUN:
            return []
        cands = self.index.get(tuple(self.seq[-self.NGRAM:]))
        if not cands:
            return []
        best, run = None, 0
        for p in cands:
            if p >= len(self.seq):
                continue
            r = self._run(p)
            if r > run:
                best, run = p, r
        if best is None or run < self.MIN_RUN:
            return []
        k = min(self.k, limit)
        d = self.seq[best + 1:best + 1 + k]      # seq[best] is the guess for the pending token, checked as is
        if any(t < 0 for t in d):                # never draft into an image's stand-in tokens
            d = d[:next(i for i, t in enumerate(d) if t < 0)]
        return d if len(d) >= 2 and self.seq[best] >= 0 else []

    def update(self, drafted, accepted):
        self.stats[0] += 1
        self.stats[1] += drafted
        self.stats[2] += accepted
        if accepted == drafted:
            self.k = min(self.MAX, self.k * 2 + 1)
        else:
            self.k = max(self.MIN, accepted + 2)
            if accepted == 0:
                self.wait = self.COOLDOWN


class Engine:
    """Continuous batching: requests join and leave a running decode batch between steps.
    - Each sequence has its own caches; a batched step runs the projections, experts and head for all rows at once
      and each row's attention against its own cache (family.batch_cache). One sequence alone uses the compiled
      decode step.
    - Prompts prefill in chunks between decode steps, taking turns so decoding users keep getting tokens.
    - Admission: at most `slots` sequences, and their caches must fit the memory budget. A cache starts at prompt +
      min(max_tokens, reply_reserve) (in BANDS) and grows a band at a time while the reply runs on, so a request
      without a max_tokens (no cap: up to the context) doesn't hold memory other users could start with. Finished sequences' caches stay for prompt reuse while memory allows (`keep`)."""

    def __init__(self, model_path, ctx, prefill_chunk, options=None):
        options = options or {}
        info = mx.device_info()
        self.limit = int(info["max_recommended_working_set_size"])
        small = self.limit < 20 * 2**30
        mx.set_wired_limit(self.limit)
        # Freed buffers MLX keeps for reuse stay in memory (wired) and aren't in the peak figures. On a 16 GB Mac
        # 512 MB of them squeezed macOS into swap; 128 MB keeps the reuse that matters.
        mx.set_cache_limit((128 if small else 512) * 2**20)
        self.family = for_model(model_path)
        log.info("nuemac %s, family %s", __version__, self.family.NAME)
        self.model_path = model_path
        self.model, self.tok = self.family.load(model_path, options)
        self.decoder = None
        if options.get("compiled_decode", True) and hasattr(self.family, "make_decoder"):
            self.decoder = self.family.make_decoder(self.model, options)
        self.dec_seq = None              # the sequence the compiled decoder holds
        self.ctx = ctx
        self.chunk = prefill_chunk           # while other sequences decode: they wait ~one chunk between tokens
        # Nothing decoding: no one waits on a chunk, so a bigger one can read the prompt faster. Measured with
        # Inkling 2 Mini oQ3e on a 24 GB M4 Pro, 256 -> 1024 tokens: +6-9% prompt speed (1.6K-18K tokens), same
        # accuracy, +1.1-1.7 GiB peak memory (2048: no faster, +3 GiB). On a 16 GB M4 it was no faster and a 94K
        # prompt ran out of GPU memory, so "auto" (<= 0) keeps 256 below a 20 GiB GPU limit.
        alone = int(options.get("prefill_chunk_alone", 0))
        if alone <= 0:
            alone = 1024 if self.limit >= 20 * 2**30 else prefill_chunk
        self.chunk_alone = max(prefill_chunk, alone)
        self.family_hint("prefill_chunk", self.chunk_alone)
        self.slots = max(1, int(options.get("slots", 4)))
        if not hasattr(self.family, "batch_cache"):
            self.slots = 1
        self.keep = max(0, int(options.get("keep_caches", 2)))
        self.reply_reserve = max(256, int(options.get("reply_reserve", 8192)))
        self.draft = options.get("draft", "copy") == "copy" and not os.environ.get("NUEMAC_NO_DRAFT")
        weights = mx.get_active_memory()
        # Plan for what macOS needs too, not just the GPU limit: GPU memory is wired, so whatever the engine takes
        # macOS can't page out, and on a 16 GB Mac (limit 14.5 GB) it was left ~1.5 GB: logged-in apps ran from
        # swap and prompts stalled for a minute at a time. The engine now plans total use <= RAM - headroom (3 GiB
        # by default; on 24 GB+ Macs that is already the GPU limit, so nothing changes there).
        mem = int(info.get("memory_size", 0)) or self.limit
        headroom = float(options.get("headroom_gb", -1))
        if headroom < 0:
            headroom = 3.0
        plan = min(self.limit, mem - int(headroom * 2**30))
        # what prefill workspaces, a lazily loaded vision tower and MLX's buffer pool need on top of the caches
        reserve = float(options.get("reserve_gb", -1))
        if reserve < 0:
            reserve = 0.8 if small else 1.5
        self.budget = plan - weights - int(reserve * 2**30)
        full = self._bytes(band_for(ctx, ctx)) if hasattr(self, "family") else 0
        if self.budget < full:                  # one full-context conversation must always fit
            log.warning("memory plan leaves %.2f GiB for caches, a full %d-token context needs %.2f GiB: using that "
                        "(macOS gets less headroom)", self.budget / 2**30, ctx, full / 2**30)
            self.budget = full
        self.stop_ids = self.family.stop_ids(self.tok)
        self.inbox = queue.Queue()
        self.waiting, self.prefilling, self.decoding = [], [], []
        self.idle = []                   # finished sequences kept for prompt reuse, oldest first
        self.decode_credit = 0.0         # decode time owed before the next prefill chunk
        log.info("loaded %s: %.2f GiB weights, ctx %d, %d slots, memory plan %.1f of %.1f GiB (GPU limit %.1f), "
                 "cache budget %.2f GiB, prefill chunk %d (%d alone), stop ids %s", model_path, weights / 2**30, ctx,
                 self.slots, plan / 2**30, mem / 2**30, self.limit / 2**30, self.budget / 2**30, self.chunk,
                 self.chunk_alone, sorted(self.stop_ids))

    def family_hint(self, name, value):
        """Tell the family an engine setting its cache sizing depends on (if it cares)."""
        hint = getattr(self.family, "set_hint", None)
        if hint is not None:
            hint(name, value)

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
            req.want = len(req.ids) + req.max_tokens
            need = band_for(len(req.ids) + min(req.max_tokens, self.reply_reserve), self.ctx)
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
        part = req.todo[: self.chunk if self.decoding else self.chunk_alone]
        if req.images and any(t < 0 for t in part):
            h = self.model.model(None, req.cache, input_embeddings=self._image_embeds(req, part))
        else:
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
            had_images, req.images = bool(req.images), {}   # features and pixels only matter while reading the prompt
            if had_images and hasattr(self.family, "release_vision") and                     not any(r.images for r in self.prefilling + self.waiting if r is not req):
                self.family.release_vision(self.model)
            log.info("prompt read: %d tokens in %.1f s, peak GPU memory %.2f GiB", len(req.tokens) - req.reused,
                     req.t_prompt, mx.get_peak_memory() / 2**30)
            mx.reset_peak_memory()
            req.snap = (len(req.tokens), self.family.snapshot(req.cache))
            req.y = req.sampler(logits)
            mx.eval(req.y)
            req.t1 = time.time()
            self.prefilling.remove(req)
            self.decoding.append(req)
            mx.clear_cache()

    def _image_embeds(self, req, part):
        """Prompt embeddings for a chunk holding image stand-ins: token embeddings, with each image's features (run
        through the vision tower the first time a chunk reaches that image) in place of its image tokens."""
        s = len(req.tokens)                   # absolute position of part[0]
        img_tok = self.family.image_token_id(self.model)
        clean = [img_tok if t < 0 else t for t in part]
        emb = self.family.embed_tokens(self.model, mx.array(clean)[None])
        pieces, i = [], 0
        while i < len(part):
            if part[i] >= 0:
                j = i
                while j < len(part) and part[j] >= 0:
                    j += 1
                pieces.append(emb[:, i:j])
            else:
                pid = part[i]
                j = i
                while j < len(part) and part[j] == pid:
                    j += 1
                img = req.images[pid]
                if img.feats is None:
                    t0 = time.time()
                    img.feats = self.family.encode_image(self.model, img.pixels, img.grid)
                    mx.eval(img.feats)
                    img.pixels = None
                    log.info("image %dx%d blocks (%d tokens) encoded in %.2f s", img.grid[2], img.grid[1],
                             img.count, time.time() - t0)
                off = s + i - img.start
                pieces.append(img.feats[off:off + j - i][None].astype(emb.dtype))
            i = j
        return pieces[0] if len(pieces) == 1 else mx.concatenate(pieces, axis=1)

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

    def _draft_step(self, r):
        """One copy-draft cycle for r, decoding alone. Returns False when there was nothing to draft."""
        if r.drafter is None:
            r.drafter = CopyDrafter()
        dr = r.drafter
        dr.extend(r.tokens[len(dr.seq):])       # catch up with tokens decoded normally since the last check
        # room: the reply's token budget and the cache (prompt + max_tokens keys)
        limit = min(r.max_tokens - len(r.out) - 3, r.max_len - len(r.tokens) - 3, CopyDrafter.MAX)
        d = dr.propose(limit) if limit >= 2 else []   # known tokens only: no wait on the pending token r.y
        if not d:
            return False
        t0 = time.perf_counter()
        self._use_decoder(None)
        snap = self.family.snapshot(r.cache)
        t1 = time.perf_counter()
        x = mx.concatenate([r.y.reshape(1).astype(mx.int32), mx.array(d, dtype=mx.int32)])[None]
        logits = self.model(x, cache=r.cache)[0]          # [K + 1, vocab]
        s = r.sampler(logits)
        mx.eval(s)
        s = s.tolist()
        tok0 = r.y.item()
        t2 = time.perf_counter()
        dr.secs[0] += t1 - t0
        dr.secs[1] += t2 - t1
        j = 0
        while j < len(d) and d[j] == s[j]:
            j += 1
        dr.update(len(d), j)
        if j < len(d):                          # roll the cache back to tok0 + the accepted drafts
            self.family.restore(r.cache, snap)
            mx.eval(self.model.model(mx.array([tok0] + d[:j], dtype=mx.int32)[None], r.cache))
            dr.secs[2] += time.perf_counter() - t2
            dr.secs[3] += 1
        emitted = [tok0] + d[:j]
        r.tokens.extend(emitted)                # the cache holds all of them, even past a stop token
        r.y = mx.array([s[j]], dtype=mx.int32)
        for tok in emitted:
            r.out.append(tok)
            stop = (tok in self.stop_ids and not r.ignore_eos) or r.cancelled
            if not stop and r.on_token is not None and r.on_token(tok) is False:
                stop = True
            if stop:
                self.decoding.remove(r)
                self._finish(r, "stop")
                return True
        return True

    GROW_MARGIN = 64                     # grow a cache this many keys before it fills (drafts need room too)

    def _grow(self):
        """Give replies running past their cache's length the next band. Without the memory for it, a reply waits
        while others run on (they finish and free memory); one with nobody else running grows regardless, as
        admission does."""
        for r in list(self.decoding):
            if len(r.tokens) + self.GROW_MARGIN < r.max_len or r.max_len >= min(r.want, self.ctx):
                r.paused = False
                continue
            need = band_for(min(len(r.tokens) + self.GROW_MARGIN + 1, r.want), self.ctx)
            others = [x for x in self.decoding + self.prefilling if x is not r and not x.paused]
            while self._in_use() + self._bytes(need) > self.budget and self.idle:
                self.idle.pop(0)
                mx.clear_cache()
            if self._in_use() + self._bytes(need) > self.budget and others:
                if not r.paused:
                    log.info("reply at %d tokens waits for memory to grow its cache", len(r.out))
                r.paused = True
                continue
            if self.dec_seq is r:
                self._use_decoder(None)       # the compiled decoder holds the buffers; give them back first
            self.family.fit_cache(r.cache, need)
            log.info("reply at %d tokens: cache %d -> %d keys", len(r.out), r.max_len, need)
            r.max_len, r.paused = need, False

    def _decode_step(self):
        t = time.time()
        self._grow()
        running = [r for r in self.decoding if not r.paused]
        if not running:                       # every reply waits for memory: let the prompts being read go on
            self.decode_credit = 0
            return
        if (self.draft and len(self.decoding) == 1 and running and not self.prefilling and not self.waiting
                and len(self.decoding[0].out) + 3 < self.decoding[0].max_tokens):
            if self._draft_step(self.decoding[0]):
                self.decode_credit -= time.time() - t
                return
        # a sequence about to emit its last allowed token is not stepped (its cache has no room past it)
        last = [r for r in running if len(r.out) + 1 >= r.max_tokens]
        step = [r for r in running if r not in last]
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
                "predicted_per_second": (n - 1) / t_gen if n > 1 and t_gen > 0 else 0.0, "finish": finish,
                "draft_checks": req.drafter.stats[0] if req.drafter else 0,
                "draft_n": req.drafter.stats[1] if req.drafter else 0,
                "draft_accepted": req.drafter.stats[2] if req.drafter else 0}

    def _finish(self, req, finish):
        if req.drafter is not None and req.drafter.stats[0]:
            st, sc = req.drafter.stats, req.drafter.secs
            log.info("copy drafts: %d checks, %d drafted, %d accepted; switch %.2f s, checks %.2f s, re-runs %d %.2f s,"
                     " %d tokens in %.2f s", st[0], st[1], st[2], sc[0], sc[1], sc[3], sc[2], len(req.out),
                     time.time() - req.t1)
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
                                  "cache_n", "draft_checks", "draft_n", "draft_accepted")}


class BadRequest(Exception):
    pass


MAX_IMAGE_BYTES = 20 * 2**20
MAX_IMAGE_TOKENS = 2304          # 40 px blocks per image; larger images are scaled down (2304 = 1920x1920 px area)


def image_parts(msgs):
    """Image content parts in prompt order (the order the chat template writes their image tokens)."""
    for m in msgs:
        c = m.get("content") if isinstance(m, dict) else None
        if isinstance(c, list):
            for p in c:
                if isinstance(p, dict) and p.get("type") in ("image", "input_image", "image_url"):
                    yield p


def image_bytes(part):
    v = part.get("image_url") or part.get("image") or part.get("url")
    if isinstance(v, dict):
        v = v.get("url")
    if not isinstance(v, str) or not v:
        raise BadRequest("image part without an image URL")
    if v.startswith("data:"):
        head, _, data = v.partition(",")
        if ";base64" not in head:
            raise BadRequest("image data URLs must be base64")
        try:
            return base64.b64decode(data, validate=False)
        except ValueError as e:
            raise BadRequest(f"bad base64 image: {e}") from None
    if v.startswith(("http://", "https://")):
        with urllib.request.urlopen(urllib.request.Request(v, headers={"User-Agent": "nuemac"}), timeout=20) as r:
            data = r.read(MAX_IMAGE_BYTES + 1)
        if len(data) > MAX_IMAGE_BYTES:
            raise BadRequest(f"image over {MAX_IMAGE_BYTES >> 20} MB")
        return data
    raise BadRequest("image URL must be a data: URL or http(s)")


def expand_images(ids, msgs):
    """Each image part's single image token -> one stand-in id per 40 px block (PromptImage). Returns (ids, images)."""
    parts = list(image_parts(msgs))
    if not parts:
        return ids, {}
    fam = ENGINE.family
    if not (hasattr(fam, "vision_ready") and fam.vision_ready(ENGINE.model)):
        raise BadRequest("this model build reads text only: it has no vision tower")
    img_tok = fam.image_token_id(ENGINE.model)
    pos = [i for i, t in enumerate(ids) if t == img_tok]
    if len(pos) != len(parts):
        raise BadRequest(f"{len(parts)} images but {len(pos)} image places in the prompt")
    out, images, last = [], {}, 0
    for k, (p, part) in enumerate(zip(pos, parts)):
        data = image_bytes(part)
        try:
            pixels, grid = fam.prepare_image(data, MAX_IMAGE_TOKENS)
        except Exception as e:  # noqa: BLE001
            raise BadRequest(f"image {k + 1} could not be read: {e}") from None
        out.extend(ids[last:p])
        pid = image_stand_in(hashlib.sha256(data).hexdigest(), k)
        images[pid] = PromptImage(pixels, grid, len(out))
        out.extend([pid] * images[pid].count)
        last = p + 1
    out.extend(ids[last:])
    return out, images


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
                 "meta": {"n_ctx": ENGINE.ctx, "vision": bool(getattr(ENGINE.family, "vision_ready", None)
                                                              and ENGINE.family.vision_ready(ENGINE.model))}}]})
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
        except BadRequest as e:
            return self._json(400, {"error": {"message": str(e), "type": "invalid_request_error"}})
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
        return expand_images(ENGINE.tok.encode(text, add_special_tokens=False), msgs)

    def chat(self, body):
        ids, images = self._prompt_ids(body)
        # no max_tokens: no cap, the reply may run to the end of the context (admission trims it to what's left)
        max_tokens = int(body.get("max_tokens") or body.get("max_completion_tokens") or ENGINE.ctx)
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
                      cache_prompt=bool(body.get("cache_prompt", True)), on_token=on_token, images=images)
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
    global ENGINE, MODEL_ID, MAX_IMAGE_TOKENS
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True, help="MLX model directory")
    ap.add_argument("--alias", default=MODEL_ID)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--ctx", type=int, default=98304)
    ap.add_argument("--prefill-chunk", type=int, default=256,
                    help="prompt tokens per step while other requests are generating (they wait about one step)")
    ap.add_argument("--prefill-chunk-alone", type=int, default=0,
                    help="prompt tokens per step when no request is generating; 0 = auto (1024 with a GPU limit of "
                         "20 GiB or more, else --prefill-chunk)")
    ap.add_argument("--expect-omlx", default="", help="oMLX version the model's recipe was tested with")
    ap.add_argument("--slots", type=int, default=4,
                    help="most sequences decoding at once (continuous batching); 1 = one request at a time")
    ap.add_argument("--keep-caches", type=int, default=2,
                    help="finished conversations whose caches stay for prompt reuse (memory permitting)")
    ap.add_argument("--mem-headroom-gb", type=float, default=-1,
                    help="memory to leave macOS and other apps; the engine plans its own use within RAM minus this "
                         "(and the GPU limit). -1 = 3 GiB")
    ap.add_argument("--reply-reserve", type=int, default=8192,
                    help="reply tokens a request's cache starts with; it grows past this as the reply runs on")
    ap.add_argument("--draft", default="copy", choices=("copy", "off"),
                    help="copy: draft tokens from the context while one request decodes alone (rewrites, edits)")
    ap.add_argument("--no-compiled-decode", action="store_true",
                    help="decode eagerly (one MLX graph built per token) instead of the compiled decode step")
    ap.add_argument("--attn-bits", type=int, default=8, choices=(4, 5, 6, 8),
                    help="inkling2: requantize the 8-bit attention projections")
    ap.add_argument("--head-bits", type=int, default=6, choices=(4, 5, 6, 8),
                    help="inkling2: requantize the 8-bit output head (6: near-lossless, faster)")
    ap.add_argument("--vision", default="auto", choices=("auto", "resident", "lazy", "off"),
                    help="inkling2: keep the vision tower in memory (resident), load it per image (lazy), or off; "
                         "auto = lazy below a 20 GiB GPU limit")
    ap.add_argument("--vision-bits", type=int, default=-1, choices=(-1, 0, 4, 8),
                    help="inkling2: quantize the vision tower's linear layers (0: as stored; -1 auto: 8 below a "
                         "20 GiB GPU limit, else 0)")
    ap.add_argument("--max-image-tokens", type=int, default=MAX_IMAGE_TOKENS,
                    help="40 px blocks per image; larger images are scaled down to fit")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    global WORKER
    MODEL_ID = a.alias
    MAX_IMAGE_TOKENS = max(1, a.max_image_tokens)
    WORKER = EngineThread(a.model, a.ctx, a.prefill_chunk,
                          {"expect_omlx": a.expect_omlx, "head_bits": a.head_bits, "attn_bits": a.attn_bits,
                           "compiled_decode": not a.no_compiled_decode, "slots": a.slots, "draft": a.draft,
                           "keep_caches": a.keep_caches, "reply_reserve": a.reply_reserve,
                           "headroom_gb": a.mem_headroom_gb,
                           "prefill_chunk_alone": a.prefill_chunk_alone,
                           "vision_bits": a.vision_bits, "vision": a.vision})
    WORKER.start()
    WORKER.ready.wait()
    if WORKER.error is not None:
        raise SystemExit(f"model failed to load: {WORKER.error}")
    srv = ThreadingHTTPServer((a.host, a.port), Handler)
    log.info("listening on http://%s:%d (model id %s)", a.host, a.port, MODEL_ID)
    srv.serve_forever()


if __name__ == "__main__":
    main()
