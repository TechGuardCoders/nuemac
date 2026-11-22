"""Compiled single-token decode for Inkling 2.

One decode token is ~9,800 MLX graph nodes and ~3,600 GPU launches. Most are small casts and element-wise ops,
each costing ~4-5 us on an M4 Pro, which adds ~16 ms to a ~28 ms token. Compiling the whole token into one graph
lets MLX fuse those chains and drops Python's per-token graph building. That needs every shape fixed from token
to token:
- the 40 sliding-window visits keep a ring of `window` keys (a token's slot is its position mod window); the
  relative bias is looked up by distance, so key order doesn't matter;
- the 8 global visits write their 8-bit codes into the preallocated buffers with a dynamic slice_update, and the
  split-K kernel reads the key count from an array, over a fixed grid per context band (BUCKETS);
- the short-conv states pass through as arrays.
Prefill stays eager (Engine.generate). `Decoder.begin` moves the caches into this layout after the prompt and
`Decoder.end` moves them back, so prompt reuse and snapshots work as before.
"""
import mlx.core as mx
from mlx_lm.models.cache import ArraysCache

from .q8attn import D, _pass1, _pass2

# (most keys, chunk): a fixed grid per band, chosen per token from the Python-side position.
BUCKETS = ((4096, 256), (16384, 512), (65536, 1024), (1 << 31, 2048))
SG, PF = 4, 1


def bucket_for(n):
    for i, (most, _chunk) in enumerate(BUCKETS):
        if n <= most:
            return i
    return len(BUCKETS) - 1


class RingKV:
    """A sliding-window visit during decode: `window` slots, slot = position % window."""

    def __init__(self, k, v, offset):
        self.k, self.v, self.offset = k, v, offset

    def decode_attend(self, attn, q, k, v, rel, sink):
        W = self.k.shape[2]
        o = self.offset
        slot = (o % W).reshape(1)
        self.k = mx.slice_update(self.k, k, slot, axes=(2,))
        self.v = mx.slice_update(self.v, v, slot, axes=(2,))
        d = (o - mx.arange(W, dtype=mx.int32) + W) % W            # distance of each slot's key
        bias = mx.take(rel[0, 0], d, axis=-1)                      # [hq, W]
        mask = mx.where((d <= o)[None, :], bias.astype(q.dtype), mx.array(-mx.inf, q.dtype))
        return mx.fast.scaled_dot_product_attention(
            q, self.k, self.v, scale=attn.scale, mask=mask.reshape(1, -1, 1, W),
            sinks=sink[0, 0].astype(q.dtype))


class Q8DecodeKV:
    """A global visit during decode: fixed 8-bit buffers, dynamic write, split-K read over a fixed grid."""

    def __init__(self, arrays, offset, bucket, group=64, bits=8):
        self.kq, self.ks, self.kb, self.vq, self.vs, self.vb = arrays
        self.offset, self.bucket, self.group, self.bits = offset, bucket, group, bits

    def arrays(self):
        return (self.kq, self.ks, self.kb, self.vq, self.vs, self.vb)

    def decode_attend(self, attn, q, k, v, rel, sink):
        o = self.offset
        at = o.reshape(1)
        kq, ks, kb = mx.quantize(k, group_size=self.group, bits=self.bits)
        vq, vs, vb = mx.quantize(v, group_size=self.group, bits=self.bits)
        self.kq = mx.slice_update(self.kq, kq, at, axes=(2,))
        self.ks = mx.slice_update(self.ks, ks, at, axes=(2,))
        self.kb = mx.slice_update(self.kb, kb, at, axes=(2,))
        self.vq = mx.slice_update(self.vq, vq, at, axes=(2,))
        self.vs = mx.slice_update(self.vs, vs, at, axes=(2,))
        self.vb = mx.slice_update(self.vb, vb, at, axes=(2,))
        hq, hkv, maxlen = q.shape[1], self.kq.shape[1], self.kq.shape[2]
        rep = hq // hkv
        most, chunk = BUCKETS[self.bucket]
        chunks = (min(most, maxlen) + chunk - 1) // chunk
        P = chunks * SG
        params = mx.concatenate([(o + 1).reshape(1), mx.array([maxlen, P], dtype=mx.int32)]).astype(mx.int32)
        fparams = mx.array([attn.scale], dtype=mx.float32)
        pm, pl, pacc = _pass1(
            inputs=[q.reshape(hq * D), self.kq, self.ks, self.kb, self.vq, self.vs, self.vb,
                    rel[0, 0], params, fparams],
            template=[("REP", rep), ("RELN", rel.shape[-1]), ("CHUNK", chunk), ("SG", SG), ("PF", PF)],
            grid=(chunks * 32 * SG, hkv, 1),
            threadgroup=(32 * SG, 1, 1),
            output_shapes=[(hkv, P, rep), (hkv, P, rep), (hkv, P, rep, D)],
            output_dtypes=[mx.float32, mx.float32, mx.float32],
        )
        (out,) = _pass2(
            inputs=[pm, pl, pacc, sink[0, 0], params],
            template=[("REP", rep)],
            grid=(D, hq, 1),
            threadgroup=(D, 1, 1),
            output_shapes=[(hq * D,)],
            output_dtypes=[mx.bfloat16],
        )
        return out.reshape(1, hq, 1, D).astype(q.dtype)


class Decoder:
    """Runs decode steps through one compiled graph per context band."""

    def __init__(self, model, window_cls, q8_cls):
        self.model = model
        self.window_cls, self.q8_cls = window_cls, q8_cls
        self.fns = {}
        self.state = None

    # -- layout -------------------------------------------------------------------------------------------
    def begin(self, cache):
        """Take the eager caches (after prefill) into decode layout. The eager objects drop their buffers so the
        compiled graph can update them in place; end() gives them back."""
        self.cache = cache
        kinds, flat = [], []
        offset = None
        for kv, conv in ((c[0], c[1]) for c in cache):
            if isinstance(kv, self.window_cls):
                o = kv.offset
                W = kv.window
                B, H, _, Dh = kv.keys.shape
                m = min(kv.n, W - 1, o)
                rk = mx.zeros((B, H, W, Dh), kv.keys.dtype)
                rv = mx.zeros((B, H, W, kv.values.shape[3]), kv.values.dtype)
                if m:
                    idx = (mx.arange(o - m, o) % W).astype(mx.int32)
                    rk[:, :, idx, :] = kv.keys[:, :, kv.n - m: kv.n, :]
                    rv[:, :, idx, :] = kv.values[:, :, kv.n - m: kv.n, :]
                kinds.append("w")
                flat += [rk, rv]
            elif isinstance(kv, self.q8_cls):
                o = kv.offset
                kinds.append("g")
                flat += [kv.kq, kv.ks, kv.kb, kv.vq, kv.vs, kv.vb]
                kv.kq = kv.ks = kv.kb = kv.vq = kv.vs = kv.vb = None
            else:
                raise TypeError(f"unsupported cache {type(kv).__name__}")
            if offset is None:
                offset = o
            elif o != offset:
                raise ValueError("layer visits disagree on the position")
            flat += list(conv.cache)
        self.kinds, self.offset = kinds, offset
        mx.eval(flat)
        self.state = flat

    def end(self):
        """Give the caches back to the eager objects (contiguous window keys, the same 8-bit buffers)."""
        if self.state is None:
            return
        mx.eval(self.state)
        it = iter(self.state)
        o = self.offset
        for kind, c in zip(self.kinds, self.cache):
            kv, conv = c[0], c[1]
            if kind == "w":
                rk, rv = next(it), next(it)
                W = rk.shape[2]
                m = min(o, W - 1)
                idx = (mx.arange(o - m, o) % W).astype(mx.int32)
                kv.keys, kv.values = rk[:, :, idx, :], rv[:, :, idx, :]
                kv.n, kv.offset = m, o
            else:
                kv.kq, kv.ks, kv.kb, kv.vq, kv.vs, kv.vb = (next(it) for _ in range(6))
                kv.offset = o
            conv.cache = [next(it) for _ in range(4)]
        mx.eval([c[0].keys if k == "w" else c[0].kq for k, c in zip(self.kinds, self.cache)])
        # Drop every reference: a kept cache list would pin its 8-bit buffers (4.6 GB at 256K) after the engine
        # replaces the cache, and the next prefill runs out of GPU memory.
        self.state = self.cache = None

    # -- one token ---------------------------------------------------------------------------------------
    def _unflatten(self, flat, o, bucket):
        it = iter(flat)
        caches = []
        for kind in self.kinds:
            if kind == "w":
                kv = RingKV(next(it), next(it), o)
            else:
                kv = Q8DecodeKV(tuple(next(it) for _ in range(6)), o, bucket)
            conv = ArraysCache(4)
            conv.cache = [next(it) for _ in range(4)]
            caches.append([kv, conv])
        return caches

    @staticmethod
    def _flatten(caches):
        flat = []
        for kv, conv in caches:
            flat += [kv.k, kv.v] if isinstance(kv, RingKV) else list(kv.arrays())
            flat += list(conv.cache)
        return flat

    def _fn(self, bucket):
        fn = self.fns.get(bucket)
        if fn is None:
            def step(tok, o, *flat):
                caches = self._unflatten(flat, o, bucket)
                logits = self.model(tok, cache=caches)[:, -1, :]
                return (logits, *self._flatten(caches))

            fn = self.fns[bucket] = mx.compile(step)
        return fn

    def step(self, tok):
        """tok: [1, 1] int32 array. Returns logits [1, vocab] (lazy); the state advances one position."""
        fn = self._fn(bucket_for(self.offset + 1))
        out = fn(tok.astype(mx.int32), mx.array(self.offset, dtype=mx.int32), *self.state)
        self.state = list(out[1:])
        self.offset += 1
        return out[0]
