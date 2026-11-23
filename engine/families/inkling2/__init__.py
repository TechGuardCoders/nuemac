"""Inkling 2 (model types inkling2_loop / inkling2): the first Nuemac family.

Runs on oMLX's Inkling 2 model code (MLX) with the layout a 16 GB Mac needs:
- the 40 sliding-window layer visits keep only their last 1,024 keys (oMLX
  keeps every key on all 48 visits, ~192 KB/token);
- the 8 global visits keep full history as 8-bit affine K/V (group 64),
  allocated once at the context size, ~17 KB/token;
- decode attention on the global visits runs a split-K Metal kernel straight
  from the 8-bit codes (q8attn.py) once the context passes FLASH_MIN tokens;
- fewer bytes per token: the loop controllers stay bf16 (oMLX upcasts them to
  fp32) and the 8-bit output head is requantized to 6 bits (head_bits).
Measurements behind every choice: docs/recipes/inkling-2-mini.md.
"""
import logging
import os

import mlx.core as mx
from mlx_lm.models.cache import ArraysCache, CacheList, KVCache

from .. import ContextFull

log = logging.getLogger("nuemac.inkling2")

NAME = "inkling2"
MODEL_TYPES = frozenset({"inkling2_loop", "inkling2"})
TESTED_OMLX = ("0.8.2", "0.8.3")

# Below this many cached keys the plain MLX-ops path decodes faster end to end:
# the custom kernel's own time is lower at every depth, but on a base M4 it cost
# ~1.4 ms per token at short context (measured A/B), more than it saves there.
FLASH_MIN = 12288


class WindowKVCache(KVCache):
    """Sliding-window visit: keeps the last window-1 keys plus the new ones,
    contiguous and in order (what oMLX's banded kernels expect)."""

    def __init__(self, window, slack=256):
        super().__init__()
        self.window = window
        self.slack = slack
        self.n = 0

    def update_and_fetch(self, keys, values):
        L = keys.shape[2]
        keep = self.window - 1
        if self.keys is None or self.n + L > self.keys.shape[2]:
            B, H, _, D = keys.shape
            n_old = min(self.n, keep)
            cap = n_old + L + self.slack
            nk = mx.zeros((B, H, cap, D), keys.dtype)
            nv = mx.zeros((B, H, cap, values.shape[3]), values.dtype)
            if self.keys is not None and n_old:
                nk[..., :n_old, :] = self.keys[..., self.n - n_old : self.n, :]
                nv[..., :n_old, :] = self.values[..., self.n - n_old : self.n, :]
            self.keys, self.values, self.n = nk, nv, n_old
        self.keys[..., self.n : self.n + L, :] = keys
        self.values[..., self.n : self.n + L, :] = values
        self.n += L
        self.offset += L
        return self.keys[..., : self.n, :], self.values[..., : self.n, :]

    def snapshot(self):
        return (self.keys, self.values, self.n, self.offset)

    def restore(self, snap):
        self.keys, self.values, self.n, self.offset = snap


class Q8GlobalKVCache(KVCache):
    """Global visit: full history as 8-bit affine codes, allocated once.
    Prefill fetches a bf16 copy for the flash kernel; decode reads the codes."""

    bits = 8
    group = 64

    def __init__(self, max_len):
        super().__init__()
        self.max_len = max_len
        self.kq = None

    def _alloc(self, keys):
        B, H, _, D = keys.shape
        w = D * self.bits // 32
        g = D // self.group
        z = lambda last, dt: mx.zeros((B, H, self.max_len, last), dt)  # noqa: E731
        self.kq, self.vq = z(w, mx.uint32), z(w, mx.uint32)
        self.ks, self.kb = z(g, keys.dtype), z(g, keys.dtype)
        self.vs, self.vb = z(g, keys.dtype), z(g, keys.dtype)
        self.dtype = keys.dtype

    def append(self, keys, values):
        if self.kq is None:
            self._alloc(keys)
        a, b = self.offset, self.offset + keys.shape[2]
        if b > self.max_len:
            raise ContextFull(f"prompt + reply need {b} tokens; context is {self.max_len}")
        kq, ks, kb = mx.quantize(keys, group_size=self.group, bits=self.bits)
        vq, vs, vb = mx.quantize(values, group_size=self.group, bits=self.bits)
        self.kq[..., a:b, :] = kq
        self.ks[..., a:b, :] = ks
        self.kb[..., a:b, :] = kb
        self.vq[..., a:b, :] = vq
        self.vs[..., a:b, :] = vs
        self.vb[..., a:b, :] = vb
        self.offset = b

    def codes(self):
        n = self.offset
        return ((self.kq[..., :n, :], self.ks[..., :n, :], self.kb[..., :n, :]),
                (self.vq[..., :n, :], self.vs[..., :n, :], self.vb[..., :n, :]))

    def update_and_fetch(self, keys, values):
        self.append(keys, values)
        (kq, ks, kb), (vq, vs, vb) = self.codes()
        k = mx.dequantize(kq, ks, kb, group_size=self.group, bits=self.bits)
        v = mx.dequantize(vq, vs, vb, group_size=self.group, bits=self.bits)
        return k.astype(self.dtype), v.astype(self.dtype)

    def snapshot(self):
        # Codes below offset are never rewritten, so the offset is enough.
        return self.offset

    def restore(self, snap):
        self.offset = snap

    def grow(self, max_len):
        """Reallocate for max_len keys, keeping the first `offset` codes (a reused cache meeting a longer reply)."""
        if max_len <= self.max_len:
            return
        if self.kq is not None:
            old = (self.kq, self.ks, self.kb, self.vq, self.vs, self.vb)
            n = self.offset
            self.max_len = max_len
            new = []
            for a in old:
                b = mx.zeros(a.shape[:2] + (max_len,) + a.shape[3:], a.dtype)
                if n:
                    b[..., :n, :] = a[..., :n, :]
                new.append(b)
            mx.eval(new)
            self.kq, self.ks, self.kb, self.vq, self.vs, self.vb = new
        else:
            self.max_len = max_len


def q8_decode_attention(q, kc, vc, rel, sink, scale, group, bits):
    """One query over 8-bit K/V via quantized_matmul (short contexts)."""
    B, hq, L, D = q.shape
    hkv = kc[0].shape[1]
    s = kc[0].shape[2]
    rep = hq // hkv
    n = min(rel.shape[-1], s)
    tail = rel[0, 0, :, :n][:, ::-1]
    if n < s:
        tail = mx.concatenate([mx.zeros((hq, s - n), dtype=tail.dtype), tail], -1)
    qg = (q * scale).reshape(B, hkv, rep, L, D)
    kc = tuple(mx.expand_dims(x, -3) for x in kc)
    vc = tuple(mx.expand_dims(x, -3) for x in vc)
    scores = mx.quantized_matmul(qg, *kc, transpose=True, group_size=group, bits=bits)
    scores = scores.astype(mx.float32) + tail.reshape(B, hkv, rep, 1, s).astype(mx.float32)
    sk = sink[0, 0].astype(mx.float32).reshape(B, hkv, rep, 1, 1)
    p = mx.softmax(mx.concatenate([scores, sk], axis=-1), axis=-1, precise=True)
    out = mx.quantized_matmul(p[..., :s].astype(q.dtype), *vc, transpose=False,
                              group_size=group, bits=bits)
    return out.reshape(B, hq, L, D)


ATTEND_SIGS = (["self", "q", "k", "v", "rel", "sink", "kv"],              # oMLX 0.8.2
               ["self", "q", "k", "v", "rel", "sink", "kv", "state"])     # 0.8.3: a window visit's state may hold its K/V


VERIFY_MAX_ROWS = 16


def q8_rows_attention(q, kv, rel, sink, scale):
    """L new rows (already appended) over the 8-bit global cache, causal: row i sits at position P + i, sees keys
    up to it, with the relative bias for distances below R. q: [1, hq, L, D]; rel: [1, L, hq, R]; sink: [1, L, hq].
    Same math as q8_decode_attention, one row at a time, for draft checks."""
    (kq, ks, kb), (vq, vs, vb) = kv.codes()
    B, hq, L, D = q.shape
    hkv, s = kq.shape[1], kq.shape[2]
    rep, R = hq // hkv, rel.shape[-1]
    P = s - L
    qg = (q * scale).reshape(B, hkv, rep, L, D)
    kc = tuple(mx.expand_dims(x, -3) for x in (kq, ks, kb))
    vc = tuple(mx.expand_dims(x, -3) for x in (vq, vs, vb))
    scores = mx.quantized_matmul(qg, *kc, transpose=True, group_size=kv.group, bits=kv.bits).astype(mx.float32)
    T = min(s, R + L)                                           # the keys any row's bias or mask touches
    d = (P + mx.arange(L))[:, None] - (s - T + mx.arange(T))[None, :]          # [L, T] distance row -> key
    rel_t = rel[0].transpose(1, 0, 2).astype(mx.float32)                       # [hq, L, R]
    idx = mx.broadcast_to(mx.clip(d, 0, R - 1)[None], (hq, L, T))
    bias = mx.take_along_axis(rel_t, idx, axis=2)
    bias = mx.where((d < R)[None], bias, 0.0)
    bias = mx.where((d < 0)[None], -mx.inf, bias).reshape(B, hkv, rep, L, T)
    if T < s:
        scores = mx.concatenate([scores[..., : s - T], scores[..., s - T:] + bias], axis=-1)
    else:
        scores = scores + bias
    sk = sink[0].transpose(1, 0).astype(mx.float32).reshape(B, hkv, rep, L, 1)   # [hq, L] per row
    p = mx.softmax(mx.concatenate([scores, sk], axis=-1), axis=-1, precise=True)
    out = mx.quantized_matmul(p[..., :s].astype(q.dtype), *vc, transpose=False, group_size=kv.group, bits=kv.bits)
    return out.reshape(B, hq, L, D)


def check_omlx(expect):
    """The family leans on oMLX internals: fail early and clearly if they moved."""
    import importlib.metadata as md
    import inspect

    have = md.version("omlx")
    from omlx.patches.inkling2 import model as ink

    sig = list(inspect.signature(ink.Inkling2Attention.attend).parameters)
    if sig not in ATTEND_SIGS or not hasattr(ink, "kernels"):
        raise SystemExit(f"oMLX {have}: Inkling2Attention.attend changed ({sig}); "
                         f"this Nuemac family was written for oMLX {' / '.join(TESTED_OMLX)}")
    tested = tuple(expect.replace(",", " ").split()) if isinstance(expect, str) else tuple(expect or ())
    if tested and have not in tested:
        log.warning("oMLX %s installed; Nuemac was tested with %s", have, " / ".join(tested))
    return have


def install_attend_patch():
    from omlx.patches.inkling2 import model as ink
    from .q8attn import q8_flash_decode

    orig = ink.Inkling2Attention.attend

    def attend(self, q, k, v, rel, sink, kv, state=None):
        if hasattr(kv, "decode_attend"):   # compiled decode (fastpath.py)
            return kv.decode_attend(self, q, k, v, rel, sink)
        if isinstance(kv, Q8GlobalKVCache) and q.shape[0] == 1 and q.shape[2] == 1:
            kv.append(k, v)
            if kv.offset > FLASH_MIN:
                return q8_flash_decode(q, kv, rel, sink, self.scale)
            kc, vc = kv.codes()
            return q8_decode_attention(q, kc, vc, rel, sink, self.scale, kv.group, kv.bits)
        if isinstance(kv, Q8GlobalKVCache) and q.shape[0] == 1 and q.shape[2] <= VERIFY_MAX_ROWS:
            # a few rows (a draft being checked): read the 8-bit codes, no bf16 copy of the whole history
            kv.append(k, v)
            return q8_rows_attention(q, kv, rel, sink, self.scale)
        return orig(self, q, k, v, rel, sink, kv)   # kv is never None here, so 0.8.3's `state` is unused

    ink.Inkling2Attention.attend = attend
    if hasattr(ink.Inkling2TextModel, "visit_caches"):
        install_visit_caches_patch(ink)


def install_visit_caches_patch(ink):
    """oMLX 0.8.3 lays a cache list out as one entry per window visit and two per global visit, and maps them to
    visits in visit_caches. Nuemac's caches stay one [KV, conv state] pair per visit (make_cache, batch_cache,
    the compiled decoder): hand those through as they are."""
    orig = ink.Inkling2TextModel.visit_caches

    def visit_caches(self, cache):
        if len(cache) == self.num_visits and all(isinstance(c, (list, tuple, CacheList)) for c in cache):
            return [(c[0], c[1]) for c in cache]
        return orig(self, cache)

    ink.Inkling2TextModel.visit_caches = visit_caches


def install_bf16_loop_patch():
    """The loop controllers (loop_down, loop_gain_forget) are bf16 weights that
    oMLX keeps as fp32 copies: ~0.23 GB read per token in the looped layers.
    Keep them bf16 and run the two small matmuls in bf16 (fp32 accumulation
    inside the matmul, one bf16 rounding, as the reference does). Greedy tokens
    were identical to the fp32 path on our checks; decode +0.4 tok/s."""
    from omlx.patches.inkling2 import model as ink

    import os

    from .mixkernel import mix_messages_row

    fused = not os.environ.get("NUEMAC_NO_FUSED_MIX")   # debug switch: the MLX-ops path

    def _mix_messages(x, message, down, gain_forget):
        n, b, length, hidden = x.shape
        rows = b * length
        if fused and rows == 1 and hidden % 256 == 0 and down.shape[-1] == 64:
            return mix_messages_row(x, message, down, gain_forget)   # decode: 2 fused kernels instead of ~12
        z = mx.stack([ink._unit_rms(x), ink._unit_rms(message)], axis=1)
        z = z.astype(mx.bfloat16).reshape(2 * n, rows, hidden)
        low = (z @ down.reshape(2 * n, hidden, -1)).astype(mx.float32)
        low = low.reshape(n, 2, rows, -1).sum(axis=1).astype(mx.bfloat16)
        gates = (low @ gain_forget).astype(mx.float32).reshape(n, b, length, 2 * hidden)
        gain = mx.tanh(gates[..., :hidden])
        forget = mx.tanh(gates[..., hidden:])
        return (1.0 - forget) * x + gain * message

    ink._mix_messages = _mix_messages


def _requant(lin, bits):
    """A QuantizedLinear at `bits` (group 64) from an existing quantized one."""
    import mlx.nn as nn

    w = mx.dequantize(lin.weight, lin.scales, lin.biases, group_size=lin.group_size, bits=lin.bits)
    q = nn.QuantizedLinear(w.shape[1], w.shape[0], bias=False, group_size=64, bits=bits)
    q.weight, q.scales, q.biases = mx.quantize(w, group_size=64, bits=bits)
    mx.eval(q.weight, q.scales, q.biases)
    return q


ATTN_PROJS = ("in_proj", "q_proj", "k_proj", "v_proj", "o_proj")


def shrink_weights(model, head_bits, attn_bits=8):
    """bf16 loop controllers; optionally requantize the 8-bit output head and attention projections.
    6-bit head: mean KL 0.00027 against the 8-bit head (8-bit requantized to
    itself: 0.00008), top-1 98.9% (99.5%); decode +0.7 tok/s on a base M4.
    attn_bits < 8 requantizes every attention projection (in/q/k/v/o, ~1.37 GB per token at 8 bits)."""

    for layer in model.model.layers:
        if hasattr(layer, "loop_down"):
            layer.loop_down = layer.loop_down.astype(mx.bfloat16)
            layer.loop_gain_forget = layer.loop_gain_forget.astype(mx.bfloat16)
    head = model.lm_head
    # only ever lower the head's width: a build that already stores it at 4 or 6 bits keeps it as stored
    if head_bits and head_bits < getattr(head, "bits", 16):
        model.lm_head = _requant(head, head_bits)
    if attn_bits and attn_bits != 8 or os.environ.get("NUEMAC_REQUANT_ATTN8"):
        for layer in model.model.layers:
            for name in ATTN_PROJS:
                lin = getattr(layer.attn, name, None)
                if lin is not None and hasattr(lin, "bits"):
                    setattr(layer.attn, name, _requant(lin, attn_bits or 8))
            mx.clear_cache()
    mx.eval(model.parameters())
    mx.clear_cache()


# -- the family interface the engine calls ------------------------------------

def load(model_path, options):
    """Check oMLX, patch it, load the model, then shrink the weights."""
    log.info("inkling2 family on oMLX %s", check_omlx(options.get("expect_omlx") or TESTED_OMLX))
    from omlx.utils.model_loading import load_text_model

    install_attend_patch()
    install_bf16_loop_patch()
    model, tok = load_text_model(model_path)
    mx.eval(model.parameters())
    nrot = install_rotation(model, model_path)
    if nrot:
        log.info("inkling2: %d expert projections in a Hadamard-rotated basis", nrot)
    shrink_weights(model, int(options.get("head_bits", 6)), int(options.get("attn_bits", 8)))
    setup_vision(model, dict(options, model_path=model_path))
    return model, tok


# -- images ---------------------------------------------------------------------
# A build that ships the vision tower (config vision_config + vision_tower.* tensors) reads images: the processor cuts
# an image into 40 px blocks, one image token each (a 1280x720 screenshot is 32x18 = 576 tokens); the tower (a 23-layer
# ViT, ~300M parameters) turns them into embeddings that replace the image tokens' embeddings in the prompt.
IMAGE_TOKEN = "<|unused_200054|>"
BLOCK = 40


# Where the tower lives: "resident" keeps it in memory; "lazy" loads it when a prompt's image is first read and
# frees it once no prompt waits on an image (~0.3 s per load). Measured on a 16 GB M4 (11.2 GB build, 94K-token
# prompt): a resident 8-bit tower left short-context speed and quality unchanged but cut decode at 94K from 15.1 to
# 12.2 tok/s (the last GPU memory), so "auto" is lazy below a 20 GiB GPU limit.
VISION = {"mode": None, "bits": 0, "path": None, "config": None, "hidden": 0}


def vision_ready(model):
    return VISION["mode"] in ("resident", "lazy")


def _small_gpu():
    return mx.device_info()["max_recommended_working_set_size"] < 20 * 2**30


def _quantize_tower(tower, bits):
    import mlx.nn as nn
    if bits:
        nn.quantize(tower, group_size=64, bits=bits,
                    class_predicate=lambda _p, m: isinstance(m, nn.Linear) and m.weight.shape[-1] % 64 == 0)
    mx.eval(tower.parameters())


def setup_vision(model, options):
    """vision: resident | lazy | off | auto (lazy below a 20 GiB GPU limit, else resident).
    vision_bits 8/4 quantizes the tower's linear layers (group 64); 0 keeps them as stored (bf16, 607 MiB); -1 (auto)
    picks 8 below a 20 GiB GPU limit. Measured on Inkling 2 Mini: 8-bit linears (325 MiB) keep the image features at
    cosine 0.997-0.999 of bf16 with the same answers on the vision checks; 4-bit (174 MiB) drops to 0.95-0.97."""
    import json

    from mlx.utils import tree_flatten

    if getattr(model, "vision_tower", None) is None:
        return
    mode = options.get("vision", "auto") or "auto"
    if mode == "auto":
        mode = "lazy" if _small_gpu() else "resident"
    bits = int(options.get("vision_bits", -1))
    if bits < 0:
        bits = 8 if _small_gpu() else 0
    path = options.get("model_path")
    VISION.update(mode=mode, bits=bits, path=path, hidden=model.args.text.hidden_size if hasattr(model.args, "text")
                  else model.args.hidden_size)
    if path:
        VISION["config"] = json.load(open(os.path.join(path, "config.json"))).get("vision_config")
    if mode == "lazy" and not VISION["config"]:
        mode = VISION["mode"] = "resident"           # no config to rebuild it from
    if mode in ("off", "lazy"):
        model.pop("vision_tower")
        mx.clear_cache()
        log.info("inkling2: vision tower %s", "off" if mode == "off" else
                 f"loaded on demand ({bits}-bit linears)" if bits else "loaded on demand (as stored)")
        return
    _quantize_tower(model.vision_tower, bits)
    size = sum(v.nbytes for _, v in tree_flatten(model.vision_tower.parameters()))
    log.info("inkling2: vision tower resident, %.0f MiB (%s)", size / 2**20,
             f"{bits}-bit linears" if bits else "as stored")


def _tower(model):
    """The vision tower, loading it from the build when it is not in memory (lazy mode)."""
    import glob
    import time

    t = getattr(model, "vision_tower", None)
    if t is not None:
        return t
    from omlx.patches.inkling2.vision import VisionArgs, VisionTower

    t0 = time.time()
    t = VisionTower(VisionArgs.from_dict(VISION["config"]), VISION["hidden"])
    w = []
    for f in sorted(glob.glob(os.path.join(VISION["path"], "*.safetensors"))):
        if os.path.basename(f) == "mtp.safetensors":
            continue
        d = mx.load(f)
        w += [(k[len("vision_tower."):], v) for k, v in d.items() if k.startswith("vision_tower.")]
        del d
    t.load_weights(w)
    _quantize_tower(t, VISION["bits"])
    model.vision_tower = t
    log.info("inkling2: vision tower loaded in %.2f s", time.time() - t0)
    return t


def release_vision(model):
    """Lazy mode: free the tower once no prompt waits on an image."""
    if VISION["mode"] == "lazy" and getattr(model, "vision_tower", None) is not None:
        model.pop("vision_tower")
        mx.clear_cache()


def image_token_id(model):
    return int(getattr(model.args, "image_token_id", 200054))


def prepare_image(data, max_tokens):
    """Image bytes -> (pixel blocks [n, 40, 40, 3] float32, (1, rows, cols)). CPU only (any thread). An image over
    max_tokens blocks is scaled down (aspect kept) to fit; EXIF rotation is applied first."""
    import io
    import math

    from omlx.patches.inkling2.processing import Inkling2ImageProcessor
    from PIL import Image, ImageOps

    im = Image.open(io.BytesIO(data))
    im = ImageOps.exif_transpose(im)
    w, h = im.size
    n = math.ceil(w / BLOCK) * math.ceil(h / BLOCK)
    if max_tokens and n > max_tokens:
        s = math.sqrt(max_tokens / n)
        while True:
            nw, nh = max(BLOCK, int(w * s)), max(BLOCK, int(h * s))
            if math.ceil(nw / BLOCK) * math.ceil(nh / BLOCK) <= max_tokens:
                break
            s *= 0.97
        im = im.resize((nw, nh), Image.LANCZOS)
    out = Inkling2ImageProcessor().preprocess(im)
    _t, rows, cols = (int(v) for v in out["image_grid_thw"][0])
    return out["pixel_values"], (1, rows, cols)


def encode_image(model, pixels, grid):
    """Pixel blocks of one image -> its image-token embeddings [rows * cols, hidden] (engine thread)."""
    return _tower(model)(mx.array(pixels), [list(grid)])


def embed_tokens(model, ids):
    """The prompt embeddings the text model starts from (normalized); image features replace these unnormalized."""
    return model.model.embed_norm(model.model.embed(ids))


ROTATION_FILE = "nuemac_rotation.json"


def rotation_signs(dim, seed):
    import numpy as np
    return mx.array(np.random.default_rng(seed).choice([-1.0, 1.0], size=dim).astype("float32"))


def rotate(x, signs, block):
    """Signed blockwise Walsh-Hadamard of the last axis, orthonormal: W x == (rotate(W rows)) (rotate(x))."""
    import math
    shp = x.shape
    y = (x.astype(mx.float32) * signs).reshape(*shp[:-1], shp[-1] // block, block)
    return mx.hadamard_transform(y, scale=1.0 / math.sqrt(block)).reshape(shp).astype(x.dtype)


def install_rotation(model, model_path):
    """Experts quantized in a Hadamard-rotated basis (the build lists them in nuemac_rotation.json: block size and
    sign seed per projection) get their input rotated the same way at run time, on every path (compiled decode,
    batches, draft checks, prefill). Rotation spreads a weight row's outliers across its block, so a 2-bit grid fits
    it far better (the idea behind QuIP#, QuaRot and Bonsai's rotated basis). Returns how many projections."""
    path = os.path.join(model_path, ROTATION_FILE)
    if not os.path.exists(path):
        return 0
    import json

    from omlx.patches.inkling2 import model as ink

    if not getattr(ink.ExpertProj, "_nuemac_rot", False):
        orig = ink.ExpertProj.__call__

        def call(self, x, idx, sorted_indices):
            rot = self.__dict__.get("_rot")
            if rot is not None:
                x = rotate(x, *rot)
            return orig(self, x, idx, sorted_indices)

        ink.ExpertProj.__call__ = call
        ink.ExpertProj._nuemac_rot = True
    spec = json.load(open(path))
    n = 0
    for mpath, r in spec.items():
        m = model
        for part in mpath.split("."):
            m = m[int(part)] if part.isdigit() else getattr(m, part)
        din = m.scales.shape[-1] * m.group_size
        signs = rotation_signs(din, int(r["seed"]))
        mx.eval(signs)
        m.__dict__["_rot"] = (signs, int(r["block"]))
        n += 1
    return n


def make_decoder(model, options):
    """Compiled single-token decode (fastpath.py): same tokens as the eager path, less per-token overhead."""
    from .fastpath import Decoder

    return Decoder(model, WindowKVCache, Q8GlobalKVCache)


# -- batched decode: B sequences, each with its own caches ---------------------------------------------------------

class BatchKV:
    """One layer visit's KV for a batched decode step: row i attends to sequence i's own cache, so sequences of
    any lengths batch together. One launch reads every row's cache (slots.py); NUEMAC_NO_SLOTS_KERNEL=1 runs
    the single-row paths row by row instead."""

    def __init__(self, kvs):
        self.kvs = kvs
        self.offset = mx.array([kv.offset for kv in kvs], dtype=mx.int32)   # log_tau reads per-row positions

    def decode_attend(self, attn, q, k, v, rel, sink):
        from omlx.patches.inkling2 import model as ink

        from . import slots

        if slots.ENABLED and isinstance(self.kvs[0], WindowKVCache) and q.shape[2] == 1:
            # sliding visits: append each row's key, then one launch reads every row's window (slots.py)
            for i, kv in enumerate(self.kvs):
                kv.update_and_fetch(k[i:i + 1], v[i:i + 1])
            return slots.window_slots_attend(q, self.kvs, rel, sink, attn.scale, attn.window)
        if slots.ENABLED and isinstance(self.kvs[0], Q8GlobalKVCache) and q.shape[2] == 1:
            for i, kv in enumerate(self.kvs):
                kv.append(k[i:i + 1], v[i:i + 1])
            return slots.global_slots_attend(q, self.kvs, rel, sink, attn.scale)
        attend = ink.Inkling2Attention.attend   # Nuemac's patched attend
        rows = [attend(attn, q[i:i + 1], k[i:i + 1], v[i:i + 1], rel[i:i + 1], sink[i:i + 1], kv)
                for i, kv in enumerate(self.kvs)]
        return mx.concatenate(rows, axis=0)


class BatchConv:
    """One layer visit's short-conv states for a batched step: rows stacked on read, split back on write."""

    def __init__(self, convs):
        self.convs = convs

    def __getitem__(self, slot):
        states = [c[slot] for c in self.convs]
        if any(s is None for s in states):
            return None
        return mx.concatenate(states, axis=0)

    def __setitem__(self, slot, value):
        for i, c in enumerate(self.convs):
            c[slot] = value[i:i + 1]


def batch_cache(caches):
    """Per-visit [BatchKV, BatchConv] over B sequences' caches (each from make_cache)."""
    return [[BatchKV([c[v][0] for c in caches]), BatchConv([c[v][1] for c in caches])]
            for v in range(len(caches[0]))]


HINTS = {"prefill_chunk": 256}


def set_hint(name, value):
    """Engine settings the cache estimate depends on (the largest prefill chunk sizes the window caches)."""
    HINTS[name] = value


def cache_bytes(model, max_len):
    """Estimated bytes one sequence's caches take: 8-bit global K/V for max_len keys, the window caches."""
    args = model.args.text if hasattr(model.args, "text") else model.args
    sched = model.model._schedule
    n_global = sum(1 for idx, _p in sched if not args.is_sliding(idx))
    n_window = len(sched) - n_global
    per_key = 2 * args.num_key_value_heads * (args.head_dim + 2 * 2 * (args.head_dim // 64))   # codes + scales/biases
    window_keys = args.sliding_window_size + HINTS["prefill_chunk"] + 256                       # keep + chunk + slack
    win = 2 * args.swa_num_key_value_heads * args.swa_head_dim * 2 * window_keys
    return n_global * per_key * max_len + n_window * win


def fit_cache(cache, max_len):
    """Make a (reused) cache hold max_len keys."""
    for c in cache:
        if isinstance(c[0], Q8GlobalKVCache):
            c[0].grow(max_len)


def make_cache(model, ctx):
    """Caches for one sequence; `ctx` is the most keys it will hold (the 8-bit global buffers' length)."""
    args = model.args.text if hasattr(model.args, "text") else model.args
    out = []
    for idx, _pass in model.model._schedule:
        if args.is_sliding(idx):
            kv = WindowKVCache(args.sliding_window_size)
        else:
            kv = Q8GlobalKVCache(ctx)
        out.append(CacheList(kv, ArraysCache(4)))
    return out


def snapshot(cache):
    return [(c[0].snapshot(), list(c[1].cache)) for c in cache]


def restore(cache, state):
    for c, (kvs, conv) in zip(cache, state):
        c[0].restore(kvs)
        c[1].cache = list(conv)


def stop_ids(tok):
    ids = set()
    tid = tok.convert_tokens_to_ids("<|content_model_end_sampling|>")
    if isinstance(tid, int) and tid >= 0:
        ids.add(tid)
    ids.update(int(x) for x in (getattr(tok, "eos_token_ids", None) or []))
    return ids


def output_parser(tok, model_path):
    from omlx.adapter.output_parser import InklingOutputParserSession

    return InklingOutputParserSession(tok, model_path)
