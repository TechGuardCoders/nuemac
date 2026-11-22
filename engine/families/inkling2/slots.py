"""Batched decode attention: one query row per sequence, each against its own cache, many sequences a launch
(8 over the sliding-window visits' bf16 caches, 4 over the global visits' 8-bit caches).

Adapted from mlx-serve's `relslots.metal` + `relcomb.metal` (Inkling 2 plugin, MIT, Copyright (c) 2026 David Dalcu,
https://github.com/ddalcu/mlx-serve). Changes: our window caches are separate bf16 buffers of their own capacity
(their row strides come in as CAPS), the output gate stays outside (oMLX applies it), the combine writes
[B, hq, 1, D], and a second variant reads the global visits' 8-bit affine caches (Nuemac's own layout).

Pass 1: grid (splits * 32 * NSG, HKV, NS). Threadgroup (split, KV head, slot) takes KB keys of that slot's window;
its SIMD groups take the keys in turn and each lane owns D / 32 columns. The G query heads that share a KV head
share each key's read; an online softmax (log2 domain) per query head, the sink as a zero-value logit on split 0.
Pass 2: one thread per output element merges the splits.

mlx-serve's notice for the adapted kernels:

    MIT License. Copyright (c) 2026 David Dalcu
    Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
    documentation files (the "Software"), to deal in the Software without restriction, including without limitation
    the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and
    to permit persons to whom the Software is furnished to do so, subject to the following conditions:
    The above copyright notice and this permission notice shall be included in all copies or substantial portions
    of the Software.
    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
    THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
    CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
    IN THE SOFTWARE.
"""
import os

import mlx.core as mx

MAX_SLOTS = 8          # a slot's K and V take a buffer each; a Metal kernel binds at most 31
KB = 128               # keys per threadgroup (128-512 measured within 1%; mlx-serve uses 256)
NSG = 8                # SIMD groups per threadgroup
ENABLED = not os.environ.get("NUEMAC_NO_SLOTS_KERNEL")

_PASS1 = """
    constexpr float LOG2E = 1.4426950408889634f;
    constexpr int G = HQ / HKV;
    constexpr int DL = (D + 31) / 32;
    const int slot = threadgroup_position_in_grid.z;
    const int kvh = threadgroup_position_in_grid.y;
    const int split = threadgroup_position_in_grid.x;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const device T* kp[8] = {k0, k1, k2, k3, k4, k5, k6, k7};
    const device T* vp[8] = {v0, v1, v2, v3, v4, v5, v6, v7};
    const int S = KEYS[slot];
    const size_t cap = size_t(CAPS[slot]);
    const int first = WINDOW > 0 ? max(0, S - WINDOW) : 0;
    const int kbeg = first + split * KB;
    const int kend = min(S, kbeg + KB);
    const float qscale = scale[0] * LOG2E;
    float qv[G][DL], acc[G][DL], m[G], l[G];
    for (int g = 0; g < G; g++) {
        const int h = kvh * G + g;
        for (int i = 0; i < DL; i++) {
            const int c = lane * DL + i;
            qv[g][i] = c < D ? float(q[(slot * HQ + h) * D + c]) * qscale : 0.0f;
            acc[g][i] = 0.0f;
        }
        m[g] = -1e30f;                 // a finite floor: a split with no visible key joins at weight 0
        l[g] = 0.0f;
        if (split == 0 && sg == 0) {
            m[g] = float(sinks[slot * HQ + h]) * LOG2E;
            l[g] = 1.0f;
        }
    }
    const device T* kh = kp[slot] + size_t(kvh) * cap * D;
    const device T* vh = vp[slot] + size_t(kvh) * cap * D;
    for (int j = kbeg + int(sg); j < kend; j += NSG) {
        float kd[DL], vd[DL];
        for (int i = 0; i < DL; i++) {
            const int c = lane * DL + i;
            kd[i] = c < D ? float(kh[size_t(j) * D + c]) : 0.0f;
            vd[i] = c < D ? float(vh[size_t(j) * D + c]) : 0.0f;
        }
        const int d = S - 1 - j;
        for (int g = 0; g < G; g++) {
            float dot = 0.0f;
            for (int i = 0; i < DL; i++) dot += qv[g][i] * kd[i];
            float s = simd_sum(dot);
            if (d < R) s += float(rel[(slot * HQ + kvh * G + g) * R + d]) * LOG2E;
            const float mn = max(m[g], s);
            const float f = fast::exp2(m[g] - mn);
            const float p = fast::exp2(s - mn);
            l[g] = l[g] * f + p;
            for (int i = 0; i < DL; i++) acc[g][i] = acc[g][i] * f + p * vd[i];
            m[g] = mn;
        }
    }
    threadgroup float sm[NSG][G];
    threadgroup float sl[NSG][G];
    threadgroup float so[NSG][G][DL * 32];
    for (int g = 0; g < G; g++) {
        if (lane == 0) {
            sm[sg][g] = m[g];
            sl[sg][g] = l[g];
        }
        for (int i = 0; i < DL; i++) so[sg][g][lane * DL + i] = acc[g][i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (int(sg) < G) {                 // SIMD group g joins query head g's partials
        const int g = sg;
        float mx_ = -INFINITY;
        for (int t = 0; t < NSG; t++) mx_ = max(mx_, sm[t][g]);
        float ls = 0.0f;
        float os[DL];
        for (int i = 0; i < DL; i++) os[i] = 0.0f;
        for (int t = 0; t < NSG; t++) {
            const float w = fast::exp2(sm[t][g] - mx_);
            ls += w * sl[t][g];
            for (int i = 0; i < DL; i++) os[i] += w * so[t][g][lane * DL + i];
        }
        const size_t at = (size_t(split) * NS + slot) * HQ + kvh * G + g;
        for (int i = 0; i < DL; i++) {
            const int c = lane * DL + i;
            if (c < D) PART[at * D + c] = os[i];
        }
        if (lane == 0) {
            ML[at * 2] = mx_;
            ML[at * 2 + 1] = ls;
        }
    }
"""

_PASS2 = """
    const uint d = thread_position_in_grid.x;
    const uint lh = thread_position_in_grid.y;
    float m = -INFINITY;
    for (int s = 0; s < SPLITS; s++) m = max(m, ML[(s * LH + lh) * 2]);
    float num = 0.0f, den = 0.0f;
    for (int s = 0; s < SPLITS; s++) {
        const float w = precise::exp2(ML[(s * LH + lh) * 2] - m);
        num += w * PART[(s * LH + lh) * D + d];
        den += w * ML[(s * LH + lh) * 2 + 1];
    }
    OUT[lh * D + d] = T(num / den);
"""

# The same read over the global visits' 8-bit affine caches (Q8GlobalKVCache: codes [1, H, cap, D/4] u32, scales and
# biases [1, H, cap, D/64]). A lane owns one code word (4 columns); its group is lane / 16. Six buffers a slot, so four
# slots a launch (24 + q, rel, sinks, P, scale and the two outputs: 31).
_Q8_PASS1 = """
    constexpr float LOG2E = 1.4426950408889634f;
    constexpr int G = HQ / HKV;
    const int slot = threadgroup_position_in_grid.z;
    const int kvh = threadgroup_position_in_grid.y;
    const int split = threadgroup_position_in_grid.x;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const uint grp = lane >> 4;
    const device uint* kqp[4] = {kq0, kq1, kq2, kq3};
    const device uint* vqp[4] = {vq0, vq1, vq2, vq3};
    const device T* ksp[4] = {ks0, ks1, ks2, ks3};
    const device T* kbp[4] = {kb0, kb1, kb2, kb3};
    const device T* vsp[4] = {vs0, vs1, vs2, vs3};
    const device T* vbp[4] = {vb0, vb1, vb2, vb3};
    const int S = P[1 + slot];             // P: keys a threadgroup, then each slot's keys, then its capacity
    const size_t cap = size_t(P[5 + slot]);
    const int kb_ = P[0];
    const int kbeg = split * kb_;
    const int kend = min(S, kbeg + kb_);
    const float qscale = scale[0] * LOG2E;
    float qv[G][4], qsum[G], acc[G][4], accb[G], m[G], l[G];
    for (int g = 0; g < G; g++) {
        const int h = kvh * G + g;
        qsum[g] = 0.0f;
        for (int i = 0; i < 4; i++) {
            qv[g][i] = float(q[(slot * HQ + h) * 128 + lane * 4 + i]) * qscale;
            qsum[g] += qv[g][i];
            acc[g][i] = 0.0f;
        }
        accb[g] = 0.0f;
        m[g] = -1e30f;
        l[g] = 0.0f;
        if (split == 0 && sg == 0) {
            m[g] = float(sinks[slot * HQ + h]) * LOG2E;
            l[g] = 1.0f;
        }
    }
    const size_t hrow = size_t(kvh) * cap;
    const device uint* kq = kqp[slot];
    const device uint* vq = vqp[slot];
    const device T* ksc = ksp[slot];
    const device T* kbi = kbp[slot];
    const device T* vsc = vsp[slot];
    const device T* vbi = vbp[slot];
    for (int j = kbeg + int(sg); j < kend; j += NSG) {
        const size_t r = hrow + size_t(j);
        const uint kw = kq[r * 32 + lane], vw = vq[r * 32 + lane];
        const float a = float(ksc[r * 2 + grp]), b = float(kbi[r * 2 + grp]);
        const float va = float(vsc[r * 2 + grp]), vb = float(vbi[r * 2 + grp]);
        const float c0 = float(kw & 0xffu), c1 = float((kw >> 8) & 0xffu);
        const float c2 = float((kw >> 16) & 0xffu), c3 = float(kw >> 24);
        const float v0 = float(vw & 0xffu), v1 = float((vw >> 8) & 0xffu);
        const float v2 = float((vw >> 16) & 0xffu), v3 = float(vw >> 24);
        const int d = S - 1 - j;
        for (int g = 0; g < G; g++) {
            float s = simd_sum(a * (qv[g][0] * c0 + qv[g][1] * c1 + qv[g][2] * c2 + qv[g][3] * c3) + b * qsum[g]);
            if (d < R) s += float(rel[(slot * HQ + kvh * G + g) * R + d]) * LOG2E;
            const float mn = max(m[g], s);
            const float f = fast::exp2(m[g] - mn);
            const float p = fast::exp2(s - mn);
            l[g] = l[g] * f + p;
            accb[g] = accb[g] * f + p * vb;
            const float pv = p * va;
            acc[g][0] = acc[g][0] * f + pv * v0;
            acc[g][1] = acc[g][1] * f + pv * v1;
            acc[g][2] = acc[g][2] * f + pv * v2;
            acc[g][3] = acc[g][3] * f + pv * v3;
            m[g] = mn;
        }
    }
    threadgroup float sm[NSG][G];
    threadgroup float sl[NSG][G];
    threadgroup float so[NSG][G][128];
    for (int g = 0; g < G; g++) {
        if (lane == 0) {
            sm[sg][g] = m[g];
            sl[sg][g] = l[g];
        }
        for (int i = 0; i < 4; i++) so[sg][g][lane * 4 + i] = acc[g][i] + accb[g];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (int(sg) < G) {
        const int g = sg;
        float mx_ = -INFINITY;
        for (int t = 0; t < NSG; t++) mx_ = max(mx_, sm[t][g]);
        float ls = 0.0f;
        float os[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (int t = 0; t < NSG; t++) {
            const float w = fast::exp2(sm[t][g] - mx_);
            ls += w * sl[t][g];
            for (int i = 0; i < 4; i++) os[i] += w * so[t][g][lane * 4 + i];
        }
        const size_t at = (size_t(split) * NS + slot) * HQ + kvh * G + g;
        for (int i = 0; i < 4; i++) PART[at * 128 + lane * 4 + i] = os[i];
        if (lane == 0) {
            ML[at * 2] = mx_;
            ML[at * 2 + 1] = ls;
        }
    }
"""
Q8_SLOTS = 4
MAX_SPLITS = 64        # long caches take more keys a threadgroup, so the partials stay small

_q8n = [f"{p}{i}" for p in ("kq", "vq", "ks", "kb", "vs", "vb") for i in range(Q8_SLOTS)]
_q8_pass1 = mx.fast.metal_kernel(
    name="nuemac_inkling2_slots_q8",
    input_names=["q", *_q8n, "rel", "sinks", "P", "scale"],
    output_names=["PART", "ML"],
    source=_Q8_PASS1,
)

_ks = [f"k{i}" for i in range(MAX_SLOTS)]
_vs = [f"v{i}" for i in range(MAX_SLOTS)]
_pass1 = mx.fast.metal_kernel(
    name="nuemac_inkling2_slots_window",
    input_names=["q", *_ks, *_vs, "rel", "sinks", "KEYS", "CAPS", "scale"],
    output_names=["PART", "ML"],
    source=_PASS1,
)
_pass2 = mx.fast.metal_kernel(
    name="nuemac_inkling2_slots_combine",
    input_names=["PART", "ML"],
    output_names=["OUT"],
    source=_PASS2,
)


def window_slots_attend(q, kvs, rel, sink, scale, window):
    """q [B, hq, 1, D]; kvs: B WindowKVCaches that already hold this step's key (their last); rel [B, 1, hq, R];
    sink [B, 1, hq]. Returns [B, hq, 1, D] in q's dtype, what oMLX's banded decode attention returns row by row."""
    B, hq, _, D = q.shape
    outs = []
    for a in range(0, B, MAX_SLOTS):
        group = kvs[a:a + MAX_SLOTS]
        ns = len(group)
        hkv = group[0].keys.shape[1]
        keys = [kv.n for kv in group]
        seen = max(min(n, window) if window else n for n in keys)
        splits = max(1, (seen + KB - 1) // KB)
        pad = lambda xs: xs + [xs[0]] * (MAX_SLOTS - ns)   # noqa: E731  unused slots bind slot 0's buffers
        part, ml = _pass1(
            inputs=[q[a:a + ns].reshape(ns * hq * D), *pad([kv.keys for kv in group]),
                    *pad([kv.values for kv in group]), rel[a:a + ns].reshape(ns * hq * rel.shape[-1]),
                    sink[a:a + ns].astype(q.dtype).reshape(ns * hq), mx.array(keys, dtype=mx.int32),
                    mx.array([kv.keys.shape[2] for kv in group], dtype=mx.int32),
                    mx.array([scale], dtype=mx.float32)],
            template=[("T", q.dtype), ("D", D), ("HQ", hq), ("HKV", hkv), ("R", rel.shape[-1]), ("NS", ns),
                      ("NSG", NSG), ("KB", KB), ("WINDOW", window)],
            grid=(splits * 32 * NSG, hkv, ns),
            threadgroup=(32 * NSG, 1, 1),
            output_shapes=[(splits, ns * hq, D), (splits, ns * hq, 2)],
            output_dtypes=[mx.float32, mx.float32],
        )
        (out,) = _pass2(
            inputs=[part, ml],
            template=[("T", q.dtype), ("D", D), ("LH", ns * hq), ("SPLITS", splits)],
            grid=(D, ns * hq, 1),
            threadgroup=(D, 1, 1),
            output_shapes=[(ns * hq * D,)],
            output_dtypes=[q.dtype],
        )
        outs.append(out.reshape(ns, hq, 1, D))
    return outs[0] if len(outs) == 1 else mx.concatenate(outs, axis=0)


def global_slots_attend(q, kvs, rel, sink, scale):
    """q [B, hq, 1, 128]; kvs: B Q8GlobalKVCaches that already hold this step's key; rel [B, 1, hq, R];
    sink [B, 1, hq]. Returns [B, hq, 1, 128] in q's dtype."""
    B, hq, _, D = q.shape
    assert D == 128
    outs = []
    for a in range(0, B, Q8_SLOTS):
        group = kvs[a:a + Q8_SLOTS]
        ns = len(group)
        hkv = group[0].kq.shape[1]
        keys = [kv.offset for kv in group]
        kb = max(KB, -(-max(keys) // MAX_SPLITS))
        kb = -(-kb // NSG) * NSG
        splits = max(1, -(-max(keys) // kb))
        pad = lambda xs: xs + [xs[0]] * (Q8_SLOTS - ns)   # noqa: E731
        bufs = []
        for name in ("kq", "vq", "ks", "kb", "vs", "vb"):
            bufs += pad([getattr(kv, name) for kv in group])
        part, ml = _q8_pass1(
            inputs=[q[a:a + ns].reshape(ns * hq * D), *bufs, rel[a:a + ns].reshape(ns * hq * rel.shape[-1]),
                    sink[a:a + ns].astype(q.dtype).reshape(ns * hq),
                    mx.array([kb] + pad(keys) + pad([kv.max_len for kv in group]), dtype=mx.int32),
                    mx.array([scale], dtype=mx.float32)],
            template=[("T", q.dtype), ("HQ", hq), ("HKV", hkv), ("R", rel.shape[-1]), ("NS", ns), ("NSG", NSG)],
            grid=(splits * 32 * NSG, hkv, ns),
            threadgroup=(32 * NSG, 1, 1),
            output_shapes=[(splits, ns * hq, D), (splits, ns * hq, 2)],
            output_dtypes=[mx.float32, mx.float32],
        )
        (out,) = _pass2(
            inputs=[part, ml],
            template=[("T", q.dtype), ("D", D), ("LH", ns * hq), ("SPLITS", splits)],
            grid=(D, ns * hq, 1),
            threadgroup=(D, 1, 1),
            output_shapes=[(ns * hq * D,)],
            output_dtypes=[q.dtype],
        )
        outs.append(out.reshape(ns, hq, 1, D))
    return outs[0] if len(outs) == 1 else mx.concatenate(outs, axis=0)
