"""Split-K ("flash decoding") Metal kernels: one query token against an 8-bit
affine (group 64) K/V cache, with Inkling's relative-position bias and sink.

Pass 1: each threadgroup takes CHUNK keys of one KV head; each of its SG
simdgroups runs an online softmax over a contiguous slice and writes a partial
(max, sum, acc[128]) for the REP query heads that share the KV head.
- Each lane owns 4 of the 128 dims, so a key row is one coalesced 128-byte read
  for K and one for V.
- The next key's loads are issued before the current key's math (prefetch 1).
- The running sum is rescaled only when a key beats the running max; the score
  comes from simd_sum, so the branch is uniform across the simdgroup.
- V's per-row bias is accumulated once per key, not once per dim.
Pass 2: one thread per (head, dim) merges the partials and the sink.

On a base M4 (16 GB) at 96K tokens, 8 global layer visits read 1.71 GB per
token in 20.4 ms (~84 GB/s); a plain read of the same bytes takes 15.4 ms.
The inkling2 family uses it above FLASH_MIN cached keys (see __init__.py).
"""
import mlx.core as mx

D = 128

_PASS1 = """
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint chunk = threadgroup_position_in_grid.x;
    const uint h = threadgroup_position_in_grid.y;
    const int n = params[0];
    const int maxlen = params[1];
    const int P = params[2];
    const float scale = fparams[0];
    const uint g = lane >> 4;

    float qv[REP][4];
    float qs[REP];
    float m[REP], l[REP], accb[REP];
    float acc[REP][4];
    for (int r = 0; r < REP; r++) {
        qs[r] = 0.0f;
        for (int k = 0; k < 4; k++) {
            qv[r][k] = float(q[(h * REP + r) * 128 + lane * 4 + k]) * scale;
            qs[r] += qv[r][k];
            acc[r][k] = 0.0f;
        }
        m[r] = -1e30f;
        l[r] = 0.0f;
        accb[r] = 0.0f;
    }
    const int per = CHUNK / SG;
    const int start = (int)chunk * CHUNK + (int)sg * per;
    const int end = min(start + per, n);
    const size_t hrow = (size_t)h * (size_t)maxlen;
    // software pipeline: the loads for key j+PF are issued before key j's math
    uint pkw[PF], pvw[PF];
    float pks[PF], pkb[PF], pvs[PF], pvb[PF];
    for (int i = 0; i < PF; i++) {
        const size_t row = hrow + (size_t)min(start + i, max(end - 1, start));
        pkw[i] = kq[row * 32 + lane]; pvw[i] = vq[row * 32 + lane];
        pks[i] = float(ks[row * 2 + g]); pkb[i] = float(kb[row * 2 + g]);
        pvs[i] = float(vs[row * 2 + g]); pvb[i] = float(vb[row * 2 + g]);
    }
    for (int j = start; j < end; j++) {
        const uint kw = pkw[0], vw = pvw[0];
        const float ksc = pks[0], kbi = pkb[0], vsc = pvs[0], vbi = pvb[0];
        for (int i = 0; i + 1 < PF; i++) {
            pkw[i] = pkw[i + 1]; pvw[i] = pvw[i + 1];
            pks[i] = pks[i + 1]; pkb[i] = pkb[i + 1]; pvs[i] = pvs[i + 1]; pvb[i] = pvb[i + 1];
        }
        {
            const size_t row = hrow + (size_t)min(j + PF, end - 1);
            pkw[PF - 1] = kq[row * 32 + lane]; pvw[PF - 1] = vq[row * 32 + lane];
            pks[PF - 1] = float(ks[row * 2 + g]); pkb[PF - 1] = float(kb[row * 2 + g]);
            pvs[PF - 1] = float(vs[row * 2 + g]); pvb[PF - 1] = float(vb[row * 2 + g]);
        }
        const float c0 = float(kw & 0xffu), c1 = float((kw >> 8) & 0xffu);
        const float c2 = float((kw >> 16) & 0xffu), c3 = float(kw >> 24);
        const float v0 = float(vw & 0xffu), v1 = float((vw >> 8) & 0xffu);
        const float v2 = float((vw >> 16) & 0xffu), v3 = float(vw >> 24);
        const int d = n - 1 - j;
        for (int r = 0; r < REP; r++) {
            float s = simd_sum(ksc * (qv[r][0] * c0 + qv[r][1] * c1 + qv[r][2] * c2 + qv[r][3] * c3)
                               + kbi * qs[r]);
            if (d < RELN) s += float(rel[(h * REP + r) * RELN + d]);
            if (s > m[r]) {                       // uniform across the simdgroup
                const float corr = fast::exp(m[r] - s);
                l[r] *= corr; accb[r] *= corr;
                acc[r][0] *= corr; acc[r][1] *= corr; acc[r][2] *= corr; acc[r][3] *= corr;
                m[r] = s;
            }
            const float p = fast::exp(s - m[r]);
            l[r] += p;
            accb[r] += p * vbi;
            const float pv = p * vsc;
            acc[r][0] += pv * v0; acc[r][1] += pv * v1; acc[r][2] += pv * v2; acc[r][3] += pv * v3;
        }
    }
    const size_t pidx = (size_t)chunk * SG + sg;
    for (int r = 0; r < REP; r++) {
        const size_t o = ((size_t)h * (size_t)P + pidx) * REP + r;
        if (lane == 0) { pm[o] = m[r]; pl[o] = l[r]; }
        for (int k = 0; k < 4; k++) pacc[o * 128 + lane * 4 + k] = acc[r][k] + accb[r];
    }
"""

_PASS2 = """
    const uint d = thread_position_in_threadgroup.x;
    const uint head = threadgroup_position_in_grid.y;
    const uint h = head / REP, r = head % REP;
    const int P = params[2];
    const float sk = float(sink[head]);
    float M = sk;
    for (int p = 0; p < P; p++) M = max(M, pm[((size_t)h * P + p) * REP + r]);
    float den = metal::exp(sk - M), num = 0.0f;
    for (int p = 0; p < P; p++) {
        const size_t o = ((size_t)h * P + p) * REP + r;
        const float w = metal::exp(pm[o] - M);
        den += w * pl[o];
        num += w * pacc[o * 128 + d];
    }
    out[head * 128 + d] = static_cast<bfloat16_t>(num / den);
"""


_pass1 = mx.fast.metal_kernel(
    name="inkling_q8_decode_pass1",
    input_names=["q", "kq", "ks", "kb", "vq", "vs", "vb", "rel", "params", "fparams"],
    output_names=["pm", "pl", "pacc"],
    source=_PASS1,
)
_pass2 = mx.fast.metal_kernel(
    name="inkling_q8_decode_pass2",
    input_names=["pm", "pl", "pacc", "sink", "params"],
    output_names=["out"],
    source=_PASS2,
)


def _chunk_for(n):
    # base M4: 1024 is best through ~48K keys, 2048 beyond
    return 1024 if n <= 49152 else 2048


def q8_flash_decode(q, cache, rel, sink, scale, sg=4, pf=1):
    """q: [1, hq, 1, 128] bf16; cache: Q8GlobalKVCache holding n keys (after
    append); rel: [1, 1, hq, R]; sink: [1, 1, hq]. Returns [1, hq, 1, 128]."""
    hq = q.shape[1]
    hkv = cache.kq.shape[1]
    rep = hq // hkv
    n = cache.offset
    reln = rel.shape[-1]
    chunk = _chunk_for(n)
    chunks = (n + chunk - 1) // chunk
    P = chunks * sg
    params = mx.array([n, cache.max_len, P], dtype=mx.int32)
    fparams = mx.array([scale], dtype=mx.float32)
    pm, pl, pacc = _pass1(
        inputs=[q.reshape(hq * D), cache.kq, cache.ks, cache.kb, cache.vq, cache.vs, cache.vb,
                rel[0, 0], params, fparams],
        template=[("REP", rep), ("RELN", reln), ("CHUNK", chunk), ("SG", sg), ("PF", pf)],
        grid=(chunks * 32 * sg, hkv, 1),
        threadgroup=(32 * sg, 1, 1),
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
