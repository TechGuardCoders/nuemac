"""Inkling 2's loop-controller mixing for one decode row, as two Metal kernels.

Reference (oMLX `_mix_messages`, with Nuemac's bf16 controllers): for each controlled input i < n,
    z      = [unit_rms(x_i), unit_rms(m_i)]                       rounded to bf16
    low    = bf16(bf16(z_x @ down[i, 0]) + bf16(z_m @ down[i, 1]))   [64]
    gates  = bf16(low @ gain_forget[i])                            [2H]
    out_i  = (1 - tanh(gates[H:])) * x_i + tanh(gates[:H]) * m_i
As MLX ops that is ~12 small kernels per call and 48 calls per decode token (~2.8 ms on an M4 Pro).
Kernel A: one threadgroup per (input, half, slice of H) writes fp32 partial dot products for the 64 outputs.
Kernel B: one threadgroup per (input, block of 256 channels) sums the partials with the reference's bf16 roundings,
then computes its channels' gate and forget values and the mix.
"""
import mlx.core as mx

H_SLICES = 8      # slices of the hidden size in kernel A
RANK = 64         # controller rank (loop_config gate_rank)
TG_A = 256        # 64 outputs x 4 lanes over h
TG_B = 256

_A = """
    // grid: (TG_A * H_SLICES, 2 halves, n inputs)
    const uint t = thread_position_in_threadgroup.x;
    const uint slice = threadgroup_position_in_grid.x;
    const uint half_ = threadgroup_position_in_grid.y;
    const uint i = threadgroup_position_in_grid.z;
    const device float* src = (half_ == 0 ? x : m) + (size_t)i * H;
    threadgroup float red[TG_A];
    // unit RMS of the whole row (each threadgroup recomputes it; the row is 9 KB)
    float ss = 0.0f;
    for (uint h = t; h < H; h += TG_A) { const float v = src[h]; ss += v * v; }
    red[t] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = TG_A / 2; s > 0; s >>= 1) {
        if (t < s) red[t] += red[t + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float inv = metal::rsqrt(red[0] / float(H) + 1e-6f);
    // partial dots: thread (k, p) sums h = h0 + p, h0 + p + 4, ...
    const uint k = t % RANK, p = t / RANK;
    const uint per = H / H_SLICES;
    const uint h0 = slice * per;
    const device bfloat16_t* w = down + ((size_t)(i * 2 + half_) * H) * RANK;
    float acc = 0.0f;
    for (uint h = h0 + p; h < h0 + per; h += TG_A / RANK) {
        const float zv = float(static_cast<bfloat16_t>(src[h] * inv));   // the reference rounds z to bf16
        acc += zv * float(w[(size_t)h * RANK + k]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    red[t] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (p == 0) {
        float s = 0.0f;
        for (uint q = 0; q < TG_A / RANK; q++) s += red[q * RANK + k];
        part[(((size_t)i * 2 + half_) * H_SLICES + slice) * RANK + k] = s;
    }
"""

_B = """
    // grid: (TG_B * ceil(H / TG_B), 1, n inputs)
    const uint t = thread_position_in_threadgroup.x;
    const uint blk = threadgroup_position_in_grid.x;
    const uint i = threadgroup_position_in_grid.z;
    threadgroup float low[RANK];
    if (t < RANK) {
        float a = 0.0f, b = 0.0f;
        for (uint s = 0; s < H_SLICES; s++) {
            a += part[(((size_t)i * 2 + 0) * H_SLICES + s) * RANK + t];
            b += part[(((size_t)i * 2 + 1) * H_SLICES + s) * RANK + t];
        }
        // bf16(bf16(a) + bf16(b)), as the reference's two bf16 matmul outputs summed then rounded
        low[t] = float(static_cast<bfloat16_t>(float(static_cast<bfloat16_t>(a)) + float(static_cast<bfloat16_t>(b))));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint h = blk * TG_B + t;
    if (h >= H) return;
    const device bfloat16_t* gf = gain_forget + (size_t)i * RANK * 2 * H;
    float g = 0.0f, f = 0.0f;
    for (uint k = 0; k < RANK; k++) {
        const float l = low[k];
        g += l * float(gf[(size_t)k * 2 * H + h]);
        f += l * float(gf[(size_t)k * 2 * H + H + h]);
    }
    const float gain = metal::precise::tanh(float(static_cast<bfloat16_t>(g)));
    const float forget = metal::precise::tanh(float(static_cast<bfloat16_t>(f)));
    const size_t o = (size_t)i * H + h;
    out[o] = (1.0f - forget) * x[o] + gain * m[o];
"""

_ka = mx.fast.metal_kernel(name="inkling_mix_low", input_names=["x", "m", "down"], output_names=["part"], source=_A)
_kb = mx.fast.metal_kernel(name="inkling_mix_gates", input_names=["x", "m", "gain_forget", "part"],
                           output_names=["out"], source=_B)


def mix_messages_row(x, message, down, gain_forget):
    """x, message: [n, 1, 1, H] fp32; down: [n, 2, H, 64] bf16; gain_forget: [n, 64, 2H] bf16."""
    n, b, length, H = x.shape
    xf = x.reshape(n * H)
    mf = message.astype(mx.float32).reshape(n * H)
    tmpl = [("H", H), ("RANK", RANK), ("H_SLICES", H_SLICES), ("TG_A", TG_A), ("TG_B", TG_B)]
    (part,) = _ka(inputs=[xf, mf, down], template=tmpl, grid=(TG_A * H_SLICES, 2, n), threadgroup=(TG_A, 1, 1),
                  output_shapes=[(n * 2 * H_SLICES * RANK,)], output_dtypes=[mx.float32])
    (out,) = _kb(inputs=[xf, mf, gain_forget, part], template=tmpl,
                 grid=(TG_B * ((H + TG_B - 1) // TG_B), 1, n), threadgroup=(TG_B, 1, 1),
                 output_shapes=[(n * H,)], output_dtypes=[mx.float32])
    return out.reshape(n, b, length, H)
