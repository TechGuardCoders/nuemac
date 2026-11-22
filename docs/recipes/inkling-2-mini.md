# Inkling 2 Mini (`inkling2_loop`) on a 16 GB Mac mini

Nuemac's first family: Thinking Machines' Inkling 2 Mini at 96K context on a base Mac mini (M4, 10-core GPU,
16 GB, macOS 26.5). This page explains why each choice in [`models/inkling-2-mini.conf`](../../models/inkling-2-mini.conf)
and [`engine/families/inkling2/`](../../engine/families/inkling2/) is what it is. Measured 2026-11-20/21. The
method: measure the ceiling first, change one thing per experiment, run a control on both sides, and stop after
three changes land inside the noise.

| at 96K context, same bench | **Nuemac** | llama.cpp b12600 (AtomicChat AD-IQ3_XXS) |
| --- | --- | --- |
| decode, short prompt | **25.9 tok/s** (0.4.0; 23.5 on 0.3.0) | 17.25 tok/s |
| prefill, 4K / 16K prompt | **186 / 173 tok/s** | 68 / 61 tok/s |
| 32K prompt: first token / decode | **202 s / 18.8 tok/s** | 690 s / 12.2 tok/s |
| 94K prompt: first token / decode | **11.8 min / 13.7 tok/s** | 42.6 min / 11.2 tok/s |
| known-answer questions | 12/12 | 12/12 |
| memory free under load | ~13% | ~1-2% |

## What decides the speed

- 24 layers; layers 6-17 run three times per token, so there are **48 layer passes**. 40 of them are sliding-window
  attention (1,024 tokens) and 8 are global, with 8 KV heads × 128 dims. Mixture of experts: 128 experts, 8
  active, expert width 1,152, hidden size 2,304. Vocabulary 201K with a soft-capped head.
- The weights read per token are 3.35 GB: 1.08 GB of experts (2-bit gate/up, 3-bit down), 1.78 GB of dense
  layers (8-bit attention projections) and a 0.49 GB head. At the ~111 GB/s this Mac actually reads, that is
  ~30 ms per token, a ~33 tok/s ceiling at this quantization.
- A short-context token takes ~43 ms on Nuemac: MoE ~15 ms, attention math ~3 ms, and dense layers, head and
  small ops ~25 ms. MLX's expert matvecs already run at 79-87 GB/s and the head at 98 GB/s. What's left is
  mostly bytes and small-op overhead, not kernel speed.

## Three fixes to the cache

MLX runs this looped mixture-of-experts model faster than llama.cpp: [oMLX](https://github.com/jundot/omlx)
0.8.2 decodes it at 24.8 tok/s against 17.25. Out of the box, though, oMLX fits only about 7K tokens of
context on 16 GB: a 10K-token prompt stops at its memory limit. The model weights take 10.2 GiB, which leaves
about 2.5 GiB, and the cache was using ~192 KB per token.

1. **Sliding layers keep a window, not the whole history.** Inkling runs 48 layer visits per token (layers
   6-17 three times). 40 of those visits are sliding-window attention over the last 1,024 tokens, but oMLX
   gives every visit a full-length cache. Nuemac gives the sliding visits a 1,024-key window and keeps full
   history only on the 8 global visits. The output is token-for-token identical (64 of 64 greedy tokens on a
   prompt longer than the window).
2. **The 8 global visits store 8-bit keys and values** (affine, group 64), allocated once at the full context:
   1.6 GiB at 96K instead of 3 GiB at bf16. bf16 does fit on paper; in practice it ran out of GPU memory with
   macOS and a few background apps running. Output is still identical to the bf16 reference on the same check.
3. **A decode kernel that reads the 8-bit cache directly.** Plain MLX ops (`quantized_matmul`) reach only
   ~23 GB/s on a single query against a long cache: 73 ms per token at 96K across the 8 visits. Nuemac's
   split-K ("flash decoding") Metal kernel, `engine/families/inkling2/q8attn.py`, spreads the keys over many threadgroups, one
   coalesced 128-byte read per key row, with Inkling's relative-position bias and attention sink folded in:
   20.3 ms at 96K (~84 GB/s), 7.1 ms at 32K. Nuemac uses it once the context passes 12K tokens; below
   that the plain path decodes faster end to end (see below).

| global attention, one decode step (8 visits) | 16K | 32K | 96K |
| --- | --- | --- | --- |
| bf16, MLX SDPA (oMLX's path) | 6.6 ms | 17.3 ms | 65.4 ms |
| 8-bit, `quantized_matmul` | 6.1 ms | 17.9 ms | 73.2 ms |
| 8-bit, Nuemac kernel, first version | 4.2 ms | 7.9 ms | 22.3 ms |
| **8-bit, Nuemac kernel, current** | **3.9 ms** | **7.1 ms** | **20.3 ms** |
| reading the same bytes once (floor) | | | 15.4 ms |

### Tuning the kernel

The M4's spec is 120 GB/s. A plain MLX reduction over the same 1.71 GB reads it in 15.4 ms (~111 GB/s), and
that is the practical floor. The first kernel took 22.3 ms (77 GB/s). Timing each piece showed where the rest
went. About 1.7 ms was the partial merge, run as ~8 small MLX ops per visit. The rest was the main loop
waiting: one key's loads were issued only after the previous key's reduce-max-exp chain, and every key paid
for a rescale.

| change | 96K | kept? |
| --- | --- | --- |
| merge and sink in one Metal kernel instead of MLX ops | -1.0 ms | yes |
| issue the next key's loads before this key's math (prefetch 1) | -1.4 ms | yes |
| rescale only when a key beats the running max (uniform branch) | -0.4 ms | yes |
| store K/V scale and bias interleaved per row: one 8-byte read, not four | -0.3 ms | no: within noise end to end |
| prefetch 2 keys | +0.4 ms | no (registers) |
| 16-byte reads, 4 keys per simdgroup | +1.8 ms | no (registers) |
| block softmax over 4-16 keys | +4 to +64 ms | no (registers spill) |
| chunk 512-4096 keys, 2-8 simdgroups | within 1 ms | chunk picked by depth |

The remaining ~5 ms to the floor is instruction latency: three cross-lane sums per key on a 10-core GPU. The
next steps there: reduce the three heads' sums together (fewer shuffles), let one lane own one key for the
scores (no cross-lane sums at all), or compute scores as small matrix multiplies (simdgroup matrices).

**Kernel time is not decode time.** The kernel is faster than the plain MLX path at every depth in isolation,
even with the 8 visits chained back to back as the model runs them (0.66 against 0.68 ms at 400 tokens).
Yet in an alternating end-to-end A/B on the server, using it at every depth cost ~0.8 tok/s at short context
(21.6 against 22.4), while gaining ~0.5 tok/s at 32K. Swapping one piece at a time in one loaded process
pinned it on the kernel's presence at short context: about 1.4 ms per token, more than its own run time.
So Nuemac switches to it above 12K tokens. With that threshold the end-to-end A/B is neutral: short
context 22.36 / 22.41 against 22.50 / 22.38, and 32K decode 17.2 / 17.4 against 17.3 / 17.3. The kernel's
0.8 ms saving at 32K is inside run-to-run noise. Lesson: measure the whole token, A/B alternating, before
trusting a kernel benchmark. The one variant that decoded faster at 32K end to end (18.0 / 18.3) was the
packed-scale cache, which issues fewer small write operations per token. That points at per-operation
overhead, which is next on the list.

## Fewer bytes per token

Once the kernels were near the floor, the lever was bytes. Profiling a short-context token by ablation (one
component off per process) and chained microbenchmarks:

| part (48 passes, 1 token) | time | effective speed |
| --- | --- | --- |
| expert gate+up, 2-bit (`gather_qmm`) | 8.1 ms | 79 GB/s |
| expert down, 3-bit | 5.1 ms | 87 GB/s |
| router + activation + combine | 1.9 ms | |
| output head, 8-bit, 0.49 GB | 5.0 ms | 98 GB/s |

Two trims, each checked on its own:
- **Loop controllers in bf16.** oMLX upcast these bf16 weights to fp32 copies, ~0.23 GB read per token. Nuemac
  keeps them bf16 and runs their two small matmuls in bf16, with fp32 accumulation and one rounding, as the
  reference does. Greedy tokens were identical on two prompts. Decode +0.4 tok/s.
- **Output head at 6 bits**, requantized at load (`--head-bits`). Measured by feeding the same hidden states
  through both heads, 3,072 positions of real text:

| head | size | mean KL vs 8-bit | top-1 agreement | decode |
| --- | --- | --- | --- | --- |
| 8-bit requantized to itself (noise floor) | 0.49 GB | 0.00008 | 99.5% | |
| **6-bit (default)** | 0.38 GB | 0.00027 | 98.9% | +0.7 tok/s |
| 5-bit | 0.32 GB | 0.00075 | 98.6% | |
| 4-bit | 0.26 GB | 0.0032 | 96.9% | +1.4 tok/s |

For scale, the whole model's quantization error is on the order of KL 0.05-0.1.

End-to-end A/B, alternating, two runs each: short-context decode 22.39 / 22.45 -> **23.46 / 23.53 tok/s**, 32K
decode 17.4 / 18.0 -> 18.1 / 18.6. Quality check 12/12 on every run. Skipping the per-pass conv-state copies on
1-token steps (192 small ops per token) gave only +0.1 tok/s and was left out.

## Why a separate server

Why a separate server rather than a patch to oMLX's: oMLX's batch engine merges every request's cache into
mlx-lm batch caches and its SSD prefix cache slices caches by token position, and neither fits a window cache.
Nuemac gives each sequence its own window and global caches and batches sequences itself (see
[concurrent requests](#concurrent-requests-050)), keeps the cache between turns of a conversation (a new turn only processes its new tokens), and speaks both llama-server's API (`/health`,
`/completion` with `timings`) and OpenAI's (`/v1/chat/completions` with streaming, `reasoning_content` and tool
calls through oMLX's own Inkling output parser). Other gotchas on the way:

- MLX streams belong to the thread that created them, so the model loads and every request runs on one engine
  thread.
- Without `mx.set_wired_limit`, macOS compresses the weights under memory pressure and decode collapses.
- Prefill computes logits for the last position only. The 201K-token vocabulary makes a full-chunk logits
  tensor 0.8 GB at 1,024 tokens. Prefill runs in 256-token chunks.

The weights are mlx-community's `Inkling-2-Mini-oQ2.5e-mtp` (oMLX's mixed-precision oQ format, 10.9 GB on
disk). The drafter file in that repo is not downloaded: see [the drafter](#speculative-decoding-the-drafter).

---

The rest of this page is the llama.cpp build we compared against: how it fits, and what we learned on the way.
It installs from [inkling2-mini-mac-recipe](https://github.com/TechGuardCoders/inkling2-mini-mac-recipe).

## The model in one paragraph

Inkling 2 Mini is a mixture-of-experts model: 24 layers, 128 experts with 8 active per token, and a loop:
layers 6-18 run three times per token, so each token makes 48 layer passes. Attention is mostly sliding
window (1,024 tokens) with 4 global layers, so its KV cache is small: about 18.6 KB per token at q8_0, or 1.7
GiB for 96K tokens. Support in upstream llama.cpp arrived on 2026-11-20 (`inkling2_loop`, PR #32694,
release b12600).

## Which GGUF

| file | size | on llama.cpp b12600 + Metal |
| --- | --- | --- |
| **AtomicChat `Inkling-2-Mini-AD-IQ3_XXS`** | 12.58 GB | **works.** Perplexity 3.26 on our docs corpus, 12/12 known-answer questions |
| unsloth `Mainline_Support/Inkling-2-Mini-UD-IQ3_XXS` | 11.66 GB | **wrong output.** Perplexity 43.1; it loops in its reasoning; 0/12 |
| unsloth non-Mainline files | | do not load ("wrong number of tensors"); they are for unsloth's own fork |

The unsloth Mainline file keeps 20 of its 24 expert `down` tensors in NVFP4 with separate per-expert scale
tensors (weights up to |2688|). Metal has no NVFP4 kernels yet (PR #20456 is open), so those tensors run on the
CPU and decode drops to 9.8 tok/s. Converting just those 20 tensors to IQ4_NL put everything on the GPU at 18
tok/s, but the output was still garbage. The original file itself measures perplexity 43 on CPU, so the
problem is the file plus master, not the conversion. AtomicChat folds the scales into the weights (704 tensors
against 744) and uses IQ4_NL for every `down` tensor.

Lesson: **a speed benchmark will happily measure a broken model.** Our first harness's loop detector did not
catch it. `nuemac bench` ends with 12 known-answer questions for that reason.

## Memory: what actually fills 16 GB

The weights are 11.5 GiB and the 96K q8_0 KV cache is 1.7 GiB. The part people miss is the **compute buffer**:
about 13 MiB per token of micro-batch (`-ub`) at long context. At the llama.cpp default `-ub 512` that is 6.6
GiB, and the model never loads.

| `-ub` at 96K | llama.cpp's projection | result |
| --- | --- | --- |
| 64 | 14,492 / 14,499 MiB | too tight |
| 48 | 14,184 MiB | **runs out of GPU memory** mid-request (191 errors), despite 315 MiB "spare" |
| **32** | 13,875 MiB | **stable**, including a full 93,822-token prompt |

macOS also lets the GPU wire only about two thirds of memory by default (~10.7 GB on 16 GB). Raise it:

```bash
sudo sysctl iogpu.wired_limit_mb=14500
```

It resets at reboot; the installer offers a small boot job to keep it. Leaving ~1.5 GB for macOS works, but
only with nothing big running beside it: a Docker/Colima VM holding 10 GB caused Metal out-of-memory errors on
every request until we stopped it. Expect the system to report ~1-2% free memory under full load.

Context sizes: 128K with q8_0 KV does not fit with any `-ub`. 96K q8_0 is the largest that does.

## Speed: where the time goes

Ceiling from the bytes per token (weights read per emitted token: ~1.3 GB of experts, ~1.5 GB of dense layers
read on all 48 passes, 0.3 GB output head) and the M4's ~110 GB/s measured bandwidth: about 25 tok/s practical
at short context.

We measure about 18 tok/s, and the gap is not bandwidth. Profiling each kernel (`test-backend-ops perf`):
Q8_0 matrix-vector runs at 109 GB/s and IQ3_XXS at 82 GB/s (the IQ2/IQ3 kernels are compute-bound on this GPU).
Adding up the kernels gives ~32.5 ms per token; the measured step is ~55.5 ms. The other ~23 ms is per-operation
overhead: 1,500+ small operations per token across 48 looped layer passes. So **fewer bytes (smaller quants)
will not help much; fewer graph steps per token would**.

Things that made no difference or made it worse (5 controls at 17.9-18.1 tok/s):

| change | result |
| --- | --- |
| `-t 8` | 11% slower (`-t` also sets the Metal command-buffer threads) |
| `-t 2` | no change |
| `--load-mode none` (no mmap; replaces the removed `--no-mmap`) | no change, slower start |
| `-fa off` | refused: a quantized V cache requires flash attention, and f16 V does not fit |

## Speculative decoding (the drafter)

A draft model should cut graph steps. Inkling 2 Mini ships with one: Thinking Machines' DSpark head
(`mtp.safetensors`).

- **llama.cpp** (unsloth's `dspark-Inkling-2-Mini-Q8_0` on PR #32218, not merged): it does not fit at 96K
  (it needs ~985 MiB more; ~958 MiB is free at `-ub 32`). At 64K it accepts 1-4% of drafted tokens and decode
  falls to 6.6 tok/s. The likely cause: the draft's `target_layer_ids` (5, 23, 41, 48) count looped layer
  *visits* (48), not layers (24).
- **oMLX** ("Lightning MTP"): the drafter works, accepting 56-76% of drafted tokens, but it is not faster on a
  base M4. Checking two tokens at once costs ~1.8x one step, because the two tokens route to almost twice as
  many experts, and expert weights are what this chip's memory bandwidth is spent on. Measured: depth 1 gives
  21-24 tok/s; depth 2 gives 25.7 on code and 19.8 on prose; adaptive gives 21-25. Without the drafter it is
  24.8.

On the **M4 Pro** (stock oMLX, oQ3e, temperature 0, 400 tokens, one fresh server per setting):

| setting | essay | code | prose (TCP) | accepted | tokens per verify |
| --- | --- | --- | --- | --- | --- |
| drafter off | **41.6** | **41.4** | **40.8** | | |
| depth 1 | 33.2 | 38.2 | 32.3 | 51-76% | 1.52-1.77 |
| depth 2 | 29.6 | 40.5 | 31.2 | 48-75% | 1.73-2.35 |
| depth 3 | 26.8 | 38.6 | 27.7 | 49-72% | 1.86-2.65 |
| adaptive | 32.6 | 40.8 | 32.6 | | |

The drafter itself is cheap (under 1 ms per cycle). The verify pass is not: checking 2, 3 or 4 tokens costs
1.92x, 2.39x and 2.83x a decode step, so each setting needs more tokens per verify than it gets. Timing the
target alone in Nuemac (2K context) shows where:

| one target pass over L new tokens | L=1 | L=2 | L=3 | L=4 |
| --- | --- | --- | --- | --- |
| whole model | 32.0 ms | 52.7 | 64.9 | 77.8 |
| without the MoE | 22.0 | 34.9 | 38.2 | 41.3 |
| experts (difference) | 10.0 | 17.7 | 26.8 | 36.5 |

- Experts grow linearly, ~9 ms per token: with 128 experts and 8 per token, each extra token reads new experts.
- The rest jumps 13 ms from one token to two, then adds ~3 ms per token. Most of that jump is in the loop
  passes (layers 6-17's second and third visits): the loop-controller mixing and other small ops leave their
  single-token kernels. MLX's batched matmul is slowest at exactly two rows (42 us vs 14 us for the
  controller's down projection).
- Attention is a small part. A per-row path that reuses the single-token kernels (exact against sequential
  decode) took a 2-token verify from 53.3 to 50.1 ms.

Even with that jump removed, depth 1 would gain ~15% on code and nothing on prose, because prose accepts too
few drafts. So Nuemac does not use the DSpark head. It drafts from the context instead, where a check accepts
~14 tokens: see [copy drafting](#copy-drafting-060). ~25 tok/s at short context is close to this chip's bandwidth ceiling
for these weights; 30 would take a lower-bit (lossier) quant or a faster chip.

## Long context

At a full window the model still finds what you put in it: 3/3 needles at 38,472 tokens and 3/3 at 93,822
tokens. It is slow to read that much on this GPU: a 94K-token prompt took 42.6 minutes before the first word
(37 tok/s average prefill), and decode at that depth was 11.2 tok/s. Prompts up to a few thousand tokens start
in seconds.

## 24 GB M4 Pro: 256K context

On a Mac mini M4 Pro (16-core GPU, 24 GB, macOS 26.6) the same engine runs a larger quant with a longer window.
Nothing in the family changes; only the model conf does
([`models/inkling-2-mini-oq3e.conf`](../../models/inkling-2-mini-oq3e.conf)). Both candidates were measured in
one session with the same bench (2026-11-21):

| same bench, M4 Pro 24 GB | **oQ3e, 256K** (shipped) | oQ4e, 128K |
| --- | --- | --- |
| weights | 12.1 GiB | 15.0 GiB |
| decode, short prompt | **35.6 tok/s** | 34.5 tok/s |
| prefill, 4K / 16K prompt | 319 / 301 tok/s | 314 / 301 tok/s |
| 32K prompt: first token / decode | 115 s / **29.4 tok/s** | 116 s / 26.2 tok/s |
| 120K prompt: first token / decode | 9.2 min / **21.4 tok/s** | 9.3 min / 19.9 tok/s |
| fact planted at 32K and at 120K | found, found | found, found |
| 12 known-answer questions | 12/12 | 12/12 |
| system memory free under load | 17-19% | 14-15% |

- **oQ3e wins on every row.** It is smaller, so it decodes faster, and it leaves room for twice the context.
  The 12 questions and two needles did not separate the two quants; a larger quality set might.
- **Against the 16 GB build:** 1.5x decode (35.6 vs 23.5), 1.7x prefill, and 2.7x the context.
- **Stock oMLX was faster at short context** (39.9 tok/s, MTP off, 16K window) until 0.4.0; most of that gap was
  the server's decode loop, see [Per-token overhead](#per-token-overhead-040). Stock oMLX's full-history cache
  (~192 KB per token) would fit only ~45K tokens on 24 GB.
- The drafter does not pay on this chip either; see [Speculative decoding](#speculative-decoding-the-drafter).

## Per-token overhead (0.4.0)

At 0.3.2 an M4 Pro token took ~28 ms while its weights need ~12 ms to read at full bandwidth (3.16 GB: experts
1.34, attention projections 1.33, head 0.38, the rest 0.11). Where the other ~16 ms went, measured on the M4 Pro:

- **One token is a graph of 9,808 MLX nodes**: about half are free views, the rest ~3,600 GPU launches (1,180
  dtype casts, ~900 small fused element-wise kernels, 397 RMSNorms, 300 concatenates, 240 sums, 208 cache writes)
  and only ~420 matmuls. A dependent small op costs 4-5 us alone, but inside a pipelined token it costs ~1 us:
  removing ~400 of them saved ~0.3 ms.
- **The decode loop left the GPU idle.** It waited for each token before building the next step's graph, so the
  GPU sat idle for the ~2.8 ms Python takes to build it. Queueing the next step first (as mlx-lm does): server
  decode 35.9 -> 39.4 tok/s. This was most of the gap to stock oMLX.
- **A compiled decode step** (`engine/families/inkling2/fastpath.py`): fixed-shape caches let one `mx.compile`
  graph serve every token. The 40 window visits keep a 1,024-slot ring (slot = position mod window, bias looked
  up by distance); the 8 global visits write their 8-bit codes with a dynamic `slice_update`, updated in place,
  and the split-K kernel reads the key count from an array over a fixed grid per context band. Same greedy tokens
  as eager decoding, and a handoff between them mid-reply is exact. +1.8%.
- **Fused loop-controller mixing** (`mixkernel.py`): the mixing that feeds each looped pass its messages was ~12
  ops per call, 48 calls per token (2.8 ms). Two Metal kernels, with the reference's bf16 roundings: same greedy
  tokens, +4%.
- **Window attention as a split-K kernel** over the ring: same tokens, +0.5-1%.

| server decode, M4 Pro 24 GB, alternating A/B | tok/s |
| --- | --- |
| 0.3.2 | 35.9 |
| + queue the next step first | 39.4 |
| + compiled decode step | 40.1 |
| + fused mixing and ring-attention kernels | **42.1** |
| + 6-bit attention projections (`--attn-bits 6`, opt-in) | 43.7 |

Tried and not shipped:
- **A fused 3-bit expert kernel** (both expert projections, the SwiGLU and the routing sum in two kernels). Exact,
  but 181 us per visit against MLX's 156 us: MLX's `gather_qmm` already reads these 3-bit experts at 174-209 GB/s,
  and its loads pre-scale the inputs so codes dequantize without shifts. A faster one is a bigger project worth
  ~2 ms per token at best.
- **6-bit attention projections by default.** Over 2,048 teacher-forced tokens (this recipe's prose and the
  server's code), against stock 8-bit: re-rounding the 8-bit weights alone gives mean KL 0.0045 and 97.6% top-1
  agreement (a looped model amplifies small weight changes); 6-bit gives 0.0057 and 97.1%; 5-bit 0.0101 and 95.8%.
  6-bit still answers 12/12 and finds the 32K needle, but it is a quality trade, so it stays a flag.

## Concurrent requests (0.5.0)

Up to `--slots` requests (default 4) run at once. They join and leave one decode batch between steps, the
iteration-level scheduling vLLM uses, and long prompts prefill in 256-token chunks that take turns with decoding:

- **An engine thread owns every MLX call.** HTTP threads hand it a request and read back events (streamed text,
  done, error); a client that disconnects cancels its request.
- **Each sequence keeps its own caches**, sized to prompt + `max_tokens` in bands (4K, 8K, 16K ... up to the
  context). A request whose cache would not fit the memory budget waits for a running one to finish. Finished
  conversations' caches stay (`--keep-caches`, default 2) for prompt reuse while memory allows.
- **One request alone runs the compiled decode step**, so it is as fast as 0.4.0 and gives the same text. With two
  or more, a batched step runs the projections, experts and head for all rows at once, and every row's attention
  against its own cache in one kernel launch (`slots.py`, adapted from
  [mlx-serve](https://github.com/ddalcu/mlx-serve)'s Inkling 2 slots kernel, with an 8-bit variant for the global
  visits).

| short prompts, 200-token greedy replies (totals include prefill) | M4 Pro 24 GB (oQ3e) | M4 16 GB (oQ2.5e) |
| --- | --- | --- |
| 1 request | 42.4 tok/s | 25.5 tok/s |
| 2 at once, total | 45.7 tok/s (~24 each) | 29.1 tok/s (~15 each) |
| 4 at once, total | 60.4 tok/s (~16 each) | 36.3 tok/s (~9.5 each) |
| two quality checks at once | 12/12, 12/12 | 12/12, 12/12 |
| a long prompt prefilling while another reply streams | 39K tokens; longest gap 0.95 s | 39K tokens; longest gap 1.5 s |

Why 4 requests give ~1.4x, not 4x: a 4-row step on the M4 Pro takes 66 ms against 24 ms for one. Stubbing out
attention entirely saves only ~14% of it; the rest is expert weights. Each row picks its own 8 of 128 experts per
visit, so 4 rows read about 29 experts' weights a visit, ~3.7x one row's bytes. The slots kernel took the
attention part from 69.5 to 66.4 ms (-4.5%); the expert reads are bandwidth.

Greedy output is not batch-invariant: a row batched with others rounds differently than it does alone, so its
tokens can drift from the single-request text after a few hundred characters (the text stays coherent). Two
identical prompts sent together need not come back identical. vLLM and other batching servers behave the same way
by default. `--slots 1` restores one request at a time; `NUEMAC_NO_SLOTS_KERNEL=1` falls back to per-row attention.

## Copy drafting (0.6.0)

oMLX 0.8.3 showed what does pay on this chip: copying from the context. When a reply repeats something already in
the prompt or the reply (a rewritten file, an edit, quoted code), the next tokens can be read off the earlier copy
and checked in one forward pass. A 16-row check costs ~9 decode steps here (each row routes to its own experts),
so it pays only when nearly all of a draft is right. Stock oMLX 0.8.3 with its drafter on the M4 Pro: a 240-line
file rewrite 38.9 -> 56.9 tok/s, code 42.0 -> 46.3, but prose 42.0 -> 36.5, because its DSpark drafts are checked
on every token.

Nuemac drafts only copies, and only when the copy is certain enough to be worth a check:

- The lookup uses tokens already known, so a step that finds nothing waits on nothing (the first version read
  the pending token every step, which undid the queue-first decode loop and cost prose 13%).
- Up to 8 earlier positions of the last 4 tokens are candidates; the one with the longest run of equal tokens
  before it wins, and a draft needs a run of 12 (a run of 8 drafted common code lines and cost code 5%).
- Drafts start at 4 tokens and double up to 15 while they keep matching. A miss on the first token pauses
  drafting for 8 tokens.
- The check feeds the pending token plus the draft. Each row's token is sampled with the request's own sampler,
  and the draft is kept up to its first difference, so temperature sampling stays exact.
- A partial match restores the cache snapshot and re-runs the kept tokens (the short-conv states can't be cut
  back otherwise); full matches, the common case, need no re-run.
- The 8 global visits read a check's few rows straight from the 8-bit cache (`q8_rows_attention`), instead of
  dequantizing the whole history.
- Only while one request is decoding: with several, each extra row costs every request.

| M4 Pro 24 GB, oQ3e, tok/s | oMLX 0.8.2, off | **0.8.2, copy** | 0.8.3, off | **0.8.3, copy** |
| --- | --- | --- | --- | --- |
| prose, temperature 0 / 1 | 42.8 / 42.2 | 42.8 / 42.3 | 43.8 / 43.3 | 43.8 / 43.4 |
| code, temperature 0 / 1 | 42.6 / 42.1 | 41.9 / 41.6 | 43.7 / 43.4 | 43.1 / 42.9 |
| 240-line file rewrite, temperature 0 / 1 | 40.8 / 40.7 | **66.3 / 69.5** | 42.1 / 41.9 | **69.1 / 72.5** |

On the rewrite, 2,801-2,919 of ~2,900 drafted tokens were accepted in ~200 checks (~14 per check). The quality
check scores 12/12 with drafting on; 2 and 4 requests at once are unchanged (45.8 and 60.6 tok/s total). As with
batching, a drafted stretch can round differently from one-token steps, so greedy text may differ from 0.5.0 after
a draft. `--draft off` turns it off.

Nuemac also runs on oMLX 0.8.3 now (0.8.3 maps cache entries to layer visits itself; Nuemac keeps its own
per-visit caches). 0.8.3's smaller forward pass gives Nuemac ~2.5% more decode and ~12% faster prompt reading.

## Ideas not tried

- An 8-bit prefill kernel. Prefill falls from 190 to 135 tok/s by 94K, partly because every
  256-token chunk dequantizes the whole global cache for the flash kernel. Reading the 8-bit keys directly, as
  decode does, would remove that copy.
- Lower-bit attention weights (Q8_0 to Q6_K/Q5_K) for the llama.cpp build: an estimated +6-9% decode, at some
  quality cost.
- Operator fusion for the `inkling2_loop` graph in llama.cpp, the real fix for its per-operation overhead
  (upstream work).
- More GPU: done, see [24 GB M4 Pro](#24-gb-m4-pro-256k-context).
- A faster 3-bit expert kernel than MLX's (see [Per-token overhead](#per-token-overhead-040)).
- Engines checked and not usable here today: TensorFold 1.3.2 runs Inkling 2 Small on CUDA only; sushi 1.7
  serves four other model families.
