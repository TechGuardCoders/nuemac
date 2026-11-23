# Changelog

Speeds are from `nuemac bench`: a base Mac mini M4 (10-core GPU, 16 GB) at 96K context unless a line says otherwise.

## 0.7.0 - 2026-11-23: images

- Inkling 2 Mini reads images. Send OpenAI-style `image_url` parts (base64 `data:` URLs or http(s) links) in user or
  tool messages; both pinned builds ship the vision tower. Each 40 px block of an image is one prompt token (a
  1280x720 screenshot is 576); images over `--max-image-tokens` (2304) are scaled down to fit, EXIF rotation applied.
  `/v1/models` reports `"vision": true`. A build without a vision tower answers an image with a 400.
  - Earlier releases ignored images: the template's image token went in with no picture behind it.
- Where the tower lives (`--vision`): on a 16 GB Mac (GPU limit under 20 GiB) it loads when a prompt's image is first
  read (0.1-0.3 s) and is freed afterwards, so text-only work keeps its memory; on 24 GB it stays loaded. Measured
  on the base M4 16 GB with the 11.2 GB custom build at a 94K-token prompt: decode 15.1 tok/s loaded on demand,
  12.2 with the tower kept resident.
- `--vision-bits` (auto: 8 on 16 GB Macs, as stored on 24 GB): 8-bit linears (325 MiB instead of 607) keep the image
  features at cosine 0.997-0.999 of the released bf16 tower, with the same answers.
- Vision checks (base M4, 6/6 at bf16, 8-bit, and loaded on demand): reading an invoice (number and total), colors
  and positions, counting, the open tickets in a 1280x800 screenshot, a follow-up question reusing the cached
  image, and a different image at the same place (never served from the first image's cache). A 3840x2160 photo
  (scaled to 2304 tokens) takes 4.5 s to encode, 17 s to read in all, and is described correctly.
- Prompt reuse keys each image by its bytes and place: the image tokens are stored as per-image stand-in ids.
  Copy drafting never drafts into them.

## 0.6.1 - 2026-11-23: faster prompt reading on 24 GB Macs

- Bigger prompt steps when nothing is generating (`--prefill-chunk-alone`, auto). While other requests decode,
  prompts are still read 256 tokens at a time so those requests wait at most about one step. With nothing
  decoding there is no one to wait, so a Mac with a GPU limit of 20 GiB or more reads 1024 tokens per step.
  M4 Pro 24 GB (oQ3e): 320 -> 348 tok/s at 1.6K tokens, 329 -> 349 at 6K, 318 -> 337 at 18K, a 120K-token prompt
  in 8.2 min (243 tok/s, needle found); same accuracy (top-1 agreement with 0.6.0 99%); peak GPU memory +1.1-1.7
  GiB. On the 16 GB M4 it was no faster and a 94K prompt ran out of GPU memory, so it keeps 256 there.
- `MLX_MAX_OPS_PER_BUFFER` defaults to 200 (MLX ends a Metal command buffer after 40 ops on M4 chips): M4 Pro
  decode +2%, 4 requests at once +5%. An explicit environment value still wins.
- The output head is only ever lowered to `--head-bits`, never raised: a build that stores it at 4 or 6 bits is
  served as stored.
- Builds whose experts were quantized in a Hadamard-rotated basis (listed in `nuemac_rotation.json`) get their
  inputs rotated to match.
- `MODEL_PATH` in `~/.nuemac/settings` serves your own build of the active model; the installer now keeps other
  settings when it rewrites `PORT`.
- The log shows each prompt's read time and peak GPU memory.

## 0.6.0 - 2026-11-22: copy drafting, oMLX 0.8.3

- Copy drafting: while one request decodes alone, a reply that repeats its context (a rewritten file, an edit,
  quoted code) is drafted from that context and checked up to 16 tokens per forward pass. M4 Pro 24 GB (oQ3e),
  rewriting a 240-line file: 40.8 -> 66.3 tok/s on oMLX 0.8.2 (69.5 at temperature 1), 42.1 -> 69.1 on 0.8.3
  (72.5); ~14 tokens accepted per check. Base M4 16 GB (oQ2.5e): 24.4 -> 43.5. Prose is unchanged, code within 1.5%. Quality check 12/12; 2 and 4
  requests at once unchanged. `--draft off` turns it off.
  - Drafts come only from long exact runs (12 tokens), the lookup never waits on the GPU, and temperature
    sampling stays exact (each drafted token is kept only if it equals the token the request's sampler draws).
  - A check's rows read the 8-bit global cache directly (`q8_rows_attention`), checked against oMLX's reference
    attention (1.2% relative, bf16 rounding).
  - As with batching, greedy text after a drafted stretch may differ from 0.5.0's (different rounding).
  - Response `timings` gain `draft_checks`, `draft_n`, `draft_accepted`.
- oMLX 0.8.3 support (0.8.2 still works): 0.8.3 maps cache entries to layer visits itself, and Nuemac now hands
  its per-visit caches through. On 0.8.3 Nuemac decodes ~2.5% faster (M4 Pro 43.8 tok/s) and reads prompts ~12%
  faster. Homebrew still installs 0.8.2.

## 0.5.0 - 2026-11-22: continuous batching

Several requests run at once: they join and leave one decode batch between steps (vLLM-style iteration-level
scheduling). Measured on the M4 Pro 24 GB (oQ3e, 200-token greedy replies, short prompts):
one request 42.2-42.6 tok/s, the same text as 0.4.0; 2 at once 45.7 tok/s total (~24 each); 4 at once 60.4 total
(~16 each). Base M4 16 GB (oQ2.5e): 25.5 alone, the same text as 0.4.0; 29.1 total for 2; 36.3 total for 4.
Two quality checks run at the same time score 12/12 each on both Macs.
- `engine/server.py`: an engine thread owns every MLX call and schedules sequences; HTTP threads submit requests
  and read their events (streamed deltas, done, error). A client that disconnects cancels its request.
- Prompts prefill in chunks taking turns with decoding, so streams keep flowing: a 39K-token prompt prefilled while
  another reply streamed 636 chunks, the longest gap 0.95 s (one chunk); 1.5 s on the base M4.
- Each sequence keeps its own caches, sized to prompt + max_tokens in bands (4K to 512K). A request waits when its
  cache would not fit the memory budget; finished conversations' caches stay for prompt reuse while memory allows.
- One request alone still runs the compiled decode step. A batched step runs the projections, experts and head
  for all rows at once and every row's attention in one launch (`engine/families/inkling2/slots.py`, adapted from
  mlx-serve's Inkling 2 slots kernel, MIT): -4.5% step time at 4 rows. Most of a batched step is expert reads (4
  rows touch ~29 of 128 experts a visit), so throughput grows ~1.4x at 4 rows, not 4x.
- Greedy output is not batch-invariant: a row batched with others can take different tokens than it would alone
  (kernel numerics), as in other batching servers.
- `--slots N` (default 4; 1 = one request at a time), `--keep-caches N` (default 2), `NUEMAC_NO_SLOTS_KERNEL=1`.

## 0.4.0 - 2026-11-21: less per-token overhead

Lossless decode speed-ups (same greedy tokens), end to end through the server:
M4 Pro 24 GB 35.9 -> 42.3 tok/s (32K: 29.4 -> 33.8); base M4 16 GB 23.5 -> 25.9 tok/s (32K: 18.8 -> 20.7).
12/12 on both (`nuemac bench --depth 32768`, results/2026-11-21-nuemac-0.4.0-*.json).
- The decode loop queues the next step before waiting for the current token. It used to wait first, leaving the
  GPU idle while Python built the next graph (~2.8 ms per token on an M4 Pro): 35.9 -> 39.4 tok/s.
- Compiled decode step (`engine/families/inkling2/fastpath.py`): fixed-shape caches (a ring for the window
  visits, in-place 8-bit writes for the global ones, a fixed split-K grid per context band) so one `mx.compile`
  graph serves every token. `--no-compiled-decode` turns it off.
- Fused loop-controller mixing (two Metal kernels instead of ~12 ops, 48 times a token) and window attention as a
  split-K kernel over the ring.
- `--attn-bits 6` (opt-in) requantizes the attention projections: +3.9% (43.7 tok/s on the M4 Pro), mean KL 0.0057
  against 8-bit where re-rounding alone gives 0.0045. Off by default.
- Debug switches: `NUEMAC_NO_FUSED_MIX=1`, `NUEMAC_NO_RING_KERNEL=1`.

## 0.3.2 - 2026-11-21

- Fix: multi-step tool use. OpenAI clients send an assistant tool call's arguments back as a JSON string; Inkling's
  chat template raised on that and the server dropped the connection, so agents (Hermes) failed on the turn after
  their first tool call. The server now parses those arguments before applying the template.
- Errors in a request now come back as an HTTP 500 with the message instead of a dropped connection.

## 0.3.1 - 2026-11-21: 24 GB Macs

- New model `inkling-2-mini-oq3e`: Inkling 2 Mini oQ3e at 256K context for 24 GB Macs. On a Mac mini M4 Pro
  (16-core GPU, 24 GB): 35.6 tok/s decode, 319/301 tok/s prefill, 29.4 tok/s at 32K, 21.4 at 120K, 12/12, both
  needles found. The installer picks it by default on Macs with 24 GB or more.
- The GPU-limit step uses each model's need (`GPU_MB` in its conf) instead of a fixed 14.5 GB, and the boot job
  keeps that value. Models can set a minimum memory (`MIN_MEM_MB`).

## 0.3.0 - 2026-11-21: Nuemac

- The project is now an engine with model families and recipes, following TensorFold's layout:
  - `engine/server.py` is model-agnostic;
  - `engine/families/inkling2/` holds everything Inkling-specific;
  - `nuemac` replaces the `mini-llm` CLI, with a new `models` command;
  - docs gain `supported-models.md` and `recipes/`.
- Inkling 2 Mini reads fewer bytes per token:
  - the loop controllers stay bf16 instead of fp32 copies; greedy tokens are unchanged;
  - the output head is requantized from 8 to 6 bits at load (`--head-bits`). Mean KL 0.00027 against the
    8-bit head, against a 0.00008 noise floor.
  - Decode went from 22.4 to 23.5 tok/s, and 32K decode from 17.9 to 18.8; quality check 12/12.
- The installer reuses an earlier `inkling2-mini-mac-recipe` install's verified model files and stops its
  `mini-llm` service.
- The llama.cpp and Ternary Bonsai builds are no longer installable models here. They stay as comparison
  numbers; the original recipe repo still installs them.

## 0.2.0 - 2026-11-21

- Decode-attention kernel tuned: it prefetches the next key, rescales only when the running max changes, and
  merges partials and the sink in one kernel. 96K: 22.3 -> 20.3 ms per token for the 8 global passes
  (77 -> 84 GB/s; the floor is 15.4 ms).
- Used above 12K tokens. Below that the plain MLX path wins end to end (A/B: the kernel at every depth cost
  0.8 tok/s at short context).

## 0.1.0 - 2026-11-21

- Inkling 2 Mini at 96K on MLX, on a 16 GB Mac:
  - window caches for the 40 sliding passes;
  - 8-bit preallocated caches for the 8 global passes;
  - a split-K Metal decode kernel.
- Results: 22.4 tok/s decode (llama.cpp: 17.25), prefill 190 / 175 tok/s (68 / 61), 94K prompt answered after
  11.7 min (42.6).
- Single-stream server with llama-server and OpenAI APIs, a one-command installer, and the benchmark.
