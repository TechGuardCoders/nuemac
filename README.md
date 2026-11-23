# Nuemac

**Private inference at the edge.**

<p align="center"><img src="docs/assets/infographic.png" alt="Nuemac: private inference at the edge. Inkling 2 Mini at 25.9 tok/s with a 96K context on a base 16 GB M4 Mac mini, up to 4 requests at once" width="820"></p>

Nuemac gives the Mac you already own a new life as a private AI machine. It's a local LLM engine for
memory-tight Apple Silicon: big open models with long contexts, on a 16 GB Mac mini, served as an
OpenAI-compatible API on your own machine. Prompts, documents and answers never leave it, and there's no
per-token bill.

Under the hood it runs on [MLX](https://github.com/ml-explore/mlx), with its own KV-cache layouts and Metal
kernels for each model family. One command installs it and a model, and one more benchmarks it on your Mac.

```bash
curl -fsSL https://raw.githubusercontent.com/TechGuardCoders/nuemac/main/install.sh | bash
```

Landing page: <https://techguardcoders.github.io/nuemac/>

## Why Nuemac is different

General-purpose engines like llama.cpp, mlx-lm and oMLX run thousands of models with one shared set of machinery.
That breadth is their strength. It also leaves memory and speed unused for any single model on a small machine.

Nuemac works the other way round. Each supported model gets its own cache layout, kernels and weight trims,
designed from that model's architecture and measured on the Mac it targets. Here is what that did for Inkling 2
Mini, Thinking Machines' 26.4B-parameter looped mixture-of-experts model, within a day of its release.

**Half the memory its makers ask for.** The model card lists "Apple silicon devices with 32GB or more". Nuemac
runs it on a base 16 GB Mac mini with a 96K-token context, and 10-13% of memory still free under load.

**Long context without giving up speed.** On 16 GB the other paths each gave something up:

| on a 16 GB M4 Mac mini | stock MLX port (oMLX) | llama.cpp | **Nuemac** |
| --- | --- | --- | --- |
| longest context that fits | ~7K tokens | 96K | **96K** |
| decode, short prompt | 24.8 tok/s | 17.3 tok/s | **25.9 tok/s** |
| prompt reading | ~188 tok/s | 68 tok/s | **191 tok/s** |
| a 94K-token prompt: first word after | doesn't fit | 42.6 min | **11.8 min** (0.3.0) |

**Built around the model's own shape.** Inkling runs its middle layers three times per token: 48 attention passes,
and 40 of them only ever look back 1,024 tokens. Nuemac keeps full history only for the 8 passes that read it, and
stores it at 8 bits. A Metal kernel written for exactly this layout, with Inkling's relative-position bias and
attention sink built in, reads that history straight from the 8-bit codes. At 96K that is 20 ms per token, where
plain MLX operations took 73 ms. The stock port kept full history on all 48 passes, about 192 KB per token.

**Measured down to the byte.** Profiling a whole token showed the expert kernels already running near the chip's
bandwidth, so the last gain came from reading less. The loop controllers stay at their native bf16 instead of
fp32 copies, and the output head is cut to 6 bits after measuring it (KL 0.0003 against the 8-bit head). Together
that is +5% decode with the same answers.

**Every number can be checked.**
- A quality gate before any speed claim: 12 known-answer questions, a fact planted at depth, and token-for-token
  agreement with the reference implementation.
- An alternating end-to-end A/B for every engine change. One kernel won in isolation but lost on short prompts,
  so it now only switches on above 12K tokens.
- The raw results are in [`results/`](results/). The experiments that failed are written up next to the ones that
  worked, in the [recipe](docs/recipes/inkling-2-mini.md).

**Several users at once.** Up to 4 requests run together (continuous batching): they join and leave one decode
batch between steps, and long prompts are read in chunks between the replies already streaming. One request
alone is as fast as before and gives the same text.

| short prompts, 200-token replies | 16 GB M4 Mac mini | 24 GB M4 Pro Mac mini |
| --- | --- | --- |
| 1 request | 25.5 tok/s | 42.4 tok/s |
| 2 at once, total | 29.1 tok/s (~15 each) | 45.7 tok/s (~24 each) |
| 4 at once, total | 36.3 tok/s (~9.5 each) | 60.4 tok/s (~16 each) |
| longest pause in a reply while a 39K-token prompt is read | 1.5 s | 0.95 s |

Four requests give ~1.4x, not 4x, because each picks its own experts and together they read ~3.7x the expert
weights of one. Details: [concurrent requests](docs/recipes/inkling-2-mini.md#concurrent-requests-050).

**Private by construction.** The server listens on loopback only (`127.0.0.1`). While serving, it held no
connections beyond its own local clients. Weights are downloaded once, at install. There's no account, no
telemetry and no per-token bill.

## Supported models

| Model | Machine checked | Speed | Recipe |
| --- | --- | --- | --- |
| **Inkling 2 Mini** (Thinking Machines), MLX oQ2.5e, 96K context: `inkling-2-mini` | Mac mini M4, 16 GB | **25.9 tok/s** decode, 191 tok/s prefill, 20.7 tok/s at 32K; 36.3 tok/s total at 4 requests; 43.5 tok/s rewriting a file | [inkling-2-mini](docs/recipes/inkling-2-mini.md) |
| **Inkling 2 Mini**, MLX oQ3e, 256K context: `inkling-2-mini-oq3e` | Mac mini M4 Pro, 24 GB | **42.3 tok/s** decode, 318 tok/s prefill, 33.8 tok/s at 32K; 60.4 tok/s total at 4 requests; 66-72 tok/s rewriting a file | [inkling-2-mini, 24 GB](docs/recipes/inkling-2-mini.md#24-gb-m4-pro-256k-context) |

The installer picks the build for your Mac's memory: the 16 GB build below 24 GB, the 256K build at 24 GB or more.

Pinned downloads, memory needs and limits: [docs/supported-models.md](docs/supported-models.md). How a model
gets here: [adding a family](docs/recipes/adding-a-family.md).

## Inkling 2 Mini on a 16 GB Mac mini

Same Mac, 96K context on every row, one model loaded, `nuemac bench --depth 32768`, server timings,
2026-11-21. The other two columns are the best builds we found on other engines, kept for comparison.

| | **Nuemac** | llama.cpp b12600 | Ternary Bonsai 2 27B (PrismML llama.cpp) |
| --- | --- | --- | --- |
| decode, short prompt | **25.9 tok/s** | 17.3 tok/s | 10.9 tok/s |
| prefill, 4K / 16K prompt | **191 / 176 tok/s** | 68 / 61 tok/s | 58 / 55 tok/s |
| 32K prompt: first token | **201 s** | 690 s | 647 s |
| 32K prompt: decode | **20.7 tok/s** | 12.2 tok/s | 8.7 tok/s |
| fact planted at 32K | found | found | found |
| 12 known-answer questions | 12/12 | 12/12 | 11/12 |

Nuemac column: 0.4.0. At a nearly full window (a 94K-token prompt, measured on 0.3.0) Nuemac answers after 11.8
minutes and decodes at 13.7 tok/s;
llama.cpp takes 42.6 minutes and decodes at 11.2. Raw results: [`results/`](results/). The llama.cpp and Bonsai setups install from
[inkling2-mini-mac-recipe](https://github.com/TechGuardCoders/inkling2-mini-mac-recipe), where this project
started.

What Nuemac does for this model (details and every experiment in the [recipe](docs/recipes/inkling-2-mini.md)):

1. **Window caches for the 40 sliding-window layer passes.** The stock MLX port kept full history on all 48
   passes (~192 KB per token) and fit ~7K tokens on 16 GB. Output is token-for-token identical.
2. **8-bit keys and values for the 8 global passes**, allocated once: 1.6 GB at 96K.
3. **A split-K Metal decode kernel** that reads that 8-bit cache directly: 20 ms per token at 96K, against
   73 ms for plain MLX ops.
4. **Fewer bytes per token.** The loop controllers stay bf16 instead of fp32 copies, and the output head runs
   at 6 bits (KL 0.0003 against 8-bit). Together: +5% decode.
5. **Continuous batching** (0.5.0): up to 4 requests share each decode step, with every row's attention in one
   Metal launch over its own window and 8-bit caches. 4 at once: 36.3 tok/s total.
6. **Copy drafting** (0.6.0): a reply that repeats its context (a rewritten file, an edit) is drafted from that
   context and checked up to 16 tokens at a time: a file rewrite goes 40.8 -> 66-72 tok/s on the M4 Pro and 24.4 -> 43.5
   on the 16 GB M4, prose unchanged. [Details](docs/recipes/inkling-2-mini.md#copy-drafting-060).

## Requirements

- Apple Silicon Mac, 16 GB or more, macOS 14 or newer, ~14 GB free disk (~16 GB for the 24 GB build).
- [Homebrew](https://brew.sh). The installer adds [oMLX](https://github.com/jundot/omlx) from its tap for MLX,
  mlx-lm and the model code Nuemac runs on.
- The GPU memory limit raised to what the model needs: 14.5 GB on 16 GB Macs (macOS allows ~10.7 GB by
  default), 19.3 GB for the 256K build on 24 GB (default ~16 GB). The installer asks before running
  `sudo sysctl iogpu.wired_limit_mb=...`, and can keep it after reboot.
- Close big apps while it runs. A Docker VM holding 10 GB will not leave room.

## What the installer does

It's a short script, so you can read [`install.sh`](install.sh) first. `--dry-run` prints the plan and stops.
It then:
1. Checks the Mac.
2. Installs oMLX (Homebrew) and Nuemac.
3. Downloads the model from Hugging Face at a pinned revision, checking each file's SHA-256. Downloads resume
   if interrupted.
4. Asks before raising the GPU limit.
5. Starts a login service on `127.0.0.1:8080` (loopback only).
6. Runs 12 known-answer questions, because a fast model that outputs garbage is the most common failure.

If you used the earlier `inkling2-mini-mac-recipe`, the installer reuses its verified model files and stops its
`mini-llm` service.

## Use it

```bash
curl http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "inkling-2-mini",
  "messages": [{"role": "user", "content": "Explain mixture-of-experts in two sentences."}]
}'
```

Any OpenAI-compatible client works with base URL `http://127.0.0.1:8080/v1`; the API key can be anything. The
server supports streaming, `reasoning_content`, tool calls and `reasoning_effort` (`low` / `medium` / `high`).
It also speaks llama-server's `/completion` with `timings`. Up to 4 requests run at once (continuous batching,
`--slots`), and each conversation's cache is kept, so a new turn only processes its new tokens. A reply that
repeats its context, such as a rewritten file, is drafted from it (`--draft`, on by default).

| command | what it does |
| --- | --- |
| `nuemac status` | engine, model and server state |
| `nuemac models` | supported models and which are installed |
| `nuemac bench` | decode, prefill and the quality check (~10 min); `--depth 32768` adds a needle test |
| `nuemac stop` / `start` / `restart` / `logs` | the server |
| `nuemac uninstall` | remove the login service (files stay in `~/.nuemac`) |

Settings live in `~/.nuemac/settings` (shell `KEY=value` lines; `nuemac restart` after a change):
`PORT`, `SERVER_ARGS` (extra `engine/server.py` options, e.g. `--slots 2`), and `MODEL_PATH`, a folder holding
your own build of the active model (same family, e.g. a different quantization) to serve instead of the
downloaded weights.

## Repository

```
engine/server.py                 the engine: HTTP APIs, continuous batching, multi-turn cache reuse
engine/families/inkling2/        Inkling 2: cache layout, Metal decode kernels (q8attn.py, slots.py), weight trims
models/<model>.conf              pinned weights, file hashes, context, tested oMLX version
bin/nuemac, install.sh           CLI and installer
bench/                           the benchmark (curl + plutil, no Python)
docs/supported-models.md         what runs, on what, with which limits
docs/recipes/                    one page per model, plus how to add a family
results/                         raw benchmark JSON behind every number above
```

## Credits

Nuemac is a thin layer over other people's work. Full notices are in [NOTICE](NOTICE).

- [Thinking Machines Lab](https://huggingface.co/thinkingmachines/Inkling-2-Mini) for Inkling 2 Mini.
- [oMLX](https://github.com/jundot/omlx) for the MLX model code, chat handling and output parser, and (0.8.3,
  [#5528](https://github.com/jundot/omlx/pull/5528)) the context-copy drafting that Nuemac's copy drafter follows.
  [MLX and mlx-lm](https://github.com/ml-explore) are by Apple.
- [mlx-community](https://huggingface.co/mlx-community/Inkling-2-Mini-oQ2.5e-mtp) for the MLX weights.
- [mlx-serve](https://github.com/ddalcu/mlx-serve) by David Dalcu: Nuemac's batched decode attention is adapted
  from its Inkling 2 slots kernel (MIT).
- [TensorFold](https://github.com/ashhart/TensorFold) for the engine-plus-recipes layout this repo follows, and
  [Strata](https://github.com/Niko1221/Strata) for kernel and dispatch ideas.
- Comparison builds: [llama.cpp](https://github.com/ggml-org/llama.cpp),
  [AtomicChat](https://huggingface.co/AtomicChat/Inkling-2-Mini-GGUF),
  [PrismML](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf).
- [Sybil Solutions](https://github.com/sybil-solutions/inference-optimisation-skill) for the tuning method.

## License

Apache-2.0 for Nuemac's engine, kernels, scripts and docs ([LICENSE](LICENSE)). Models and dependencies keep
their own licenses ([NOTICE](NOTICE)). Your numbers will differ: run `nuemac bench`.
