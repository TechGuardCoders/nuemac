# Changelog

Speeds are from `nuemac bench`: a base Mac mini M4 (10-core GPU, 16 GB) at 96K context unless a line says otherwise.

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
