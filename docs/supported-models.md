# Supported models

Which checkpoints Nuemac serves, on which machines, and what each needs. Install Nuemac and a model with the
[quick start](../README.md). `nuemac models` lists these with their install state. `nuemac use <model>` switches
to one, downloading it the first time.

"Machine checked" is where the recipe was measured, not a promise for every Mac. Context, the KV cache and
macOS itself need memory beyond the weights. Each model's recipe lists its measured headroom.

| Model / format | Weights (pinned) | Machine checked and memory | Recipe / limits |
| --- | --- | --- | --- |
| **Inkling 2 Mini**, MLX oQ2.5e (mixed 2/3-bit experts, 8-bit attention), 96K context | [`mlx-community/Inkling-2-Mini-oQ2.5e-mtp@d4a4f3f`](https://huggingface.co/mlx-community/Inkling-2-Mini-oQ2.5e-mtp/tree/d4a4f3f555fdf6f06102de1865d07e8ac5effe6d), 10.9 GB (the drafter file is skipped) | Mac mini M4 (10-core GPU), 16 GB, macOS 26.5. GPU wired limit 14.5 GB; ~12 GB in use at 96K; 10-13% system memory free under load. | [inkling-2-mini](recipes/inkling-2-mini.md). Text only (vision and audio towers not wired up). Up to 4 requests at once (`--slots`); see [concurrent requests](recipes/inkling-2-mini.md#concurrent-requests-050). Tested with oMLX 0.8.2 and 0.8.3; refuses to start if the oMLX internals it patches have changed. |

| **Inkling 2 Mini**, MLX oQ3e (3-bit experts, 8-bit attention), 256K context: `inkling-2-mini-oq3e` | [`mlx-community/Inkling-2-Mini-oQ3e-mtp@5552e32`](https://huggingface.co/mlx-community/Inkling-2-Mini-oQ3e-mtp/tree/5552e32ec7a50fa8e9b0bfe2002aab72f8b78975), 13.1 GB (the drafter file is skipped) | Mac mini M4 Pro (16-core GPU), 24 GB, macOS 26.6. GPU wired limit 19.3 GB or more; 12.1 GiB weights + 4.6 GB global cache at 256K; 17-19% system memory free under load. Needs 24 GB. | [inkling-2-mini, 24 GB](recipes/inkling-2-mini.md#24-gb-m4-pro-256k-context). Same limits as the 16 GB build. |

## Engine requirements

- macOS 14+, Apple Silicon. Raise the GPU limit to the model's need: `sudo sysctl iogpu.wired_limit_mb=14500` for
  the 16 GB build, `19300` for the 256K build (the installer offers this).
- oMLX from Homebrew (`brew tap jundot/omlx https://github.com/jundot/omlx && brew install jundot/omlx/omlx`; the
  installer does this) provides MLX, mlx-lm and the upstream model code.
  Nuemac runs with oMLX's Python, isolated (`python -I`), and does not use oMLX's server.

## Next candidates

Families with sliding-window or hybrid attention benefit most from Nuemac's cache layouts on 16-24 GB Macs. A
model gets a row here only after its recipe passes the quality gate (token match, known-answer questions,
needle at depth) and an end-to-end A/B against the best existing engine. See
[adding a family](recipes/adding-a-family.md).
