# Adding a family

A family is a Python package in `engine/families/<name>/` that the engine calls for everything model-specific.
`engine/families/inkling2/` is the reference.

## The interface

```python
NAME = "inkling2"
MODEL_TYPES = frozenset({"inkling2_loop", "inkling2"})   # config.json model_type values
TESTED_OMLX = "0.8.2"

def load(model_path, options) -> (model, tokenizer)  # check + patch upstream code, load, trim weights
def make_cache(model, ctx) -> list                    # one cache entry per layer (or layer visit)
def snapshot(cache) -> state                          # cheap: references and offsets, not copies
def restore(cache, state)                             # back to the end of the last prompt
def stop_ids(tokenizer) -> set[int]
def output_parser(tokenizer, model_path)              # .process_token(id), .finalize() (oMLX's interface)
```

Register it in `engine/families/__init__.py`. Then add `models/<model>.conf` with the pinned revision and every
file's SHA-256, and a recipe page here.

## The order that worked

1. **Ceilings.** Weights read per token (count each layer pass, active experts only), the KV bytes per token
   the architecture needs, and the measured bandwidth floor (a plain MLX reduction over 1-2 GB). Write the target
   and the ceiling down.
2. **Quality gate first.** 12 known-answer questions (`nuemac qa`). Greedy tokens against the upstream
   implementation, 64 tokens on a prompt longer than any sliding window. A fact planted at depth.
3. **Memory.** Measure KV bytes per token at two prompt lengths and compare with the minimum. A 5-10x gap means
   caches are kept that aren't needed (full caches on sliding layers, per-visit caches on looped layers).
4. **Profile a whole token** before writing kernels. Ablate one component per process (fresh process, since
   `mx.compile` caches traces), then microbenchmark each matvec chained as the model runs it, in GB/s. Beware
   MLX laziness: a stub that makes the output independent of the model skips the whole forward pass.
5. **Kernels only where an op is far from the floor.** Use `mx.fast.metal_kernel`. Pass whole preallocated
   buffers plus offsets (inputs are made contiguous, which copies), and put per-call values in a params array,
   not in template arguments.
6. **Bytes when kernels are near the floor.** fp32 copies of bf16 weights, the output head's bits, side tensors.
   Measure head changes with the same hidden states through both heads (KL, top-1), with the original requantized
   to itself as the noise floor.
7. **End-to-end A/B, alternating, two runs or more each** (`nuemac bench --prefill "" --depth 32768`). A kernel
   that wins in isolation can lose in the whole token. Noise is about ±0.4 tok/s run to run.

The Inkling recipe has the full record: kept and rejected changes, with their measured effect.

## Done means

- A row in [supported models](../supported-models.md) with the pinned download, the machine checked and the
  limits.
- A recipe page: what decides the speed, what worked (tables), how to run it.
- Raw bench JSON in `results/`, quality check 12/12, needle found at depth.
- A CHANGELOG entry.
