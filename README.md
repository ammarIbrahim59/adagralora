# AdaGraLoRA

**Layer-wise adaptive block allocation for GraLoRA.**

GraLoRA splits a LoRA adapter into `k × k` sub-blocks, where `k` is one global number shared by
every adapted layer. This project asks whether *different layers want different `k`*, and
whether a cheap gradient signal can pick a better allocation than the best uniform one.

The comparison is unusually clean, because GraLoRA's trainable parameter count is **exactly
invariant to `k`**: `gralora_A` is `[k, in/k, r]` and `gralora_B` is `[k, r, out/k]`, so their
element counts are `in·r` and `out·r` at any legal `k`. Every allocation strategy is therefore
budget-matched by construction — no capacity confound, no balancing logic anywhere.

The hypothesis, the decision rule, and what a negative result would mean were all fixed in
advance in [`HYPOTHESIS.md`](HYPOTHESIS.md), before any training run.

## Status

| Phase | State |
|---|---|
| 0 — repo, environment, constraint validator, mixed-k patcher, checkpoint I/O | ✅ complete, 57 CPU tests |
| 1 — constraint sweep for the target model | via `python -m adagralora.patching` |
| 2 — data pipeline | not started |
| 3 — checkpoint round trip | ✅ complete (`tests/test_io_roundtrip.py`) |
| 4–8 — trainer, sweep, analysis, writeup | not started |

No training run has happened yet, so there are no results to report.

## Setup

```bash
pip install -r requirements.txt && pip install -e .
python environment_check.py     # run this first on any new machine
pytest -q                       # expect 57 passed, CPU only, no downloads
```

`environment_check.py` reports package versions, verifies that
`GraloraLayer.update_layer` still has the signature the patcher depends on, confirms that
GraLoRA has no bitsandbytes backend, reads your GPU's compute capability, and prints the
exact `--precision` flag to use on that machine.

## Which `k` values are legal?

Three divisibility constraints must hold for a layer to accept a given `k`:

```
r % k == 0            in_features % k == 0            out_features % k == 0
```

Check them for a real model before writing any training config — if a rank has no legal `k`
for some layer, drop that rank now rather than mid-sweep:

```bash
python -m adagralora.patching --model Qwen/Qwen2.5-0.5B-Instruct --r 16 32 64 128
```

It reads the real config from `transformers` when it can and falls back to an offline table
(`KNOWN_MODELS`) otherwise. Exit code is non-zero if any module has no legal `k`.

## Usage

```python
from transformers import AutoModelForCausalLM
from adagralora import build_mixed_gralora, save_mixed_adapter, load_mixed_adapter

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

# Layers not named in k_map fall back to default_k.
peft_model = build_mixed_gralora(
    model, r=32,
    k_map={"model.layers.0.mlp.down_proj": 8, "model.layers.1.self_attn.q_proj": 4},
    default_k=2,
    task_type="CAUSAL_LM",
)

save_mixed_adapter(peft_model, "runs/my-adapter")

reloaded = load_mixed_adapter(
    AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct"),
    "runs/my-adapter",
    task_type="CAUSAL_LM",
)
```

Every constraint is validated for the whole k-map *before* a single tensor is allocated, so an
illegal allocation fails immediately instead of part-way through building a multi-GB model.

## Why checkpoints need a sidecar

`adapter_config.json` has room for exactly one global `gralora_k`. Saving a mixed-k adapter the
normal way and reloading it with `PeftModel.from_pretrained` rebuilds every layer at that single
`k`, so the saved tensors no longer fit their layers. It fails reproducibly with a shape-mismatch
`RuntimeError` — and only *after* training, when the run is already spent.

`save_mixed_adapter` writes a `k_map.json` sidecar next to the adapter, and `load_mixed_adapter`
rebuilds the layers at their saved `k` *before* loading any weights.
`tests/test_io_roundtrip.py::test_naive_reload_of_a_mixed_adapter_fails` pins down the original
failure so it can't quietly come back.

## Known constraints

- **`peft` is pinned to exactly `0.20.0`.** The mixed-k patcher re-invokes
  `GraloraLayer.update_layer` per layer, so it depends on that method's internal signature.
  `transformers` and `trl` are left unpinned above their floors. A test and an
  `environment_check.py` line both fail loudly if the signature moves.
- **4-bit training is impossible.** PEFT's GraLoRA has no bitsandbytes backend — there is no
  `bnb.py` in `peft/tuners/gralora/`. Passing `--load_in_4bit` raises
  `Target module ... is not supported`. Do not add the flag.
- **bf16 is unavailable on Turing (T4).** `environment_check.py` detects this and tells you to
  use fp16; watch for NaN loss if you do.

## Layout

```
src/adagralora/patching.py   constraint validation + the mixed-k builder + the sweep CLI
src/adagralora/io_utils.py   k_map.json sidecar, the custom loader, run provenance
environment_check.py         pre-flight check for a new machine
HYPOTHESIS.md                pre-registration: hypothesis, decision rule, negative-result policy
tests/                       57 CPU tests; no GPU, no model download
```

## License

Apache-2.0.
