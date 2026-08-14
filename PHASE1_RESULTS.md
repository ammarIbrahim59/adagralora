# Phase 1 — Constraint Sweep Results (COMPLETE)

Status: ✅ done. Ranks confirmed, grid locked. Phase 2 (data pipeline) can proceed
on top of this without re-running the sweep.

## 1.1 — Legal k per rank, primary model

`Qwen/Qwen2.5-0.5B-Instruct`, swept via:

```
python -m adagralora.patching --model Qwen/Qwen2.5-0.5B-Instruct --r 16 32 64 128
```

| r   | q_proj | k_proj | v_proj | o_proj | gate_proj | up_proj | down_proj | legal k (all modules) |
|-----|--------|--------|--------|--------|-----------|---------|-----------|------------------------|
| 16  | 2,4,8  | 2,4,8  | 2,4,8  | 2,4,8  | 2,4,8     | 2,4,8   | 2,4,8     | **2, 4, 8** |
| 32  | 2,4,8  | 2,4,8  | 2,4,8  | 2,4,8  | 2,4,8     | 2,4,8   | 2,4,8     | **2, 4, 8** |
| 64  | 2,4,8  | 2,4,8  | 2,4,8  | 2,4,8  | 2,4,8     | 2,4,8   | 2,4,8     | **2, 4, 8** |
| 128 | 2,4,8  | 2,4,8  | 2,4,8  | 2,4,8  | 2,4,8     | 2,4,8   | 2,4,8     | **2, 4, 8** |

No rank had to be dropped — every rank in the target set has k ∈ {2, 4, 8} legal on
every projection.

## 1.2 — Locked-in grid

Given 1.1's clean result, the recommended grid from the workflow doc is used as-is,
no changes:

- **Model:** `Qwen/Qwen2.5-0.5B-Instruct` (primary)
- **Ranks:** {16, 32, 64, 128}
- **k:** {2, 4, 8}
- **Stretch goal model:** `Qwen/Qwen2.5-1.5B-Instruct` (see 1.3)

## 1.3 — Stretch model in `KNOWN_MODELS`

`Qwen/Qwen2.5-1.5B-Instruct` is present in `KNOWN_MODELS` in `patching.py`
(hidden_size=1536, intermediate_size=8960, num_attention_heads=12,
num_key_value_heads=2, head_dim=128). Swept the same way:

```
python -m adagralora.patching --model Qwen/Qwen2.5-1.5B-Instruct --r 16 32 64 128
```

| r   | q_proj | k_proj | v_proj | o_proj | gate_proj | up_proj | down_proj | legal k (all modules) |
|-----|--------|--------|--------|--------|-----------|---------|-----------|------------------------|
| 16  | 2,4,8  | 2,4,8  | 2,4,8  | 2,4,8  | 2,4,8     | 2,4,8   | 2,4,8     | **2, 4, 8** |
| 32  | 2,4,8  | 2,4,8  | 2,4,8  | 2,4,8  | 2,4,8     | 2,4,8   | 2,4,8     | **2, 4, 8** |
| 64  | 2,4,8  | 2,4,8  | 2,4,8  | 2,4,8  | 2,4,8     | 2,4,8   | 2,4,8     | **2, 4, 8** |
| 128 | 2,4,8  | 2,4,8  | 2,4,8  | 2,4,8  | 2,4,8     | 2,4,8   | 2,4,8     | **2, 4, 8** |

Same result: every target rank is fully legal for the stretch model too.

Cross-checked by `test_patching.py::test_known_models_dims_match_transformers_config`,
which asserts the `KNOWN_MODELS` entry for this model matches a real
`transformers.Qwen2Config(hidden_size=1536, intermediate_size=8960,
num_attention_heads=12, num_key_value_heads=2)` — not just internal
self-consistency.

## How this was verified

- `python -m adagralora.patching` run manually for both models, all four ranks (output above).
- `python environment_check.py` — independently re-runs the same sweep on startup; reported `[ok]` for both models at all four ranks.
- `pytest -q` — 116/116 passed, including 17 tests in `tests/test_patching.py` covering `KNOWN_MODELS` correctness and legal-k logic.

**Caveat:** all checks above ran against the offline `KNOWN_MODELS` fallback table
(no network route to huggingface.co in this environment — CLI output shows
`dims : KNOWN_MODELS offline table`). On a machine with real internet access,
`_dims_for_model` will instead hit the live `AutoConfig.from_pretrained` path; the
test referenced above guarantees the two paths agree, but it's worth a quick glance
that the live-path output on the GPU machine still shows the same dims before
trusting it for Phase 4 training.

## Next

Phase 2 (data pipeline) can start against this locked grid: `Qwen2.5-0.5B-Instruct`,
r ∈ {16, 32, 64, 128}, k ∈ {2, 4, 8}, stretch model `Qwen2.5-1.5B-Instruct` available
and pre-validated at the same ranks.
