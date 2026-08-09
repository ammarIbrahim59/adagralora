"""Constraint validation and the mixed-k GraLoRA builder.

PEFT's ``GraloraConfig`` carries a single global ``gralora_k``. This module lets
every targeted linear layer get its own ``k`` by re-invoking
``GraloraLayer.update_layer`` per layer with a per-layer config clone. That means
it depends on ``update_layer``'s internal signature, which is why ``peft`` is
pinned exactly in ``requirements.txt``.

Three divisibility constraints must hold for a layer to accept a given ``k``:

    r % k == 0            (rank splits evenly across sub-blocks)
    in_features % k == 0  (input is chunked into k slices)
    out_features % k == 0 (output is chunked into k slices)

All of them are checked for the whole k-map *before* a single tensor is
allocated, so an illegal allocation fails immediately rather than part-way
through building a multi-GB model.

Parameter-count invariance
--------------------------
``gralora_A`` has shape ``[k, in_features // k, r]`` and ``gralora_B`` has shape
``[k, r, out_features // k]``, so their element counts are ``in_features * r``
and ``out_features * r`` for *any* legal ``k``. The trainable parameter count is
therefore exactly invariant to the allocation. Every allocator strategy is
budget-matched by construction, and no balancing logic is needed downstream.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from typing import Iterable, Mapping, Optional, Sequence

import torch.nn as nn

from peft import GraloraConfig, get_peft_model
from peft.tuners.gralora import GraloraLayer

DEFAULT_TARGET_MODULES: tuple[str, ...] = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)

DEFAULT_K_CANDIDATES: tuple[int, ...] = (2, 4, 8)

#: Offline fallback dimensions, so the constraint sweep runs without a network
#: round trip or a model download. Values are ``(in_features, out_features)``.
KNOWN_MODELS: dict[str, dict[str, tuple[int, int]]] = {
    # hidden=896, intermediate=4864, 14 heads x 64, 2 KV heads
    "Qwen/Qwen2.5-0.5B-Instruct": {
        "q_proj": (896, 896),
        "k_proj": (896, 128),
        "v_proj": (896, 128),
        "o_proj": (896, 896),
        "gate_proj": (896, 4864),
        "up_proj": (896, 4864),
        "down_proj": (4864, 896),
    },
    # hidden=1536, intermediate=8960, 12 heads x 128, 2 KV heads
    "Qwen/Qwen2.5-1.5B-Instruct": {
        "q_proj": (1536, 1536),
        "k_proj": (1536, 256),
        "v_proj": (1536, 256),
        "o_proj": (1536, 1536),
        "gate_proj": (1536, 8960),
        "up_proj": (1536, 8960),
        "down_proj": (8960, 1536),
    },
}

#: Prefix PEFT prepends to every module path once a model is wrapped.
_PEFT_PREFIX = "base_model.model."


class KConstraintError(ValueError):
    """Raised when a requested ``k`` violates a GraLoRA divisibility constraint."""


def canonical_name(name: str) -> str:
    """Strip PEFT's wrapper prefix so names match the unwrapped model's paths."""
    if name.startswith(_PEFT_PREFIX):
        return name[len(_PEFT_PREFIX) :]
    return name


def validate_k(r: int, in_features: int, out_features: int, k: int, *, where: str = "layer") -> None:
    """Raise :class:`KConstraintError` unless ``k`` is legal for this layer.

    Reports *every* violated constraint at once rather than the first, so a bad
    sweep grid can be fixed in one pass.
    """
    if not isinstance(k, int) or k <= 0:
        raise KConstraintError(f"{where}: gralora_k must be a positive integer, got {k!r}")
    # Without this, r <= 0 divides evenly by every k and the sweep reports an
    # impossible rank as fully legal; the builder would only fail later, inside
    # update_layer, part-way through wrapping.
    if not isinstance(r, int) or r <= 0:
        raise KConstraintError(f"{where}: r must be a positive integer, got {r!r}")

    problems = []
    if r % k != 0:
        problems.append(f"r={r} is not divisible by k={k}")
    if in_features % k != 0:
        problems.append(f"in_features={in_features} is not divisible by k={k}")
    if out_features % k != 0:
        problems.append(f"out_features={out_features} is not divisible by k={k}")
    if problems:
        raise KConstraintError(f"{where}: " + "; ".join(problems))


def is_legal_k(r: int, in_features: int, out_features: int, k: int) -> bool:
    """Non-raising form of :func:`validate_k`."""
    try:
        validate_k(r, in_features, out_features, k)
    except KConstraintError:
        return False
    return True


def legal_k_values(
    r: int,
    in_features: int,
    out_features: int,
    candidates: Iterable[int] = DEFAULT_K_CANDIDATES,
) -> list[int]:
    """Return the subset of ``candidates`` legal for this layer, ascending."""
    return sorted(k for k in set(candidates) if is_legal_k(r, in_features, out_features, k))


def module_dims(
    model: nn.Module,
    target_modules: Sequence[str] = DEFAULT_TARGET_MODULES,
) -> dict[str, tuple[int, int]]:
    """Map every targeted ``nn.Linear``'s path to ``(in_features, out_features)``.

    Matches the same way PEFT does: a module is targeted when its dotted path
    ends with one of ``target_modules``.
    """
    suffixes = tuple(target_modules)
    dims: dict[str, tuple[int, int]] = {}
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        canon = canonical_name(name)
        if canon.split(".")[-1] in suffixes:
            dims[canon] = (module.in_features, module.out_features)
    return dims


def validate_k_map(
    dims: Mapping[str, tuple[int, int]],
    r: int,
    k_map: Optional[Mapping[str, int]] = None,
    default_k: int = 2,
) -> dict[str, int]:
    """Resolve and fully validate a k-map against known layer dimensions.

    Returns the resolved map covering every layer in ``dims``. Raises
    :class:`KConstraintError` on the first illegal entry, or :class:`KeyError`
    if ``k_map`` names a layer that does not exist (a silent typo here would
    otherwise degrade to "uniform" without warning).
    """
    k_map = dict(k_map or {})
    unknown = sorted(set(k_map) - set(dims))
    if unknown:
        raise KeyError(
            f"k_map names {len(unknown)} layer(s) not present among the targeted modules: {unknown[:5]}"
        )

    resolved: dict[str, int] = {}
    for name, (in_f, out_f) in dims.items():
        raw = k_map.get(name, default_k)
        # Coercion is wanted for the values that genuinely round-trip as integers
        # (a JSON 4, a numpy int) but not for the ones int() would truncate: an
        # allocator that emits 2.9 would build at k=2 and then record 2 in
        # k_map.json, indistinguishable downstream from a k that was asked for.
        if isinstance(raw, bool) or int(raw) != raw:
            raise KConstraintError(f"{name}: gralora_k must be an integer, got {raw!r}")
        k = int(raw)
        validate_k(r, in_f, out_f, k, where=name)
        resolved[name] = k
    return resolved


def trainable_parameter_count(model: nn.Module) -> int:
    """Total number of trainable parameters (used to assert budget matching)."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_mixed_gralora(
    model: nn.Module,
    r: int,
    *,
    k_map: Optional[Mapping[str, int]] = None,
    default_k: int = 2,
    target_modules: Sequence[str] = DEFAULT_TARGET_MODULES,
    alpha: int = 64,
    gralora_dropout: float = 0.0,
    hybrid_r: int = 0,
    init_weights: bool = True,
    adapter_name: str = "default",
    task_type: Optional[str] = None,
    autocast_adapter_dtype: bool = True,
):
    """Attach GraLoRA to ``model`` with a per-layer ``k``.

    ``k_map`` maps a layer's dotted path (as it appears in the *unwrapped*
    model) to its ``k``; any layer left out gets ``default_k``.

    The model is first wrapped with ``gralora_k=1`` — legal for every layer,
    since 1 divides everything — and each layer is then rebuilt at its target
    ``k``. This keeps construction from tripping over a ``default_k`` that is
    illegal for some layer, and costs nothing extra: the allocation is the same
    size at every ``k``.

    ``autocast_adapter_dtype`` means the same thing it does in
    ``get_peft_model``: keep the adapter tensors in fp32 over a half-precision
    base.
    """
    dims = module_dims(model, target_modules)
    if not dims:
        raise ValueError(
            f"No nn.Linear modules matched target_modules={list(target_modules)}. "
            "Check the module names for this architecture."
        )
    resolved = validate_k_map(dims, r, k_map, default_k)

    config_kwargs = dict(
        r=r,
        gralora_k=1,
        hybrid_r=hybrid_r,
        alpha=alpha,
        gralora_dropout=gralora_dropout,
        target_modules=list(target_modules),
        init_weights=init_weights,
    )
    if task_type is not None:
        config_kwargs["task_type"] = task_type

    peft_model = get_peft_model(
        model,
        GraloraConfig(**config_kwargs),
        adapter_name=adapter_name,
        autocast_adapter_dtype=autocast_adapter_dtype,
    )

    patched = 0
    for name, module in peft_model.named_modules():
        if not isinstance(module, GraloraLayer):
            continue
        canon = canonical_name(name)
        if canon not in resolved:
            raise KeyError(f"GraLoRA layer {canon!r} was wrapped but has no resolved k entry")
        k = resolved[canon]
        layer_config = dataclasses.replace(peft_model.peft_config[adapter_name], gralora_k=k)
        module.update_layer(adapter_name, getattr(module, "module_name", canon), r, config=layer_config)
        patched += 1

    if patched != len(resolved):
        raise RuntimeError(
            f"Expected to patch {len(resolved)} GraLoRA layers but patched {patched}. "
            "PEFT may have skipped a targeted module."
        )

    # get_peft_model upcasts the adapters to fp32 once, at injection time, and
    # update_layer ends by moving each rebuilt tensor to the base layer's dtype —
    # so every layer patched above came back down to fp16/bf16 on a half-precision
    # base. Half-precision adapter weights are the usual cause of NaN loss there,
    # which is exactly what PEFT's default was protecting against.
    peft_model.base_model._cast_adapter_dtype(
        adapter_name=adapter_name, autocast_adapter_dtype=autocast_adapter_dtype
    )

    # The stored config keeps one global k for compatibility with PEFT's loader;
    # the authoritative per-layer map is written out beside it as k_map.json.
    peft_model.peft_config[adapter_name].gralora_k = _representative_k(resolved)
    return peft_model


def _representative_k(resolved: Mapping[str, int]) -> int:
    """The single k stored in ``adapter_config.json``.

    Uniform maps store their real value. Mixed maps store the most common k
    purely as a human-readable hint — ``k_map.json`` is what the loader reads.
    """
    counts: dict[int, int] = {}
    for k in resolved.values():
        counts[k] = counts.get(k, 0) + 1
    return max(counts, key=lambda k: (counts[k], -k))


def current_k_map(peft_model: nn.Module, adapter_name: str = "default") -> dict[str, int]:
    """Read back the per-layer k actually installed on a wrapped model."""
    k_map: dict[str, int] = {}
    for name, module in peft_model.named_modules():
        if isinstance(module, GraloraLayer) and adapter_name in module.gralora_k:
            k_map[canonical_name(name)] = int(module.gralora_k[adapter_name])
    return k_map


# --------------------------------------------------------------------------
# CLI: constraint sweep for a real model (Phase 1.1)
# --------------------------------------------------------------------------


def _dims_for_model(model_name: str, target_modules: Sequence[str]) -> tuple[dict[str, tuple[int, int]], str]:
    """Get per-module-type dims, preferring the real config over the offline table."""
    try:
        from transformers import AutoConfig

        cfg = AutoConfig.from_pretrained(model_name)
        hidden = cfg.hidden_size
        inter = cfg.intermediate_size
        n_heads = cfg.num_attention_heads
        head_dim = getattr(cfg, "head_dim", None) or hidden // n_heads
        n_kv = getattr(cfg, "num_key_value_heads", n_heads)
        dims = {
            "q_proj": (hidden, n_heads * head_dim),
            "k_proj": (hidden, n_kv * head_dim),
            "v_proj": (hidden, n_kv * head_dim),
            "o_proj": (n_heads * head_dim, hidden),
            "gate_proj": (hidden, inter),
            "up_proj": (hidden, inter),
            "down_proj": (inter, hidden),
        }
        source = "transformers AutoConfig"
    except Exception as exc:  # offline, gated repo, unusual architecture
        if model_name not in KNOWN_MODELS:
            raise SystemExit(
                f"Could not load config for {model_name!r} ({exc.__class__.__name__}: {exc}) "
                f"and it is not in KNOWN_MODELS. Add its dims to KNOWN_MODELS in patching.py."
            )
        dims = dict(KNOWN_MODELS[model_name])
        source = "KNOWN_MODELS offline table"

    # Silently dropping an unrecognised name would leave the sweep reporting
    # "all legal" over an empty set of modules — a green light on nothing.
    missing = [m for m in target_modules if m not in dims]
    if missing:
        raise SystemExit(
            f"{missing} are not known projection names for this architecture; "
            f"known: {sorted(dims)}"
        )

    return {m: dims[m] for m in target_modules}, source


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m adagralora.patching",
        description="Report which gralora_k values are legal for a model at a given rank.",
    )
    parser.add_argument("--model", required=True, help="HF model id, e.g. Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--r", type=int, nargs="+", required=True, help="rank(s) to check")
    parser.add_argument(
        "--k",
        type=int,
        nargs="+",
        default=list(DEFAULT_K_CANDIDATES),
        help=f"candidate k values (default: {' '.join(map(str, DEFAULT_K_CANDIDATES))})",
    )
    parser.add_argument("--target-modules", nargs="+", default=list(DEFAULT_TARGET_MODULES))
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = parser.parse_args(argv)

    dims, source = _dims_for_model(args.model, args.target_modules)
    report: dict[str, dict[str, list[int]]] = {}
    ok_overall = True

    for r in args.r:
        per_module = {m: legal_k_values(r, i, o, args.k) for m, (i, o) in dims.items()}
        report[str(r)] = per_module
        if any(not v for v in per_module.values()):
            ok_overall = False

    if args.json:
        print(json.dumps({"model": args.model, "source": source, "dims": {k: list(v) for k, v in dims.items()}, "legal_k": report}, indent=2))
        return 0 if ok_overall else 1

    print(f"model : {args.model}")
    print(f"dims  : {source}")
    print()
    width = max(len(m) for m in dims)
    for r in args.r:
        per_module = report[str(r)]
        common = set(args.k)
        for legal in per_module.values():
            common &= set(legal)
        print(f"r = {r}")
        for m, (in_f, out_f) in dims.items():
            legal = per_module[m]
            mark = " " if legal else "!"
            shown = ", ".join(map(str, legal)) if legal else "NONE"
            print(f"  {mark} {m:<{width}}  in={in_f:<6} out={out_f:<6}  legal k: {shown}")
        print(f"    -> legal for every targeted module: {sorted(common) or 'NONE'}")
        print()

    if not ok_overall:
        print("At least one module has no legal k at some rank. Drop that rank from the sweep.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
