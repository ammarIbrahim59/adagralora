"""Checkpoint I/O for mixed-k GraLoRA adapters.

Why this module exists
----------------------
``adapter_config.json`` has room for exactly one global ``gralora_k``. Saving a
mixed-k adapter the normal way and reloading it with ``PeftModel.from_pretrained``
rebuilds every layer at that single ``k``, so the checkpoint's tensors no longer
match the layers they are being loaded into. The failure is a reproducible
``RuntimeError`` about mismatched shapes — and it only surfaces *after* training,
when the run is already spent.

The fix is a ``k_map.json`` sidecar written next to the adapter, plus a loader
that rebuilds the layers at their saved ``k`` *before* any weights are loaded.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
from typing import Any, Mapping, Optional, Sequence

import torch.nn as nn

from peft import PeftModel
from peft.utils.save_and_load import load_peft_weights, set_peft_model_state_dict

from adagralora.patching import (
    build_mixed_gralora,
    current_k_map,
    module_dims,
    trainable_parameter_count,
)

K_MAP_FILENAME = "k_map.json"
METADATA_FILENAME = "run_metadata.json"
#: Either name PEFT will have written the adapter tensors under.
_WEIGHT_FILENAMES = ("adapter_model.safetensors", "adapter_model.bin")
_K_MAP_FORMAT_VERSION = 1


def save_mixed_adapter(
    peft_model: PeftModel,
    output_dir: str,
    adapter_name: str = "default",
    **save_kwargs: Any,
) -> str:
    """Save a GraLoRA adapter together with its per-layer k-map.

    Returns the path to the written ``k_map.json``.
    """
    os.makedirs(output_dir, exist_ok=True)
    peft_model.save_pretrained(output_dir, selected_adapters=[adapter_name], **save_kwargs)

    # PEFT writes non-default adapters into a subdirectory of their own.
    adapter_dir = output_dir if adapter_name == "default" else os.path.join(output_dir, adapter_name)

    k_map = current_k_map(peft_model, adapter_name)
    if not k_map:
        raise ValueError(f"No GraLoRA layers found for adapter {adapter_name!r}; nothing to save.")

    config = peft_model.peft_config[adapter_name]
    payload = {
        "format_version": _K_MAP_FORMAT_VERSION,
        "adapter_name": adapter_name,
        "r": int(config.r),
        "alpha": int(config.alpha),
        "hybrid_r": int(config.hybrid_r),
        "gralora_dropout": float(config.gralora_dropout),
        "target_modules": sorted(config.target_modules) if config.target_modules else None,
        "uniform": len(set(k_map.values())) == 1,
        "k_map": dict(sorted(k_map.items())),
    }

    path = os.path.join(adapter_dir, K_MAP_FILENAME)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    return path


def read_k_map(adapter_dir: str) -> dict[str, Any]:
    """Load the ``k_map.json`` sidecar, with a clear error if it is missing."""
    path = os.path.join(adapter_dir, K_MAP_FILENAME)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"{path} not found. This adapter was not saved with save_mixed_adapter, so its "
            "per-layer k allocation is unknown and it cannot be safely reloaded."
        )
    with open(path, encoding="utf-8") as fh:
        payload = json.load(fh)
    version = payload.get("format_version")
    if version != _K_MAP_FORMAT_VERSION:
        raise ValueError(f"Unsupported k_map format_version {version!r} (expected {_K_MAP_FORMAT_VERSION}).")
    return payload


def _resolve_adapter_dir(adapter_dir: str, adapter_name: str) -> str:
    """Find the directory the sidecar actually landed in.

    ``save_mixed_adapter`` follows PEFT and puts a non-default adapter one level
    down, so the directory a caller saved *to* is not the directory the sidecar
    is *in*. Accept either, or the loader tells an operator their checkpoint was
    never saved by ``save_mixed_adapter`` when in fact it was.

    The subdirectory has to win over the directory itself: a sweep that saves a
    default adapter and a named one into one run dir leaves a k_map.json at both
    levels, and only the nested one belongs to the requested name.
    """
    if adapter_name != "default":
        nested = os.path.join(adapter_dir, adapter_name)
        if os.path.isfile(os.path.join(nested, K_MAP_FILENAME)):
            return nested
    return adapter_dir


def load_mixed_adapter(
    base_model: nn.Module,
    adapter_dir: str,
    adapter_name: str = "default",
    *,
    task_type: Optional[str] = None,
    strict: bool = True,
) -> PeftModel:
    """Rebuild a mixed-k adapter onto ``base_model`` and load its weights.

    Layers are constructed at their saved ``k`` first, so the shapes match
    before any tensor is copied in.

    ``adapter_dir`` may be either the directory ``save_mixed_adapter`` was given
    or, for a non-default ``adapter_name``, the subdirectory it wrote into.

    ``task_type`` overrides what the checkpoint recorded; left unset, the
    saved value is used, so an evaluate/resume path cannot silently downgrade
    a ``PeftModelForCausalLM`` to a bare ``PeftModel``.
    """
    adapter_dir = _resolve_adapter_dir(adapter_dir, adapter_name)
    payload = read_k_map(adapter_dir)
    # Resolution can only pick a directory; the sidecar is the one record of which
    # adapter is actually in it. Without this check a run dir holding more than one
    # hands back whichever one resolution landed on, under the requested name.
    saved_name = payload.get("adapter_name")
    if saved_name is not None and saved_name != adapter_name:
        raise RuntimeError(
            f"{os.path.join(adapter_dir, K_MAP_FILENAME)} was saved from adapter {saved_name!r}, "
            f"but {adapter_name!r} was requested. Point adapter_dir at that adapter's own "
            f"directory, or load it under the name it was saved with."
        )
    # load_peft_weights treats a directory with no weights file as a Hub repo id, so
    # a checkpoint that lost its tensors (a copy that skipped the large binaries)
    # would rebuild the whole model and then fail complaining about a repo named
    # "runs/seed0" — or actually go looking for one online.
    if not any(os.path.isfile(os.path.join(adapter_dir, name)) for name in _WEIGHT_FILENAMES):
        raise FileNotFoundError(
            f"{adapter_dir} has a {K_MAP_FILENAME} but none of {list(_WEIGHT_FILENAMES)}. "
            "This checkpoint is incomplete — its adapter weights were never written or did not "
            "survive the copy."
        )

    k_map = {name: int(k) for name, k in payload["k_map"].items()}

    with open(os.path.join(adapter_dir, "adapter_config.json"), encoding="utf-8") as fh:
        adapter_config = json.load(fh)

    target_modules = payload.get("target_modules") or adapter_config.get("target_modules")
    if target_modules is None:
        raise ValueError("Cannot determine target_modules from the saved adapter.")

    if task_type is None:
        task_type = adapter_config.get("task_type")

    # The mirror of the check below: a checkpoint covering layers this base model
    # does not have. The builder's validator would raise a bare KeyError naming
    # them but never naming the cause, and the cause is the same wrong-base-model
    # mistake, so it gets the same error type and the same question.
    targeted = module_dims(base_model, sorted(target_modules))
    absent = sorted(set(k_map) - set(targeted))
    if absent:
        raise RuntimeError(
            f"The checkpoint covers layers this base model does not target.\n"
            f"  in the checkpoint, not targeted here : {absent[:5]}{' ...' if len(absent) > 5 else ''}\n"
            f"The base model has {len(targeted)} targeted layers and the checkpoint covers "
            f"{len(k_map)} — is this the base model the adapter was trained on?"
        )

    peft_model = build_mixed_gralora(
        base_model,
        r=int(payload["r"]),
        k_map=k_map,
        default_k=int(adapter_config.get("gralora_k", 2)),
        target_modules=sorted(target_modules),
        alpha=int(payload["alpha"]),
        gralora_dropout=float(payload["gralora_dropout"]),
        hybrid_r=int(payload["hybrid_r"]),
        # This is the branch that zero-initialises gralora_B, so a tensor that
        # fails to load stays exactly zero and is detectable instead of being
        # masked by plausible-looking random values.
        init_weights=True,
        adapter_name=adapter_name,
        task_type=task_type,
    )

    # A postcondition of the builder, checked before any weight is copied: once
    # the tensors are in, a k disagreement only surfaces as a shape mismatch.
    reloaded = current_k_map(peft_model, adapter_name)
    if reloaded != k_map:
        # Diffed both ways round: the disagreement this check uniquely catches is a
        # base model whose targeted layers the checkpoint never covered (an eval or
        # resume pointed at the wrong-size base), and that shows up only as names
        # the saved map has never heard of.
        uncovered = sorted(set(reloaded) - set(k_map))
        absent = sorted(set(k_map) - set(reloaded))
        differing = sorted(n for n in set(k_map) & set(reloaded) if reloaded[n] != k_map[n])
        raise RuntimeError(
            f"Rebuilt k-map does not match the saved one.\n"
            f"  different k                          : {differing[:5]}{' ...' if len(differing) > 5 else ''}\n"
            f"  targeted here, not in the checkpoint : {uncovered[:5]}{' ...' if len(uncovered) > 5 else ''}\n"
            f"  in the checkpoint, not targeted here : {absent[:5]}{' ...' if len(absent) > 5 else ''}\n"
            f"The base model has {len(reloaded)} targeted layers and the checkpoint covers "
            f"{len(k_map)} — is this the base model the adapter was trained on?"
        )

    state_dict = load_peft_weights(adapter_dir)
    result = set_peft_model_state_dict(peft_model, state_dict, adapter_name=adapter_name)

    if strict:
        unexpected = list(getattr(result, "unexpected_keys", []) or [])
        # Base-model and non-adapter keys are legitimately absent from an adapter
        # checkpoint; only missing *adapter* tensors indicate a real problem.
        missing = [k for k in (getattr(result, "missing_keys", []) or []) if "gralora" in k]
        if missing or unexpected:
            raise RuntimeError(
                f"Adapter state dict did not match the rebuilt model.\n"
                f"  missing gralora keys : {missing[:5]}{' ...' if len(missing) > 5 else ''}\n"
                f"  unexpected keys      : {unexpected[:5]}{' ...' if len(unexpected) > 5 else ''}"
            )

    return peft_model


def _git(args: Sequence[str], cwd: Optional[str] = None) -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", *args],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        return out.stdout.strip()
    except Exception:
        return None


def _git_commit(cwd: Optional[str] = None) -> Optional[str]:
    return _git(["rev-parse", "HEAD"], cwd)


def _git_dirty(cwd: Optional[str] = None) -> Optional[bool]:
    """Whether the tree had uncommitted edits. ``None`` if git could not answer.

    A commit hash alone is misleading provenance when the run was launched with
    local edits: it names code that never executed.
    """
    status = _git(["status", "--porcelain"], cwd)
    return None if status is None else bool(status)


def save_run_metadata(
    output_dir: str,
    extra: Optional[Mapping[str, Any]] = None,
    *,
    peft_model: Optional[PeftModel] = None,
    adapter_name: str = "default",
) -> str:
    """Write provenance for a run so every number in the paper traces to a commit.

    Deliberately avoids wall-clock-only identifiers: the git commit (plus
    whether the tree was dirty when it ran) and the resolved k-map are what
    make a result reproducible.
    """
    os.makedirs(output_dir, exist_ok=True)

    repo = os.path.dirname(os.path.abspath(__file__))
    metadata: dict[str, Any] = {
        "git_commit": _git_commit(repo),
        "git_dirty": _git_dirty(repo),
        "python": platform.python_version(),
        "platform": platform.platform(),
    }

    try:
        import torch

        metadata["torch"] = torch.__version__
        metadata["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            metadata["gpu"] = torch.cuda.get_device_name(0)
            # Named for what it is: the allocator's high-water mark since the process
            # started, not this run's peak. Nothing here owns a "run begins" moment at
            # which to call reset_peak_memory_stats, so a driver that trains several
            # allocations in one process records the max over all of them — a number
            # that would look like "memory does not vary with k" if it were reported
            # as this run's.
            metadata["peak_vram_bytes_process"] = int(torch.cuda.max_memory_allocated())
    except Exception:
        pass

    for module_name in ("peft", "transformers", "trl", "datasets"):
        try:
            metadata[module_name] = __import__(module_name).__version__
        except Exception:
            metadata[module_name] = None

    if peft_model is not None:
        metadata["k_map"] = current_k_map(peft_model, adapter_name)
        # Via the helper the budget-matching tests already pin, so the number in the
        # paper and the number those tests assert cannot drift apart.
        metadata["trainable_params"] = trainable_parameter_count(peft_model)

    if extra:
        # Flat layout keeps downstream aggregation simple, so a caller key that
        # collides with a provenance field has to be a hard error rather than a
        # silent overwrite of the very thing this file exists to record.
        collisions = sorted(set(extra) & set(metadata))
        if collisions:
            raise ValueError(f"extra may not overwrite recorded provenance fields: {collisions}")
        metadata.update(dict(extra))

    path = os.path.join(output_dir, METADATA_FILENAME)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2, default=str)
    return path
