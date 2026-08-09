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
from typing import Any, Mapping, Optional

import torch.nn as nn

from peft import PeftModel
from peft.utils.save_and_load import load_peft_weights, set_peft_model_state_dict

from adagralora.patching import build_mixed_gralora, current_k_map

K_MAP_FILENAME = "k_map.json"
METADATA_FILENAME = "run_metadata.json"
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
    """
    payload = read_k_map(adapter_dir)
    k_map = {name: int(k) for name, k in payload["k_map"].items()}

    with open(os.path.join(adapter_dir, "adapter_config.json"), encoding="utf-8") as fh:
        adapter_config = json.load(fh)

    target_modules = payload.get("target_modules") or adapter_config.get("target_modules")
    if target_modules is None:
        raise ValueError("Cannot determine target_modules from the saved adapter.")

    peft_model = build_mixed_gralora(
        base_model,
        r=int(payload["r"]),
        k_map=k_map,
        default_k=int(adapter_config.get("gralora_k", 2)),
        target_modules=sorted(target_modules),
        alpha=int(payload["alpha"]),
        gralora_dropout=float(payload["gralora_dropout"]),
        hybrid_r=int(payload["hybrid_r"]),
        # Weights are about to be overwritten; skip the init RNG work.
        init_weights=True,
        adapter_name=adapter_name,
        task_type=task_type,
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

    reloaded = current_k_map(peft_model, adapter_name)
    if reloaded != k_map:
        differing = [n for n in k_map if reloaded.get(n) != k_map[n]]
        raise RuntimeError(f"Rebuilt k-map does not match the saved one at: {differing[:5]}")

    return peft_model


def _git_commit(cwd: Optional[str] = None) -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        return out.stdout.strip()
    except Exception:
        return None


def save_run_metadata(
    output_dir: str,
    extra: Optional[Mapping[str, Any]] = None,
    *,
    peft_model: Optional[PeftModel] = None,
    adapter_name: str = "default",
) -> str:
    """Write provenance for a run so every number in the paper traces to a commit.

    Deliberately avoids wall-clock-only identifiers: the git commit and the
    resolved k-map are what make a result reproducible.
    """
    os.makedirs(output_dir, exist_ok=True)

    metadata: dict[str, Any] = {
        "git_commit": _git_commit(os.path.dirname(os.path.abspath(__file__))),
        "python": platform.python_version(),
        "platform": platform.platform(),
    }

    try:
        import torch

        metadata["torch"] = torch.__version__
        metadata["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            metadata["gpu"] = torch.cuda.get_device_name(0)
            metadata["peak_vram_bytes"] = int(torch.cuda.max_memory_allocated())
    except Exception:
        pass

    for module_name in ("peft", "transformers", "trl", "datasets"):
        try:
            metadata[module_name] = __import__(module_name).__version__
        except Exception:
            metadata[module_name] = None

    if peft_model is not None:
        metadata["k_map"] = current_k_map(peft_model, adapter_name)
        metadata["trainable_params"] = sum(
            p.numel() for p in peft_model.parameters() if p.requires_grad
        )

    if extra:
        metadata.update(dict(extra))

    path = os.path.join(output_dir, METADATA_FILENAME)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2, default=str)
    return path
