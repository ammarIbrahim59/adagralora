"""AdaGraLoRA — layer-wise adaptive block allocation for GraLoRA.

Phase 0 surface: constraint validation, the mixed-k patcher, and checkpoint I/O
that survives a save -> reload round trip with a non-uniform k allocation.

Names are resolved lazily (PEP 562) so that ``python -m adagralora.patching``
does not import the submodule twice and warn about it.
"""

from __future__ import annotations

from typing import Any

__version__ = "0.1.0"

_EXPORTS = {
    "DEFAULT_TARGET_MODULES": "adagralora.patching",
    "KNOWN_MODELS": "adagralora.patching",
    "KConstraintError": "adagralora.patching",
    "build_mixed_gralora": "adagralora.patching",
    "current_k_map": "adagralora.patching",
    "legal_k_values": "adagralora.patching",
    "module_dims": "adagralora.patching",
    "trainable_parameter_count": "adagralora.patching",
    "validate_k": "adagralora.patching",
    "validate_k_map": "adagralora.patching",
    "K_MAP_FILENAME": "adagralora.io_utils",
    "METADATA_FILENAME": "adagralora.io_utils",
    "load_mixed_adapter": "adagralora.io_utils",
    "read_k_map": "adagralora.io_utils",
    "save_mixed_adapter": "adagralora.io_utils",
    "save_run_metadata": "adagralora.io_utils",
}

__all__ = ["__version__", *sorted(_EXPORTS)]


def __getattr__(name: str) -> Any:
    module_path = _EXPORTS.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(module_path), name)


def __dir__() -> list[str]:
    return sorted(__all__)
