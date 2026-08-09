"""Assumptions the mixed-k patcher makes about the installed PEFT.

These are cheap tripwires: if a peft upgrade changes any of them, the patcher
breaks silently or subtly, and a training run is what pays for it.
"""

from __future__ import annotations

import inspect
import os
import re


def test_update_layer_has_the_signature_the_patcher_calls():
    from peft.tuners.gralora import GraloraLayer

    params = list(inspect.signature(GraloraLayer.update_layer).parameters)
    assert params == ["self", "adapter_name", "module_name", "r", "config"]


def test_gralora_config_carries_a_single_global_k():
    from peft import GraloraConfig

    assert "gralora_k" in GraloraConfig.__dataclass_fields__


def test_gralora_config_enforces_rank_divisibility_itself():
    import pytest
    from peft import GraloraConfig

    with pytest.raises(ValueError, match="divisible"):
        GraloraConfig(r=12, gralora_k=8, target_modules=["q_proj"])


def test_gralora_has_no_bitsandbytes_backend():
    """4-bit training is not an option; --load_in_4bit must never be added."""
    import peft.tuners.gralora as gralora_pkg

    files = os.listdir(os.path.dirname(gralora_pkg.__file__))
    assert "bnb.py" not in files


def _floor(requires_python: str) -> tuple[int, ...]:
    """Parse a '>=X.Y' specifier into a comparable (major, minor)."""
    return tuple(int(part) for part in requires_python.strip().lstrip(">=").split(".")[:2])


def test_declared_python_floor_is_at_least_what_the_pinned_peft_needs():
    """A floor below peft's promises an install pip cannot actually resolve.

    The CI matrix starts at 3.10, so nothing else exercises the declared floor.
    """
    from importlib.metadata import metadata

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(root, "pyproject.toml"), encoding="utf-8") as fh:
        declared = re.search(r'requires-python\s*=\s*"([^"]+)"', fh.read()).group(1)

    assert _floor(declared) >= _floor(metadata("peft")["Requires-Python"])


def test_adagralora_exposes_its_phase_zero_surface():
    import adagralora

    for name in ("build_mixed_gralora", "save_mixed_adapter", "load_mixed_adapter", "validate_k"):
        assert hasattr(adagralora, name), name
