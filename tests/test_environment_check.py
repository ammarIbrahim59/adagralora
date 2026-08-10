"""The pre-flight checker's own FAIL paths.

CI runs `python environment_check.py` and asserts only that a *healthy* machine
exits 0, so every guard inside it was reachable by no test: the version floors,
the constraint sweep, and — the one that matters — the tripwire that catches
`GraloraLayer.update_layer` drifting away from the signature the mixed-k patcher
calls. That tripwire is the whole reason peft is pinned, and it is also the check
most likely to be softened by someone unblocking an upgrade.
"""

from __future__ import annotations

import importlib.util
import os

import pytest


@pytest.fixture(scope="module")
def env_check():
    """Import the checker by path; it lives at the repo root, not in the package."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    spec = importlib.util.spec_from_file_location(
        "environment_check", os.path.join(root, "environment_check.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_version_tuple_parses_release_and_prerelease_versions(env_check):
    assert env_check._version_tuple("2.9.1") == (2, 9, 1)
    assert env_check._version_tuple("4.57.0.dev0") == (4, 57, 0)
    assert env_check._version_tuple("0.20.0") < (0, 21)


def test_check_gralora_fails_when_update_layers_signature_drifts(env_check, monkeypatch, capsys):
    """The pin's reason for existing, simulated by an upgrade that adds a parameter."""
    from peft.tuners.gralora import GraloraLayer

    def update_layer(self, adapter_name, module_name, r, config, init_lora_weights=True):
        raise AssertionError("not called")

    monkeypatch.setattr(GraloraLayer, "update_layer", update_layer)
    assert env_check.check_gralora() is False
    assert "update_layer signature changed" in capsys.readouterr().out


def test_check_packages_fails_below_a_version_floor(env_check, monkeypatch, capsys):
    monkeypatch.setattr(env_check, "MIN_VERSIONS", {"torch": (999, 0)})
    assert env_check.check_packages() is False
    assert "below the floor" in capsys.readouterr().out


def test_check_constraints_fails_for_a_model_with_no_legal_k(env_check, monkeypatch, capsys):
    """An odd in_features leaves k in {2,4,8} illegal at every planned rank."""
    from adagralora.patching import KNOWN_MODELS

    monkeypatch.setitem(KNOWN_MODELS, "some-org/odd-dims", {"q_proj": (97, 96)})
    assert env_check.check_constraints() is False
    assert "no k legal for every module" in capsys.readouterr().out


def test_main_exits_non_zero_when_any_single_check_fails(env_check, monkeypatch):
    """Every sub-check has to be able to fail the run on its own."""
    monkeypatch.setattr(env_check, "check_constraints", lambda: False)
    assert env_check.main() == 1
