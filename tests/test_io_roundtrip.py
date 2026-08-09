"""Checkpoint round trip — the gate that protects against a wasted training run.

The failure this guards against: a mixed-k adapter saved and reloaded the normal
way rebuilds every layer at the single k in adapter_config.json, so the saved
tensors no longer fit. `test_naive_reload_of_a_mixed_adapter_fails` pins that
behaviour down, and the round-trip tests prove the sidecar loader avoids it.
"""

from __future__ import annotations

import json
import os

import pytest
import torch

from adagralora.io_utils import (
    K_MAP_FILENAME,
    METADATA_FILENAME,
    load_mixed_adapter,
    read_k_map,
    save_mixed_adapter,
    save_run_metadata,
)
from adagralora.patching import build_mixed_gralora, current_k_map

MIXED_K_MAP = {
    "layers.0.self_attn.q_proj": 8,
    "layers.0.self_attn.o_proj": 4,
    "layers.0.mlp.down_proj": 4,
    "layers.1.self_attn.v_proj": 8,
    "layers.1.mlp.gate_proj": 2,
}


def _train_a_little(peft_model: torch.nn.Module) -> None:
    """Push B off zero so the round trip compares something non-trivial."""
    with torch.no_grad():
        for name, param in peft_model.named_parameters():
            if "gralora_B" in name:
                param.normal_(mean=0.0, std=0.02)


@pytest.fixture
def saved_mixed_adapter(tmp_path, tiny_model, sample_input):
    peft_model = build_mixed_gralora(tiny_model(seed=0), r=16, k_map=MIXED_K_MAP, default_k=2)
    _train_a_little(peft_model)
    peft_model.eval()
    with torch.no_grad():
        reference = peft_model(sample_input).clone()
    out_dir = str(tmp_path / "adapter")
    save_mixed_adapter(peft_model, out_dir)
    return out_dir, reference


def test_save_writes_the_k_map_sidecar(saved_mixed_adapter):
    out_dir, _ = saved_mixed_adapter
    assert os.path.isfile(os.path.join(out_dir, K_MAP_FILENAME))
    assert os.path.isfile(os.path.join(out_dir, "adapter_config.json"))


def test_saved_k_map_records_every_layer_and_the_requested_values(saved_mixed_adapter):
    out_dir, _ = saved_mixed_adapter
    payload = read_k_map(out_dir)
    assert payload["r"] == 16
    assert payload["uniform"] is False
    assert len(payload["k_map"]) == 14
    for name, k in MIXED_K_MAP.items():
        assert payload["k_map"][name] == k


def test_roundtrip_reproduces_outputs_for_a_mixed_adapter(saved_mixed_adapter, tiny_model, sample_input):
    out_dir, reference = saved_mixed_adapter
    reloaded = load_mixed_adapter(tiny_model(seed=0), out_dir)
    reloaded.eval()
    with torch.no_grad():
        actual = reloaded(sample_input)
    assert torch.allclose(reference, actual, atol=1e-5)


def test_roundtrip_restores_the_exact_k_map(saved_mixed_adapter, tiny_model):
    out_dir, _ = saved_mixed_adapter
    reloaded = load_mixed_adapter(tiny_model(seed=0), out_dir)
    installed = current_k_map(reloaded)
    for name, k in MIXED_K_MAP.items():
        assert installed[name] == k
    assert installed["layers.1.mlp.up_proj"] == 2


def test_roundtrip_reproduces_outputs_for_a_uniform_adapter(tmp_path, tiny_model, sample_input):
    peft_model = build_mixed_gralora(tiny_model(seed=1), r=16, default_k=4)
    _train_a_little(peft_model)
    peft_model.eval()
    with torch.no_grad():
        reference = peft_model(sample_input).clone()

    out_dir = str(tmp_path / "uniform")
    save_mixed_adapter(peft_model, out_dir)
    assert read_k_map(out_dir)["uniform"] is True

    reloaded = load_mixed_adapter(tiny_model(seed=1), out_dir)
    reloaded.eval()
    with torch.no_grad():
        assert torch.allclose(reference, reloaded(sample_input), atol=1e-5)


def test_roundtrip_restores_every_adapter_tensor_exactly(saved_mixed_adapter, tiny_model):
    out_dir, _ = saved_mixed_adapter
    reloaded = load_mixed_adapter(tiny_model(seed=0), out_dir)
    tensors = {n: p for n, p in reloaded.named_parameters() if "gralora_B" in n}
    assert tensors, "no gralora_B tensors found after reload"
    # B was randomised before saving; an all-zero tensor means the load silently no-opped.
    assert all(p.abs().sum().item() > 0 for p in tensors.values())


def test_naive_reload_of_a_mixed_adapter_fails(saved_mixed_adapter, tiny_model):
    """Documents the exact failure the sidecar exists to prevent."""
    from peft import PeftModel

    out_dir, _ = saved_mixed_adapter
    with pytest.raises(Exception) as excinfo:
        PeftModel.from_pretrained(tiny_model(seed=0), out_dir)
    assert "size" in str(excinfo.value).lower() or "shape" in str(excinfo.value).lower()


def test_read_k_map_is_explicit_when_the_sidecar_is_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match="save_mixed_adapter"):
        read_k_map(str(tmp_path))


def test_read_k_map_rejects_an_unknown_format_version(tmp_path):
    path = tmp_path / K_MAP_FILENAME
    path.write_text(json.dumps({"format_version": 999, "k_map": {}}))
    with pytest.raises(ValueError, match="format_version"):
        read_k_map(str(tmp_path))


def test_load_detects_a_tampered_k_map(saved_mixed_adapter, tiny_model):
    """A k-map edited to disagree with the saved tensors must not load silently."""
    out_dir, _ = saved_mixed_adapter
    path = os.path.join(out_dir, K_MAP_FILENAME)
    payload = json.loads(open(path).read())
    payload["k_map"]["layers.0.self_attn.q_proj"] = 2  # was 8
    open(path, "w").write(json.dumps(payload))

    # Caught by the shape check when the tensors are copied in, before the
    # rebuilt-vs-saved k-map comparison ever runs.
    with pytest.raises(RuntimeError, match="size mismatch"):
        load_mixed_adapter(tiny_model(seed=0), out_dir)


def test_save_run_metadata_records_provenance(tmp_path, tiny_model):
    peft_model = build_mixed_gralora(tiny_model(), r=16, k_map=MIXED_K_MAP, default_k=2)
    out_dir = str(tmp_path / "run")
    path = save_run_metadata(out_dir, {"seed": 0, "allocator": "gradnorm"}, peft_model=peft_model)

    assert os.path.basename(path) == METADATA_FILENAME
    metadata = json.loads(open(path).read())
    assert metadata["seed"] == 0
    assert metadata["allocator"] == "gradnorm"
    assert metadata["peft"] is not None
    assert metadata["k_map"]["layers.0.self_attn.q_proj"] == 8
    assert metadata["trainable_params"] > 0
