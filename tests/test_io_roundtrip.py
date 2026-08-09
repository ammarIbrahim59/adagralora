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
    """Push every adapter tensor off its initialisation.

    B has to move because it starts at zero. A has to move too, for a subtler
    reason: the save-side and load-side models are built from the same seed
    through the same RNG-consuming sequence, so a freshly initialised A is
    bit-identical to the saved one and every assertion below would hold even if
    A were never loaded at all.
    """
    with torch.no_grad():
        for name, param in peft_model.named_parameters():
            if "gralora" in name:
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


def test_a_uniform_adapter_is_readable_by_stock_peft(tmp_path, tiny_model, sample_input):
    """The uniform baselines are the arm most likely to be handed to an off-the-shelf tool.

    Nothing else pins the builder's write-back of the real k into
    adapter_config.json: drop it and the file records the k=1 scaffold, which
    every sidecar-aware test survives because the sidecar is what they read.
    `test_naive_reload_of_a_mixed_adapter_fails` does not cover it either — it
    asserts a shape mismatch, which is exactly what the broken write-back
    produces too.
    """
    from peft import PeftModel

    peft_model = build_mixed_gralora(tiny_model(seed=1), r=16, default_k=4)
    _train_a_little(peft_model)
    peft_model.eval()
    with torch.no_grad():
        reference = peft_model(sample_input).clone()

    out_dir = str(tmp_path / "uniform")
    save_mixed_adapter(peft_model, out_dir)
    assert json.loads(open(os.path.join(out_dir, "adapter_config.json")).read())["gralora_k"] == 4

    reloaded = PeftModel.from_pretrained(tiny_model(seed=1), out_dir)
    reloaded.eval()
    with torch.no_grad():
        assert torch.allclose(reference, reloaded(sample_input), atol=1e-5)


def test_a_mixed_adapter_records_its_representative_k(saved_mixed_adapter):
    """The stored k is only a hint for a mixed map, but it has to be an honest one."""
    from adagralora.patching import _representative_k

    out_dir, _ = saved_mixed_adapter
    recorded = json.loads(open(os.path.join(out_dir, "adapter_config.json")).read())["gralora_k"]
    assert recorded == _representative_k(read_k_map(out_dir)["k_map"])


def test_roundtrip_restores_every_adapter_tensor_exactly(saved_mixed_adapter, tiny_model):
    """Compares against the file, not against the model's own output.

    Outputs can agree for the wrong reason — see `_train_a_little` — and they
    also cannot distinguish two same-shaped A tensors swapped between layers.
    """
    from safetensors.torch import load_file

    out_dir, _ = saved_mixed_adapter
    reloaded = load_mixed_adapter(tiny_model(seed=0), out_dir)
    live = dict(reloaded.named_parameters())

    saved = load_file(os.path.join(out_dir, "adapter_model.safetensors"))
    assert len(saved) == 28, "expected an A and a B for each of the 14 targeted layers"
    # B was randomised before saving; all-zero tensors would make equality trivial.
    b_tensors = [t for name, t in saved.items() if name.endswith("gralora_B")]
    assert b_tensors and all(t.abs().sum().item() > 0 for t in b_tensors)

    for name, tensor in saved.items():
        # Saved keys are adapter-name-agnostic; the live parameter carries the name.
        assert torch.equal(tensor, live[f"{name}.default"]), name


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

    # The rebuilt-vs-saved comparison runs first but cannot fire here: the model
    # was rebuilt *from* the tampered map, so the two agree. Only the tensor
    # shapes still remember the real k.
    with pytest.raises(RuntimeError, match="size mismatch"):
        load_mixed_adapter(tiny_model(seed=0), out_dir)


def test_load_onto_a_base_model_with_extra_targeted_layers_names_them(saved_mixed_adapter, tiny_model):
    """The realistic operator error: an eval or resume pointed at the wrong base model.

    The layers the checkpoint never covered are the only evidence of it, so the
    error has to name them rather than report an empty diff.
    """
    out_dir, _ = saved_mixed_adapter
    with pytest.raises(RuntimeError, match="Rebuilt k-map") as excinfo:
        load_mixed_adapter(tiny_model(seed=0, n_layers=4), out_dir)
    assert "layers.2." in str(excinfo.value)


def test_load_onto_a_base_model_missing_targeted_layers_names_them(saved_mixed_adapter, tiny_model):
    """The mirror of the case above: a 2-layer checkpoint pointed at a 1-layer base.

    At least as likely as the other direction (evaluating a 1.5B adapter against
    the 0.5B base), so it gets the same error and the same question rather than a
    bare KeyError out of the constraint validator.
    """
    out_dir, _ = saved_mixed_adapter
    with pytest.raises(RuntimeError, match="is this the base model the adapter was trained on") as excinfo:
        load_mixed_adapter(tiny_model(seed=0, n_layers=1), out_dir)
    assert "layers.1." in str(excinfo.value)


def test_roundtrip_of_a_named_adapter(tmp_path, tiny_model, sample_input):
    """A sweep holding two allocators as two named adapters must be able to reload them.

    PEFT puts a non-default adapter one directory down, so the path the caller
    saved to is not the path the sidecar is in — and both have to work.
    """
    peft_model = build_mixed_gralora(
        tiny_model(seed=3), r=16, k_map=MIXED_K_MAP, default_k=2, adapter_name="alloc"
    )
    _train_a_little(peft_model)
    peft_model.eval()
    with torch.no_grad():
        reference = peft_model(sample_input).clone()

    out_dir = str(tmp_path / "named")
    assert save_mixed_adapter(peft_model, out_dir, adapter_name="alloc") == os.path.join(
        out_dir, "alloc", K_MAP_FILENAME
    )

    for path in (out_dir, os.path.join(out_dir, "alloc")):
        reloaded = load_mixed_adapter(tiny_model(seed=3), path, adapter_name="alloc")
        reloaded.eval()
        with torch.no_grad():
            assert torch.allclose(reference, reloaded(sample_input), atol=1e-5), path
        installed = current_k_map(reloaded, "alloc")
        for name, k in MIXED_K_MAP.items():
            assert installed[name] == k


def test_a_named_adapter_is_not_shadowed_by_a_default_one(tmp_path, tiny_model):
    """Two allocators in one run dir is the sweep layout this loader exists for.

    PEFT leaves the default adapter's files at the top level and the named one a
    directory down, so a k_map.json sits at both levels and only the nested one
    belongs to the requested name. Picking the wrong one loads the wrong weights
    under the right name, which nothing downstream can detect.
    """
    out_dir = str(tmp_path / "run")
    save_mixed_adapter(build_mixed_gralora(tiny_model(seed=0), r=16, default_k=2), out_dir)
    save_mixed_adapter(
        build_mixed_gralora(tiny_model(seed=0), r=16, default_k=8, adapter_name="alloc"),
        out_dir,
        adapter_name="alloc",
    )

    reloaded = load_mixed_adapter(tiny_model(seed=0), out_dir, adapter_name="alloc")
    assert set(current_k_map(reloaded, "alloc").values()) == {8}


def test_load_refuses_a_sidecar_saved_from_a_different_adapter(saved_mixed_adapter, tiny_model):
    """A mistyped adapter name must not silently deliver whichever adapter is there."""
    out_dir, _ = saved_mixed_adapter
    with pytest.raises(RuntimeError, match="saved from adapter 'default'"):
        load_mixed_adapter(tiny_model(seed=0), out_dir, adapter_name="alloc")


def test_roundtrip_restores_a_non_default_config(tmp_path, tiny_model, sample_input):
    """alpha, hybrid_r and dropout must come back from the sidecar, not from defaults.

    scaling is alpha / (r + hybrid_r), so an alpha silently reset to the
    GraloraConfig default rescales every adapter at reload time.
    """
    peft_model = build_mixed_gralora(
        tiny_model(seed=2),
        r=16,
        k_map=MIXED_K_MAP,
        default_k=2,
        alpha=16,
        hybrid_r=8,
        gralora_dropout=0.1,
    )
    _train_a_little(peft_model)
    peft_model.eval()
    with torch.no_grad():
        reference = peft_model(sample_input).clone()

    out_dir = str(tmp_path / "nondefault")
    save_mixed_adapter(peft_model, out_dir)
    payload = read_k_map(out_dir)
    assert payload["alpha"] == 16
    assert payload["hybrid_r"] == 8
    assert payload["gralora_dropout"] == pytest.approx(0.1)

    reloaded = load_mixed_adapter(tiny_model(seed=2), out_dir)
    reloaded.eval()
    with torch.no_grad():
        assert torch.allclose(reference, reloaded(sample_input), atol=1e-5)


def _drop_one_tensor(adapter_dir: str) -> str:
    """Truncate the checkpoint the way a killed writer would: one tensor short."""
    from safetensors.torch import load_file, save_file

    path = os.path.join(adapter_dir, "adapter_model.safetensors")
    tensors = load_file(path)
    victim = next(name for name in tensors if "gralora_B" in name)
    del tensors[victim]
    save_file(tensors, path)
    return victim


def test_strict_load_rejects_a_checkpoint_missing_a_tensor(saved_mixed_adapter, tiny_model):
    """The one failure mode shapes cannot catch: a tensor that simply isn't there."""
    out_dir, _ = saved_mixed_adapter
    _drop_one_tensor(out_dir)
    with pytest.raises(RuntimeError, match="missing gralora keys"):
        load_mixed_adapter(tiny_model(seed=0), out_dir)


def test_non_strict_load_tolerates_a_missing_tensor(saved_mixed_adapter, tiny_model):
    out_dir, _ = saved_mixed_adapter
    victim = _drop_one_tensor(out_dir)
    reloaded = load_mixed_adapter(tiny_model(seed=0), out_dir, strict=False)
    # B initialises to zero, so the tensor that never loaded is still exactly zero.
    dropped = dict(reloaded.named_parameters())[f"{victim}.default"]
    assert dropped.abs().sum().item() == 0.0


def test_roundtrip_restores_the_task_type(tmp_path, tiny_causal_lm):
    """A resume path that forgets the keyword must not get a differently-typed wrapper.

    Re-saving a downgraded wrapper writes task_type: null back into the
    checkpoint, so the metadata is lost permanently.
    """
    peft_model = build_mixed_gralora(
        tiny_causal_lm(),
        r=16,
        k_map={"model.layers.0.self_attn.q_proj": 8},
        default_k=2,
        task_type="CAUSAL_LM",
    )
    assert type(peft_model).__name__ == "PeftModelForCausalLM"

    out_dir = str(tmp_path / "causal")
    save_mixed_adapter(peft_model, out_dir)

    reloaded = load_mixed_adapter(tiny_causal_lm(), out_dir)
    assert type(reloaded).__name__ == "PeftModelForCausalLM"


def test_an_explicit_task_type_overrides_what_the_checkpoint_recorded(tmp_path, tiny_causal_lm):
    """The other direction from the test above: the caller wins over the sidecar.

    A checkpoint saved before the task_type was being recorded is exactly the one
    an operator has to override by hand, and only this direction catches a
    regression that makes the saved value unconditionally win.
    """
    peft_model = build_mixed_gralora(tiny_causal_lm(), r=16, default_k=2)
    assert type(peft_model).__name__ == "PeftModel"

    out_dir = str(tmp_path / "untyped")
    save_mixed_adapter(peft_model, out_dir)
    assert json.loads(open(os.path.join(out_dir, "adapter_config.json")).read())["task_type"] is None

    reloaded = load_mixed_adapter(tiny_causal_lm(), out_dir, task_type="CAUSAL_LM")
    assert type(reloaded).__name__ == "PeftModelForCausalLM"


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


def test_save_run_metadata_records_whether_the_tree_was_dirty(tmp_path):
    """A commit hash alone names code that may never have run."""
    metadata = json.loads(open(save_run_metadata(str(tmp_path / "run"))).read())
    assert "git_dirty" in metadata
    # Tied to the sibling field rather than asserted outright: in a source tarball
    # with no .git both are legitimately None, but wherever git can answer at all
    # it has to answer this too, or a broken status call degrades every run's
    # provenance to "unknown" with nothing noticing.
    if metadata["git_commit"] is not None:
        assert isinstance(metadata["git_dirty"], bool)


def test_save_run_metadata_refuses_to_let_extra_shadow_provenance(tmp_path):
    with pytest.raises(ValueError, match="git_commit"):
        save_run_metadata(str(tmp_path / "run"), {"git_commit": "deadbeef"})
