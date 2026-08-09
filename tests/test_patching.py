"""Constraint validation and the mixed-k builder."""

from __future__ import annotations

import pytest
import torch

from adagralora.patching import (
    DEFAULT_TARGET_MODULES,
    KNOWN_MODELS,
    KConstraintError,
    _representative_k,
    build_mixed_gralora,
    canonical_name,
    current_k_map,
    is_legal_k,
    legal_k_values,
    module_dims,
    trainable_parameter_count,
    validate_k,
    validate_k_map,
)


# --- validate_k ------------------------------------------------------------


def test_validate_k_accepts_a_legal_combination():
    validate_k(r=16, in_features=96, out_features=24, k=8)


def test_validate_k_rejects_rank_not_divisible():
    with pytest.raises(KConstraintError, match=r"r=12 is not divisible by k=8"):
        validate_k(r=12, in_features=96, out_features=24, k=8)


def test_validate_k_rejects_in_features_not_divisible():
    with pytest.raises(KConstraintError, match=r"in_features=100 is not divisible"):
        validate_k(r=16, in_features=100, out_features=24, k=8)


def test_validate_k_rejects_out_features_not_divisible():
    with pytest.raises(KConstraintError, match=r"out_features=24 is not divisible by k=16"):
        validate_k(r=16, in_features=96, out_features=24, k=16)


def test_validate_k_reports_every_violation_at_once():
    with pytest.raises(KConstraintError) as excinfo:
        validate_k(r=12, in_features=100, out_features=30, k=8)
    message = str(excinfo.value)
    assert "r=12" in message and "in_features=100" in message and "out_features=30" in message


def test_validate_k_rejects_non_positive_k():
    with pytest.raises(KConstraintError, match="positive integer"):
        validate_k(r=16, in_features=96, out_features=24, k=0)


def test_validate_k_includes_the_layer_name_in_the_error():
    with pytest.raises(KConstraintError, match="layers.0.self_attn.k_proj"):
        validate_k(r=16, in_features=96, out_features=24, k=16, where="layers.0.self_attn.k_proj")


def test_is_legal_k_is_the_non_raising_form():
    assert is_legal_k(16, 96, 24, 8) is True
    assert is_legal_k(16, 96, 24, 16) is False


# --- legal_k_values --------------------------------------------------------


def test_legal_k_values_filters_to_the_legal_subset():
    assert legal_k_values(r=16, in_features=96, out_features=24, candidates=(2, 4, 8, 16)) == [2, 4, 8]


def test_legal_k_values_can_return_empty():
    assert legal_k_values(r=3, in_features=96, out_features=24, candidates=(2, 4, 8)) == []


def test_legal_k_values_dedupes_and_sorts():
    assert legal_k_values(r=16, in_features=96, out_features=96, candidates=(8, 2, 2, 4)) == [2, 4, 8]


# --- name handling and discovery ------------------------------------------


def test_canonical_name_strips_the_peft_wrapper_prefix():
    assert canonical_name("base_model.model.layers.0.self_attn.q_proj") == "layers.0.self_attn.q_proj"
    assert canonical_name("layers.0.self_attn.q_proj") == "layers.0.self_attn.q_proj"


def test_module_dims_finds_every_targeted_linear(tiny_model):
    dims = module_dims(tiny_model())
    # 7 targeted projections in each of 2 blocks
    assert len(dims) == 14
    assert dims["layers.0.self_attn.k_proj"] == (96, 24)
    assert dims["layers.1.mlp.down_proj"] == (192, 96)


def test_module_dims_respects_a_narrower_target_list(tiny_model):
    dims = module_dims(tiny_model(), target_modules=("q_proj", "v_proj"))
    assert set(name.split(".")[-1] for name in dims) == {"q_proj", "v_proj"}
    assert len(dims) == 4


# --- validate_k_map --------------------------------------------------------


def test_validate_k_map_fills_unlisted_layers_with_the_default(tiny_model):
    dims = module_dims(tiny_model())
    resolved = validate_k_map(dims, r=16, k_map={"layers.0.mlp.down_proj": 8}, default_k=2)
    assert resolved["layers.0.mlp.down_proj"] == 8
    assert resolved["layers.1.mlp.down_proj"] == 2
    assert len(resolved) == len(dims)


def test_validate_k_map_rejects_an_unknown_layer_name(tiny_model):
    dims = module_dims(tiny_model())
    with pytest.raises(KeyError, match="not present"):
        validate_k_map(dims, r=16, k_map={"layers.9.mlp.down_proj": 4})


def test_validate_k_map_rejects_an_illegal_entry(tiny_model):
    dims = module_dims(tiny_model())
    with pytest.raises(KConstraintError, match="k_proj"):
        validate_k_map(dims, r=16, k_map={"layers.0.self_attn.k_proj": 16}, default_k=2)


# --- building --------------------------------------------------------------


def test_build_uniform_installs_the_requested_k(tiny_model, sample_input):
    peft_model = build_mixed_gralora(tiny_model(), r=16, default_k=4)
    k_map = current_k_map(peft_model)
    assert len(k_map) == 14
    assert set(k_map.values()) == {4}
    assert peft_model(sample_input).shape == sample_input.shape


def test_build_mixed_installs_a_different_k_per_layer(tiny_model):
    requested = {
        "layers.0.self_attn.q_proj": 8,
        "layers.0.mlp.down_proj": 4,
        "layers.1.self_attn.v_proj": 2,
    }
    peft_model = build_mixed_gralora(tiny_model(), r=16, k_map=requested, default_k=2)
    installed = current_k_map(peft_model)
    for name, k in requested.items():
        assert installed[name] == k
    assert installed["layers.1.mlp.up_proj"] == 2


def test_build_rejects_an_illegal_k_before_allocating(tiny_model, monkeypatch):
    import peft

    def explode(*args, **kwargs):
        raise AssertionError("get_peft_model was called despite an illegal k")

    monkeypatch.setattr("adagralora.patching.get_peft_model", explode)
    with pytest.raises(KConstraintError):
        build_mixed_gralora(tiny_model(), r=16, k_map={"layers.0.self_attn.k_proj": 16})


def test_build_rejects_a_target_list_that_matches_nothing(tiny_model):
    with pytest.raises(ValueError, match="No nn.Linear modules matched"):
        build_mixed_gralora(tiny_model(), r=16, target_modules=("nonexistent_proj",))


def test_build_is_identity_at_initialisation(tiny_model, sample_input):
    """B initialises to zero, so a freshly attached adapter must be a no-op."""
    base = tiny_model()
    with torch.no_grad():
        expected = base(sample_input)
    peft_model = build_mixed_gralora(base, r=16, default_k=4)
    with torch.no_grad():
        actual = peft_model(sample_input)
    assert torch.allclose(expected, actual, atol=1e-6)


def test_adapter_changes_the_output_once_b_is_non_zero(tiny_model, sample_input):
    peft_model = build_mixed_gralora(tiny_model(), r=16, default_k=4)
    with torch.no_grad():
        before = peft_model(sample_input).clone()
        for name, param in peft_model.named_parameters():
            if "gralora_B" in name:
                param.add_(0.05)
        after = peft_model(sample_input)
    assert not torch.allclose(before, after, atol=1e-6)


def test_only_adapter_parameters_are_trainable(tiny_model):
    peft_model = build_mixed_gralora(tiny_model(), r=16, default_k=4)
    trainable = [n for n, p in peft_model.named_parameters() if p.requires_grad]
    assert trainable, "no trainable parameters after wrapping"
    assert all("gralora" in n for n in trainable)


# --- parameter-count invariance -------------------------------------------


@pytest.mark.parametrize("k", [1, 2, 4, 8])
def test_trainable_parameter_count_is_invariant_to_uniform_k(tiny_model, k):
    counts = trainable_parameter_count(build_mixed_gralora(tiny_model(), r=16, default_k=k))
    reference = trainable_parameter_count(build_mixed_gralora(tiny_model(), r=16, default_k=1))
    assert counts == reference


def test_trainable_parameter_count_is_invariant_for_a_mixed_map(tiny_model):
    uniform = trainable_parameter_count(build_mixed_gralora(tiny_model(), r=16, default_k=2))
    mixed = trainable_parameter_count(
        build_mixed_gralora(
            tiny_model(),
            r=16,
            k_map={"layers.0.self_attn.q_proj": 8, "layers.1.mlp.down_proj": 4},
            default_k=2,
        )
    )
    assert mixed == uniform


def test_parameter_count_matches_the_closed_form(tiny_model):
    """A totals in_features * r, B totals out_features * r, for any k."""
    model = tiny_model()
    dims = module_dims(model)
    r = 16
    expected = sum((in_f + out_f) * r for in_f, out_f in dims.values())
    assert trainable_parameter_count(build_mixed_gralora(model, r=r, default_k=4)) == expected


# --- representative k ------------------------------------------------------


def test_representative_k_is_the_value_itself_when_uniform():
    assert _representative_k({"a": 4, "b": 4, "c": 4}) == 4


def test_representative_k_is_the_most_common_when_mixed():
    assert _representative_k({"a": 2, "b": 2, "c": 8}) == 2


# --- offline model table ---------------------------------------------------


@pytest.mark.parametrize("model_name", sorted(KNOWN_MODELS))
@pytest.mark.parametrize("r", [16, 32, 64, 128])
def test_known_models_have_a_legal_k_at_every_planned_rank(model_name, r):
    common = set((2, 4, 8))
    for in_f, out_f in KNOWN_MODELS[model_name].values():
        common &= set(legal_k_values(r, in_f, out_f))
    assert common, f"{model_name} has no k legal for every module at r={r}"


def test_known_models_cover_every_default_target_module():
    for model_name, dims in KNOWN_MODELS.items():
        assert set(dims) == set(DEFAULT_TARGET_MODULES), model_name
