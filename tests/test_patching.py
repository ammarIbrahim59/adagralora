"""Constraint validation and the mixed-k builder."""

from __future__ import annotations

import json

import pytest
import torch
from peft.tuners.gralora import GraloraLayer

from adagralora.patching import (
    DEFAULT_TARGET_MODULES,
    KNOWN_MODELS,
    KConstraintError,
    _dims_for_model,
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


def test_validate_k_rejects_a_non_positive_r():
    with pytest.raises(KConstraintError, match="r must be a positive integer"):
        validate_k(r=0, in_features=96, out_features=24, k=2)


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


def test_legal_k_values_is_empty_for_an_impossible_rank():
    """0 % k == 0 for every k, so an unguarded r reports an impossible rank as fine."""
    assert legal_k_values(r=0, in_features=96, out_features=96) == []


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


def test_validate_k_map_rejects_a_non_integral_k(tiny_model):
    """An allocator ranking is continuous; int() would truncate 2.9 to a plausible 2."""
    dims = module_dims(tiny_model())
    with pytest.raises(KConstraintError, match="must be an integer"):
        validate_k_map(dims, r=16, k_map={"layers.0.self_attn.q_proj": 2.9}, default_k=2)


def test_validate_k_map_rejects_a_bool_k(tiny_model):
    """bool is an int, so only an explicit check keeps True from becoming k=1."""
    dims = module_dims(tiny_model())
    with pytest.raises(KConstraintError, match="must be an integer"):
        validate_k_map(dims, r=16, k_map={"layers.0.self_attn.q_proj": True}, default_k=2)


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


def test_build_rejects_a_non_positive_rank_before_allocating(tiny_model, monkeypatch):
    """peft would catch r=0 too, but only part-way through wrapping the model."""

    def explode(*args, **kwargs):
        raise AssertionError("get_peft_model was called despite an impossible rank")

    monkeypatch.setattr("adagralora.patching.get_peft_model", explode)
    with pytest.raises(KConstraintError, match="r must be a positive integer"):
        build_mixed_gralora(tiny_model(), r=0, default_k=2)


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


@pytest.mark.parametrize("default_k", [1, 2, 4, 8])
def test_requested_k_reaches_the_tensor_shapes(tiny_model, default_k):
    """The recorded k is only a scalar; this pins that the blocks are really built at it.

    A regression that records k but chunks at a different granularity would
    survive every count-based test and the save/load round trip alike, since
    both sides would be wrong identically.
    """
    peft_model = build_mixed_gralora(
        tiny_model(), r=16, k_map={"layers.0.self_attn.q_proj": 8}, default_k=default_k
    )
    installed = current_k_map(peft_model)
    checked = 0
    for name, module in peft_model.named_modules():
        if not isinstance(module, GraloraLayer):
            continue
        k = installed[canonical_name(name)]
        assert tuple(module.gralora_A["default"].shape) == (k, module.in_features // k, 16)
        assert tuple(module.gralora_B["default"].shape) == (k, 16, module.out_features // k)
        checked += 1
    assert checked == 14
    assert installed["layers.0.self_attn.q_proj"] == 8


def test_gradients_reach_every_rebuilt_adapter_parameter(tiny_model, sample_input):
    """update_layer reassigns into a ParameterDict; the results must stay leaves."""
    peft_model = build_mixed_gralora(tiny_model(), r=16, k_map={"layers.0.mlp.down_proj": 4}, default_k=2)
    trainable = [(n, p) for n, p in peft_model.named_parameters() if p.requires_grad]
    assert all(p.is_leaf for _, p in trainable), "a rebuilt adapter tensor is not a leaf"

    peft_model(sample_input).sum().backward()
    ungraded = [n for n, p in trainable if p.grad is None]
    assert not ungraded, f"no gradient reached: {ungraded[:5]}"


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_adapters_stay_fp32_over_a_half_precision_base(tiny_model, sample_input, dtype):
    """The fp16 path the README warns about for NaN loss.

    Rebuilding a layer moves its fresh tensors to the base layer's dtype, which
    undoes the one upcast get_peft_model does at injection time. Nothing else in
    the suite leaves fp32, so the divergence is silent everywhere else.
    """
    peft_model = build_mixed_gralora(tiny_model().to(dtype), r=16, default_k=4)
    assert {p.dtype for n, p in peft_model.named_parameters() if "gralora" in n} == {torch.float32}
    with torch.no_grad():
        assert peft_model(sample_input.to(dtype)).dtype is dtype


def test_the_fp32_upcast_can_be_opted_out_of(tiny_model):
    """Same contract as get_peft_model's keyword of the same name."""
    peft_model = build_mixed_gralora(
        tiny_model().to(torch.bfloat16), r=16, default_k=4, autocast_adapter_dtype=False
    )
    assert {p.dtype for n, p in peft_model.named_parameters() if "gralora" in n} == {torch.bfloat16}


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


# --- sweep CLI -------------------------------------------------------------


@pytest.fixture
def offline(monkeypatch):
    """Force the KNOWN_MODELS path, so the CLI tests never touch the network."""
    import transformers

    def unavailable(*args, **kwargs):
        raise OSError("no network in tests")

    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", unavailable)


@pytest.fixture
def local_config(monkeypatch):
    """Serve a config built in-process, so the *preferred* live path runs offline.

    Same fields as the real Qwen2.5 configs; nothing is fetched.
    """
    import transformers
    from transformers import Qwen2Config

    fields = {
        "Qwen/Qwen2.5-0.5B-Instruct": dict(
            hidden_size=896, intermediate_size=4864, num_attention_heads=14, num_key_value_heads=2
        ),
        "Qwen/Qwen2.5-1.5B-Instruct": dict(
            hidden_size=1536, intermediate_size=8960, num_attention_heads=12, num_key_value_heads=2
        ),
    }

    def from_config(model_name, *args, **kwargs):
        return Qwen2Config(**fields[model_name])

    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", from_config)


@pytest.mark.parametrize("model_name", sorted(KNOWN_MODELS))
def test_dims_from_the_live_config_agree_with_the_offline_table(local_config, model_name):
    """Pins the two dim sources to each other.

    Every other CLI test forces the fallback, so the derivation Phase 1 actually
    uses is otherwise never executed, and drift between it and KNOWN_MODELS
    (a renamed config field, a head_dim that stops being hidden // heads) would
    only show up as a wrong sweep.
    """
    dims, source = _dims_for_model(model_name, DEFAULT_TARGET_MODULES)
    assert source == "transformers AutoConfig"
    assert dims == KNOWN_MODELS[model_name]


def test_dims_for_model_falls_back_to_the_offline_table(offline):
    dims, source = _dims_for_model("Qwen/Qwen2.5-0.5B-Instruct", DEFAULT_TARGET_MODULES)
    assert set(dims) == set(DEFAULT_TARGET_MODULES)
    assert "KNOWN_MODELS" in source


def test_dims_for_model_says_how_to_onboard_an_unknown_model(offline):
    """Offline is the normal state in CI, so this is the first error a second model hits.

    It is also the only place that says what to do about it.
    """
    with pytest.raises(SystemExit, match="KNOWN_MODELS"):
        _dims_for_model("some-org/not-in-the-table", DEFAULT_TARGET_MODULES)


def test_dims_for_model_rejects_an_unresolvable_module_name(offline):
    with pytest.raises(SystemExit, match="fc1"):
        _dims_for_model("Qwen/Qwen2.5-0.5B-Instruct", ("q_proj", "fc1"))


def test_sweep_cli_does_not_report_success_after_checking_nothing(offline):
    """A typo'd module name must not buy a green light on an unvalidated sweep."""
    from adagralora.patching import main

    with pytest.raises(SystemExit):
        main(["--model", "Qwen/Qwen2.5-0.5B-Instruct", "--r", "16", "--target-modules", "fc1", "fc2"])


def test_sweep_cli_reports_success_for_a_workable_grid(offline, capsys):
    from adagralora.patching import main

    assert main(["--model", "Qwen/Qwen2.5-0.5B-Instruct", "--r", "16", "--json"]) == 0
    assert "q_proj" in capsys.readouterr().out


def test_sweep_cli_reports_success_on_the_human_readable_path(offline, capsys):
    """Every other success-path test passes --json, and every printed-path one exits 1."""
    from adagralora.patching import main

    assert main(["--model", "Qwen/Qwen2.5-0.5B-Instruct", "--r", "16"]) == 0
    assert "legal for every targeted module: [2, 4, 8]" in capsys.readouterr().out


def test_sweep_cli_fails_on_a_rank_with_no_legal_k(offline, capsys):
    """The exit code the README promises, on the human-readable path it documents."""
    from adagralora.patching import main

    assert main(["--model", "Qwen/Qwen2.5-0.5B-Instruct", "--r", "3"]) == 1
    captured = capsys.readouterr()
    assert "NONE" in captured.out
    assert "no legal k" in captured.err


def test_sweep_cli_fails_on_an_impossible_rank(offline, capsys):
    """r=0 divides by every k, so an unguarded sweep greenlights it and exits 0."""
    from adagralora.patching import main

    assert main(["--model", "Qwen/Qwen2.5-0.5B-Instruct", "--r", "0"]) == 1
    assert "NONE" in capsys.readouterr().out


def test_sweep_cli_fails_on_a_rank_with_no_legal_k_in_json_mode(offline, capsys):
    """A machine-readable sweep must not disagree with the printed one about pass/fail."""
    from adagralora.patching import main

    assert main(["--model", "Qwen/Qwen2.5-0.5B-Instruct", "--r", "3", "--json"]) == 1
    assert json.loads(capsys.readouterr().out)["legal_k"]["3"]["q_proj"] == []
