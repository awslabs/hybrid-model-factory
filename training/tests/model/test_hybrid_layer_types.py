"""
Tests that the hybrid Qwen2, Qwen3, Qwen3-MoE, and Llama configs fill config.layer_types with transformers' names
(such as "full_attention", "sliding_attention", and "linear_attention"), which they derive from
hybrid_override_pattern with hybrid_layer_types, while the decoder layer classes still come from the pattern itself.
Older checkpoints hold the pattern's entries in layer_types, so the tests also check that such configs load with
transformers' names. The tests use tiny configs, and models with attention and SWA layers only, which run on the CPU.
"""

import json

import pytest
from transformers import AutoConfig

import hmf.model.hybrid_zoo.models.model_register  # noqa: F401  # Registers the HMF model families
from hmf.model.hybrid_zoo.models.hybrid_qwen2_5_vl.configuration_hybrid_qwen2_5_vl import HybridQwen2_5_VLConfig
from hmf.model.hybrid_zoo.models.utils import LAYER_TYPE_PATTERNS, PATTERN_TO_LAYER_TYPE, hybrid_layer_types

SMALL = {
    "vocab_size": 97, "hidden_size": 64, "intermediate_size": 128, "num_attention_heads": 4,
    "num_key_value_heads": 2, "head_dim": 16, "max_position_embeddings": 256,
}
FAMILIES = {
    "hybrid_qwen2": {},
    "hybrid_qwen3": {},
    "hybrid_qwen3_moe": {"moe_intermediate_size": 32, "num_experts": 4, "num_experts_per_tok": 2},
    "hybrid_llama": {},
}
EVERY_ENTRY = "*-M2-BMF-GDN-GKA-SWA-M2*-BMF*-GDN*-GKA*-SWA*-GDN>GKA-*DA"
EVERY_ENTRY_LAYER_TYPES = [
    "full_attention", "linear_attention", "linear_attention", "linear_attention", "linear_attention",
    "sliding_attention", "full_attention", "full_attention", "full_attention", "full_attention", "full_attention",
    "linear_attention", "full_attention",
]


def _config(model_type, pattern, **extra):
    return AutoConfig.for_model(
        model_type, num_hidden_layers=len(pattern.split("-")), hybrid_override_pattern=pattern,
        **SMALL, **FAMILIES[model_type], **extra,
    )


def test_every_pattern_entry_has_a_layer_type():
    assert set(PATTERN_TO_LAYER_TYPE) == set(LAYER_TYPE_PATTERNS)


def test_attention_layers_keep_the_base_model_kind():
    base = ["sliding_attention", "full_attention", "sliding_attention", "full_attention"]
    assert hybrid_layer_types("*-GKA*-*DA-GKA", base) == base[:3] + ["linear_attention"]


def test_unknown_pattern_entry_raises():
    with pytest.raises(ValueError, match="GKKA"):
        hybrid_layer_types("*-GKKA", ["full_attention"] * 2)


def test_pattern_of_the_wrong_length_raises():
    with pytest.raises(ValueError, match="has 2 layers, but the model has 3"):
        hybrid_layer_types("*-GKA", ["full_attention"] * 3)


@pytest.mark.parametrize("model_type", FAMILIES)
def test_layer_types_follow_the_pattern(model_type):
    config = _config(model_type, EVERY_ENTRY)
    assert config.layer_types == EVERY_ENTRY_LAYER_TYPES
    # Transformers' own checks, which save_pretrained also runs
    config.validate()


@pytest.mark.parametrize("model_type", FAMILIES)
def test_config_with_an_unknown_pattern_entry_raises(model_type):
    with pytest.raises(ValueError, match="GKKA"):
        _config(model_type, "*-GKKA")


@pytest.mark.parametrize("model_type", FAMILIES)
def test_old_config_with_pattern_entries_in_layer_types_loads(model_type):
    saved = _config(model_type, "*-GKA-SWA").to_dict()
    saved["layer_types"] = ["*", "GKA", "SWA"]
    config = type(_config(model_type, "*")).from_dict(saved)
    assert config.layer_types == ["full_attention", "linear_attention", "sliding_attention"]


@pytest.mark.parametrize("model_type", FAMILIES)
def test_saved_config_keeps_its_layer_types(model_type, tmp_path):
    _config(model_type, "*-GKA-SWA*-SWA").save_pretrained(tmp_path)
    saved = json.loads((tmp_path / "config.json").read_text())
    expected = ["full_attention", "linear_attention", "full_attention", "sliding_attention"]
    assert saved["layer_types"] == expected
    assert AutoConfig.from_pretrained(tmp_path).layer_types == expected


@pytest.mark.parametrize("model_type", ["hybrid_qwen2", "hybrid_qwen3"])
def test_qwen_attention_layers_keep_the_base_sliding_layers(model_type):
    config = _config(model_type, "*-*-*-GKA", use_sliding_window=True, sliding_window=8, max_window_layers=2)
    assert config.layer_types == ["full_attention", "full_attention", "sliding_attention", "linear_attention"]


def test_qwen2_5_vl_text_config_follows_the_pattern():
    text_config = {**SMALL, "num_hidden_layers": 3, "hybrid_override_pattern": "*-GKA-SWA"}
    config = HybridQwen2_5_VLConfig(text_config=text_config)
    assert config.text_config.layer_types == ["full_attention", "linear_attention", "sliding_attention"]


@pytest.mark.parametrize("model_type", FAMILIES)
def test_attention_and_swa_layers_keep_their_masks_and_windows(model_type):
    from transformers import AutoModelForCausalLM

    config = _config(model_type, "*-SWA-SWA*", swa_config={"window_size": 4})
    model = AutoModelForCausalLM.from_config(config)
    layers = model.model.layers
    assert [layer.attention_type for layer in layers] == ["full_attention", "sliding_attention", "full_attention"]
    assert model.model.has_sliding_layers
    assert layers[0].self_attn.sliding_window is None
    assert layers[1].swa.sliding_window == 4
    assert layers[2].self_attn.sliding_window is None and layers[2].swa.sliding_window == 4


@pytest.mark.parametrize("model_type", FAMILIES)
def test_unfusing_derives_layer_types_from_the_unfused_pattern(model_type):
    from hmf.priming.fused_to_standard import get_unfused_config

    config = get_unfused_config(_config(model_type, "*DA-GKA*-SWA*-GDN>GKA-M2*"))
    assert config.hybrid_override_pattern == "*-GKA-SWA-GKA-M2"
    assert config.layer_types == [
        "full_attention", "linear_attention", "sliding_attention", "linear_attention", "linear_attention",
    ]
