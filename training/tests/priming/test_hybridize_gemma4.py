"""
Tests priming a hybrid Gemma 4 from a base Gemma 4 (construct_hybrid_gemma4), with a tiny multimodal
Gemma4ForConditionalGeneration (without its vision and audio towers) whose text model has Gemma 4 E2B's features:
sliding window and full attention layers with different head sizes, layers that share keys and values, and
per-layer embeddings. A pattern of '*' layers, and an SWA layer whose window covers the whole input, must give the
base model's logits. GKA, GDN, and Mamba2 layers must start from the weights of the attention that they replace,
with its head size, while the rest of each decoder layer is copied. The tests run on the CPU.
"""

import pytest
import torch
from transformers import AutoModelForCausalLM
from transformers.models.gemma4.configuration_gemma4 import Gemma4Config
from transformers.models.gemma4.modeling_gemma4 import Gemma4ForConditionalGeneration

import hmf.model.hybrid_zoo.models.model_register  # noqa: F401  # Registers the HMF model families
from hmf.model.hybrid_zoo.layers.hybrid_dataclasses import HybridConfig, HybridSWAConfig
from hmf.model.hybrid_zoo.models.hybrid_gemma4.modeling_hybrid_gemma4 import HybridGemma4ForCausalLM
from hmf.priming.hybridize_gemma4 import construct_hybrid_gemma4

VOCAB = 97
SLIDING, FULL = "sliding_attention", "full_attention"
# Fifteen layers in three groups of four sliding and one full, whose last five share the keys and values of layers
# 8 (sliding) and 9 (full), so that full layer 4 neither shares nor is shared
TEXT = {
    "vocab_size": VOCAB, "hidden_size": 64, "intermediate_size": 128, "num_hidden_layers": 15,
    "num_attention_heads": 4, "num_key_value_heads": 1, "head_dim": 16, "global_head_dim": 32,
    "layer_types": ([SLIDING] * 4 + [FULL]) * 3, "sliding_window": 4, "num_kv_shared_layers": 5,
    "hidden_size_per_layer_input": 8, "vocab_size_per_layer_input": VOCAB, "final_logit_softcapping": 30.0,
    "max_position_embeddings": 256, "pad_token_id": 0,
}


@pytest.fixture(scope="module")
def base():
    config = Gemma4Config(text_config=TEXT, vision_config=None, audio_config=None)
    torch.manual_seed(0)
    return Gemma4ForConditionalGeneration(config).eval()


def _ids():
    return torch.randint(3, VOCAB, (2, 12), generator=torch.Generator().manual_seed(1))


def _prime(base, pattern, **layer_configs):
    hybrid_config = HybridConfig(base_model_name_or_path="base", hybrid_override_pattern=pattern, output_dir="out",
                                 **layer_configs)
    return construct_hybrid_gemma4(base, base.config, hybrid_config).eval()


def _base_layer(base, layer_idx):
    return base.model.language_model.layers[layer_idx]


@pytest.mark.parametrize(
    "pattern, layer_configs",
    [("-".join(["*"] * 15), {}), ("*-*-*-*-SWA" + "-*" * 10, {"swa": HybridSWAConfig(window_size=64)})],
    ids=["attention", "swa_with_a_window_wider_than_the_input"],
)
def test_primed_model_gives_the_logits_of_its_base_model(base, pattern, layer_configs):
    model = _prime(base, pattern, **layer_configs)
    with torch.no_grad():
        assert torch.equal(model(_ids()).logits, base(input_ids=_ids()).logits)


def test_hybrid_layers_start_from_the_attention_they_replace(base):
    model = _prime(base, "GKA-GDN-M2-*-GKA" + "-*" * 10)
    layers = model.model.layers
    assert torch.equal(layers[0].gka.q_proj.weight, _base_layer(base, 0).self_attn.q_proj.weight)
    assert torch.equal(layers[4].gka.k_proj.weight, _base_layer(base, 4).self_attn.k_proj.weight)
    assert torch.equal(layers[1].gdn.v_proj.weight, _base_layer(base, 1).self_attn.v_proj.weight)
    assert torch.equal(layers[2].mamba.out_proj.weight, _base_layer(base, 2).self_attn.o_proj.weight)
    # Layer 4 replaces full attention, whose heads are twice as large
    assert layers[0].gka.q_proj.weight.shape == (4 * 16, 64) and layers[4].gka.q_proj.weight.shape == (4 * 32, 64)


def test_rest_of_each_hybrid_layer_is_copied(base):
    model = _prime(base, "GKA-*-*-*-GKA" + "-*" * 10)
    base_layer = _base_layer(base, 4)
    for name in ("mlp.up_proj", "post_attention_layernorm", "pre_feedforward_layernorm", "per_layer_input_gate",
                 "per_layer_projection", "post_per_layer_input_norm"):
        assert torch.equal(model.model.layers[4].get_submodule(name).weight, base_layer.get_submodule(name).weight)
    assert torch.equal(model.model.layers[4].layer_scalar, base_layer.layer_scalar)


def test_primed_model_reloads_as_a_hybrid_gemma4(base, tmp_path):
    model = _prime(base, "*-*-*-*-SWA" + "-*" * 10, swa=HybridSWAConfig(window_size=3))
    model.save_pretrained(tmp_path)
    reloaded = AutoModelForCausalLM.from_pretrained(tmp_path).eval()
    assert type(reloaded) is HybridGemma4ForCausalLM and reloaded.config.swa_config["window_size"] == 3
    with torch.no_grad():
        assert torch.equal(reloaded(_ids()).logits, model(_ids()).logits)


@pytest.mark.parametrize(
    "pattern, match",
    [("BMF" + "-*" * 14, "supports the layer types"), ("-".join(["*"] * 14) + "-GKA", "share keys and values")],
    ids=["bmojo", "a_layer_that_shares_keys_and_values"],
)
def test_unsupported_pattern_raises(base, pattern, match):
    with pytest.raises(NotImplementedError, match=match):
        _prime(base, pattern)


def test_heads_above_the_gka_limit_are_split_at_priming(base, monkeypatch):
    # The tiny full attention layers have heads of 32, so a limit of 16 splits them in two, as GKA's limit of 128
    # splits Gemma 4's heads of 256 and 512
    import hmf.model.hybrid_zoo.layers.gated_kalmanet.gka_config as gka_config
    from hmf.priming.hybrid_layer_init import extract_attention_weights, split_attention_heads

    monkeypatch.setattr(gka_config, "GKA_MAX_HEAD_DIM", 16)
    model = _prime(base, "*-*-*-*-GKA" + "-*" * 10)
    gka = model.model.layers[4].gka
    assert (gka.head_dim, gka.num_q_heads, gka.num_k_heads) == (16, 8, 2)
    expected = split_attention_heads(extract_attention_weights(_base_layer(base, 4).self_attn), 16)
    assert torch.equal(gka.q_proj.weight, expected.q_weight) and torch.equal(gka.o_proj.weight, expected.o_weight)
    assert torch.equal(gka.k_proj.weight, _base_layer(base, 4).self_attn.k_proj.weight)
