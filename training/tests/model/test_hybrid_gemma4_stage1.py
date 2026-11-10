"""
Tests the Stage 1 (layerwise distillation) layers of hybrid Gemma 4, '*DA' and the fused 'X*', from priming through
the HKD trainer's loss to unfusing, with a tiny multimodal Gemma4ForConditionalGeneration whose text model has the
features of Gemma 4 E2B: sliding window and full attention layers with different head sizes, layers that share keys
and values, and per-layer embeddings. The attention stream must give Gemma 4's own layer outputs, and the hybrid
stream those of the unfused model, including the keys and values that a fused source computes for the hybrid
stream's sharing layers. GKA's kernels need CUDA, so GKA's forward is replaced by a linear stand-in built from its
own projections, which the unfused model gets too. The tests run on the CPU.
"""

from types import SimpleNamespace

import pytest
import torch
from transformers.models.gemma4.configuration_gemma4 import Gemma4Config
from transformers.models.gemma4.modeling_gemma4 import Gemma4ForConditionalGeneration

import hmf.model.hybrid_zoo.models.model_register  # noqa: F401  # Registers the HMF model families
from hmf.model.hybrid_zoo.layers.hybrid_dataclasses import HybridConfig, HybridSWAConfig
from hmf.model.hybrid_zoo.models.hybrid_gemma4.modeling_hybrid_gemma4 import HybridGemma4ForCausalLM
from hmf.model.hybrid_zoo.models.utils import hybrid_layer_types
from hmf.priming.fused_to_standard import get_unfused_config, get_unfused_weights
from hmf.priming.hybridize_gemma4 import construct_hybrid_gemma4
from hmf.train.hkd.trainer import CustomTrainer

VOCAB = 97
SLIDING, FULL = "sliding_attention", "full_attention"
# Fifteen layers in three groups of four sliding and one full, whose last five share the keys and values of layers
# 8 (sliding) and 9 (full), as in tests/priming/test_hybridize_gemma4.py
TEXT = {
    "vocab_size": VOCAB, "hidden_size": 64, "intermediate_size": 128, "num_hidden_layers": 15,
    "num_attention_heads": 4, "num_key_value_heads": 1, "head_dim": 16, "global_head_dim": 32,
    "layer_types": ([SLIDING] * 4 + [FULL]) * 3, "sliding_window": 4, "num_kv_shared_layers": 5,
    "hidden_size_per_layer_input": 8, "vocab_size_per_layer_input": VOCAB, "final_logit_softcapping": 30.0,
    "max_position_embeddings": 256, "pad_token_id": 0,
}
ALL_DUAL_ATTENTION = "-".join(["*DA"] * 15)
# GKA at both sources (8 and 9), whose keys and values the '*DA' sharing layers and the SWA at layer 12 read, and at
# a sliding and a full layer that neither share nor are shared
SOURCES = "GKA*-*DA-*DA-*DA-GKA*-*DA-*DA-*DA-GKA*-GKA*-*DA-*DA-SWA*-*DA-*DA"
SWA = {"swa": HybridSWAConfig(window_size=6)}


@pytest.fixture(scope="module")
def base():
    config = Gemma4Config(text_config=TEXT, vision_config=None, audio_config=None)
    config.text_config._attn_implementation = "eager"
    torch.manual_seed(0)
    return Gemma4ForConditionalGeneration(config).eval()


def _ids():
    return torch.randint(3, VOCAB, (2, 12), generator=torch.Generator().manual_seed(1))


def _inputs():
    return {"input_ids": _ids(), "attention_mask": torch.ones(2, 12, dtype=torch.long)}


def _replace_gka(model):
    """Replaces the forward of each GKA mixer with a linear map through its own query and output projections."""
    for layer in model.model.layers:
        gka = getattr(layer, "gka", None)
        if gka is not None:
            gka.forward = lambda hidden_states, gka=gka, **kwargs: (gka.o_proj(gka.q_proj(hidden_states)), None, None)
    return model


def _prime(base, pattern):
    hybrid_config = HybridConfig(base_model_name_or_path="base", hybrid_override_pattern=pattern, output_dir="out",
                                 **SWA)
    model = construct_hybrid_gemma4(base, base.config, hybrid_config).eval()
    model.config._attn_implementation = "eager"
    return _replace_gka(model)


def _unfuse(fused):
    config = get_unfused_config(fused.config)
    model = HybridGemma4ForCausalLM(config).eval()
    weights, missing = get_unfused_weights(fused.state_dict(), set(model.state_dict()))
    model.load_state_dict(weights, strict=False)
    return _replace_gka(model), missing


def _layer_outputs(model, **kwargs):
    """Returns the output of each decoder layer of a model's forward."""
    outputs = []
    hooks = [layer.register_forward_hook(lambda m, a, out: outputs.append(out)) for layer in model.layers]
    with torch.no_grad():
        model(input_ids=_ids(), use_cache=False, **kwargs)
    for hook in hooks:
        hook.remove()
    return outputs


def _stage1_outputs(model):
    with torch.no_grad():
        return list(model.model.stage1_layer_outputs(**_inputs()))


def _trainer(target="all", pre_lm_head=True):
    """Returns a stand-in for the HKD trainer with only what compute_lw_mse_loss uses."""
    return SimpleNamespace(lw_distill_target=target, distill_pre_lm_head=pre_lm_head)


def test_pattern_builds_stage1_layers_with_the_masks_of_their_two_streams(base):
    layers = _prime(base, SOURCES).model.layers
    names = [type(layer).__name__ for layer in layers]
    assert names[0] == "Gemma4FusedAttGKADecoderLayer" and names[1] == "Gemma4DualAttDecoderLayer"
    assert names[12] == "Gemma4FusedAttSWADecoderLayer"
    # The attention stream uses the base kind's mask, and the hybrid stream the mask of its mixer
    assert (layers[4].attention_type, layers[4].hybrid_attention_type) == (FULL, FULL)
    assert (layers[0].attention_type, layers[0].hybrid_attention_type) == (SLIDING, FULL)
    assert (layers[12].attention_type, layers[12].hybrid_attention_type) == (SLIDING, "swa")
    assert (layers[1].attention_type, layers[1].hybrid_attention_type) == (SLIDING, SLIDING)


def test_fused_layers_start_from_the_attention_they_replace(base):
    layer = _prime(base, SOURCES).model.layers[4]
    base_attention = base.model.language_model.layers[4].self_attn
    assert torch.equal(layer.self_attn.k_proj.weight, base_attention.k_proj.weight)
    assert torch.equal(layer.gka.q_proj.weight, base_attention.q_proj.weight)
    assert torch.equal(layer.mlp.up_proj.weight, base.model.language_model.layers[4].mlp.up_proj.weight)


def test_fused_sources_compute_keys_and_values_for_the_hybrid_stream_only_when_later_layers_read_them(base):
    layers = _prime(base, SOURCES).model.layers
    assert layers[8].computes_source_kv and layers[9].computes_source_kv
    assert not layers[0].computes_source_kv
    # With GKA at every sharing layer, no layer reads the sources' keys and values
    layers = _prime(base, "*DA-" * 8 + "GKA*-GKA*" + "-GKA*" * 5).model.layers
    assert not layers[8].computes_source_kv and not layers[9].computes_source_kv


def test_attention_stream_gives_the_outputs_of_gemma4_at_every_layer(base):
    expected = _layer_outputs(base.model.language_model)
    outputs = _stage1_outputs(_prime(base, SOURCES))
    for layer_idx, (layer_outputs, base_output) in enumerate(zip(outputs, expected)):
        assert torch.equal(layer_outputs[1], base_output), layer_idx


def test_dual_attention_layers_give_the_outputs_of_gemma4_on_both_streams(base):
    expected = _layer_outputs(base.model.language_model)
    for layer_idx, (layer_outputs, base_output) in enumerate(zip(_stage1_outputs(_prime(base, ALL_DUAL_ATTENTION)),
                                                                 expected)):
        hybrid_out, attn_out, hybrid_mixer_out, attn_mixer_out, hybrid_mlp_out, attn_mlp_out = layer_outputs
        assert torch.equal(hybrid_out, base_output) and torch.equal(attn_out, base_output), layer_idx
        assert torch.equal(hybrid_mixer_out, attn_mixer_out) and torch.equal(hybrid_mlp_out, attn_mlp_out), layer_idx


def test_hybrid_stream_gives_the_outputs_of_the_unfused_model_at_every_layer(base):
    fused = _prime(base, SOURCES)
    unfused, missing = _unfuse(fused)
    assert missing == set()
    expected = _layer_outputs(unfused.model)
    for layer_idx, (layer_outputs, unfused_output) in enumerate(zip(_stage1_outputs(fused), expected)):
        assert torch.equal(layer_outputs[0], unfused_output), layer_idx


def test_unfusing_gives_the_hybrid_layers_and_the_source_keys_and_values(base):
    fused = _prime(base, SOURCES)
    unfused, _ = _unfuse(fused)
    config = unfused.config
    assert config.hybrid_override_pattern == "GKA-*-*-*-GKA-*-*-*-GKA-GKA-*-*-SWA-*-*"
    assert config.layer_types == hybrid_layer_types(config.hybrid_override_pattern, TEXT["layer_types"])
    assert config.base_layer_types == TEXT["layer_types"]
    for source in (8, 9):
        source_kv, attention = unfused.model.layers[source].source_kv, fused.model.layers[source].self_attn
        assert torch.equal(source_kv.k_proj.weight, attention.k_proj.weight)
        assert torch.equal(source_kv.v_proj.weight, attention.v_proj.weight)
        assert torch.equal(source_kv.k_norm.weight, attention.k_norm.weight)


def test_trainer_loss_of_a_model_of_dual_attention_layers_is_zero(base):
    model = _prime(base, ALL_DUAL_ATTENTION)
    with torch.no_grad():
        assert CustomTrainer.compute_lw_mse_loss(_trainer(), model, _inputs()).item() == 0.0


def test_trainer_loss_compares_the_two_streams_at_every_layer_and_before_the_lm_head(base):
    model = _prime(base, SOURCES)
    outputs = _stage1_outputs(model)
    mse = torch.nn.functional.mse_loss
    expected = sum(mse(out[0], out[1]) for out in outputs) / len(outputs)
    expected += mse(model.model.norm(outputs[-1][0]), model.model.norm(outputs[-1][1]))
    with torch.no_grad():
        loss = CustomTrainer.compute_lw_mse_loss(_trainer("decoder"), model, _inputs())
    assert loss.item() > 0 and torch.allclose(loss, expected)


def test_only_the_hybrid_stream_carries_gradients(base):
    model = _prime(base, SOURCES).train()
    for layer_outputs in model.model.stage1_layer_outputs(**_inputs()):
        assert layer_outputs[0].requires_grad and not any(t.requires_grad for t in layer_outputs[1::2])


def test_gradient_checkpointing_gives_the_gka_layers_the_same_gradients_as_without_it(base):
    def gka_grads(checkpointing):
        model = _prime(base, SOURCES).train()
        if checkpointing:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        CustomTrainer.compute_lw_mse_loss(_trainer(), model, _inputs()).backward()
        return [layer.gka.q_proj.weight.grad for layer in model.model.layers if hasattr(layer, "gka")]

    reference = gka_grads(checkpointing=False)
    assert all(grad is not None and grad.abs().sum() > 0 for grad in reference)
    assert all(torch.equal(a, b) for a, b in zip(gka_grads(checkpointing=True), reference))


def test_forward_of_a_stage1_model_raises_and_points_to_stage1_layer_outputs(base):
    with pytest.raises(ValueError, match="stage1_layer_outputs"):
        _prime(base, ALL_DUAL_ATTENTION)(_ids())
