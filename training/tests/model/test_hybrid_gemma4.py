"""
Tests hybrid Gemma 4 against transformers' Gemma 4 text model, with a tiny config that has the features of Gemma 4
E2B: sliding window and full attention layers whose head sizes differ, keys and values that the last layers share,
per-layer embeddings, a double-width MLP in the sharing layers, and a final logit softcapping. With every layer a
'*' and the weights of transformers' model, hybrid Gemma 4 must give the same logits and generated tokens. The tests
also check the config's layer_types and base_layer_types, the layers whose keys and values are shared, the head
sizes of hybrid layers, and SWA layers. They run on the CPU, apart from the GKA and GDN tests, which need CUDA and
fla.
"""

import importlib.util
import json

import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig
from transformers.models.gemma4.modeling_gemma4 import Gemma4ForCausalLM

import hmf.model.hybrid_zoo.models.model_register  # noqa: F401  # Registers the HMF model families
from hmf.model.hybrid_zoo.models.hybrid_gemma4.configuration_hybrid_gemma4 import HybridGemma4TextConfig
from hmf.model.hybrid_zoo.models.hybrid_gemma4.modeling_hybrid_gemma4 import HybridGemma4ForCausalLM

VOCAB = 97
SLIDING, FULL = "sliding_attention", "full_attention"
# Ten layers in two groups of four sliding and one full, of which the last four share the keys and values of layer 4
# (full) and layer 5 (sliding), as E2B's last 20 layers share those of layers 13 and 14
TINY = {
    "vocab_size": VOCAB, "hidden_size": 64, "intermediate_size": 128, "num_hidden_layers": 10,
    "num_attention_heads": 4, "num_key_value_heads": 1, "head_dim": 16, "global_head_dim": 32,
    "layer_types": ([SLIDING] * 4 + [FULL]) * 2, "sliding_window": 4, "num_kv_shared_layers": 4,
    "hidden_size_per_layer_input": 8, "vocab_size_per_layer_input": VOCAB, "use_double_wide_mlp": True,
    "final_logit_softcapping": 30.0, "max_position_embeddings": 256, "pad_token_id": 0, "bos_token_id": 2,
    "eos_token_id": 1,
}
# Fifteen layers whose last five share keys and values, so that full layer 4 neither shares nor is shared
WIDER = {**TINY, "num_hidden_layers": 15, "layer_types": ([SLIDING] * 4 + [FULL]) * 3, "num_kv_shared_layers": 5,
         "use_double_wide_mlp": False}
ALL_ATTENTION = "-".join(["*"] * 10)

needs_cuda = pytest.mark.skipif(
    not torch.cuda.is_available() or importlib.util.find_spec("fla") is None,
    reason="GDN and GKA kernels need CUDA and fla",
)


def _ids(batch=2, seq=12, seed=1):
    return torch.randint(3, VOCAB, (batch, seq), generator=torch.Generator().manual_seed(seed))


def _models(attn, pattern=ALL_ATTENTION, **extra):
    """Returns transformers' Gemma 4 and a hybrid Gemma 4 with its weights, both in eval mode."""
    torch.manual_seed(0)
    reference_config = Gemma4TextConfig(**TINY)
    reference_config._attn_implementation = attn
    reference = Gemma4ForCausalLM(reference_config).eval()
    config = HybridGemma4TextConfig(**TINY, hybrid_override_pattern=pattern, **extra)
    config._attn_implementation = attn
    model = HybridGemma4ForCausalLM(config).eval()
    model.load_state_dict(reference.state_dict(), strict=True)
    return reference, model


@pytest.mark.parametrize("attn", ["eager", "sdpa"])
def test_all_attention_hybrid_matches_gemma4_from_transformers(attn):
    reference, model = _models(attn)
    ids = _ids()
    mask = torch.ones_like(ids)
    mask[0, :3] = 0
    with torch.no_grad():
        assert torch.equal(model(ids, attention_mask=mask).logits, reference(ids, attention_mask=mask).logits)


@pytest.mark.parametrize("attn", ["eager", "sdpa"])
def test_all_attention_hybrid_generates_as_gemma4_from_transformers_does(attn):
    reference, model = _models(attn)
    ids = _ids()
    mask = torch.ones_like(ids)
    mask[0, :3] = 0
    kwargs = {"attention_mask": mask, "max_new_tokens": 8, "do_sample": False, "pad_token_id": 0}
    with torch.no_grad():
        assert torch.equal(model.generate(ids, **kwargs), reference.generate(ids, **kwargs))


def test_layer_types_follow_the_pattern_while_base_layer_types_keep_the_base_layout():
    config = HybridGemma4TextConfig(**WIDER, hybrid_override_pattern="GKA-*-*-*-SWA" + "-*" * 10,
                                    swa_config={"window_size": 3})
    assert config.base_layer_types == WIDER["layer_types"]
    assert config.layer_types[:5] == ["linear_attention", SLIDING, SLIDING, SLIDING, SLIDING]
    assert config.kv_sharing_sources() == {SLIDING: 8, FULL: 9}
    assert config.per_layer_config[4].head_dim == 32 and config.per_layer_config[0].head_dim == 16


def test_saved_config_keeps_base_layer_types(tmp_path):
    HybridGemma4TextConfig(**WIDER, hybrid_override_pattern="*-*-*-*-GKA" + "-*" * 10).save_pretrained(tmp_path)
    saved = json.loads((tmp_path / "config.json").read_text())
    assert saved["base_layer_types"] == WIDER["layer_types"] and saved["layer_types"][4] == "linear_attention"
    config = AutoConfig.from_pretrained(tmp_path)
    assert config.layer_types[4] == "linear_attention" and config.per_layer_config[4].head_dim == 32


@pytest.mark.parametrize("layer", [4, 5, 6, 9])
def test_hybrid_layer_at_a_layer_that_shares_keys_and_values_raises(layer):
    entries = ["*"] * 10
    entries[layer] = "GKA"
    with pytest.raises(NotImplementedError, match=f"Layers \\[{layer}\\] share keys and values"):
        HybridGemma4TextConfig(**TINY, hybrid_override_pattern="-".join(entries))


def test_hybrid_layers_get_the_head_size_of_the_attention_they_replace():
    config = HybridGemma4TextConfig(**WIDER, hybrid_override_pattern="GKA-*-*-*-GKA" + "-*" * 10)
    model = HybridGemma4ForCausalLM(config)
    assert model.model.layers[0].gka.q_proj.weight.shape == (4 * 16, 64)
    assert model.model.layers[4].gka.q_proj.weight.shape == (4 * 32, 64)


@pytest.mark.parametrize("attn", ["eager", "sdpa"])
def test_swa_layers_decode_from_the_cache_as_a_full_forward_does(attn):
    config = HybridGemma4TextConfig(**WIDER, hybrid_override_pattern="SWA-*-*-*-SWA" + "-*" * 10,
                                    swa_config={"window_size": 3})
    config._attn_implementation = attn
    torch.manual_seed(0)
    model = HybridGemma4ForCausalLM(config).eval()
    assert model.model.layers[4].swa.head_dim == 32 and model.model.layers[4].swa.sliding_window == 3
    ids = _ids()
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=6, do_sample=False, pad_token_id=0, return_dict_in_generate=True,
                             output_logits=True)
        full = model(out.sequences).logits
    steps = torch.stack(out.logits, dim=1)
    assert (steps - full[:, ids.shape[1] - 1 : -1]).abs().max() < 1e-4


def test_saved_model_reloads_with_the_same_outputs(tmp_path):
    config = HybridGemma4TextConfig(**TINY, hybrid_override_pattern="SWA" + "-*" * 9, swa_config={"window_size": 3})
    torch.manual_seed(0)
    model = HybridGemma4ForCausalLM(config).eval()
    model.save_pretrained(tmp_path)
    reloaded = AutoModelForCausalLM.from_pretrained(tmp_path).eval()
    assert type(reloaded) is HybridGemma4ForCausalLM
    with torch.no_grad():
        assert torch.equal(reloaded(_ids()).logits, model(_ids()).logits)


@needs_cuda
@pytest.mark.parametrize("mixer", ["GKA", "GDN"])
def test_linear_layers_decode_from_the_cache_as_a_full_forward_does(mixer):
    config = HybridGemma4TextConfig(**WIDER, hybrid_override_pattern=f"{mixer}-*-*-*-{mixer}" + "-*" * 10)
    config._attn_implementation = "sdpa"
    torch.manual_seed(0)
    model = HybridGemma4ForCausalLM(config).cuda().eval()
    ids = _ids().cuda()
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=6, do_sample=False, pad_token_id=0, return_dict_in_generate=True,
                             output_logits=True)
        full = model(out.sequences).logits
    steps = torch.stack(out.logits, dim=1)
    assert ((steps - full[:, ids.shape[1] - 1 : -1]).abs().max() / full.abs().max()) < 0.05


@needs_cuda
def test_gka_outputs_match_between_cached_decoding_and_the_full_forward():
    """
    GKA's own outputs, before Gemma 4's norm after the mixer scales them to unit size, must match between the
    full forward (GKA's chunked kernel) and cached decoding (its one-token kernel), at a sliding window layer and at
    a full attention layer. The model's logits differ more than in hybrid Qwen3, since that norm scales up GKA's
    small outputs in these tiny models, along with the kernels' difference.
    """
    config = HybridGemma4TextConfig(**WIDER, hybrid_override_pattern="GKA-*-*-*-GKA" + "-*" * 10)
    config._attn_implementation = "sdpa"
    torch.manual_seed(0)
    model = HybridGemma4ForCausalLM(config).cuda().eval()
    outputs = {}
    hooks = [
        layer.gka.register_forward_hook(lambda module, args, output, i=i: outputs.setdefault(i, []).append(output[0]))
        for i, layer in enumerate(model.model.layers) if hasattr(layer, "gka")
    ]
    ids, prefill = _ids(seq=24).cuda(), 8
    with torch.no_grad():
        model(ids, use_cache=False)
        cache = model(ids[:, :prefill], use_cache=True).past_key_values
        for t in range(prefill, ids.shape[1]):
            cache = model(ids[:, t : t + 1], past_key_values=cache, use_cache=True).past_key_values
    for hook in hooks:
        hook.remove()
    for full, *cached in outputs.values():
        cached = torch.cat(cached, dim=1)
        assert ((cached - full).abs().max() / full.abs().max()) < 0.05
