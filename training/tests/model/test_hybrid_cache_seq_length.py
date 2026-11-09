"""
Tests that HybridCache counts the tokens each layer has processed, which get_seq_length returns as the position of
the next token.

Each layer adds its new tokens to its own count when it updates its state, so the count needs no updates from the
model and is also correct when the inner model runs alone. The cache keeps a count for each layer type, since a
B'MOJO layer and its SSM share a layer index, and cache composition sets the counts of the caches it builds, which
can exceed the attention cache's length. The CPU tests use the cache directly and a tiny hybrid Qwen3 with attention
layers only, and the CUDA test runs one whose first layer is GDN, so that the positions come from GDN's count.
"""

import importlib.util

import pytest
import torch

from hmf.model.hybrid_zoo.models.cache import HybridCache
from hmf.model.hybrid_zoo.models.hybrid_qwen3.configuration_hybrid_qwen3 import HybridQwen3Config
from hmf.model.hybrid_zoo.models.hybrid_qwen3.modeling_hybrid_qwen3 import HybridQwen3ForCausalLM

VOCAB = 97

needs_fla = pytest.mark.skipif(
    not torch.cuda.is_available() or importlib.util.find_spec("fla") is None, reason="GDN needs CUDA and fla"
)


def _config(pattern="*-*"):
    config = HybridQwen3Config(
        vocab_size=VOCAB, hidden_size=64, intermediate_size=128, num_hidden_layers=len(pattern.split("-")),
        num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=256,
        hybrid_override_pattern=pattern,
    )
    config._attn_implementation = "sdpa"
    return config


def _ids(batch=2, seq=12):
    return torch.randint(0, VOCAB, (batch, seq), generator=torch.Generator().manual_seed(1))


def test_each_layer_type_counts_its_own_tokens():
    cache = HybridCache(_config())
    assert cache.get_seq_length() == cache.get_seq_length(None) == 0
    keys = torch.zeros(1, 2, 5, 16)
    cache.update(keys, keys, layer_idx=0)
    cache.update(keys[:, :, :1], keys[:, :, :1], layer_idx=0)
    cache.gated_deltanet.update(recurrent_state=torch.zeros(1), conv_state=None, layer_idx=1, offset=6)
    cache.gated_kalmanet.update(recurrent_state=(None, None), conv_state=(None, None, None), layer_idx=2, offset=3)
    cache.mamba2.record_tokens(3, 4)
    assert [cache.get_seq_length(i) for i in range(5)] == [6, 6, 3, 4, 0]
    assert cache.cached_attention_length() == 6


def test_bmojo_and_its_ssm_keep_separate_counts():
    cache = HybridCache(_config())
    cache.bmojo_f.record_tokens(0, 5)
    cache.mamba2.record_tokens(0, 5)
    assert cache.get_seq_length(0) == cache.bmojo_f.seen_tokens(0) == cache.mamba2.seen_tokens(0) == 5


def test_counts_continue_from_set_seq_length():
    cache = HybridCache(_config())
    keys = torch.zeros(1, 2, 4, 16)
    cache.update(keys, keys, layer_idx=0)
    cache.gated_deltanet.key_value_memory_dict[1] = {"recurrent_state": torch.zeros(1)}
    cache.set_seq_length(10)
    assert cache.get_seq_length(0) == cache.get_seq_length(1) == 10 and cache.cached_attention_length() == 4
    cache.update(keys[:, :, :1], keys[:, :, :1], layer_idx=0)
    assert cache.get_seq_length(0) == 11 and cache.cached_attention_length() == 5


def test_copy_has_its_own_counts_and_clear_resets_them():
    cache = HybridCache(_config())
    cache.gated_kalmanet.record_tokens(1, 3)
    copy = cache.copy()
    copy.gated_kalmanet.record_tokens(1, 2)
    assert cache.get_seq_length(1) == 3 and copy.get_seq_length(1) == 5
    cache.clear()
    assert cache.get_seq_length(1) == 0 and copy.get_seq_length(1) == 5


def test_generate_counts_every_token():
    torch.manual_seed(0)
    model = HybridQwen3ForCausalLM(_config()).eval()
    ids = _ids()
    with torch.no_grad():
        out = model.generate(input_ids=ids, max_new_tokens=4, min_new_tokens=4, do_sample=False, pad_token_id=0,
                             eos_token_id=None, return_dict_in_generate=True)
    # The last generated token isn't fed back through the model, so the cache holds one token fewer
    assert out.past_key_values.get_seq_length() == out.sequences.shape[1] - 1


def _inner_model_steps(model, ids, prefill):
    """Returns the logits of a prefill followed by one-token steps through the inner model, without its ForCausalLM."""
    cache, steps = HybridCache(model.config), []
    hidden = model.model(ids[:, :prefill], past_key_values=cache, use_cache=True).last_hidden_state
    steps.append(model.lm_head(hidden))
    for i in range(prefill, ids.shape[1]):
        hidden = model.model(ids[:, i : i + 1], past_key_values=cache, use_cache=True).last_hidden_state
        steps.append(model.lm_head(hidden))
    return torch.cat(steps, dim=1)


def test_inner_model_steps_match_a_full_forward():
    torch.manual_seed(0)
    model = HybridQwen3ForCausalLM(_config()).eval()
    ids = _ids()
    with torch.no_grad():
        full = model(ids).logits
        steps = _inner_model_steps(model, ids, prefill=8)
    assert torch.allclose(steps, full, atol=1e-4, rtol=1e-4)


@needs_fla
def test_inner_model_steps_match_a_full_forward_when_layer_0_is_gdn():
    # The attention layer's positions come from layer 0's count, so they would restart at 0 if GDN didn't count its
    # tokens
    torch.manual_seed(0)
    model = HybridQwen3ForCausalLM(_config("GDN-*")).to("cuda", torch.bfloat16).eval()
    ids = _ids().cuda()
    with torch.no_grad():
        full = model(ids).logits.float()
        steps = _inner_model_steps(model, ids, prefill=8).float()
    error = ((steps - full).abs().max() / full.abs().max()).item()
    assert error < 0.05, f"one-token steps differ from the full forward by {error:.3g} of its scale"
