"""
Tests that a HybridCache and its tensors are freed as soon as nothing uses it.

The per-layer-type caches (Mamba2Cache, BMojoCache, GatedDeltaNetCache, and GatedKalmaNetCache) keep their states in
the HybridCache that creates them, and hold only a weak reference to it, because a strong one would form a reference
cycle with the HybridCache's references to them. Python frees such a cycle only when its garbage collector runs, so a
finished cache's GPU memory would stay allocated until then, and a loop of batched B'MOJO generate() calls on an 8B
model would run out of memory. The tests turn the garbage collector off and use tiny hybrid Qwen3 models with
attention layers only, which run on the CPU.
"""

import gc
import weakref

import pytest
import torch

from hmf.model.hybrid_zoo.models.cache import HybridCache
from hmf.model.hybrid_zoo.models.hybrid_qwen3.configuration_hybrid_qwen3 import HybridQwen3Config
from hmf.model.hybrid_zoo.models.hybrid_qwen3.modeling_hybrid_qwen3 import HybridQwen3ForCausalLM

VOCAB = 97


@pytest.fixture(autouse=True)
def no_garbage_collector():
    gc.collect()
    gc.disable()
    yield
    gc.enable()


def _config():
    config = HybridQwen3Config(
        vocab_size=VOCAB, hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, head_dim=16, max_position_embeddings=256, hybrid_override_pattern="*-*",
    )
    config._attn_implementation = "sdpa"
    return config


def _ids(batch=2, seq=8):
    return torch.randint(0, VOCAB, (batch, seq), generator=torch.Generator().manual_seed(1))


def test_cache_and_its_tensors_are_freed_without_the_garbage_collector():
    cache = HybridCache(_config())
    keys = torch.zeros(1, 2, 5, 16)
    cache.update(keys, keys, layer_idx=0)
    state = torch.zeros(1, 4)
    cache.mamba2.key_value_memory_dict[1] = (state, state)
    cache_ref, keys_ref, state_ref = weakref.ref(cache), weakref.ref(keys), weakref.ref(state)
    del cache, keys, state
    assert cache_ref() is None and keys_ref() is None and state_ref() is None


def test_generate_frees_its_cache_once_its_output_is_gone():
    torch.manual_seed(0)
    model = HybridQwen3ForCausalLM(_config()).eval()
    with torch.no_grad():
        out = model.generate(input_ids=_ids(), max_new_tokens=3, do_sample=False, pad_token_id=0,
                             return_dict_in_generate=True)
    assert isinstance(out.cache, HybridCache)
    cache_ref = weakref.ref(out.cache)
    del out
    assert cache_ref() is None


def test_parts_read_and_write_the_state_of_their_cache():
    cache = HybridCache(_config())
    cache.mamba2.key_value_memory_dict[0] = "mamba"
    cache.gated_kalmanet.key_value_memory_dict = {1: "gka"}
    cache.mamba2.record_tokens(0, 7)
    assert cache._mamba_memory_dict == {0: "mamba"} and cache._gated_kalmanet_memory_dict == {1: "gka"}
    assert cache.mamba2.seen_tokens(0) == cache.get_seq_length(0) == 7
    for part in (cache.mamba2, cache.bmojo_f, cache.gated_deltanet, cache.gated_kalmanet):
        assert part.parent is cache


def test_part_of_a_freed_cache_raises_an_error_naming_it():
    cache = HybridCache(_config())
    part = cache.gated_kalmanet
    del cache
    with pytest.raises(ReferenceError, match="GatedKalmaNetCache"):
        part.key_value_memory_dict


def test_copy_has_its_own_parts():
    cache = HybridCache(_config())
    cache.gated_deltanet.key_value_memory_dict[0] = {"recurrent_state": torch.ones(1, 2)}
    copy = cache.copy()
    assert copy.gated_deltanet.parent is copy
    copy.gated_deltanet.key_value_memory_dict[0]["recurrent_state"].zero_()
    assert cache.gated_deltanet.key_value_memory_dict[0]["recurrent_state"].sum() == 2
