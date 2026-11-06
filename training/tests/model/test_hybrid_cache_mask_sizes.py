"""
Tests that HybridCache.get_mask_sizes makes attention masks cover the cached tokens as well as the new ones.

Without it, transformers' create_causal_mask sized the masks to the new tokens only. A left-padded batch decoding
from the cache then attended to its cached pad tokens under SDPA, eager, and flash attention, and a step of several
tokens after a prefill, as in chunked prefill or when reusing a cache, fell back to SDPA's top-left causal mask,
which hides most of the context. The tests use tiny Qwen3.5 HMF models with attention layers only; flash
attention test needs CUDA.
"""

import importlib.util

import pytest
import torch

from hmf.model.hybrid_zoo.models.cache import HybridCache
from hmf.model.hybrid_zoo.models.qwen3_5_hmf.configuration_qwen3_5_hmf import Qwen3_5HMFTextConfig
from hmf.model.hybrid_zoo.models.qwen3_5_hmf.modeling_qwen3_5_hmf import Qwen3_5HMFForCausalLM

VOCAB = 97


def _model(attn="sdpa", n_layers=4, seed=0):
    cfg = Qwen3_5HMFTextConfig(
        vocab_size=VOCAB, hidden_size=64, intermediate_size=128, num_hidden_layers=n_layers,
        num_attention_heads=4, num_key_value_heads=2, head_dim=32, max_position_embeddings=512,
        rope_parameters={
            "rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 0.25,
            "mrope_section": [2, 1, 1], "mrope_interleaved": True,
        },
        layer_types=["full_attention"] * n_layers,
        hybrid_override_pattern="-".join(["*"] * n_layers),
    )
    cfg._attn_implementation = attn
    torch.manual_seed(seed)
    return Qwen3_5HMFForCausalLM(cfg).eval()


def _ids(batch=2, seq=12, seed=1):
    return torch.randint(0, VOCAB, (batch, seq), generator=torch.Generator().manual_seed(seed))


def _incremental(model, ids, chunks, attention_mask=None):
    """Returns the logits of feeding ids through one HybridCache in chunks of the given sizes."""
    with torch.no_grad():
        cache, start, steps = HybridCache(model.config), 0, []
        for n in chunks:
            kwargs = {} if attention_mask is None else {"attention_mask": attention_mask[:, : start + n]}
            out = model(ids[:, start : start + n], past_key_values=cache, use_cache=True, **kwargs)
            cache, start = out.past_key_values, start + n
            steps.append(out.logits)
    return torch.cat(steps, dim=1)


def _left_padded_mask(ids, pad=4):
    mask = torch.ones_like(ids)
    mask[0, :pad] = 0  # only row 0 is padded
    return mask


def test_mask_sizes_count_cached_tokens():
    cache = HybridCache(_model().config)
    assert cache.get_mask_sizes(torch.arange(5), 0) == (5, 0)  # nothing cached yet
    k = torch.zeros(1, 2, 7, 16)
    cache.update(k, k, layer_idx=3)
    # Asked about a layer without K/V (e.g. a GDN layer 0): still the attention length.
    assert cache.get_mask_sizes(torch.arange(7, 10), 0) == (10, 0)
    assert cache.get_mask_sizes(torch.arange(7, 8), 3) == (8, 0)


@pytest.mark.parametrize("attn", ["sdpa", "eager"])
@pytest.mark.parametrize("chunks", [[8, 1, 1, 1, 1], [5, 3, 4]])
def test_left_padded_decode_equals_full_forward(attn, chunks):
    model = _model(attn)
    ids = _ids()
    mask = _left_padded_mask(ids)
    with torch.no_grad():
        full = model(ids, attention_mask=mask).logits
    inc = _incremental(model, ids, chunks, attention_mask=mask)
    # Row 0's padded positions produce arbitrary logits in both paths; compare its real tokens.
    torch.testing.assert_close(inc[0, 4:], full[0, 4:], rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(inc[1], full[1], rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("attn", ["sdpa", "eager"])
@pytest.mark.parametrize("chunks", [[5, 3, 4], [1] * 6 + [6], [3] * 4])
def test_chunked_prefill_equals_full_forward(attn, chunks):
    model = _model(attn)
    ids = _ids()
    with torch.no_grad():
        full = model(ids).logits
    torch.testing.assert_close(_incremental(model, ids, chunks), full, rtol=1e-5, atol=1e-5)


def test_generate_left_padded_matches_unpadded():
    model = _model()
    ids = _ids(seq=8)
    mask = _left_padded_mask(ids)
    with torch.no_grad():
        padded = model.generate(ids, attention_mask=mask, max_new_tokens=6, do_sample=False)
        alone = model.generate(ids[:1, 4:], max_new_tokens=6, do_sample=False)
    assert torch.equal(padded[0, 8:], alone[0, 4:])


@pytest.mark.skipif(
    not torch.cuda.is_available() or importlib.util.find_spec("flash_attn") is None,
    reason="flash-attention-2 needs CUDA and flash_attn",
)
@pytest.mark.parametrize("chunks", [[8, 1, 1, 1, 1], [5, 3, 4]])
def test_flash_attention_left_padded_decode(chunks):
    model = _model("flash_attention_2").to("cuda", torch.bfloat16)
    ids = _ids().cuda()
    mask = _left_padded_mask(ids)
    with torch.no_grad():
        full = model(ids, attention_mask=mask).logits
    inc = _incremental(model, ids, chunks, attention_mask=mask)
    err = (inc - full).float().abs().amax(-1)
    # bf16 noise is ~1e-2; the bug gave ~0.5 on the padded row.
    assert err[0, 4:].max() < 0.1 and err[1].max() < 0.1, err
