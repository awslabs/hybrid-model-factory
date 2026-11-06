"""
The masks that GDN and GKA layers receive under every attention backend.

Models pass their linear sequence mixers the mask they build for attention, which is a 2D padding mask under
flash attention and a 4D mask under SDPA and eager. GDN accepted only the 2D mask and GKA refused any mask in
eval mode, so with HybridCache.get_mask_sizes covering the cached tokens, SDPA and eager crashed on cached
multi-token steps. padding_mask_for_new_tokens reduces either mask to the padding of the new tokens.

The CPU tests check the helper on the masks transformers builds. The GPU tests need CUDA and fla. They check
that GDN gives bit-identical outputs for a 2D mask and the 4D masks with the same padding, so flash attention
runs are unchanged, and that tiny Qwen3.5 GDN hybrids decode from the cache as a full forward does under SDPA,
eager, and flash attention. For tiny Qwen3 GKA hybrids, which decode one token at a time from a cache, they check
that one-token decoding matches a full forward, that a step of several tokens on top of a cache raises, and that
generate() and cache composition run without raising.
"""

import importlib.util

import pytest
import torch
from transformers.masking_utils import create_causal_mask

from hmf.model.hybrid_zoo.layers.mask_utils import padding_mask_for_new_tokens
from hmf.model.hybrid_zoo.models.cache import HybridCache
from hmf.model.hybrid_zoo.models.qwen3_5_hmf.configuration_qwen3_5_hmf import Qwen3_5HMFTextConfig

VOCAB = 97
HIDDEN = 64

needs_cuda = pytest.mark.skipif(
    not torch.cuda.is_available() or importlib.util.find_spec("fla") is None,
    reason="GDN and GKA kernels need CUDA and fla",
)
needs_flash = pytest.mark.skipif(
    not torch.cuda.is_available() or importlib.util.find_spec("flash_attn") is None,
    reason="flash-attention-2 needs CUDA and flash_attn",
)
BACKENDS = ["sdpa", "eager", pytest.param("flash_attention_2", marks=needs_flash)]


def _qwen3_5_config(pattern, attn="sdpa"):
    n_layers = len(pattern.split("-"))
    cfg = Qwen3_5HMFTextConfig(
        vocab_size=VOCAB, hidden_size=HIDDEN, intermediate_size=128, num_hidden_layers=n_layers,
        num_attention_heads=4, num_key_value_heads=2, head_dim=32, max_position_embeddings=512,
        linear_num_key_heads=2, linear_num_value_heads=2, linear_key_head_dim=32, linear_value_head_dim=32,
        rope_parameters={
            "rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 0.25,
            "mrope_section": [2, 1, 1], "mrope_interleaved": True,
        },
        layer_types=["full_attention"] * n_layers,
        hybrid_override_pattern=pattern,
    )
    cfg._attn_implementation = attn
    return cfg


def _mask_for(attn, padding, q_len, past=0):
    """Returns the mask create_causal_mask builds for q_len new tokens after past cached ones, with this padding."""
    cfg = _qwen3_5_config("*", attn)
    cache = None
    if past:
        cache = HybridCache(cfg)
        keys = torch.zeros(padding.shape[0], 2, past, 32)
        cache.update(keys, keys, layer_idx=0)
    return create_causal_mask(
        config=cfg,
        inputs_embeds=torch.zeros(padding.shape[0], q_len, HIDDEN),
        attention_mask=padding,
        cache_position=torch.arange(past, past + q_len),
        past_key_values=cache,
    )


def _padding(total, left=None, right=None):
    """Returns a [2, total] padding mask whose row 0 is padded on the left or the right and whose row 1 is not."""
    mask = torch.ones(2, total, dtype=torch.long)
    if left:
        mask[0, :left] = 0
    if right:
        mask[0, -right:] = 0
    return mask


@pytest.mark.parametrize("attn", ["sdpa", "eager"])
@pytest.mark.parametrize("side", ["left", "right"])
def test_prefill_padding_matches_the_2d_mask(attn, side):
    padding = _padding(8, **{side: 3})
    mask = _mask_for(attn, padding, q_len=8)
    assert mask is not None and mask.dim() == 4
    assert torch.equal(padding_mask_for_new_tokens(mask, 8), padding.int())
    assert torch.equal(padding_mask_for_new_tokens(padding, 8), padding.int())


@pytest.mark.parametrize("attn", ["sdpa", "eager"])
@pytest.mark.parametrize("q_len", [1, 3])
def test_padding_in_the_cache_only_gives_none(attn, q_len):
    # Left padding sits in the cached prefix, so none of the new tokens is padding.
    padding = _padding(5 + q_len, left=3)
    mask = _mask_for(attn, padding, q_len=q_len, past=5)
    assert mask is not None and mask.dim() == 4
    assert padding_mask_for_new_tokens(mask, q_len) is None
    assert padding_mask_for_new_tokens(padding, q_len) is None


@pytest.mark.parametrize("attn", ["sdpa", "eager"])
def test_unpadded_step_after_prefill_gives_none(attn):
    # The 4D mask SDPA and eager build for a multi-token step after a prefill, which crashed GDN.
    mask = _mask_for(attn, torch.ones(2, 8, dtype=torch.long), q_len=3, past=5)
    assert mask is not None and mask.dim() == 4
    assert padding_mask_for_new_tokens(mask, 3) is None


def test_no_mask_and_no_padding_give_none():
    assert padding_mask_for_new_tokens(None, 4) is None
    assert padding_mask_for_new_tokens(torch.ones(2, 4, dtype=torch.long), 4) is None


def test_other_shapes_are_rejected():
    with pytest.raises(ValueError):
        padding_mask_for_new_tokens(torch.ones(2, 4, 4), 4)


@needs_cuda
@torch.no_grad()
def test_gdn_layer_is_bit_identical_for_2d_and_4d_masks():
    from hmf.model.hybrid_zoo.layers.gated_deltanet.modules import GatedDeltaNetLayer

    torch.manual_seed(0)
    layer = GatedDeltaNetLayer(_qwen3_5_config("GDN"), layer_idx=0).to("cuda", torch.bfloat16).eval()
    x = torch.randn(2, 8, HIDDEN, device="cuda", dtype=torch.bfloat16)
    padding = _padding(8, left=3)
    reference = layer(x.clone(), attention_mask=padding.cuda())[0]
    for attn in ["sdpa", "eager"]:
        mask = _mask_for(attn, padding, q_len=8)
        mask = mask.cuda() if mask.dtype == torch.bool else mask.to("cuda", torch.bfloat16)
        assert torch.equal(layer(x.clone(), attention_mask=mask)[0], reference), attn

    # A 4D mask without padding among the new tokens is the same as no mask.
    unpadded = torch.ones(2, 8, dtype=torch.long)
    no_mask = layer(x[:, :3].clone())[0]
    for attn in ["sdpa", "eager"]:
        mask = _mask_for(attn, unpadded, q_len=3, past=5)
        mask = mask.cuda() if mask.dtype == torch.bool else mask.to("cuda", torch.bfloat16)
        assert torch.equal(layer(x[:, :3].clone(), attention_mask=mask)[0], no_mask), attn


def _qwen3_5_gdn_model(attn):
    from hmf.model.hybrid_zoo.models.qwen3_5_hmf.modeling_qwen3_5_hmf import Qwen3_5HMFForCausalLM

    torch.manual_seed(0)
    model = Qwen3_5HMFForCausalLM(_qwen3_5_config("GDN-*-GDN-*", attn))
    return model.to("cuda", torch.bfloat16).eval()


def _qwen3_gka_model(attn):
    from hmf.model.hybrid_zoo.models.hybrid_qwen3.configuration_hybrid_qwen3 import HybridQwen3Config
    from hmf.model.hybrid_zoo.models.hybrid_qwen3.modeling_hybrid_qwen3 import HybridQwen3ForCausalLM

    torch.manual_seed(0)
    cfg = HybridQwen3Config(
        vocab_size=VOCAB, hidden_size=128, intermediate_size=256, num_hidden_layers=2, num_attention_heads=2,
        num_key_value_heads=1, head_dim=64, hybrid_override_pattern="GKA-*",
    )
    cfg._attn_implementation = attn
    return HybridQwen3ForCausalLM(cfg).to("cuda", torch.bfloat16).eval()


def _ids(batch=2, seq=12, seed=1):
    return torch.randint(0, VOCAB, (batch, seq), generator=torch.Generator().manual_seed(seed)).cuda()


@torch.no_grad()
def _incremental(model, ids, chunks, attention_mask=None):
    """Returns the logits of feeding ids through one cache in chunks of the given sizes."""
    cache, start, steps = None, 0, []
    for n in chunks:
        kwargs = {} if attention_mask is None else {"attention_mask": attention_mask[:, : start + n]}
        out = model(ids[:, start : start + n], past_key_values=cache, use_cache=True, **kwargs)
        cache, start = out.past_key_values, start + n
        steps.append(out.logits)
    return torch.cat(steps, dim=1)


def _max_err(a, b):
    """Returns the largest absolute difference, relative to the largest reference logit."""
    a, b = a.float(), b.float()
    return ((a - b).abs().max() / b.abs().max()).item()


# bf16 noise on these tiny models is about 1e-2 of the logit scale; a wrong mask or a lost state is near 1.
TOL = 0.05


@needs_cuda
@pytest.mark.parametrize("attn", BACKENDS)
@pytest.mark.parametrize("chunks", [[5, 3, 4], [8, 1, 1, 1, 1], [3] * 4])
def test_gdn_hybrid_chunked_prefill_equals_full_forward(attn, chunks):
    model = _qwen3_5_gdn_model(attn)
    ids = _ids()
    with torch.no_grad():
        full = model(ids).logits
    assert _max_err(_incremental(model, ids, chunks), full) < TOL


@needs_cuda
@pytest.mark.parametrize("attn", BACKENDS)
@pytest.mark.parametrize("chunks", [[8, 1, 1, 1, 1], [5, 3, 4]])
def test_gdn_hybrid_left_padded_decode(attn, chunks):
    model = _qwen3_5_gdn_model(attn)
    ids = _ids()
    mask = _padding(12, left=4).cuda()
    with torch.no_grad():
        full = model(ids, attention_mask=mask).logits
        alone = model(ids[:1, 4:]).logits
    inc = _incremental(model, ids, chunks, attention_mask=mask)
    # The padded row's real tokens read as they do without padding, in a full forward and from the cache.
    assert _max_err(full[0, 4:], alone[0]) < TOL
    assert _max_err(inc[0, 4:], alone[0]) < TOL
    assert _max_err(inc[1], full[1]) < TOL


@needs_cuda
@needs_flash
@pytest.mark.parametrize("attn", ["sdpa", "eager"])
def test_gdn_hybrid_backends_agree_with_flash_attention(attn):
    # Flash attention runs never changed, so they are the reference for SDPA and eager.
    ids = _ids()
    mask = _padding(12, left=4).cuda()
    flash = _incremental(_qwen3_5_gdn_model("flash_attention_2"), ids, [8, 1, 1, 1, 1], attention_mask=mask)
    other = _incremental(_qwen3_5_gdn_model(attn), ids, [8, 1, 1, 1, 1], attention_mask=mask)
    assert _max_err(other[0, 4:], flash[0, 4:]) < TOL
    assert _max_err(other[1], flash[1]) < TOL


# GKA decodes one token at a time from a cache and raises on a step of several tokens on top of a cached state,
# which its decoding kernel cannot compute. The tests below check that one-token decoding is right, that the
# error fires where it should, and that generate() and cache composition, which only make one-token steps on top
# of a cache, never trigger it.


@needs_cuda
@pytest.mark.parametrize("attn", BACKENDS)
@pytest.mark.parametrize("chunks", [[8, 1, 1, 1, 1], [1] * 6])
def test_gka_hybrid_one_token_decode_equals_full_forward(attn, chunks):
    model = _qwen3_gka_model(attn)
    ids = _ids(seq=sum(chunks))
    with torch.no_grad():
        full = model(ids).logits
    assert _max_err(_incremental(model, ids, chunks), full) < TOL


@needs_cuda
@pytest.mark.parametrize("attn", BACKENDS)
@pytest.mark.parametrize("chunks", [[8, 4], [5, 3, 4]])
def test_gka_multi_token_step_after_prefill_raises(attn, chunks):
    model = _qwen3_gka_model(attn)
    with pytest.raises(NotImplementedError, match="one token at a time"):
        _incremental(model, _ids(), chunks)


@needs_cuda
@pytest.mark.parametrize("attn", BACKENDS)
def test_gka_still_refuses_padding(attn):
    model = _qwen3_gka_model(attn)
    with torch.no_grad(), pytest.raises(NotImplementedError, match="varlen"):
        model(_ids(), attention_mask=_padding(12, left=4).cuda())


@needs_cuda
@pytest.mark.parametrize("attn", BACKENDS)
def test_gka_generate_matches_a_forward_without_cache(attn):
    """generate() prefills the prompt and decodes one token at a time, so the new error must not fire."""
    model = _qwen3_gka_model(attn)
    prompt = torch.randint(0, VOCAB - 1, (2, 8), generator=torch.Generator().manual_seed(2)).cuda()
    with torch.no_grad():
        out = model.generate(
            prompt, attention_mask=torch.ones_like(prompt), max_new_tokens=8, do_sample=False,
            pad_token_id=VOCAB - 1, return_dict_in_generate=True, output_logits=True,
        )
        # Each step's logits, against a full forward over the generated sequence without a cache.
        full = model(out.sequences).logits
    steps = torch.stack(out.logits, dim=1)
    assert _max_err(steps, full[:, prompt.shape[1] - 1 : -1]) < TOL


class _NewlineTokenizer:
    """The one tokenizer method cache composition uses, which looks up the newline token."""

    def encode(self, text, add_special_tokens=False):
        return [VOCAB - 2]


@needs_cuda
@pytest.mark.parametrize("compose_type", ["full_kv_only", "concat_kv_only", "concat_kv_soup_ssm", "sw_soup_ssm"])
def test_gka_cache_composition_generates(compose_type):
    """
    Composition prefills chunks into fresh caches and feeds the query one token at a time, so the error must not
    fire.
    """
    from hmf.model.hybrid_zoo.models.cache_compose import wrap_for_composition

    model = wrap_for_composition(_qwen3_gka_model("flash_attention_2" if _has_flash() else "sdpa"), _NewlineTokenizer())
    gen = torch.Generator().manual_seed(3)
    prefix = torch.randint(0, VOCAB - 2, (1, 4), generator=gen).cuda()
    context = torch.randint(0, VOCAB - 2, (1, 40), generator=gen).cuda()
    question = torch.randint(0, VOCAB - 2, (1, 6), generator=gen).cuda()
    with torch.no_grad():
        out = model.generate(
            input_ids=torch.cat([context, question], dim=1), max_new_tokens=4, do_sample=False,
            pad_token_id=VOCAB - 1, return_dict_in_generate=True, output_logits=True, use_cache=True,
            compose_type=compose_type, num_chunks=2, sequential_positions=False,
            prefix_input_ids=prefix, suffix_input_ids=question,
        )
    assert len(out.logits) == 4 and all(torch.isfinite(step).all() for step in out.logits)


def _has_flash():
    return importlib.util.find_spec("flash_attn") is not None
