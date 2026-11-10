"""
Tests hybrid Gemma 4 under flash_attn: fa2, where each attention layer runs flash attention 2 if its heads are small
enough for it, and SDPA otherwise. In Gemma 4, the sliding window layers, whose heads have 256 dimensions, run flash
attention 2, and the full attention layers, whose heads have 512, run SDPA with masks in SDPA's format. An HMF SWA
layer follows the layer that it replaces. The model of test_hybrid_gemma4.py has full attention heads of 32, so the
tests lower the head size limit to 16, which sends its full layers to SDPA. On the CPU, the tests check which mask
each layer asks for. On a GPU with flash attention, they check that the outputs, the
gradients, cached decoding, and Stage 1's two streams match those of the same weights with SDPA throughout.

The GPU tests run in fp16, as flash attention runs in fp16 and bf16 only, and fp16 rounds eight times finer than
bf16. Their attention's query and key norms are set to head_dim**-0.25, so that the scores scale as 1/sqrt(head_dim)
does in other models (Gemma 4 scales them by 1, which with untrained norms of 1 makes a random model's attention so
sharp that rounding changes it). Two correct kernels then differ by fp16's rounding over a few layers, about 1e-3
relative, so the bound is 1e-2 relative, set before the tests ran.
"""

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_hybrid_gemma4 import TINY, VOCAB  # noqa: E402

import hmf.model.hybrid_zoo.models.hybrid_gemma4.modeling_hybrid_gemma4 as modeling  # noqa: E402
from hmf.model.hybrid_zoo.models.hybrid_gemma4.configuration_hybrid_gemma4 import HybridGemma4TextConfig  # noqa: E402
from hmf.model.hybrid_zoo.models.hybrid_gemma4.modeling_hybrid_gemma4 import (  # noqa: E402
    Gemma4RMSNorm,
    HybridGemma4ForCausalLM,
)

TOLERANCE = 1e-2
SWA = {"swa_config": {"window_size": 3}}
# The tiny model's layers: sliding 0 to 3 and 5 to 8, full 4 and 9; layers 6 to 9 share the keys and values of
# layer 5 (sliding) and layer 4 (full)
ALL_ATTENTION = "-".join(["*"] * 10)
SWA_EVERYWHERE = "*-SWA-*-*-SWA-*-*-SWA-*-SWA"  # at a sliding, a full, a sharing sliding, and a sharing full layer
GKA_AT_SOURCES = "GKA-*-*-*-GKA-GKA-*-*-*-*"

needs_flash = pytest.mark.skipif(
    not torch.cuda.is_available() or importlib.util.find_spec("flash_attn") is None,
    reason="flash attention needs CUDA and flash_attn",
)
needs_fla = pytest.mark.skipif(importlib.util.find_spec("fla") is None, reason="GKA's kernels need fla")


@pytest.fixture(autouse=True)
def small_flash_attention_heads(monkeypatch):
    """Lowers flash attention's head size limit below the tiny model's full attention heads of 32."""
    monkeypatch.setattr(modeling, "FLASH_ATTENTION_MAX_HEAD_DIM", 16)


def _model(pattern, attn, **extra):
    torch.manual_seed(0)
    config = HybridGemma4TextConfig(**TINY, hybrid_override_pattern=pattern, **extra)
    model = HybridGemma4ForCausalLM(config).eval()
    model.config._attn_implementation = attn
    for name, module in model.named_modules():
        if isinstance(module, Gemma4RMSNorm) and name.endswith(("q_norm", "k_norm")) and module.with_scale:
            module.weight.data.fill_(module.weight.shape[0] ** -0.25)
    return model


def _pair(pattern, **extra):
    """Returns the same weights with flash attention 2 and with SDPA, in fp16 on the GPU."""
    flash = _model(pattern, "flash_attention_2", **extra).to("cuda", torch.float16)
    sdpa = _model(pattern, "sdpa", **extra).to("cuda", torch.float16)
    sdpa.load_state_dict(flash.state_dict())
    return flash, sdpa


def _batch(left_padding=False):
    gen = torch.Generator().manual_seed(1)
    ids = torch.randint(3, VOCAB, (2, 24), generator=gen)
    mask = torch.ones_like(ids)
    if left_padding:
        mask[1, :5] = 0
    else:
        mask[1, -5:] = 0
    return ids.cuda(), mask.cuda()


def _error(a, b, mask=None):
    """The largest difference between a and b at the unpadded tokens, relative to the largest magnitude of b."""
    if mask is not None:
        a, b = a[mask.bool()], b[mask.bool()]
    return ((a.float() - b.float()).abs().max() / b.float().abs().max().clamp_min(1e-6)).item()


def test_flash_attention_layers_ask_for_masks_in_its_format_and_larger_heads_for_sdpas():
    model = _model(SWA_EVERYWHERE, "flash_attention_2", **SWA)
    keys = [layer.attention_type for layer in model.model.layers]
    assert keys[0] == "sliding_attention" and keys[1] == "swa"
    assert keys[4] == "swa:sdpa" and keys[9] == "swa:sdpa"
    assert keys[2] == "sliding_attention" and keys[7] == "swa"
    assert model.model._mask_keys() == {"full_attention", "sliding_attention", "swa", "swa:sdpa"}


def test_full_attention_layers_ask_for_an_sdpa_mask_while_mixers_keep_the_model_format():
    model = _model(GKA_AT_SOURCES, "flash_attention_2")
    keys = [layer.attention_type for layer in model.model.layers]
    assert keys[0] == "full_attention" and keys[4] == "full_attention"  # GKA's mixers
    assert keys[9] == "full_attention:sdpa" and keys[1] == "sliding_attention"


def test_sdpa_models_ask_for_masks_without_the_sdpa_suffix():
    model = _model(SWA_EVERYWHERE, "sdpa", **SWA)
    assert model.model._mask_keys() == {"full_attention", "sliding_attention", "swa"}


@needs_flash
@pytest.mark.parametrize("left_padding", [False, True], ids=["right_padding", "left_padding"])
@pytest.mark.parametrize("pattern", [ALL_ATTENTION, SWA_EVERYWHERE], ids=["attention", "swa"])
def test_flash_attention_outputs_match_sdpa(pattern, left_padding):
    flash, sdpa = _pair(pattern, **SWA)
    ids, mask = _batch(left_padding)
    with torch.no_grad():
        assert _error(flash(ids, attention_mask=mask).logits, sdpa(ids, attention_mask=mask).logits, mask) < TOLERANCE


@needs_flash
@needs_fla
def test_flash_attention_outputs_match_sdpa_with_gka_at_the_sources():
    flash, sdpa = _pair(GKA_AT_SOURCES)
    ids, mask = _batch()
    with torch.no_grad():
        assert _error(flash(ids, attention_mask=mask).logits, sdpa(ids, attention_mask=mask).logits, mask) < TOLERANCE


@needs_flash
@pytest.mark.parametrize("pattern", [ALL_ATTENTION, SWA_EVERYWHERE], ids=["attention", "swa"])
def test_flash_attention_gradients_match_sdpa(pattern):
    flash, sdpa = _pair(pattern, **SWA)
    ids, mask = _batch()
    labels = ids.masked_fill(mask == 0, -100)
    for model in (flash, sdpa):
        model.train()
        model(ids, attention_mask=mask, labels=labels).loss.backward()
    worst = max(_error(f.grad, s.grad) for (_, f), (_, s) in zip(flash.named_parameters(), sdpa.named_parameters())
                if s.grad is not None and s.grad.abs().max() > 0)
    assert worst < TOLERANCE


@needs_flash
@pytest.mark.parametrize("pattern", [ALL_ATTENTION, SWA_EVERYWHERE], ids=["attention", "swa"])
def test_flash_attention_decodes_from_the_cache_as_its_full_forward_does(pattern):
    flash, _ = _pair(pattern, **SWA)
    ids, mask = _batch(left_padding=True)
    with torch.no_grad():
        full = flash(ids, attention_mask=mask).logits
        prefix = 16
        out = flash(ids[:, :prefix], attention_mask=mask[:, :prefix], use_cache=True)
        cache, steps = out.past_key_values, [out.logits[:, -1]]
        for position in range(prefix, ids.shape[1] - 1):
            out = flash(ids[:, position : position + 1], attention_mask=mask[:, : position + 1],
                        past_key_values=cache, use_cache=True)
            cache = out.past_key_values
            steps.append(out.logits[:, -1])
    assert _error(torch.stack(steps, dim=1), full[:, prefix - 1 : -1]) < TOLERANCE


@needs_flash
def test_flash_attention_runs_both_streams_of_stage1_as_sdpa_does():
    pattern = "*DA-*DA-*DA-*DA-*DA-SWA*-*DA-SWA*-*DA-*DA"
    flash, sdpa = _pair(pattern, **SWA)
    ids, mask = _batch()
    with torch.no_grad():
        flash_outputs = list(flash.model.stage1_layer_outputs(ids, mask))
        sdpa_outputs = list(sdpa.model.stage1_layer_outputs(ids, mask))
    for layer_idx, (f, s) in enumerate(zip(flash_outputs, sdpa_outputs)):
        assert _error(f[0], s[0], mask) < TOLERANCE and _error(f[1], s[1], mask) < TOLERANCE, layer_idx
