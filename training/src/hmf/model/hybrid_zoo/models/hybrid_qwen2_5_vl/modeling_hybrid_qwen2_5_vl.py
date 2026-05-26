"""Hybrid Qwen2.5-VL — Qwen2.5-VL multimodal pieces with a HybridQwen2 text
backbone.

The only structural change vs upstream Qwen2_5_VL is that the language model
inside the multimodal wrapper is `HybridQwen2Model` (some attention layers
replaced with GKA / mamba-style ops per `hybrid_override_pattern`). Vision
encoder, image processor, and the multimodal forward path are inherited
unchanged.

Why a dedicated class:
    `save_pretrained` / `from_pretrained` need to roundtrip GKA weights. Saving
    a hybrid model under model_type=`qwen2_5_vl` would silently drop them at
    load time because transformers would build a stock Qwen2_5_VLTextModel and
    discard any keys it didn't expect.

Position embeddings (MRoPE → 1D RoPE):
    Qwen2.5-VL uses MRoPE — `position_ids` is `[N, batch, seq]` carrying
    multiple positional axes (T/H/W in 3-axis form, plus a mm_token_type
    axis in newer transformers releases). HybridQwen2 attention uses
    standard 1D RoPE expecting `[batch, seq]`. We collapse to the temporal
    axis (axis 0) before delegating to the upstream forward via a
    forward-pre-hook on `language_model` — the only correct interception
    point because the parent Qwen2_5_VLModel.forward computes MRoPE
    position_ids internally before calling language_model. The collapse
    is exact for text tokens and an approximation for image tokens
    (H/W axes dropped). A future MRoPE-aware GKA layer would be the
    principled fix; the temporal-axis approximation works well in practice.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VisionTransformerPretrainedModel,
    Qwen2_5_VLForConditionalGeneration,
    Qwen2_5_VLModel,
    Qwen2_5_VLPreTrainedModel,
)

from ..hybrid_qwen2.modeling_hybrid_qwen2 import HybridQwen2Model
from .configuration_hybrid_qwen2_5_vl import HybridQwen2_5_VLConfig


def _collapse_mrope_position_ids(position_ids: torch.Tensor | None) -> torch.Tensor | None:
    """Reduce N-axis MRoPE position_ids to 1D ``[batch, seq]``.

    Qwen2.5-VL's ``compute_3d_position_ids`` returns shape ``[N, batch, seq]``
    where N is the number of MRoPE axes (3 = T/H/W in the original paper,
    4 in newer transformers releases that add a mm_token_type axis). For
    text-only positions all axes are equal; for vision positions the temporal
    (axis 0) carries the per-token sequence index, so we keep axis 0 and drop
    the rest. This is exact for text and an approximation for vision tokens.
    """
    if position_ids is None:
        return None
    if position_ids.dim() == 3 and position_ids.shape[0] in (3, 4):
        return position_ids[0]
    return position_ids


class HybridQwen2_5_VLPreTrainedModel(Qwen2_5_VLPreTrainedModel):
    config_class = HybridQwen2_5_VLConfig
    config: HybridQwen2_5_VLConfig
    base_model_prefix = "model"


class HybridQwen2_5_VLModel(Qwen2_5_VLModel):
    """Qwen2_5_VLModel with `language_model = HybridQwen2Model(...)`."""

    config_class = HybridQwen2_5_VLConfig

    def __init__(self, config: HybridQwen2_5_VLConfig):
        # Skip Qwen2_5_VLModel.__init__ (which would build a stock text model);
        # call the grandparent and assemble our own pieces.
        Qwen2_5_VLPreTrainedModel.__init__(self, config)
        self.visual = Qwen2_5_VisionTransformerPretrainedModel._from_config(
            config.vision_config
        )
        self.language_model = HybridQwen2Model(config.text_config)
        self.rope_deltas = None  # cached per upstream contract
        # Collapse 3D MRoPE position_ids → 1D right at the language-model
        # boundary. The outer Qwen2.5-VL forward computes MRoPE position_ids
        # internally *after* our wrapper override runs, so a pre-forward hook
        # on language_model is the only correct interception point.
        self.language_model.register_forward_pre_hook(
            _collapse_position_ids_hook, with_kwargs=True
        )
        self.post_init()


def _collapse_position_ids_hook(module, args, kwargs):
    """Forward pre-hook: rewrite 3D MRoPE position_ids → 1D before the
    HybridQwen2Model receives them."""
    if "position_ids" in kwargs:
        kwargs["position_ids"] = _collapse_mrope_position_ids(kwargs["position_ids"])
        return args, kwargs
    # position_ids may be passed positionally; HybridQwen2Model.forward
    # signature is (input_ids, attention_mask, position_ids, ...).
    if len(args) >= 3 and isinstance(args[2], torch.Tensor):
        new_args = list(args)
        new_args[2] = _collapse_mrope_position_ids(new_args[2])
        return tuple(new_args), kwargs
    return args, kwargs


class HybridQwen2_5_VLForConditionalGeneration(Qwen2_5_VLForConditionalGeneration):
    """Drop-in replacement for `Qwen2_5_VLForConditionalGeneration`.

    Only `self.model` differs; all generation, loss, and processor APIs are
    inherited.
    """

    config_class = HybridQwen2_5_VLConfig

    def __init__(self, config: HybridQwen2_5_VLConfig):
        Qwen2_5_VLPreTrainedModel.__init__(self, config)
        self.model = HybridQwen2_5_VLModel(config)
        self.vocab_size = config.text_config.vocab_size
        self.lm_head = nn.Linear(
            config.text_config.hidden_size,
            config.text_config.vocab_size,
            bias=False,
        )
        self.post_init()


__all__ = [
    "HybridQwen2_5_VLPreTrainedModel",
    "HybridQwen2_5_VLModel",
    "HybridQwen2_5_VLForConditionalGeneration",
]
