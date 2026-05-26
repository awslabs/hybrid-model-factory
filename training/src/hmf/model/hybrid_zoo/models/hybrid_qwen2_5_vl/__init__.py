"""Hybrid Qwen2.5-VL model: Qwen2.5-VL vision encoder + HybridQwen2 text backbone.

Allows save_pretrained / from_pretrained roundtrip on a hybrid VL model.
Without this, transformers loads the saved weights into a stock Qwen2.5-VL,
discarding the GKA layers entirely.
"""
from .configuration_hybrid_qwen2_5_vl import HybridQwen2_5_VLConfig
from .modeling_hybrid_qwen2_5_vl import (
    HybridQwen2_5_VLModel,
    HybridQwen2_5_VLForConditionalGeneration,
)

__all__ = [
    "HybridQwen2_5_VLConfig",
    "HybridQwen2_5_VLModel",
    "HybridQwen2_5_VLForConditionalGeneration",
]
