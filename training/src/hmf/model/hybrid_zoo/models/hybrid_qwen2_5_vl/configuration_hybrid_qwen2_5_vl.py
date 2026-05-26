"""Config for HybridQwen2_5_VL — extends the upstream Qwen2.5-VL config with a
hybrid text backbone.

The text_config is a HybridQwen2Config (carrying hybrid_override_pattern and
layer_types) instead of the stock Qwen2_5_VLTextConfig. Vision config is
unchanged from upstream.
"""
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import (
    Qwen2_5_VLConfig,
    Qwen2_5_VLVisionConfig,
)

from ..hybrid_qwen2.configuration_hybrid_qwen2 import HybridQwen2Config


class HybridQwen2_5_VLConfig(Qwen2_5_VLConfig):
    model_type = "hybrid_qwen2_5_vl"
    sub_configs = {
        "vision_config": Qwen2_5_VLVisionConfig,
        "text_config": HybridQwen2Config,
    }


__all__ = ["HybridQwen2_5_VLConfig"]
