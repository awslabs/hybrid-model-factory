"""HuggingFace config classes for hybrid models (vLLM only needs configs, not model implementations)."""

from transformers import AutoConfig

from .configuration_hybrid_qwen2 import HybridQwen2Config
from .configuration_hybrid_qwen3 import HybridQwen3Config
from .configuration_hybrid_qwen3_moe import HybridQwen3MoeConfig

__all__ = ["HybridQwen2Config", "HybridQwen3Config", "HybridQwen3MoeConfig", "register_configs"]


def register_configs():
    """Register config classes with HuggingFace AutoConfig."""
    AutoConfig.register("hybrid_qwen2", HybridQwen2Config)
    AutoConfig.register("hybrid_qwen3", HybridQwen3Config)
    AutoConfig.register("hybrid_qwen3_moe", HybridQwen3MoeConfig)
