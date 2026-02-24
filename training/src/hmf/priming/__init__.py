"""
Priming module for converting Transformer models to Hybrid architectures.

This module provides functionality for:
- Converting pretrained Transformers to Hybrid models (prime-init)
- Converting fused Hybrid models to standard format (prime-unfuse)
"""

from .hybridize_model import hybridize_model, load_config, verify_hybrid_config
from .fused_to_standard import load_model, get_model_type, get_unfused_config


__all__ = [
    "hybridize_model",
    "load_config", 
    "verify_hybrid_config",
    "load_model",
    "get_model_type",
    "get_unfused_config",
]
