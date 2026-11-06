"""
Priming module for converting Transformer models to Hybrid architectures.

This module provides functionality for:
- Converting pretrained Transformers to Hybrid models (prime-init)
- Converting fused Hybrid models to standard format (prime-unfuse)
"""

import importlib


# Imported when first used, so that importing one priming module (e.g., decision_init on a Mac)
# does not import the hybrid priming code and its CUDA-only kernels
_EXPORTS = {
    "hybridize_model": ".hybridize_model",
    "load_config": ".hybridize_model",
    "verify_hybrid_config": ".hybridize_model",
    "load_model": ".fused_to_standard",
    "get_model_type": ".fused_to_standard",
    "get_unfused_config": ".fused_to_standard",
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(importlib.import_module(_EXPORTS[name], __name__), name)
