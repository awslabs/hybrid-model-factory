"""BMOJO-F Attention Module for vLLM V1.

This module provides dual-stream attention (s-stream and c-stream) for
BMOJO-F models, with proper KV cache separation.

IMPORTANT: BMOJO-F requires two attention streams (s-stream and c-stream)
per decoder block. The only vLLM model type that allocates two attention
modules per layer is "longcat_flash", so BMOJO-F models set
model_type="longcat_flash" in their HuggingFace config to trigger
num_attn_module=2 and proper dual-stream KV cache allocation.
"""

# Import backend module to trigger @register_backend decorator
# This ensures BMOJOFAttentionBackend is registered when plugin loads
from . import bmojof_attn_backend

__all__ = [
    "bmojof_attn_backend",
]
