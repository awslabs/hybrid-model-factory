"""
GKA forward functions for vLLM.

This module contains the high-level GKA forward API:
- gka_chebyshev_gla_prefill: Full GKA prefill (Chebyshev + Simple GLA)
- gka_chebyshev_gla_decode: Single-token decode (Chebyshev + fused recurrent GLA)

Note: GKA uses GDNAttentionBackend directly - no custom attention backend needed.
"""

# No exports needed - GKA kernels are imported directly from gka_forward.py
__all__ = []
