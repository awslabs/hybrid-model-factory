"""
Hybrid Model Factory Inference vLLM Plugin Registration

vLLM Compatibility Status:
==========================

v0.15.1
--------------------------
HybridQwen2ForCausalLM
   - Supports Attention + Mamba2 + BMOJO-F + GDN + GKA + SWA hybrid layers
   - V1 engine compatible (prefix caching supported)

HybridQwen3ForCausalLM
   - Supports Attention + Mamba2 + BMOJO-F + GDN + GKA + SWA hybrid layers
   - V1 engine compatible (prefix caching supported)

HybridQwen3MoeForCausalLM
   - Supports Attention + Mamba2 + BMOJO-F + GDN + GKA + SWA hybrid layers
   - V1 engine compatible (prefix caching supported)
   - Supports sparse MoE with expert parallelism

Migration Notes:
----------------
- All models migrated to V1 architecture
- V1 architecture provides automatic cache management via forward_context
- Supports all layer types: Attention, Mamba2, BMOJO-F, GDN, GKA, SWA
"""
import logging

logger = logging.getLogger(__name__)

def register():
    # Register the optimized GDN attention backend (overrides stock GDN_ATTN).
    # Must happen before any model initialization triggers backend resolution.
    import primed_vllm.gdn.primed_gdn_attn  # noqa: F401

    # Apply vLLM patches early, before any model initialization
    from primed_vllm import vllm_patches
    vllm_patches.apply_patches()

    from primed_vllm.hf_configs import register_configs
    register_configs()
    from vllm import ModelRegistry

    if "HybridQwen2ForCausalLM" not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model(
            "HybridQwen2ForCausalLM",
            "primed_vllm.hybridqwen2:HybridQwen2ForCausalLM"
        )
        logger.info("HybridQwen2: Successfully registered HybridQwen2ForCausalLM")
        logger.info("  vLLM v0.15.1 compatible (V1 engine, Attention + Mamba2 + GDN + GKA + SWA + BMOJO-F layers)")
    else:
        logger.info("HybridQwen2: HybridQwen2ForCausalLM already registered")

    if "HybridQwen3ForCausalLM" not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model(
            "HybridQwen3ForCausalLM",
            "primed_vllm.hybridqwen3:HybridQwen3ForCausalLM"
        )
        logger.info("HybridQwen3: Successfully registered HybridQwen3ForCausalLM")
        logger.info("  vLLM v0.15.1 compatible (V1 engine, Attention + Mamba2 + GDN + GKA + SWA + BMOJO-F layers)")
    else:
        logger.info("HybridQwen3: HybridQwen3ForCausalLM already registered")

    if "HybridQwen3MoeForCausalLM" not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model(
            "HybridQwen3MoeForCausalLM",
            "primed_vllm.hybridqwen3_moe:HybridQwen3MoeForCausalLM"
        )
        logger.info("HybridQwen3Moe: Successfully registered HybridQwen3MoeForCausalLM")
        logger.info("  vLLM v0.15.1 compatible (V1 engine, Attention + Mamba2 + GDN + GKA + SWA + BMOJO-F layers)")
    else:
        logger.info("HybridQwen3Moe: HybridQwen3MoeForCausalLM already registered")
