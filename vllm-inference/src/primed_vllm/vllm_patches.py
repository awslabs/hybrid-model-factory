"""Monkey patches for vLLM to support BMOJO-F dual-stream attention.

This module patches vLLM's KV cache initialization to correctly handle
models with multiple attention modules per decoder layer (like BMOJO-F's
s-stream and c-stream).

The issue: vLLM's `bind_kv_cache()` function uses `extract_layer_index()` to map
layer names to KV cache tensors. For BMOJO-F layers like "model.layers.1.bmojo_f.bmojo_attn.0",
this function extracts TWO integers [1, 0] (decoder layer index and attention module index).
With num_attn_module=1 (default), this causes an assertion error. With num_attn_module=2,
it correctly computes layer_index = 1*2 + 0 = 2, treating s-stream (.0) and c-stream (.1)
as separate layers.

vLLM sets num_attn_module=2 only for model_type=="longcat_flash", but BMOJO-F models use
model_type="hybrid_qwen3", so they default to num_attn_module=1.

The fix: Patch `initialize_kv_cache_tensors()` to temporarily set model_type to
"longcat_flash" for BMOJO-F models during KV cache initialization, ensuring
bind_kv_cache() uses num_attn_module=2.

Note: vLLM's tensor sharing mechanism (where multiple layers reference the same physical
GPU memory but use different blocks within it) is safe and memory-efficient. We do NOT
need to disable it for BMOJO-F models.

Usage:
    # This module is automatically imported by the plugin system
    # The patches are applied when the module is imported
    from primed_vllm import vllm_patches
"""

import logging

logger = logging.getLogger(__name__)

_patches_applied = False


def _patch_compilation_config_attention_ops():
    """Patch vLLM's CompilationConfig._attention_ops to include our custom ops.

    vLLM's @support_torch_compile uses _attention_ops to determine which custom ops
    are treated as graph break points (no_compile_ops). Ops not in this list cause
    torch.compile to trace through them, breaking cudagraph capture/replay.

    This adds all primed_vllm custom ops (Mamba2, GDN, GKA, BMOJO-F) so
    they are treated as graph break points, matching the native vLLM ops.
    """
    try:
        from vllm.config import CompilationConfig
    except ImportError as e:
        logger.warning(f"Could not import vLLM CompilationConfig for patching: {e}")
        return

    # List of custom ops that need to be added to _attention_ops
    # These are custom ops registered in primed_vllm that need to be treated
    # as graph break points for torch.compile + cudagraph compatibility
    custom_ops = [
        "vllm::mamba_mixer2_primed",  # Custom Mamba2 layer
        "vllm::gka_attention_core",            # Custom GKA layer
        "vllm::custom_gdn_attention_core",   # Custom GDN layer
        "vllm::bmojof_attention_s_forward",  # BMOJO-F s-stream attention
        "vllm::bmojof_attention_c_forward",  # BMOJO-F c-stream attention
    ]

    # Check if _attention_ops exists and is a list
    if not hasattr(CompilationConfig, '_attention_ops'):
        logger.warning("[COMPILATION PATCH] CompilationConfig._attention_ops not found, skipping patch")
        return

    original_ops = CompilationConfig._attention_ops
    if not isinstance(original_ops, list):
        logger.warning(f"[COMPILATION PATCH] _attention_ops is not a list: {type(original_ops)}, skipping patch")
        return

    # Add custom ops that aren't already in the list
    added_ops = []
    for op in custom_ops:
        if op not in original_ops:
            original_ops.append(op)
            added_ops.append(op)

    if added_ops:
        logger.info(f"[COMPILATION PATCH] Added custom ops to _attention_ops: {added_ops}")
        logger.info(f"[COMPILATION PATCH] Total _attention_ops count: {len(original_ops)}")
    else:
        logger.info("[COMPILATION PATCH] All custom ops already in _attention_ops, no changes needed")


def _patch_tokenizers_backend():
    """Patch transformers v4 to resolve the 'TokenizersBackend' class from transformers v5.

    Transformers v5 renamed PreTrainedTokenizerFast to TokenizersBackend.
    Checkpoints saved with v5 have tokenizer_class="TokenizersBackend" in their config,
    which fails on v4 with: "Tokenizer class TokenizersBackend does not exist".
    """
    import transformers
    if hasattr(transformers, "TokenizersBackend"):
        return  # v5+, nothing to patch

    from transformers import PreTrainedTokenizerFast
    setattr(transformers, "TokenizersBackend", PreTrainedTokenizerFast)

    # Patch _set_model_specific_special_tokens to handle v5-style list format.
    # v5 saves extra_special_tokens as a list of strings, but v4's method expects a dict.
    _orig = transformers.PreTrainedTokenizerBase._set_model_specific_special_tokens
    def _patched_set_model_specific_special_tokens(self, special_tokens):
        if isinstance(special_tokens, list):
            return  # v5-style list; tokens already registered via other paths
        return _orig(self, special_tokens)
    transformers.PreTrainedTokenizerBase._set_model_specific_special_tokens = _patched_set_model_specific_special_tokens

    logger.info("[TOKENIZER PATCH] Patched transformers v4 for v5 tokenizer compat")


def apply_patches():
    """Apply all BMOJO-F related patches to vLLM.

    This function is idempotent - calling it multiple times has no effect
    after the first call.
    """
    global _patches_applied
    if _patches_applied:
        logger.debug("[BMOJOF PATCH] Patches already applied, skipping")
        return

    logger.info("[PATCH] Starting to apply vLLM patches...")
    _patch_tokenizers_backend()
    _patch_compilation_config_attention_ops()
    _patch_initialize_kv_cache_tensors()
    _patches_applied = True
    logger.info("[BMOJOF PATCH] All BMOJO-F vLLM patches applied successfully")


def _patch_initialize_kv_cache_tensors():
    """Patch GPUModelRunner.initialize_kv_cache_tensors to support BMOJO-F.

    The patch modifies the model_type temporarily to "longcat_flash" for BMOJO-F
    models, ensuring bind_kv_cache() uses num_attn_module=2 for proper KV cache
    separation between s-stream and c-stream.
    """
    try:
        import vllm.v1.worker.gpu_model_runner as gpu_model_runner
    except ImportError as e:
        logger.warning(f"Could not import vLLM modules for patching: {e}")
        return

    # Store reference to original method
    original_initialize_kv_cache_tensors = (
        gpu_model_runner.GPUModelRunner.initialize_kv_cache_tensors
    )

    def patched_initialize_kv_cache_tensors(self, kv_cache_config, kernel_block_sizes):
        """Patched version that detects BMOJO-F models for num_attn_module=2.

        This method wraps the original to temporarily set model_type to "longcat_flash"
        for BMOJO-F models. This ensures bind_kv_cache() (called within
        initialize_kv_cache_tensors) sees model_type == "longcat_flash" and uses
        num_attn_module=2, giving s-stream and c-stream separate KV caches.
        """
        logger.info("[BMOJOF PATCH] initialize_kv_cache_tensors called")

        # Detect if this is a BMOJO-F model
        model_type = getattr(self.model_config.hf_config, 'model_type', '')
        is_bmojof = _is_bmojof_model(model_type, self.model_config.hf_config)

        logger.info(f"[BMOJOF PATCH] model_type='{model_type}', is_bmojof={is_bmojof}")

        if is_bmojof:
            logger.info(
                f"[BMOJOF PATCH] Detected BMOJO-F model (type={model_type}), "
                "temporarily setting model_type='longcat_flash' for num_attn_module=2"
            )
            # Store original model_type
            original_model_type = self.model_config.hf_config.model_type

            # Temporarily set to "longcat_flash" to trigger num_attn_module=2
            self.model_config.hf_config.model_type = "longcat_flash"
            logger.info(f"[BMOJOF PATCH] Changed model_type: '{original_model_type}' -> 'longcat_flash'")

            try:
                logger.info("[BMOJOF PATCH] Calling original initialize_kv_cache_tensors...")
                result = original_initialize_kv_cache_tensors(
                    self, kv_cache_config, kernel_block_sizes
                )
                logger.info("[BMOJOF PATCH] Original initialize_kv_cache_tensors completed")
            finally:
                # Restore original model_type
                self.model_config.hf_config.model_type = original_model_type
                logger.info(f"[BMOJOF PATCH] Restored model_type to '{original_model_type}'")
            return result
        else:
            # Not a BMOJO-F model, use original behavior
            logger.info("[BMOJOF PATCH] Not a BMOJO-F model, using original behavior")
            return original_initialize_kv_cache_tensors(
                self, kv_cache_config, kernel_block_sizes
            )

    # Apply the patch
    gpu_model_runner.GPUModelRunner.initialize_kv_cache_tensors = (
        patched_initialize_kv_cache_tensors
    )
    logger.info("[BMOJOF PATCH] Patched GPUModelRunner.initialize_kv_cache_tensors")


def _is_bmojof_model(model_type: str, hf_config) -> bool:
    """Detect if the model is a BMOJO-F model.

    Detection is based on:
    1. Model type containing 'bmojo' (case-insensitive)
    2. Presence of 'BF' or 'BMF' in hybrid_override_pattern

    Note: We check the pattern instead of just bmojo_config presence because
    the config always initializes bmojo_config as an empty dict even for
    non-BMOJO-F models.

    Args:
        model_type: The model type string from HF config
        hf_config: The HuggingFace config object

    Returns:
        True if this is a BMOJO-F model, False otherwise
    """
    # Check model type
    if 'bmojo' in model_type.lower():
        return True

    # Check for BMOJO-F layers in the pattern (BMF)
    pattern = getattr(hf_config, 'hybrid_override_pattern', '')
    if 'BMF' in pattern:
        return True

    return False


# Patches are now applied from primed_vllm/register.py during plugin registration
# This ensures they are applied before any model initialization
