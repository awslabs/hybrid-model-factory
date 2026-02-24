"""
Hybrid Layer Initialization Module

This module provides model-agnostic functions for initializing hybrid layers
(Mamba2, B'MOJO, Gated DeltaNet, GKA, SWA) from pretrained attention weights.

These functions handle the weight transfer from standard attention layers to their
hybrid counterparts, including special initialization strategies for gates and
projection layers.

The functions are designed to work with any transformer model (Qwen, Llama, etc.)
by using an AttentionWeights dataclass that abstracts away model-specific naming
conventions for attention parameters.

Usage:
    1. Extract attention weights using extract_attention_weights()
    2. Pass the AttentionWeights to the appropriate init function
    
Example:
    attn_weights = extract_attention_weights(attn_layer, style="standard")
    mamba2_init(attn_weights, mamba2_layer, mamba2_config, layer_idx)
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .hybridize_logger import get_logger

logger = get_logger(__name__)

# Norm layer names (for target layers)
NORM_Q = "q_norm"
NORM_K = "k_norm"
NORM_B = "B_norm"
NORM_C = "C_norm"


@dataclass
class AttentionWeights:
    """
    A model-agnostic container for attention layer weights.
    
    This dataclass abstracts away the specific naming conventions used by different
    transformer models (e.g., q_proj vs query_proj, separate vs fused QKV).
    
    Attributes:
        q_weight: Query projection weight tensor [num_heads * head_dim, hidden_size]
        k_weight: Key projection weight tensor [num_kv_heads * head_dim, hidden_size]
        v_weight: Value projection weight tensor [num_kv_heads * head_dim, hidden_size]
        o_weight: Output projection weight tensor [hidden_size, num_heads * head_dim]
        q_bias: Optional query projection bias
        k_bias: Optional key projection bias
        v_bias: Optional value projection bias
        o_bias: Optional output projection bias
        q_norm: Optional query normalization layer (for models with QK-norm like Qwen3)
        k_norm: Optional key normalization layer
    """
    q_weight: torch.Tensor
    k_weight: torch.Tensor
    v_weight: torch.Tensor
    o_weight: torch.Tensor
    q_bias: Optional[torch.Tensor] = None
    k_bias: Optional[torch.Tensor] = None
    v_bias: Optional[torch.Tensor] = None
    o_bias: Optional[torch.Tensor] = None
    q_norm: Optional[nn.Module] = None
    k_norm: Optional[nn.Module] = None


def _to_float32_if_fp8(tensor: torch.Tensor) -> torch.Tensor:
    """
    Convert FP8 tensors to float32 for arithmetic operations.
    
    FP8 dtypes (Float8_e4m3fn, Float8_e5m2) don't support arithmetic operations
    like addition directly. This helper converts them to float32 when needed.
    
    Args:
        tensor: Input tensor (may be FP8 or any other dtype)
        
    Returns:
        Tensor converted to float32 if it was FP8, otherwise unchanged
    """
    if tensor.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        return tensor.to(torch.float32)
    return tensor


def extract_attention_weights(
    attn_layer: nn.Module,
    qkv_style: str = "unfused",
    num_heads: Optional[int] = None,
    num_kv_heads: Optional[int] = None,
    head_dim: Optional[int] = None,
) -> AttentionWeights:
    """
    Extract attention weights from an attention layer into a model-agnostic format.
    
    This function extracts the four core projection weights (Q, K, V, O) and optional
    QK normalization layers from an attention layer. The returned AttentionWeights
    can then be passed to any of the hybrid layer initialization functions.
    
    Args:
        attn_layer: The source attention layer
        qkv_style: How Q, K, V projections are organized:
            - "unfused" (default): Separate q_proj, k_proj, v_proj, o_proj
              Used by: Llama, Qwen, Mistral, etc.
            - "fused": Single qkv_proj containing [Q, K, V] stacked, plus separate o_proj
              Used by: Some GPT variants, Falcon, etc.
              Requires: num_heads, head_dim (and num_kv_heads if using GQA)
        num_heads: Number of query attention heads (required for "fused" style)
        num_kv_heads: Number of key/value heads. If None, assumes num_kv_heads == num_heads (MHA).
                      Set this for GQA models where num_kv_heads < num_heads.
        head_dim: Dimension per attention head (required for "fused" style)
        
    Returns:
        AttentionWeights containing:
            - q_weight, k_weight, v_weight, o_weight: Projection weight tensors
            - q_bias, k_bias, v_bias, o_bias: Optional bias tensors (if present)
            - q_norm, k_norm: Optional normalization layers (if present, e.g., Qwen3)
            
    Raises:
        ValueError: If qkv_style is unknown or required parameters are missing
        
    Examples:
        # Unfused (e.g., Qwen2/Qwen3/Llama)
        attn_weights = extract_attention_weights(attn_layer)
        
        # Fused QKV with MHA
        attn_weights = extract_attention_weights(
            attn_layer, qkv_style="fused", num_heads=32, head_dim=128
        )
        
        # Fused QKV with GQA (fewer KV heads than Q heads)
        attn_weights = extract_attention_weights(
            attn_layer, qkv_style="fused", num_heads=32, num_kv_heads=8, head_dim=128
        )
    """
    if qkv_style == "unfused":
        return _extract_unfused(attn_layer)
    elif qkv_style == "fused":
        if num_heads is None or head_dim is None:
            raise ValueError("num_heads and head_dim are required for qkv_style='fused'")
        # Default to MHA if num_kv_heads not specified
        if num_kv_heads is None:
            num_kv_heads = num_heads
        return _extract_fused(attn_layer, num_heads, num_kv_heads, head_dim)
    else:
        raise ValueError(f"Unknown qkv_style: {qkv_style}. Valid options: 'unfused', 'fused'")


def _extract_unfused(attn_layer: nn.Module) -> AttentionWeights:
    """Extract weights from unfused q_proj/k_proj/v_proj/o_proj layout."""
    return AttentionWeights(
        q_weight=attn_layer.q_proj.weight.data,
        k_weight=attn_layer.k_proj.weight.data,
        v_weight=attn_layer.v_proj.weight.data,
        o_weight=attn_layer.o_proj.weight.data,
        q_bias=attn_layer.q_proj.bias.data if hasattr(attn_layer.q_proj, "bias") and attn_layer.q_proj.bias is not None else None,
        k_bias=attn_layer.k_proj.bias.data if hasattr(attn_layer.k_proj, "bias") and attn_layer.k_proj.bias is not None else None,
        v_bias=attn_layer.v_proj.bias.data if hasattr(attn_layer.v_proj, "bias") and attn_layer.v_proj.bias is not None else None,
        o_bias=attn_layer.o_proj.bias.data if hasattr(attn_layer.o_proj, "bias") and attn_layer.o_proj.bias is not None else None,
        q_norm=getattr(attn_layer, "q_norm", None),
        k_norm=getattr(attn_layer, "k_norm", None),
    )


def _extract_fused(
    attn_layer: nn.Module,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> AttentionWeights:
    """
    Extract weights from fused QKV projection.
    
    Assumes qkv_proj weight layout: [Q, K, V] stacked along dim 0
        - Q: [num_heads * head_dim, hidden_size]
        - K: [num_kv_heads * head_dim, hidden_size]
        - V: [num_kv_heads * head_dim, hidden_size]
    """
    qkv_weight = attn_layer.qkv_proj.weight.data
    qkv_bias = attn_layer.qkv_proj.bias.data if hasattr(attn_layer.qkv_proj, "bias") and attn_layer.qkv_proj.bias is not None else None
    
    q_dim = num_heads * head_dim
    kv_dim = num_kv_heads * head_dim
    
    q_weight = qkv_weight[:q_dim, :]
    k_weight = qkv_weight[q_dim:q_dim + kv_dim, :]
    v_weight = qkv_weight[q_dim + kv_dim:, :]
    
    q_bias = k_bias = v_bias = None
    if qkv_bias is not None:
        q_bias = qkv_bias[:q_dim]
        k_bias = qkv_bias[q_dim:q_dim + kv_dim]
        v_bias = qkv_bias[q_dim + kv_dim:]
    
    return AttentionWeights(
        q_weight=q_weight,
        k_weight=k_weight,
        v_weight=v_weight,
        o_weight=attn_layer.o_proj.weight.data,
        q_bias=q_bias,
        k_bias=k_bias,
        v_bias=v_bias,
        o_bias=attn_layer.o_proj.bias.data if hasattr(attn_layer.o_proj, "bias") and attn_layer.o_proj.bias is not None else None,
        q_norm=getattr(attn_layer, "q_norm", None),
        k_norm=getattr(attn_layer, "k_norm", None),
    )



# =============================================================================
# Utility Functions
# =============================================================================

def copy_weight_to_layer(
    source_weight: torch.Tensor,
    target_layer: nn.Linear,
    source_bias: Optional[torch.Tensor] = None,
    layer_description: str = ""
) -> Tuple[float, float]:
    """
    Copy weight tensor to a target linear layer.
    
    Handles bias mismatches gracefully:
    - If source has bias but target doesn't: skip bias copying
    - If target has bias but source doesn't: keep random initialization
    
    Args:
        source_weight: Weight tensor to copy
        target_layer: Target linear layer
        source_bias: Optional bias tensor to copy
        layer_description: Description for logging purposes
        
    Returns:
        Tuple of (weight_sum_before, weight_sum_after)
    """
    weight_sum_before = target_layer.weight.sum().item()
    
    target_layer.weight.data.copy_(source_weight)
    
    target_has_bias = hasattr(target_layer, "bias") and target_layer.bias is not None
    
    if source_bias is not None and target_has_bias:
        target_layer.bias.data.copy_(source_bias)
    elif source_bias is not None and not target_has_bias:
        if layer_description:
            logger.debug(f"{layer_description}: Source has bias but target doesn't - skipping bias copy")
    elif source_bias is None and target_has_bias:
        if layer_description:
            logger.debug(f"{layer_description}: Target has bias but source doesn't - keeping random initialization")
    
    weight_sum_after = target_layer.weight.sum().item()
    
    if layer_description:
        logger.debug(f"{layer_description}: {weight_sum_before:.6f} -> {weight_sum_after:.6f}")
    
    return weight_sum_before, weight_sum_after


def copy_layer_weights(
    source_layer: nn.Module,
    target_layer: nn.Module,
    layer_description: str = ""
) -> Tuple[float, float]:
    """
    Copy weights from source layer to target layer with logging.
    
    Handles bias mismatches gracefully:
    - If source has bias but target doesn't: skip bias copying
    - If target has bias but source doesn't: keep random initialization
    
    Args:
        source_layer: Layer to copy weights from
        target_layer: Layer to copy weights to
        layer_description: Description for logging purposes
        
    Returns:
        Tuple of (weight_sum_before, weight_sum_after)
    """
    weight_sum_before = target_layer.weight.sum().item()

    source_has_bias = hasattr(source_layer, "bias") and source_layer.bias is not None
    target_has_bias = hasattr(target_layer, "bias") and target_layer.bias is not None

    if source_has_bias != target_has_bias:
        target_layer.weight.data.copy_(source_layer.weight.data)

        if source_has_bias and not target_has_bias:
            logger.info(f"{layer_description}: Source has bias but target doesn't - skipping bias copy")
        elif target_has_bias and not source_has_bias:
            logger.info(f"{layer_description}: Target has bias but source doesn't - keeping random initialization for bias")
    else:
        target_layer.load_state_dict(source_layer.state_dict())

    weight_sum_after = target_layer.weight.sum().item()

    if layer_description:
        logger.debug(f"{layer_description}: {weight_sum_before:.6f} -> {weight_sum_after:.6f}")

    return weight_sum_before, weight_sum_after


def initialize_norm_layers(
    attn_weights: AttentionWeights,
    target_layer: nn.Module,
    layer_idx: int,
    norm_pairs: List[Tuple[str, str, str]],
    context: str = "",
) -> None:
    """
    Initialize normalization layers from attention weights to target layer.
    
    Args:
        attn_weights: Source attention weights containing norm layers
        target_layer: Target layer with norm layers to initialize
        layer_idx: Layer index for logging
        norm_pairs: List of (source_attr, target_name, display_name) tuples
            - source_attr: "q_norm" or "k_norm" (attribute on AttentionWeights)
            - target_name: Name of the target norm layer attribute
            - display_name: Name for logging
        context: Context description for logging
    """
    log_parts = []
    for source_attr, target_name, display_name in norm_pairs:
        source_norm = getattr(attn_weights, source_attr, None)
        if source_norm is None:
            continue
            
        target_norm = getattr(target_layer, target_name, None)
        if target_norm is None:
            continue

        before_sum = target_norm.weight.sum().item()
        target_norm.load_state_dict(source_norm.state_dict())
        after_sum = target_norm.weight.sum().item()

        log_parts.append(f"{display_name}: {before_sum:.6f} -> {after_sum:.6f}")

    if log_parts:
        logger.norms_update(
            layer_idx, f" {context} norm initialization: {' | '.join(log_parts)}"
        )



# =============================================================================
# Layer Initialization Functions
# =============================================================================

def mamba2_init(
    attn_weights: AttentionWeights,
    mamba2_layer: nn.Module,
    layer_idx: int,
) -> None:
    """
    Initialize Mamba2 layer from attention weights.
    
    Extends Mamba in the Llama (https://github.com/jxiw/MambaInLlama) with custom z initialization.
    
    Copies weights from attention Q/K/V projections into the combined in_proj weight,
    and initializes out_proj from o_proj. Also initializes QK normalization layers
    if present in the source model.
    
    Args:
        attn_weights: Source attention weights
        mamba2_layer: Target Mamba2 layer (must have d_inner and d_xb attributes)
        layer_idx: Layer index for logging
    """
    # Initialize out_proj from o_proj
    mamba2_layer.out_proj.weight.data.copy_(attn_weights.o_weight)
    if attn_weights.o_bias is not None and hasattr(mamba2_layer.out_proj, "bias") and mamba2_layer.out_proj.bias is not None:
        mamba2_layer.out_proj.bias.data.copy_(attn_weights.o_bias)

    in_proj_sum_before = mamba2_layer.in_proj.weight.sum().item()

    # Get dimensions from the layer itself
    d_inner = mamba2_layer.d_inner
    d_xb = mamba2_layer.d_xb

    # Copy V, K, Q projections into in_proj
    # in_proj layout: [z, v, k, q] where z=d_inner, v=d_xb, k=d_xb, q=d_inner
    # maps to [z, x, B, C] in mamba
    z_start, z_end = 0, d_inner
    v_start, v_end = d_inner, d_inner + d_xb
    k_start, k_end = d_inner + d_xb, d_inner + 2 * d_xb
    q_start, q_end = d_inner + 2 * d_xb, 2 * d_inner + 2 * d_xb

    mamba2_layer.in_proj.weight.data[v_start:v_end, :].copy_(attn_weights.v_weight)
    mamba2_layer.in_proj.weight.data[k_start:k_end, :].copy_(attn_weights.k_weight)
    mamba2_layer.in_proj.weight.data[q_start:q_end, :].copy_(attn_weights.q_weight)

    # Initialize z as 0.5(o^T + v_rep)
    groups = d_inner // d_xb  # equivalent to num_heads / num_kv_heads
    attn_v_proj_weight_rep = attn_weights.v_weight.repeat_interleave(groups, dim=0)
    attn_o_proj_weight_t = attn_weights.o_weight.transpose(0, 1)
    mamba2_layer.in_proj.weight.data[z_start:z_end].copy_(
        0.5 * (_to_float32_if_fp8(attn_o_proj_weight_t) + _to_float32_if_fp8(attn_v_proj_weight_rep))
    )

    in_proj_sum_after = mamba2_layer.in_proj.weight.sum().item()

    logger.mamba2_init(
        layer_idx,
        f"Mamba2 initialization complete. "
        f"in_proj sum: {in_proj_sum_before:.6f} -> {in_proj_sum_after:.6f}",
    )

    # Initialize QK normalization layers (B_norm from k_norm, C_norm from q_norm)
    if attn_weights.k_norm is not None and mamba2_layer.B_norm is not None:
        initialize_norm_layers(
            attn_weights,
            mamba2_layer,
            layer_idx,
            [("k_norm", NORM_B, "B"), ("q_norm", NORM_C, "C")],
            context="Mamba2 QK",
        )


def bmojo_attention_init(
    attn_weights: AttentionWeights,
    bmojo_f_attn_layer: nn.Module,
    layer_idx: int,
) -> None:
    """
    Initialize B'MOJO attention layer from attention weights.
    
    Copies Q/K/V/O projection weights and normalization layers. Also handles
    the untied projections (q_proj_, k_proj_, v_proj_) if they exist.
    
    Args:
        attn_weights: Source attention weights
        bmojo_f_attn_layer: Target B'MOJO attention layer
        layer_idx: Layer index for logging
    """

    # If the B'MOJO layer has untied weights, iterate through both tied and untied weights
    has_untied = hasattr(bmojo_f_attn_layer, "q_proj_")
    suffixes = ["", "_"] if has_untied else [""]
    
    for suffix in suffixes:
        weight_sums = {}
        
        # Q/K/V projections (both primary and untied)
        proj_mapping = [
            ("q_weight", "q_bias", f"q_proj{suffix}"),
            ("k_weight", "k_bias", f"k_proj{suffix}"),
            ("v_weight", "v_bias", f"v_proj{suffix}"),
        ]
        # O projection only for primary (no untied o_proj_)
        if not suffix:
            proj_mapping.append(("o_weight", "o_bias", "o_proj"))
        
        for weight_attr, bias_attr, proj_name in proj_mapping:
            source_weight = getattr(attn_weights, weight_attr)
            source_bias = getattr(attn_weights, bias_attr)
            target_proj = getattr(bmojo_f_attn_layer, proj_name)
            before, after = copy_weight_to_layer(source_weight, target_proj, source_bias)
            weight_sums[proj_name] = (before, after)

        log_msg = " | ".join(
            [f"{name.upper()}: {before:.6f} -> {after:.6f}" for name, (before, after) in weight_sums.items()]
        )
        label = "B'MOJO Attention untied projections" if suffix else "B'MOJO Attention initialized"
        logger.bmojo_f_init(layer_idx, f"{label}: {log_msg}")

        # Initialize normalization layers if present
        if attn_weights.q_norm is not None:
            initialize_norm_layers(
                attn_weights,
                bmojo_f_attn_layer,
                layer_idx,
                [("q_norm", f"{NORM_Q}{suffix}", f"q_norm{suffix}"), 
                 ("k_norm", f"{NORM_K}{suffix}", f"k_norm{suffix}")],
                context=f"B'MOJO Attention{' Untied' if suffix else ''} QK",
            )



def gated_deltanet_proj_init(
    attn_weights: AttentionWeights,
    gated_deltanet_layer: nn.Module,
    layer_idx: int,
) -> None:
    """
    Initialize Gated DeltaNet projection layers.
    
    Copies Q/K/V/O projections directly and initializes g_proj as 0.5(o^T + v_rep).
    
    Args:
        attn_weights: Source attention weights
        gated_deltanet_layer: Target Gated DeltaNet layer
        layer_idx: Layer index for logging
    """
    weight_sums = {}

    # Initialize Q, K, V, O projections
    proj_mapping = [
        ("q_weight", "q_bias", "q_proj"),
        ("k_weight", "k_bias", "k_proj"),
        ("v_weight", "v_bias", "v_proj"),
        ("o_weight", "o_bias", "o_proj"),
    ]
    
    for weight_attr, bias_attr, proj_name in proj_mapping:
        source_weight = getattr(attn_weights, weight_attr)
        source_bias = getattr(attn_weights, bias_attr)
        target_proj = getattr(gated_deltanet_layer, proj_name)
        before, after = copy_weight_to_layer(source_weight, target_proj, source_bias)
        weight_sums[proj_name] = (before, after)

    # Initialize g_proj if gating is enabled
    if gated_deltanet_layer.use_gate:
        g_sum_before = gated_deltanet_layer.g_proj.weight.sum().item()

        # Initialize g as 0.5(o^T + v_rep)
        groups = gated_deltanet_layer.g_proj.weight.shape[0] // attn_weights.v_weight.shape[0]
        attn_v_proj_weight_rep = attn_weights.v_weight.repeat_interleave(groups, dim=0)
        attn_o_proj_weight_t = attn_weights.o_weight.transpose(0, 1)
        gated_deltanet_layer.g_proj.weight.data.copy_(
            0.5 * (_to_float32_if_fp8(attn_o_proj_weight_t) + _to_float32_if_fp8(attn_v_proj_weight_rep))
        )

        g_sum_after = gated_deltanet_layer.g_proj.weight.sum().item()
        weight_sums["g_proj"] = (g_sum_before, g_sum_after)

    log_msg = " | ".join(
        [f"{name.upper()}: {before:.6f} -> {after:.6f}" for name, (before, after) in weight_sums.items()]
    )
    logger.gated_deltanet_init(layer_idx, f"Gated DeltaNet initialized: {log_msg}")



def gka_proj_init(
    attn_weights: AttentionWeights,
    gka_layer: nn.Module,
    layer_idx: int,
) -> None:
    """
    Initialize GKA (Gated KalmaNet) projection layers.
    
    Copies Q/K/V/O projections directly and initializes g_proj (if gating is enabled)
    using the same strategy as GDN: g = 0.5(o^T + v_rep).
    
    Args:
        attn_weights: Source attention weights
        gka_layer: Target GKA layer
        layer_idx: Layer index for logging
    """
    weight_sums = {}

    # Initialize Q, K, V, O projections
    proj_mapping = [
        ("q_weight", "q_bias", "q_proj"),
        ("k_weight", "k_bias", "k_proj"),
        ("v_weight", "v_bias", "v_proj"),
        ("o_weight", "o_bias", "o_proj"),
    ]
    
    for weight_attr, bias_attr, proj_name in proj_mapping:
        source_weight = getattr(attn_weights, weight_attr)
        source_bias = getattr(attn_weights, bias_attr)
        target_proj = getattr(gka_layer, proj_name)
        before, after = copy_weight_to_layer(source_weight, target_proj, source_bias)
        weight_sums[proj_name] = (before, after)

    # Initialize g_proj if gating is enabled
    if gka_layer.use_gate:
        g_sum_before = gka_layer.g_proj.weight.sum().item()

        # Initialize g as 0.5(o^T + v_rep) (same as GDN)
        groups = gka_layer.g_proj.weight.shape[0] // attn_weights.v_weight.shape[0]
        attn_v_proj_weight_rep = attn_weights.v_weight.repeat_interleave(groups, dim=0)
        attn_o_proj_weight_t = attn_weights.o_weight.transpose(0, 1)
        gka_layer.g_proj.weight.data.copy_(
            0.5 * (_to_float32_if_fp8(attn_o_proj_weight_t) + _to_float32_if_fp8(attn_v_proj_weight_rep))
        )

        g_sum_after = gka_layer.g_proj.weight.sum().item()
        weight_sums["g_proj"] = (g_sum_before, g_sum_after)

    log_msg = " | ".join(
        [f"{name.upper()}: {before:.6f} -> {after:.6f}" for name, (before, after) in weight_sums.items()]
    )
    logger.gka_init(layer_idx, f"GKA initialized: {log_msg}")


def swa_proj_init(
    attn_weights: AttentionWeights,
    swa_layer: nn.Module,
    layer_idx: int,
) -> None:
    """
    Initialize SWA (Sliding Window Attention) projection layers.
    
    Since Attention and SWA share the same parameters, we just need to copy over
    Q/K/V/O directly, and QK normalization layers (if present).
    
    Args:
        attn_weights: Source attention weights
        swa_layer: Target SWA layer
        layer_idx: Layer index for logging
    """
    weight_sums = {}

    # Initialize Q, K, V, O projections
    proj_mapping = [
        ("q_weight", "q_bias", "q_proj"),
        ("k_weight", "k_bias", "k_proj"),
        ("v_weight", "v_bias", "v_proj"),
        ("o_weight", "o_bias", "o_proj"),
    ]
    
    for weight_attr, bias_attr, proj_name in proj_mapping:
        source_weight = getattr(attn_weights, weight_attr)
        source_bias = getattr(attn_weights, bias_attr)
        target_proj = getattr(swa_layer, proj_name)
        before, after = copy_weight_to_layer(source_weight, target_proj, source_bias)
        weight_sums[proj_name] = (before, after)

    # Initialize normalization layers if present
    if attn_weights.q_norm is not None:
        initialize_norm_layers(
            attn_weights,
            swa_layer,
            layer_idx,
            [("q_norm", NORM_Q, "q_norm"), ("k_norm", NORM_K, "k_norm")],
            context="SWA QK",
        )

    log_msg = " | ".join(
        [f"{name.upper()}: {before:.6f} -> {after:.6f}" for name, (before, after) in weight_sums.items()]
    )
    logger.swa_init(layer_idx, f"SWA initialized: {log_msg}")



# =============================================================================
# High-Level Initialization Functions
# =============================================================================

def init_hybrid_layer(
    attn_weights: AttentionWeights,
    hybrid_layer: nn.Module,
    layer_type: str,
    layer_idx: int,
) -> None:
    """
    Initialize a hybrid layer based on its type.
    
    This is a convenience function that dispatches to the appropriate initialization
    function based on the layer type string.
    
    Args:
        attn_weights: Source attention weights
        hybrid_layer: Target hybrid decoder layer
        layer_type: Type of hybrid layer (M2, BMF, GDN, GKA, SWA)
        layer_idx: Layer index for logging
    """
    if "M2" in layer_type and "BMF" not in layer_type:
        # Pure Mamba2 layer
        mamba2_init(
            attn_weights,
            hybrid_layer.mamba,
            layer_idx=layer_idx,
        )
    
    elif "BMF" in layer_type:
        # B'MOJO layer - has both SSM mixer and attention components
        bmojo_f = hybrid_layer.bmojo_f
        
        # Initialize the B'MOJO attention component
        bmojo_attention_init(
            attn_weights,
            bmojo_f.bmojo_attn,
            layer_idx,
        )
        
        # Initialize the SSM mixer component based on what's configured
        if hasattr(bmojo_f, "mamba"):
            mamba2_init(
                attn_weights,
                bmojo_f.mamba,
                layer_idx=layer_idx,
            )
        elif hasattr(bmojo_f, "gated_deltanet"):
            gated_deltanet_proj_init(attn_weights, bmojo_f.gated_deltanet, layer_idx)
        elif hasattr(bmojo_f, "gka"):
            gka_proj_init(attn_weights, bmojo_f.gka, layer_idx)
    
    elif "GDN" in layer_type:
        gated_deltanet_proj_init(attn_weights, hybrid_layer.gated_deltanet, layer_idx)
    
    elif "GKA" in layer_type:
        gka_proj_init(attn_weights, hybrid_layer.gka, layer_idx)
    
    elif "SWA" in layer_type:
        swa_proj_init(
            attn_weights,
            hybrid_layer.swa,
            layer_idx,
        )
