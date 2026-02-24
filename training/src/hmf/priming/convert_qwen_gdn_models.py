"""
Shared Qwen GDN Model to HMF Conversion Module

This module provides model-agnostic functionality for converting HuggingFace Qwen
models that use the Gated DeltaNet (GDN) + gated attention architecture to Hybrid
Model Factory (HMF) model variants. Currently supports Qwen3-Next and Qwen3.5-MoE.

Each source model has its own weight layout in the GDN layers (e.g., fused interleaved
projections in Qwen3-Next vs separate projections in Qwen3.5-MoE), so weight extraction
is handled by model-specific functions. This module provides the shared logic that
operates on the extracted weights.

Key Conversions:
    - Attention → Attention: Direct copy (both have fused gate in q_proj)
    - Attention → SSM: De-interleave fused gate from q_proj, then init SSM
    - GDN → GDN: Copy pre-extracted weights into HMF GDN structure
    - GDN → GKA: Copy pre-extracted weights into HMF GKA structure

HMF GDN Structure (target):
    - q_proj, k_proj, v_proj: Separate projections
    - b_proj, gk_proj: Separate beta and alpha projections
    - q_conv1d, k_conv1d, v_conv1d: Separate convolutions
    - A_log, dt_bias: Decay parameters
    - g_proj: Output gate projection
    - o_norm: Output normalization
    - o_proj: Output projection

Note on "Inverse GQA":
    These Qwen models use bespoke pattern where V has more heads than Q/K, e.g.,
    - num_k_heads (Q/K heads): 16
    - num_v_heads (V heads): 32
    This is the opposite of standard GQA where KV heads < Q heads.
"""

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import torch
import torch.nn as nn

from .hybridize_logger import get_logger
from .hybrid_layer_init import AttentionWeights, copy_layer_weights

from hmf.model.hybrid_zoo.layers.gated_deltanet.gdn_config import construct_gdn_config
from hmf.model.hybrid_zoo.layers.gated_kalmanet.gka_config import construct_gka_config
from hmf.model.hybrid_zoo.layers.hybrid_dataclasses import (
    HybridGatedDeltaNetConfig,
    HybridGKAConfig,
)
from hmf.model.hybrid_zoo.models.utils import parse_override_pattern

logger = get_logger(__name__)

# Dataclasses for Extracted Weights


@dataclass
class GDNWeights:
    """
    Container for de-interleaved weights from the Qwen model's GDN layer.
    
    Attributes:
        q_weight: Query projection weight [key_dim, hidden_size]
        k_weight: Key projection weight [key_dim, hidden_size]
        v_weight: Value projection weight [value_dim, hidden_size]
        z_weight: Gate (z) projection weight [value_dim, hidden_size]
        b_weight: Beta projection weight [num_v_heads, hidden_size]
        a_weight: Alpha projection weight [num_v_heads, hidden_size]
        q_conv_weight: Query conv weight [key_dim, 1, kernel_size]
        k_conv_weight: Key conv weight [key_dim, 1, kernel_size]
        v_conv_weight: Value conv weight [value_dim, 1, kernel_size]
        A_log: Log of decay parameter A [num_v_heads]
        dt_bias: Time step bias [num_v_heads]
        norm_weight: Output norm weight [head_v_dim]
        out_proj_weight: Output projection weight [hidden_size, value_dim]
    """

    q_weight: torch.Tensor
    k_weight: torch.Tensor
    v_weight: torch.Tensor
    z_weight: torch.Tensor
    b_weight: torch.Tensor
    a_weight: torch.Tensor
    q_conv_weight: torch.Tensor
    k_conv_weight: torch.Tensor
    v_conv_weight: torch.Tensor
    A_log: torch.Tensor
    dt_bias: torch.Tensor
    norm_weight: torch.Tensor
    out_proj_weight: torch.Tensor


@dataclass
class GatedAttentionWeights:
    """
    Container for weights from Qwen model's Attention layer.
    
    Note: q_proj in Qwen3-Next and Qwen3.5 outputs [Q, gate] interleaved, so it's 2x the normal size.
    
    Attributes:
        q_weight: Query+gate projection weight [num_heads * head_dim * 2, hidden_size]
        k_weight: Key projection weight [num_kv_heads * head_dim, hidden_size]
        v_weight: Value projection weight [num_kv_heads * head_dim, hidden_size]
        o_weight: Output projection weight [hidden_size, num_heads * head_dim]
        q_norm_weight: Query norm weight [head_dim]
        k_norm_weight: Key norm weight [head_dim]
    """

    q_weight: torch.Tensor
    k_weight: torch.Tensor
    v_weight: torch.Tensor
    o_weight: torch.Tensor
    q_norm_weight: torch.Tensor
    k_norm_weight: torch.Tensor


def attn_to_attn_init(
    source_attn: nn.Module, target_attn: nn.Module, layer_idx: int
) -> None:
    """
    Initialize the HMF model from the original Qwen model's Attention.
    
    Both have the same structure (fused gate in q_proj), so this is a direct copy.
    
    Args:
        source_attn: HF Qwen layer
        target_attn: HMF Qwen layer
        layer_idx: Layer index for logging
    """
    weight_sums = {}

    # Copy projections
    for proj_name in ["q_proj", "k_proj", "v_proj", "o_proj"]:
        source_proj = getattr(source_attn, proj_name)
        target_proj = getattr(target_attn, proj_name)
        before, after = copy_layer_weights(
            source_proj, target_proj, f"Layer {layer_idx} {proj_name}"
        )
        weight_sums[proj_name] = (before, after)

    # Copy norms
    for norm_name in ["q_norm", "k_norm"]:
        source_norm = getattr(source_attn, norm_name)
        target_norm = getattr(target_attn, norm_name)
        target_norm.load_state_dict(source_norm.state_dict())

    log_msg = " | ".join(
        [
            f"{name.upper()}: {before:.6f} -> {after:.6f}"
            for name, (before, after) in weight_sums.items()
        ]
    )
    logger.info(f"Layer {layer_idx} Attention → Attention: {log_msg}")


def gdn_to_gdn_init(
    gdn_weights: GDNWeights, target_gdn: nn.Module, layer_idx: int
) -> None:
    """
    Initialize HMF GDN from pre-extracted GDN weights.

    Takes already de-interleaved/split weights (as a GDNWeights dataclass) and copies
    them into the corresponding HMF GatedDeltaNet layer parameters.

    Mapping:
        gdn_weights.q_weight        → target_gdn.q_proj
        gdn_weights.k_weight        → target_gdn.k_proj
        gdn_weights.v_weight        → target_gdn.v_proj
        gdn_weights.z_weight        → target_gdn.g_proj (output gate)
        gdn_weights.b_weight        → target_gdn.b_proj (beta)
        gdn_weights.a_weight        → target_gdn.gk_proj (alpha/decay)
        gdn_weights.q_conv_weight   → target_gdn.q_conv1d
        gdn_weights.k_conv_weight   → target_gdn.k_conv1d
        gdn_weights.v_conv_weight   → target_gdn.v_conv1d
        gdn_weights.A_log           → target_gdn.A_log
        gdn_weights.dt_bias         → target_gdn.dt_bias
        gdn_weights.norm_weight     → target_gdn.o_norm
        gdn_weights.out_proj_weight → target_gdn.o_proj

    Args:
        gdn_weights: Pre-extracted GDN weights (model-agnostic)
        target_gdn: HMF GatedDeltaNet layer to initialize
        layer_idx: Layer index for logging
    """
    weight_sums = {}

    # Copy Q, K, V projections
    before = target_gdn.q_proj.weight.sum().item()
    target_gdn.q_proj.weight.data.copy_(gdn_weights.q_weight)
    after = target_gdn.q_proj.weight.sum().item()
    weight_sums["q_proj"] = (before, after)

    before = target_gdn.k_proj.weight.sum().item()
    target_gdn.k_proj.weight.data.copy_(gdn_weights.k_weight)
    after = target_gdn.k_proj.weight.sum().item()
    weight_sums["k_proj"] = (before, after)

    before = target_gdn.v_proj.weight.sum().item()
    target_gdn.v_proj.weight.data.copy_(gdn_weights.v_weight)
    after = target_gdn.v_proj.weight.sum().item()
    weight_sums["v_proj"] = (before, after)

    # Copy g_proj from z (output gate)
    if target_gdn.use_gate:
        before = target_gdn.g_proj.weight.sum().item()
        target_gdn.g_proj.weight.data.copy_(gdn_weights.z_weight)
        after = target_gdn.g_proj.weight.sum().item()
        weight_sums["g_proj"] = (before, after)

    # Copy b_proj and gk_proj (a → gk)
    before = target_gdn.b_proj.weight.sum().item()
    target_gdn.b_proj.weight.data.copy_(gdn_weights.b_weight)
    after = target_gdn.b_proj.weight.sum().item()
    weight_sums["b_proj"] = (before, after)

    before = target_gdn.gk_proj.weight.sum().item()
    target_gdn.gk_proj.weight.data.copy_(gdn_weights.a_weight)
    after = target_gdn.gk_proj.weight.sum().item()
    weight_sums["gk_proj"] = (before, after)

    # Copy convolutions
    if target_gdn.use_short_conv:
        # HMF uses ShortConvolution which extends nn.Conv1d
        # The weight is accessed directly via .weight (shape: [D, 1, W])
        before = target_gdn.q_conv1d.weight.sum().item()
        target_gdn.q_conv1d.weight.data.copy_(gdn_weights.q_conv_weight)
        after = target_gdn.q_conv1d.weight.sum().item()
        weight_sums["q_conv"] = (before, after)

        before = target_gdn.k_conv1d.weight.sum().item()
        target_gdn.k_conv1d.weight.data.copy_(gdn_weights.k_conv_weight)
        after = target_gdn.k_conv1d.weight.sum().item()
        weight_sums["k_conv"] = (before, after)

        before = target_gdn.v_conv1d.weight.sum().item()
        target_gdn.v_conv1d.weight.data.copy_(gdn_weights.v_conv_weight)
        after = target_gdn.v_conv1d.weight.sum().item()
        weight_sums["v_conv"] = (before, after)

    # Copy A_log and dt_bias
    before = target_gdn.A_log.sum().item()
    target_gdn.A_log.data.copy_(gdn_weights.A_log)
    after = target_gdn.A_log.sum().item()
    weight_sums["A_log"] = (before, after)

    before = target_gdn.dt_bias.sum().item()
    target_gdn.dt_bias.data.copy_(gdn_weights.dt_bias)
    after = target_gdn.dt_bias.sum().item()
    weight_sums["dt_bias"] = (before, after)

    # Copy output norm
    before = target_gdn.o_norm.weight.sum().item()
    target_gdn.o_norm.weight.data.copy_(gdn_weights.norm_weight)
    after = target_gdn.o_norm.weight.sum().item()
    weight_sums["o_norm"] = (before, after)

    # Copy output projection
    before = target_gdn.o_proj.weight.sum().item()
    target_gdn.o_proj.weight.data.copy_(gdn_weights.out_proj_weight)
    after = target_gdn.o_proj.weight.sum().item()
    weight_sums["o_proj"] = (before, after)

    log_msg = " | ".join(
        [
            f"{name}: {before:.4f} -> {after:.4f}"
            for name, (before, after) in weight_sums.items()
        ]
    )
    logger.info(f"Layer {layer_idx} GDN → GDN: {log_msg}")


def gdn_to_gka_init(
    gdn_weights: GDNWeights, target_gka: nn.Module, layer_idx: int
) -> None:
    """
    Initialize HMF GKA from pre-extracted GDN weights.

    Takes already de-interleaved/split weights (as a GDNWeights dataclass) and copies
    them into the corresponding HMF GKA layer parameters.

    Mapping:
        gdn_weights.q_weight      → target_gka.q_proj
        gdn_weights.k_weight      → target_gka.k_proj
        gdn_weights.v_weight      → target_gka.v_proj
        gdn_weights.z_weight      → target_gka.g_proj (output gate)
        gdn_weights.b_weight      → target_gka.b_proj (beta gate)
        gdn_weights.a_weight      → target_gka.a_proj (decay/alpha)
        gdn_weights.A_log         → target_gka.A_log
        gdn_weights.dt_bias       → target_gka.dt_bias
        gdn_weights.q_conv_weight → target_gka.q_conv1d
        gdn_weights.k_conv_weight → target_gka.k_conv1d
        gdn_weights.v_conv_weight → target_gka.v_conv1d
        gdn_weights.norm_weight   → target_gka.o_norm
        gdn_weights.out_proj_weight → target_gka.o_proj
        (alpha_proj left randomly initialized — GKA-specific, no GDN equivalent)

    Args:
        gdn_weights: Pre-extracted GDN weights (model-agnostic)
        target_gka: HMF GKA layer to initialize
        layer_idx: Layer index for logging
    """
    weight_sums = {}

    # Copy Q, K, V projections
    before = target_gka.q_proj.weight.sum().item()
    target_gka.q_proj.weight.data.copy_(gdn_weights.q_weight)
    after = target_gka.q_proj.weight.sum().item()
    weight_sums["q_proj"] = (before, after)

    before = target_gka.k_proj.weight.sum().item()
    target_gka.k_proj.weight.data.copy_(gdn_weights.k_weight)
    after = target_gka.k_proj.weight.sum().item()
    weight_sums["k_proj"] = (before, after)

    before = target_gka.v_proj.weight.sum().item()
    target_gka.v_proj.weight.data.copy_(gdn_weights.v_weight)
    after = target_gka.v_proj.weight.sum().item()
    weight_sums["v_proj"] = (before, after)

    # Copy g_proj from z (output gate) if GKA uses gating
    if hasattr(target_gka, "g_proj") and target_gka.use_gate:
        before = target_gka.g_proj.weight.sum().item()
        target_gka.g_proj.weight.data.copy_(gdn_weights.z_weight)
        after = target_gka.g_proj.weight.sum().item()
        weight_sums["g_proj"] = (before, after)

    # Copy b_proj from beta (if GKA uses beta gate)
    if hasattr(target_gka, "b_proj") and target_gka.b_proj is not None:
        before = target_gka.b_proj.weight.sum().item()
        target_gka.b_proj.weight.data.copy_(gdn_weights.b_weight)
        after = target_gka.b_proj.weight.sum().item()
        weight_sums["b_proj"] = (before, after)

    # Copy a_proj from alpha (decay parameter)
    if hasattr(target_gka, "a_proj"):
        before = target_gka.a_proj.weight.sum().item()
        target_gka.a_proj.weight.data.copy_(gdn_weights.a_weight)
        after = target_gka.a_proj.weight.sum().item()
        weight_sums["a_proj"] = (before, after)

    # Copy A_log and dt_bias
    if hasattr(target_gka, "A_log"):
        before = target_gka.A_log.sum().item()
        target_gka.A_log.data.copy_(gdn_weights.A_log)
        after = target_gka.A_log.sum().item()
        weight_sums["A_log"] = (before, after)

    if hasattr(target_gka, "dt_bias"):
        before = target_gka.dt_bias.sum().item()
        target_gka.dt_bias.data.copy_(gdn_weights.dt_bias)
        after = target_gka.dt_bias.sum().item()
        weight_sums["dt_bias"] = (before, after)

    # Copy convolutions if present and dimensions match
    # ShortConvolution extends nn.Conv1d, so weight is accessed directly via .weight
    if hasattr(target_gka, "q_conv1d"):
        try:
            before = target_gka.q_conv1d.weight.sum().item()
            target_gka.q_conv1d.weight.data.copy_(gdn_weights.q_conv_weight)
            after = target_gka.q_conv1d.weight.sum().item()
            weight_sums["q_conv"] = (before, after)
        except RuntimeError as e:
            logger.warning(f"Layer {layer_idx}: Could not copy q_conv1d. Error: {e}")

    if hasattr(target_gka, "k_conv1d"):
        try:
            before = target_gka.k_conv1d.weight.sum().item()
            target_gka.k_conv1d.weight.data.copy_(gdn_weights.k_conv_weight)
            after = target_gka.k_conv1d.weight.sum().item()
            weight_sums["k_conv"] = (before, after)
        except RuntimeError as e:
            logger.warning(f"Layer {layer_idx}: Could not copy k_conv1d. Error: {e}")

    if hasattr(target_gka, "v_conv1d") and target_gka.use_v_conv:
        try:
            before = target_gka.v_conv1d.weight.sum().item()
            target_gka.v_conv1d.weight.data.copy_(gdn_weights.v_conv_weight)
            after = target_gka.v_conv1d.weight.sum().item()
            weight_sums["v_conv"] = (before, after)
        except RuntimeError as e:
            logger.warning(f"Layer {layer_idx}: Could not copy v_conv1d. Error: {e}")

    # Copy output norm
    before = target_gka.o_norm.weight.sum().item()
    target_gka.o_norm.weight.data.copy_(gdn_weights.norm_weight)
    after = target_gka.o_norm.weight.sum().item()
    weight_sums["o_norm"] = (before, after)

    # Copy output projection
    before = target_gka.o_proj.weight.sum().item()
    target_gka.o_proj.weight.data.copy_(gdn_weights.out_proj_weight)
    after = target_gka.o_proj.weight.sum().item()
    weight_sums["o_proj"] = (before, after)

    # Note: alpha_proj is GKA-specific and has no equivalent in GDN
    # It will remain randomly initialized
    if hasattr(target_gka, "alpha_proj") and target_gka.use_alpha_connection:
        logger.info(
            f"Layer {layer_idx}: alpha_proj left randomly initialized (no GDN equivalent)"
        )

    log_msg = " | ".join(
        [
            f"{name}: {before:.4f} -> {after:.4f}"
            for name, (before, after) in weight_sums.items()
        ]
    )
    logger.info(f"Layer {layer_idx} GDN → GKA: {log_msg}")


def attn_to_ssm_init(
    attn_weights: GatedAttentionWeights,
    target_ssm: nn.Module,
    num_heads: int,
    head_dim: int,
    layer_idx: int,
    ssm_type: str = "gdn",
) -> None:
    """
    Initialize an SSM layer (GDN, GKA, Mamba2, etc.) from pre-extracted gated attention weights.

    Handles the fused gate in q_proj by de-interleaving to extract just the Q portion,
    then dispatches to the appropriate SSM initialization function.

    Args:
        attn_weights: Pre-extracted gated attention weights (model-agnostic)
        target_ssm: Target SSM layer (GDN, GKA, etc.) to initialize
        num_heads: Number of attention heads (needed to de-interleave the fused Q+gate)
        head_dim: Attention head dimension (needed to de-interleave the fused Q+gate)
        layer_idx: Layer index for logging
        ssm_type: Type of SSM ("gdn", "gka", "mamba2", "bmojo")
    """

    # The q_proj weight is [num_heads * head_dim * 2, hidden_size]
    # First half is Q, second half is gate (interleaved by head)
    # We need to de-interleave to get just Q
    q_weight_full = attn_weights.q_weight
    hidden_size = q_weight_full.shape[1]

    # Reshape to [num_heads, head_dim * 2, hidden_size] and take first half
    q_weight_reshaped = q_weight_full.view(num_heads, head_dim * 2, hidden_size)
    q_weight = q_weight_reshaped[:, :head_dim, :].reshape(
        num_heads * head_dim, hidden_size
    )

    # Create AttentionWeights for the init functions
    extracted_attn_weights = AttentionWeights(
        q_weight=q_weight,
        k_weight=attn_weights.k_weight,
        v_weight=attn_weights.v_weight,
        o_weight=attn_weights.o_weight,
    )

    # Dispatch to appropriate init function
    if ssm_type == "gdn":
        from .hybrid_layer_init import gated_deltanet_proj_init

        gated_deltanet_proj_init(extracted_attn_weights, target_ssm, layer_idx)
    elif ssm_type == "gka":
        from .hybrid_layer_init import gka_proj_init

        gka_proj_init(extracted_attn_weights, target_ssm, layer_idx)
    elif ssm_type == "mamba2":
        from .hybrid_layer_init import mamba2_init

        mamba2_init(extracted_attn_weights, target_ssm, layer_idx)
    elif ssm_type == "bmojo":
        from .hybrid_layer_init import bmojo_attention_init

        bmojo_attention_init(extracted_attn_weights, target_ssm.bmojo_attn, layer_idx)
        # Also init the SSM component if present
        if hasattr(target_ssm, "mamba"):
            from .hybrid_layer_init import mamba2_init

            mamba2_init(extracted_attn_weights, target_ssm.mamba, layer_idx)
        elif hasattr(target_ssm, "gated_deltanet"):
            from .hybrid_layer_init import gated_deltanet_proj_init

            gated_deltanet_proj_init(
                extracted_attn_weights, target_ssm.gated_deltanet, layer_idx
            )
    else:
        raise ValueError(f"Unknown SSM type: {ssm_type}")

    logger.info(
        f"Layer {layer_idx} Attention → {ssm_type.upper()}: initialized from attention weights"
    )


def copy_shared_weights(source_model: nn.Module, target_model: nn.Module) -> List[str]:
    """
    Copy shared weights between HF Qwen3-Next and HMF Qwen3-Next models.
    
    Copies:
        - Embeddings (embed_tokens)
        - Final layer norm (norm)
        - LM head (lm_head)
        - Per-layer: input_layernorm, post_attention_layernorm, MLP
    
    Args:
        source_model: HF Qwen3-Next model
        target_model: HMF Qwen3-Next model
        
    Returns:
        List of layer names that were NOT copied (candidates for hybrid init)
    """
    hybridization_candidates = []

    # Copy embeddings
    if hasattr(source_model, "model"):
        source_base = source_model.model
        target_base = target_model.model
    else:
        source_base = source_model
        target_base = target_model

    # Embed tokens
    copy_layer_weights(
        source_base.embed_tokens, target_base.embed_tokens, "embed_tokens"
    )

    # Final norm
    target_base.norm.load_state_dict(source_base.norm.state_dict())
    logger.info("Copied final layer norm")

    # Rotary embedding (if present)
    if hasattr(source_base, "rotary_emb") and hasattr(target_base, "rotary_emb"):
        target_base.rotary_emb.load_state_dict(source_base.rotary_emb.state_dict())
        logger.info("Copied rotary embeddings")

    # LM head (if present)
    if hasattr(source_model, "lm_head") and hasattr(target_model, "lm_head"):
        copy_layer_weights(source_model.lm_head, target_model.lm_head, "lm_head")

    # Per-layer weights
    num_layers = len(source_base.layers)
    for layer_idx in range(num_layers):
        source_layer = source_base.layers[layer_idx]
        target_layer = target_base.layers[layer_idx]

        # Layer norms
        target_layer.input_layernorm.load_state_dict(
            source_layer.input_layernorm.state_dict()
        )
        target_layer.post_attention_layernorm.load_state_dict(
            source_layer.post_attention_layernorm.state_dict()
        )

        # MLP (handles both dense and MoE)
        target_layer.mlp.load_state_dict(source_layer.mlp.state_dict())

        # Track which layers need hybrid initialization
        # Check source layer type
        source_layer_type = getattr(source_layer, "layer_type", None)
        if source_layer_type == "linear_attention":
            hybridization_candidates.append(f"model.layers.{layer_idx}.linear_attn")
        elif source_layer_type == "full_attention":
            hybridization_candidates.append(f"model.layers.{layer_idx}.self_attn")
        elif hasattr(source_layer, "self_attn"):
            hybridization_candidates.append(f"model.layers.{layer_idx}.self_attn")
        elif hasattr(source_layer, "linear_attn"):
            hybridization_candidates.append(f"model.layers.{layer_idx}.linear_attn")

    logger.info(f"Copied shared weights for {num_layers} layers")
    logger.info(f"Hybridization candidates: {len(hybridization_candidates)} layers")

    return hybridization_candidates


def _get_target_gdn(target_layer: nn.Module) -> nn.Module:
    """Get the GDN module from a target layer, handling various wrapper structures."""
    if hasattr(target_layer, "gdn"):
        gdn = target_layer.gdn
        # GatedDeltaNetLayer wraps the actual GatedDeltaNet
        if hasattr(gdn, "gated_deltanet"):
            return gdn.gated_deltanet
        return gdn
    if hasattr(target_layer, "gated_deltanet"):
        return target_layer.gated_deltanet
    raise AttributeError(f"Could not find GDN module in {type(target_layer)}")


def _get_target_gka(target_layer: nn.Module) -> nn.Module:
    """Get the GKA module from a target layer, handling various wrapper structures."""
    if hasattr(target_layer, "gka"):
        gka = target_layer.gka
        # GatedKalmaNetLayer wraps the actual GKA
        if hasattr(gka, "gka"):
            return gka.gka
        return gka
    raise AttributeError(f"Could not find GKA module in {type(target_layer)}")


# Conversion Mappings
# Valid target types when source is attention (full_attention)
VALID_TARGETS_FROM_ATTENTION = {
    "*",  # Keep as attention
    "GDN",  # Pure GDN
    "GKA",  # Pure GKA
    "M2",  # Pure Mamba2
    "BMF",  # Pure B'MOJO
    "GDN*",  # Fused GDN + Attention
    "GKA*",  # Fused GKA + Attention
    "M2*",  # Fused Mamba2 + Attention
    "BMF*",  # Fused B'MOJO + Attention
    "*DA",  # Dual Attention
    "GDN>GKA",  # Fused GDN + GKA (init both from attention)
}

# Valid target types when source is GDN (linear_attention)
VALID_TARGETS_FROM_GDN = {
    "GDN",  # GDN → GDN (our implementation)
    "GKA",  # GDN → GKA
    "GDN>GKA",  # Fused GDN + GKA (for distillation)
}


def validate_conversion_pattern(
    source_layer_types: List[str], target_layer_types: Dict[int, str]
) -> None:
    """
    Validate that all source→target layer conversions are supported.
    
    Rules:
        - Attention source → any target type
        - GDN source → only GDN, GKA, or GDN>GKA
    
    Args:
        source_layer_types: List of source layer types from Qwen3-Next config
            ("full_attention" or "linear_attention")
        target_layer_types: Dict mapping layer index to target type string
        
    Raises:
        ValueError: If any conversion is not supported
    """
    errors = []

    for layer_idx, target_type in target_layer_types.items():
        if layer_idx >= len(source_layer_types):
            errors.append(
                f"Layer {layer_idx}: Target pattern has more layers than source model"
            )
            continue

        source_type = source_layer_types[layer_idx]

        if source_type == "full_attention":
            if target_type not in VALID_TARGETS_FROM_ATTENTION:
                errors.append(
                    f"Layer {layer_idx}: Unknown target type '{target_type}'. "
                    f"Valid targets from attention: {sorted(VALID_TARGETS_FROM_ATTENTION)}"
                )
        elif source_type == "linear_attention":
            if target_type not in VALID_TARGETS_FROM_GDN:
                errors.append(
                    f"Layer {layer_idx}: Cannot convert GDN → {target_type}. "
                    f"GDN source can only be converted to: {sorted(VALID_TARGETS_FROM_GDN)}"
                )
        else:
            errors.append(f"Layer {layer_idx}: Unknown source type '{source_type}'")

    if errors:
        error_msg = "Invalid conversion pattern:\n  " + "\n  ".join(errors)
        raise ValueError(error_msg)


def init_qwen_hmf_layers(
    source_model: nn.Module,
    target_model: nn.Module,
    target_layer_types: Dict[int, str],
    extract_gdn_weights_fn: Callable[[nn.Module], GDNWeights],
    extract_attn_weights_fn: Callable[[nn.Module], GatedAttentionWeights],
) -> None:
    """
    Initialize target HMF model layers from source Qwen model layers.

    Dispatches to the appropriate init function based on source and target layer types.
    Uses the provided extraction functions to obtain model-agnostic weight containers
    before passing them to shared initialization logic.

    Supported conversions:
        From Attention (full_attention):
            - "*": Keep as attention (direct copy)
            - "GDN", "GKA", "M2", "BMF": Pure SSM layers
            - "GDN*", "GKA*", "M2*", "BMF*": Fused SSM + Attention
            - "*DA": Dual Attention
            - "GDN>GKA": Fused GDN + GKA

        From GDN (linear_attention):
            - "GDN": GDN → GDN (our implementation)
            - "GKA": GDN → GKA
            - "GDN>GKA": Fused GDN + GKA (for distillation)

    Args:
        source_model: Source HF Qwen model (e.g., Qwen3Next or Qwen3.5-MoE)
        target_model: Target HMF model to initialize
        target_layer_types: Dict mapping layer index to target type string
        extract_gdn_weights_fn: Model-specific function to extract GDN weights
            from a source GDN layer into a GDNWeights dataclass
        extract_attn_weights_fn: Model-specific function to extract attention weights
            from a source attention layer into a GatedAttentionWeights dataclass
    """

    source_base = source_model.model if hasattr(source_model, "model") else source_model
    target_base = target_model.model if hasattr(target_model, "model") else target_model

    num_layers = len(source_base.layers)

    for layer_idx in range(num_layers):
        source_layer = source_base.layers[layer_idx]
        target_layer = target_base.layers[layer_idx]
        target_type = target_layer_types.get(layer_idx, "*")

        # Determine source layer type
        source_layer_type = getattr(source_layer, "layer_type", None)
        is_source_gdn = source_layer_type == "linear_attention"
        is_source_attn = source_layer_type == "full_attention"

        # From Attention source
        if is_source_attn:
            source_attn = source_layer.self_attn
            attn_weights = extract_attn_weights_fn(source_attn)
            num_heads = source_attn.config.num_attention_heads
            head_dim = source_attn.head_dim

            if target_type == "*":
                # Attention → Attention (direct copy)
                target_attn = target_layer.self_attn
                attn_to_attn_init(source_attn, target_attn, layer_idx)

            elif target_type == "GDN":
                # Attention → GDN
                target_gdn = _get_target_gdn(target_layer)
                attn_to_ssm_init(
                    attn_weights,
                    target_gdn,
                    num_heads,
                    head_dim,
                    layer_idx,
                    ssm_type="gdn",
                )

            elif target_type == "GKA":
                # Attention → GKA
                target_gka = _get_target_gka(target_layer)
                attn_to_ssm_init(
                    attn_weights,
                    target_gka,
                    num_heads,
                    head_dim,
                    layer_idx,
                    ssm_type="gka",
                )

            elif target_type == "M2":
                # Attention → Mamba2
                target_mamba = target_layer.mamba
                attn_to_ssm_init(
                    attn_weights,
                    target_mamba,
                    num_heads,
                    head_dim,
                    layer_idx,
                    ssm_type="mamba2",
                )

            elif target_type == "BMF":
                # Attention → B'MOJO
                target_bmojo = target_layer.bmojo_f
                attn_to_ssm_init(
                    attn_weights,
                    target_bmojo,
                    num_heads,
                    head_dim,
                    layer_idx,
                    ssm_type="bmojo",
                )

            elif target_type == "GDN*":
                # Attention → Fused GDN + Attention
                target_attn = target_layer.self_attn
                attn_to_attn_init(source_attn, target_attn, layer_idx)
                target_gdn = _get_target_gdn(target_layer)
                attn_to_ssm_init(
                    attn_weights,
                    target_gdn,
                    num_heads,
                    head_dim,
                    layer_idx,
                    ssm_type="gdn",
                )

            elif target_type == "GKA*":
                # Attention → Fused GKA + Attention
                target_attn = target_layer.self_attn
                attn_to_attn_init(source_attn, target_attn, layer_idx)
                target_gka = _get_target_gka(target_layer)
                attn_to_ssm_init(
                    attn_weights,
                    target_gka,
                    num_heads,
                    head_dim,
                    layer_idx,
                    ssm_type="gka",
                )

            elif target_type == "M2*":
                # Attention → Fused Mamba2 + Attention
                target_attn = target_layer.self_attn
                attn_to_attn_init(source_attn, target_attn, layer_idx)
                target_mamba = (
                    target_layer.mamba
                    if hasattr(target_layer, "mamba")
                    else target_layer.hybrid_seq_mixer
                )
                attn_to_ssm_init(
                    attn_weights,
                    target_mamba,
                    num_heads,
                    head_dim,
                    layer_idx,
                    ssm_type="mamba2",
                )

            elif target_type == "BMF*":
                # Attention → Fused B'MOJO + Attention
                target_attn = target_layer.self_attn
                attn_to_attn_init(source_attn, target_attn, layer_idx)
                target_bmojo = (
                    target_layer.bmojo_f
                    if hasattr(target_layer, "bmojo_f")
                    else target_layer.hybrid_seq_mixer
                )
                attn_to_ssm_init(
                    attn_weights,
                    target_bmojo,
                    num_heads,
                    head_dim,
                    layer_idx,
                    ssm_type="bmojo",
                )

            elif target_type == "*DA":
                # Attention → Dual Attention
                target_attn = target_layer.self_attn
                attn_to_attn_init(source_attn, target_attn, layer_idx)

            elif target_type == "GDN>GKA":
                # Attention → Fused GDN + GKA (init both from attention)
                target_gdn = _get_target_gdn(target_layer)
                attn_to_ssm_init(
                    attn_weights,
                    target_gdn,
                    num_heads,
                    head_dim,
                    layer_idx,
                    ssm_type="gdn",
                )
                target_gka = _get_target_gka(target_layer)
                attn_to_ssm_init(
                    attn_weights,
                    target_gka,
                    num_heads,
                    head_dim,
                    layer_idx,
                    ssm_type="gka",
                )

        # From GDN source
        elif is_source_gdn:
            source_gdn = source_layer.linear_attn
            gdn_weights = extract_gdn_weights_fn(source_gdn)

            if target_type == "GDN":
                # GDN → GDN (our implementation)
                target_gdn = _get_target_gdn(target_layer)
                gdn_to_gdn_init(gdn_weights, target_gdn, layer_idx)

            elif target_type == "GKA":
                # GDN → GKA
                target_gka = _get_target_gka(target_layer)
                gdn_to_gka_init(gdn_weights, target_gka, layer_idx)

            elif target_type == "GDN>GKA":
                # GDN → Fused GDN + GKA (for distillation)
                target_gdn = _get_target_gdn(target_layer)
                gdn_to_gdn_init(gdn_weights, target_gdn, layer_idx)
                target_gka = _get_target_gka(target_layer)
                gdn_to_gka_init(gdn_weights, target_gka, layer_idx)


def convert_qwen_gdn_model(
    source_model: nn.Module,
    target_model_class: type,
    target_config,
    model_name: str,
    extract_gdn_weights_fn: Callable[[nn.Module], GDNWeights],
    extract_attn_weights_fn: Callable[[nn.Module], GatedAttentionWeights],
    target_layer_types: Optional[Dict[int, str]] = None,
) -> nn.Module:
    """
    Convert a Hugging Face Qwen GDN model to an HMF model variant.

    Handles the full conversion pipeline: creating the target model, copying shared
    weights (embeddings, norms, MLP), and initializing hybrid layers using the
    provided model-specific weight extraction functions.

    Works with any Qwen model that uses the GDN + gated attention architecture
    (e.g., Qwen3-Next, Qwen3.5-MoE).

    Args:
        source_model: Source HF model (e.g., Qwen3NextForCausalLM, Qwen3_5MoeForCausalLM)
        target_model_class: HMF model class to instantiate (e.g., Qwen3NextHMFForCausalLM)
        target_config: HMF model config (e.g., Qwen3NextHMFConfig, Qwen3_5MoeHMFTextConfig)
        model_name: Display name for logging (e.g., "Qwen3-Next", "Qwen3.5-MoE")
        extract_gdn_weights_fn: Model-specific function to extract GDN weights
            from a source GDN layer into a GDNWeights dataclass
        extract_attn_weights_fn: Model-specific function to extract attention weights
            from a source attention layer into a GatedAttentionWeights dataclass
        target_layer_types: Optional dict mapping layer index to target type.
            If None, uses the hybrid_override_pattern from config.

    Returns:
        Initialized HMF model with weights converted from the source model
    """

    logger.section(f"Converting {model_name} to HMF {model_name}")

    # Determine target layer types
    if target_layer_types is None:
        from hmf.model.hybrid_zoo.models.utils import parse_override_pattern

        target_layer_types = parse_override_pattern(
            target_config.hybrid_override_pattern
        )

    # Validate conversion pattern against source model's layer types
    source_config = source_model.config
    if hasattr(source_config, "layer_types"):
        validate_conversion_pattern(source_config.layer_types, target_layer_types)
        logger.info("Conversion pattern validated successfully")
    else:
        logger.warning("Source config has no layer_types; skipping validation")

    # Create target model with random initialization
    logger.info(f"Creating target model: {target_model_class.__name__}")
    target_model = target_model_class(config=target_config)
    target_model = target_model.to(dtype=source_model.dtype)

    # Track parameter changes
    initial_param_sum = sum(p.sum().item() for p in target_model.parameters())
    logger.info(f"Initial random parameter sum: {initial_param_sum:.6f}")

    # Copy shared weights
    logger.info("Copying shared weights...")
    hybridization_candidates = copy_shared_weights(source_model, target_model)

    after_copy_param_sum = sum(p.sum().item() for p in target_model.parameters())
    logger.info(
        f"Parameter sum after copying shared weights: {after_copy_param_sum:.6f}"
    )

    # Initialize hybrid layers
    logger.info("Initializing hybrid layers...")
    init_qwen_hmf_layers(
        source_model,
        target_model,
        target_layer_types,
        extract_gdn_weights_fn=extract_gdn_weights_fn,
        extract_attn_weights_fn=extract_attn_weights_fn,
    )

    final_param_sum = sum(p.sum().item() for p in target_model.parameters())

    logger.summary(
        f"{model_name} conversion complete:\n"
        f"  Initial param sum: {initial_param_sum:.6f}\n"
        f"  After copy param sum: {after_copy_param_sum:.6f}\n"
        f"  Final param sum: {final_param_sum:.6f}\n"
    )

    return target_model


def apply_layer_configs_to_model_config(target_config, yaml_config: dict) -> None:
    """
    Apply layer-specific configs (GKA, GDN, etc.) from YAML to the target model config.
    
    This ensures that:
    1. The GKA/GDN layers are initialized with the correct parameters
    2. The saved model config contains these parameters for future loading
    
    Args:
        target_config: The target model's HF-style config (e.g., Qwen3NextHMFConfig)
        yaml_config: The YAML config dictionary containing layer configs
    """
    # Apply GKA config if present in YAML
    if "gka" in yaml_config and yaml_config["gka"] is not None:
        gka_exp_config = HybridGKAConfig(**yaml_config["gka"])
        gka_config = construct_gka_config(target_config, gka_exp_config)
        target_config.gka_config = gka_config.to_dict()
        logger.info(f"Applied GKA config from YAML: {yaml_config['gka']}")

    # Apply GDN config if present in YAML
    if "gdn" in yaml_config and yaml_config["gdn"] is not None:
        gdn_exp_config = HybridGatedDeltaNetConfig(**yaml_config["gdn"])
        gdn_config = construct_gdn_config(target_config, gdn_exp_config)
        target_config.gdn_config = gdn_config.to_dict()
        logger.info(f"Applied GDN config from YAML: {yaml_config['gdn']}")
