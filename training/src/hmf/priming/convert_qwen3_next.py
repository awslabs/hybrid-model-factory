"""
Qwen3-Next to HMF conversion. Provides model-specific weight extraction functions
for the fused/interleaved GDN projections (in_proj_qkvz, in_proj_ba) and the
gated attention layer. Delegates shared conversion logic to convert_qwen_gdn_models.
"""

import torch
import torch.nn as nn
from transformers import PretrainedConfig

from .hybridize_logger import get_logger

from hmf.model.hybrid_zoo.layers.hybrid_dataclasses import HybridGKAConfig
from hmf.model.hybrid_zoo.models.qwen3_next_hmf.configuration_qwen3_next_hmf import (
    Qwen3NextHMFConfig,
)
from hmf.model.hybrid_zoo.models.qwen3_next_hmf.modeling_qwen3_next_hmf import (
    Qwen3NextHMFForCausalLM,
)
from hmf.model.hybrid_zoo.models.utils import parse_override_pattern

from .convert_qwen_gdn_models import (
    apply_layer_configs_to_model_config,
    convert_qwen_gdn_model,
    GDNWeights,
    GatedAttentionWeights,
)

logger = get_logger(__name__)

def extract_qwen3next_gdn_weights(gdn_layer: nn.Module,) -> GDNWeights:
    """
    Extract and de-interleave weights from HF Qwen3-Next GDN layer.
    
    HF GDN uses fused projections grouped by K-heads (since num_v_heads > num_k_heads):
    
    in_proj_qkvz layout (grouped by num_k_heads groups):
        For each K-head group i (i = 0..num_k_heads-1):
            - Q_i: [head_k_dim] - query for this K-head
            - K_i: [head_k_dim] - key for this K-head
            - V_group_i: [v_per_k * head_v_dim] - values for V-heads in this group
            - Z_group_i: [v_per_k * head_v_dim] - gates for V-heads in this group
        
        Where v_per_k = num_v_heads // num_k_heads (e.g., 32/16 = 2)
        
        Example with num_k_heads=16, num_v_heads=32, head_k_dim=128, head_v_dim=128:
            Group 0: [Q0(128), K0(128), V0-V1(256), Z0-Z1(256)] = 768 dims
            Group 1: [Q1(128), K1(128), V2-V3(256), Z2-Z3(256)] = 768 dims
            ...
            Total: 16 * 768 = 12288 dims
    
    in_proj_ba layout (grouped by num_k_heads groups):
        For each K-head group i:
            - B_group_i: [v_per_k] - beta for V-heads in this group
            - A_group_i: [v_per_k] - alpha for V-heads in this group
    
    conv1d layout:
        After fix_query_key_value_ordering, Q/K/V are concatenated as [Q, K, V]
        So conv1d is: [key_dim, key_dim, value_dim] = [Q_all, K_all, V_all]
    
    Args:
        gdn_layer: HF Qwen3NextGatedDeltaNet layer
        
    Returns:
        GDNWeights with de-interleaved weights
    """
    # Get dimensions from the layer
    num_k_heads = gdn_layer.num_k_heads
    num_v_heads = gdn_layer.num_v_heads
    head_k_dim = gdn_layer.head_k_dim
    head_v_dim = gdn_layer.head_v_dim
    key_dim = gdn_layer.key_dim
    value_dim = gdn_layer.value_dim

    # Number of V heads per K head group
    v_per_k = num_v_heads // num_k_heads

    # De-interleave in_proj_qkvz
    # Weight shape: [out_features, in_features] = [key_dim*2 + value_dim*2, hidden_size]
    # Layout per K-head group: [Q (head_k_dim), K (head_k_dim), V (v_per_k * head_v_dim), Z (v_per_k * head_v_dim)]
    qkvz_weight = gdn_layer.in_proj_qkvz.weight.data

    # Calculate sizes per K-head group
    group_q_size = head_k_dim
    group_k_size = head_k_dim
    group_v_size = v_per_k * head_v_dim
    group_z_size = v_per_k * head_v_dim
    group_total = group_q_size + group_k_size + group_v_size + group_z_size

    # Reshape to [num_k_heads, group_total, hidden_size]
    hidden_size = qkvz_weight.shape[1]
    qkvz_grouped = qkvz_weight.view(num_k_heads, group_total, hidden_size)

    # Split each group and collect
    q_parts = []
    k_parts = []
    v_parts = []
    z_parts = []

    for i in range(num_k_heads):
        offset = 0
        q_parts.append(qkvz_grouped[i, offset : offset + group_q_size, :])
        offset += group_q_size
        k_parts.append(qkvz_grouped[i, offset : offset + group_k_size, :])
        offset += group_k_size
        v_parts.append(qkvz_grouped[i, offset : offset + group_v_size, :])
        offset += group_v_size
        z_parts.append(qkvz_grouped[i, offset : offset + group_z_size, :])

    q_weight = torch.cat(q_parts, dim=0)  # [key_dim, hidden_size]
    k_weight = torch.cat(k_parts, dim=0)  # [key_dim, hidden_size]
    v_weight = torch.cat(v_parts, dim=0)  # [value_dim, hidden_size]
    z_weight = torch.cat(z_parts, dim=0)  # [value_dim, hidden_size]

    # De-interleave in_proj_ba
    # Weight shape: [num_v_heads * 2, hidden_size]
    # Layout per K-head group: [B (v_per_k), A (v_per_k)]
    ba_weight = gdn_layer.in_proj_ba.weight.data

    ba_grouped = ba_weight.view(num_k_heads, v_per_k * 2, hidden_size)
    b_parts = []
    a_parts = []

    for i in range(num_k_heads):
        b_parts.append(ba_grouped[i, :v_per_k, :])
        a_parts.append(ba_grouped[i, v_per_k:, :])

    b_weight = torch.cat(b_parts, dim=0)  # [num_v_heads, hidden_size]
    a_weight = torch.cat(a_parts, dim=0)  # [num_v_heads, hidden_size]

    # De-interleave conv1d
    # After fix_query_key_value_ordering reshapes Q/K/V, they are concatenated as:
    #   mixed_qkv = torch.cat((query, key, value), dim=-1)
    # where query/key are [batch, seq, key_dim] and value is [batch, seq, value_dim]
    # So conv1d weight is [key_dim + key_dim + value_dim, 1, kernel_size]
    conv_weight = gdn_layer.conv1d.weight.data  # [conv_dim, 1, kernel_size]

    q_conv_weight = conv_weight[:key_dim, :, :]
    k_conv_weight = conv_weight[key_dim : key_dim * 2, :, :]
    v_conv_weight = conv_weight[key_dim * 2 :, :, :]

    return GDNWeights(
        q_weight=q_weight,
        k_weight=k_weight,
        v_weight=v_weight,
        z_weight=z_weight,
        b_weight=b_weight,
        a_weight=a_weight,
        q_conv_weight=q_conv_weight,
        k_conv_weight=k_conv_weight,
        v_conv_weight=v_conv_weight,
        A_log=gdn_layer.A_log.data.clone(),
        dt_bias=gdn_layer.dt_bias.data.clone(),
        norm_weight=gdn_layer.norm.weight.data.clone(),
        out_proj_weight=gdn_layer.out_proj.weight.data.clone(),
    )


def extract_qwen3next_attention_weights(
    attn_layer: nn.Module,
) -> GatedAttentionWeights:
    """
    Extract weights from HF Qwen3-Next Attention layer.
    
    Note: q_proj outputs [Q, gate] interleaved by head, so it is 2x the typical size.
    
    Args:
        attn_layer: HF Qwen3NextAttention layer
        
    Returns:
        GatedAttentionWeights with extracted weights
    """
    return GatedAttentionWeights(
        q_weight=attn_layer.q_proj.weight.data.clone(),
        k_weight=attn_layer.k_proj.weight.data.clone(),
        v_weight=attn_layer.v_proj.weight.data.clone(),
        o_weight=attn_layer.o_proj.weight.data.clone(),
        q_norm_weight=attn_layer.q_norm.weight.data.clone(),
        k_norm_weight=attn_layer.k_norm.weight.data.clone(),
    )


def construct_hybrid_qwen3_next(
    base_model: nn.Module, config: PretrainedConfig, hybrid_config
) -> nn.Module:
    """
    Entry point for hybridize_model.py to call into Qwen3-Next conversion.
    
    This function mirrors the signature of construct_hybrid_qwen() and other
    construct_hybrid_X() functions, allowing hybridize_model.py to serve as
    a single entry point for all model conversions.
    
    Args:
        base_model: HF Qwen3-Next model (source model to convert)
        config: Base model's HF config (with hybrid_override_pattern added)
        hybrid_config: HybridConfig from hybridize_model.py containing layer configs
        
    Returns:
        Converted HMF Qwen3-Next model
    """
    logger.section("Constructing Hybrid Qwen3-Next (via hybridize_model.py)")

    pattern = config.hybrid_override_pattern
    logger.info(f"Hybridization pattern: {pattern}")

    # Parse the pattern to get actual layer types
    target_layer_types = parse_override_pattern(pattern)
    layer_type_set = set(target_layer_types.values())

    # Build yaml_config with defaults if layer type is in pattern but config not provided
    # Check for actual layer types, not substrings (e.g., "GDN>GKA" should not trigger GDN config)
    yaml_config = {}
    needs_gka = any("GKA" in lt for lt in layer_type_set)
    needs_gdn = any(
        lt == "GDN" for lt in layer_type_set
    )  # Only standalone GDN, not "GDN>GKA"

    if needs_gka:
        gka_cfg = hybrid_config.gka if hybrid_config.gka else HybridGKAConfig()
        gka_dict = gka_cfg.to_dict()
        gka_dict.pop(
            "norm_eps", None
        )  # Let construct_gka_config use model's rms_norm_eps
        yaml_config["gka"] = gka_dict
        logger.info(f"GKA config: {yaml_config['gka']}")
    if needs_gdn and hybrid_config.gdn:
        yaml_config["gdn"] = hybrid_config.gdn.to_dict()
        logger.info(f"GDN config: {yaml_config['gdn']}")

    # Build target config from source config
    config_dict = config.to_dict()
    config_dict.pop("model_type", None)
    target_config = Qwen3NextHMFConfig.from_dict(config_dict)
    target_config.hybrid_override_pattern = pattern

    # Apply GKA/GDN specific configs to target_config
    apply_layer_configs_to_model_config(target_config, yaml_config)

    # Convert
    return convert_qwen_gdn_model(
        source_model=base_model,
        target_model_class=Qwen3NextHMFForCausalLM,
        target_config=target_config,
        model_name="Qwen3-Next",
        extract_gdn_weights_fn=extract_qwen3next_gdn_weights,
        extract_attn_weights_fn=extract_qwen3next_attention_weights,
        target_layer_types=target_layer_types,
    )
