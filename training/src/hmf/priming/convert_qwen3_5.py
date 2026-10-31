"""
Qwen3.5 (dense) to HMF conversion. Provides model-specific weight extraction functions
for the separate GDN projections (in_proj_qkv, in_proj_z, in_proj_b, in_proj_a) and
the gated attention layer. Delegates shared conversion logic to convert_qwen_gdn_models.
"""

import torch
import torch.nn as nn
from transformers import PretrainedConfig

from .hybridize_logger import get_logger

from hmf.model.hybrid_zoo.layers.hybrid_dataclasses import HybridGKAConfig
from hmf.model.hybrid_zoo.models.qwen3_5_hmf.configuration_qwen3_5_hmf import (
    Qwen3_5HMFConfig,
    Qwen3_5HMFTextConfig,
)
from hmf.model.hybrid_zoo.models.qwen3_5_hmf.modeling_qwen3_5_hmf import (
    Qwen3_5HMFForCausalLM,
)
from hmf.model.hybrid_zoo.models.utils import parse_override_pattern

from .convert_qwen_gdn_models import (
    apply_layer_configs_to_model_config,
    convert_qwen_gdn_model,
    GDNWeights,
    GatedAttentionWeights,
)

logger = get_logger(__name__)


def extract_qwen3_5_gdn_weights(gdn_layer: nn.Module) -> GDNWeights:
    """
    Extract weights from HF Qwen3.5 GDN layer.

    Qwen3.5 uses separate (non-interleaved) projections:
        - in_proj_qkv: Concatenated [Q, K, V] projection (no interleaving)
        - in_proj_z: Separate gate (Z) projection
        - in_proj_b: Separate beta projection
        - in_proj_a: Separate alpha projection
        - conv1d: Fused convolution for Q, K, V

    Args:
        gdn_layer: HF Qwen3_5GatedDeltaNet layer

    Returns:
        GDNWeights with extracted weights
    """
    key_dim = gdn_layer.key_dim
    value_dim = gdn_layer.value_dim

    # in_proj_qkv is simply [Q, K, V] concatenated along dim 0
    qkv_weight = gdn_layer.in_proj_qkv.weight.data
    q_weight = qkv_weight[:key_dim, :]
    k_weight = qkv_weight[key_dim : key_dim * 2, :]
    v_weight = qkv_weight[key_dim * 2 :, :]

    # Z, B, A are separate projections
    z_weight = gdn_layer.in_proj_z.weight.data.clone()
    b_weight = gdn_layer.in_proj_b.weight.data.clone()
    a_weight = gdn_layer.in_proj_a.weight.data.clone()

    # conv1d layout is [Q, K, V] concatenated
    conv_weight = gdn_layer.conv1d.weight.data
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


def extract_qwen3_5_attention_weights(
    attn_layer: nn.Module,
) -> GatedAttentionWeights:
    """
    Extract weights from HF Qwen3.5 Attention layer.

    Same structure as Qwen3.5-MoE: q_proj outputs [Q, gate] interleaved by head,
    so it is 2x the typical size.

    Args:
        attn_layer: HF Qwen3_5Attention layer

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


def construct_hybrid_qwen3_5(
    base_model: nn.Module, config: PretrainedConfig, hybrid_config
) -> nn.Module:
    """
    Entry point for hybridize_model.py to call into Qwen3.5 (dense) conversion.

    Handles the composite config structure (Qwen3_5Config wraps text_config +
    vision_config) by extracting text_config for the HMF target config.

    Args:
        base_model: HF Qwen3.5 text model (source model to convert)
        config: Base model's HF config (with hybrid_override_pattern added).
            May be a composite Qwen3_5Config or a flat Qwen3_5TextConfig.
        hybrid_config: HybridConfig from hybridize_model.py containing layer configs

    Returns:
        Converted HMF Qwen3.5 model
    """
    logger.section("Constructing Hybrid Qwen3.5 (via hybridize_model.py)")

    pattern = config.hybrid_override_pattern
    logger.info(f"Hybridization pattern: {pattern}")

    # Parse the pattern to get actual layer types
    target_layer_types = parse_override_pattern(pattern)
    layer_type_set = set(target_layer_types.values())

    # Build yaml_config with defaults if layer type is in pattern but config not provided
    yaml_config = {}
    needs_gka = any("GKA" in lt for lt in layer_type_set)
    needs_gdn = any(lt == "GDN" for lt in layer_type_set)

    if needs_gka:
        gka_cfg = hybrid_config.gka if hybrid_config.gka else HybridGKAConfig()
        gka_dict = gka_cfg.to_dict()
        gka_dict.pop("norm_eps", None)  # Let construct_gka_config use model's rms_norm_eps
        yaml_config["gka"] = gka_dict
        logger.info(f"GKA config: {yaml_config['gka']}")
    if needs_gdn and hybrid_config.gdn:
        yaml_config["gdn"] = hybrid_config.gdn.to_dict()
        logger.info(f"GDN config: {yaml_config['gdn']}")

    # Handle composite config: extract text_config if present
    source_text_config = getattr(config, "text_config", config)

    # Build target text config from source text config
    config_dict = source_text_config.to_dict()
    config_dict.pop("model_type", None)
    target_text_config = Qwen3_5HMFTextConfig.from_dict(config_dict)
    target_text_config.hybrid_override_pattern = pattern

    # Apply GKA/GDN specific configs to target_config
    apply_layer_configs_to_model_config(target_text_config, yaml_config)

    # Build composite config preserving vision config and token IDs
    if hasattr(config, "vision_config") and config.vision_config is not None:
        vision_dict = (
            config.vision_config.to_dict()
            if hasattr(config.vision_config, "to_dict")
            else config.vision_config
        )
        target_config = Qwen3_5HMFConfig(
            text_config=target_text_config.to_dict(),
            vision_config=vision_dict,
            image_token_id=getattr(config, "image_token_id", 248_056),
            video_token_id=getattr(config, "video_token_id", 248_057),
            vision_start_token_id=getattr(config, "vision_start_token_id", 248_053),
            vision_end_token_id=getattr(config, "vision_end_token_id", 248_054),
        )
    else:
        # No vision config (text-only source), use text config directly
        target_config = target_text_config

    # Convert using text config (ForCausalLM expects text config directly)
    hybrid_model = convert_qwen_gdn_model(
        source_model=base_model,
        target_model_class=Qwen3_5HMFForCausalLM,
        target_config=target_text_config,
        model_name="Qwen3.5",
        extract_gdn_weights_fn=extract_qwen3_5_gdn_weights,
        extract_attn_weights_fn=extract_qwen3_5_attention_weights,
        target_layer_types=target_layer_types,
    )

    # Swap to composite config so save_pretrained writes vision config + token IDs
    if hasattr(config, "vision_config") and config.vision_config is not None:
        hybrid_model.config = target_config
        logger.info("Attached composite config (text + vision) to model")

    return hybrid_model
