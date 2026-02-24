"""
Hybrid Qwen Model Construction Module

This module provides functionality for converting base Qwen transformer models into hybrid models
that combine different layer types (Mamba2, B'MOJO, Gated DeltaNet, GKA, SWA).
"""

from typing import List, Union

import torch.nn as nn
from transformers.configuration_utils import PretrainedConfig
from .utils import add_hybrid_configs, copy_shared_weights

# Import shared layer initialization functions
from .hybrid_layer_init import (
    extract_attention_weights,
    mamba2_init,
    bmojo_attention_init,
    gated_deltanet_proj_init,
    gka_proj_init,
    swa_proj_init,
)

from hmf.model.hybrid_zoo.layers.hybrid_dataclasses import HybridConfig

# Qwen2.5
from hmf.model.hybrid_zoo.models.hybrid_qwen2.configuration_hybrid_qwen2 import (
    HybridQwen2Config,
)
from hmf.model.hybrid_zoo.models.hybrid_qwen2.modeling_hybrid_qwen2 import (
    HybridQwen2ForCausalLM,
    HybridQwen2ForSequenceClassification,
    HybridQwen2ForTokenClassification,
    HybridQwen2Model,
    HybridQwen2ForQuestionAnswering,
    HybridQwen2PreTrainedModel,
)

# Qwen3
from hmf.model.hybrid_zoo.models.hybrid_qwen3.configuration_hybrid_qwen3 import (
    HybridQwen3Config,
)
from hmf.model.hybrid_zoo.models.hybrid_qwen3.modeling_hybrid_qwen3 import (
    HybridQwen3ForCausalLM,
    HybridQwen3ForSequenceClassification,
    HybridQwen3ForTokenClassification,
    HybridQwen3Model,
    HybridQwen3ForQuestionAnswering,
    HybridQwen3PreTrainedModel,
)

# Qwen3-MoE
from hmf.model.hybrid_zoo.models.hybrid_qwen3_moe.configuration_hybrid_qwen3_moe import (
    HybridQwen3MoeConfig,
)
from hmf.model.hybrid_zoo.models.hybrid_qwen3_moe.modeling_hybrid_qwen3_moe import (
    HybridQwen3MoeForCausalLM,
    HybridQwen3MoeForSequenceClassification,
    HybridQwen3MoeForTokenClassification,
    HybridQwen3MoeModel,
    HybridQwen3MoeForQuestionAnswering,
    HybridQwen3MoePreTrainedModel,
)

from hmf.model.hybrid_zoo.models.utils import parse_override_pattern

from .hybridize_logger import get_logger

logger = get_logger(__name__)


# =============================================================================
# Constants
# =============================================================================

# Layer type identifiers
LAYER_TYPE_M2 = "M2"
LAYER_TYPE_BMF = "BMF"
LAYER_TYPE_GDN = "GDN"
LAYER_TYPE_GKA = "GKA"
LAYER_TYPE_SWA = "SWA"

# Layer name components
SELF_ATTN = "self_attn"
MAMBA2 = "mamba"
BMOJO_F = "bmojo_f"
GATED_DELTANET = "gdn"
GKA = "gka"
SWA = "swa"
MODEL_LAYERS_PREFIX = "model.layers"


# =============================================================================
# Model Class Registry
# =============================================================================

QWEN_CLASS_REGISTRY = {
    2: {
        "ForCausalLM": HybridQwen2ForCausalLM,
        "Model": HybridQwen2Model,
        "PreTrainedModel": HybridQwen2PreTrainedModel,
        "ForSequenceClassification": HybridQwen2ForSequenceClassification,
        "ForQuestionAnswering": HybridQwen2ForQuestionAnswering,
        "ForTokenClassification": HybridQwen2ForTokenClassification,
    },
    3: {
        "ForCausalLM": HybridQwen3ForCausalLM,
        "Model": HybridQwen3Model,
        "PreTrainedModel": HybridQwen3PreTrainedModel,
        "ForSequenceClassification": HybridQwen3ForSequenceClassification,
        "ForQuestionAnswering": HybridQwen3ForQuestionAnswering,
        "ForTokenClassification": HybridQwen3ForTokenClassification,
    },
}

QWEN_MOE_CLASS_REGISTRY = {
    3: {
        "ForCausalLM": HybridQwen3MoeForCausalLM,
        "Model": HybridQwen3MoeModel,
        "PreTrainedModel": HybridQwen3MoePreTrainedModel,
        "ForSequenceClassification": HybridQwen3MoeForSequenceClassification,
        "ForQuestionAnswering": HybridQwen3MoeForQuestionAnswering,
        "ForTokenClassification": HybridQwen3MoeForTokenClassification,
    }
}


# =============================================================================
# Utility Functions
# =============================================================================

def get_qwen_class(base_model: nn.Module, version: int, is_moe: bool) -> type:
    """
    Get the appropriate hybrid model class based on the base model and version.
    
    Args:
        base_model: The base model to convert
        version: Model version (2 or 3)
        is_moe: Whether this is a MoE model
        
    Returns:
        The corresponding hybrid model class
        
    Raises:
        ValueError: If the version is unsupported or model class is unknown
    """
    model_class_name = type(base_model).__name__

    registry = QWEN_MOE_CLASS_REGISTRY if is_moe else QWEN_CLASS_REGISTRY
    base_name = f"Qwen{version}Moe" if is_moe else f"Qwen{version}"

    if version not in registry:
        raise ValueError(
            f"Unsupported Qwen version: {version}. "
            f"Supported versions: {list(registry.keys())}"
        )

    version_registry = registry[version]

    for suffix, hybrid_class in version_registry.items():
        if f"{base_name}{suffix}" in model_class_name:
            return hybrid_class

    raise ValueError(
        f"Unknown model class: {model_class_name}. "
        f"Expected one of: {[f'{base_name}{s}' for s in version_registry.keys()]}"
    )


def is_hybrid_layer_candidate(
    layer_name_prefix: str, hybridization_candidates: List[str]
) -> bool:
    """
    Check if a layer is a candidate for hybridization.
    
    Args:
        layer_name_prefix: The layer name prefix to check
        hybridization_candidates: List of candidate layer names
        
    Returns:
        True if the layer is a hybridization candidate
    """
    return any(layer_name_prefix in candidate for candidate in hybridization_candidates)


def get_layer_name(layer_idx: int, *components: str) -> str:
    """
    Construct a layer name from components.
    
    Args:
        layer_idx: Layer index
        *components: Additional name components
        
    Returns:
        Fully qualified layer name
    """
    parts = [MODEL_LAYERS_PREFIX, str(layer_idx)] + list(components)
    return ".".join(parts)



# =============================================================================
# Layer Initialization
# =============================================================================

def init_hybrid_layers(
    base_model: nn.Module,
    hybrid_model: nn.Module,
    hybridization_candidates: List[str],
    config: Union[HybridQwen2Config, HybridQwen3Config, HybridQwen3MoeConfig],
    attn_layer_name: str = SELF_ATTN,
) -> None:
    """
    Initialize hybrid layers in the model.
    
    Iterates through all layers and initializes hybrid-specific components
    (Mamba2, B'MOJO, Gated DeltaNet, GKA, SWA) based on the hybridization pattern.
    
    Args:
        base_model: Base model to copy weights from
        hybrid_model: Hybrid model to initialize
        hybridization_candidates: List of candidate layer names for hybridization
        config: Hybrid model configuration
        attn_layer_name: Name of attention layer in base model
    """
    hybrid_override_pattern = parse_override_pattern(config.hybrid_override_pattern)

    for layer_idx in range(config.num_hidden_layers):
        layer_type = hybrid_override_pattern[layer_idx]
        
        # Skip attention layers (no hybridization needed)
        if layer_type == "*":
            continue
        
        # Get source attention layer
        attn_layer = getattr(base_model.model.layers[layer_idx], attn_layer_name)
        hybrid_layer = hybrid_model.model.layers[layer_idx]
        
        # Extract attention weights once per layer
        attn_weights = extract_attention_weights(attn_layer)
        
        # Initialize based on layer type
        if LAYER_TYPE_M2 in layer_type and LAYER_TYPE_BMF not in layer_type:
            # Pure Mamba2 layer
            candidate_name = get_layer_name(layer_idx, MAMBA2)
            if is_hybrid_layer_candidate(candidate_name, hybridization_candidates):
                mamba2_init(attn_weights, hybrid_layer.mamba, layer_idx)
        
        elif LAYER_TYPE_BMF in layer_type:
            # B'MOJO layer
            candidate_name = get_layer_name(layer_idx, BMOJO_F)
            if is_hybrid_layer_candidate(candidate_name, hybridization_candidates):
                bmojo_f = hybrid_layer.bmojo_f
                
                # Initialize B'MOJO attention component
                bmojo_attention_init(attn_weights, bmojo_f.bmojo_attn, layer_idx)
                
                # Initialize SSM mixer component
                if hasattr(bmojo_f, MAMBA2):
                    mamba2_init(attn_weights, bmojo_f.mamba, layer_idx)
                elif hasattr(bmojo_f, GATED_DELTANET):
                    gated_deltanet_proj_init(attn_weights, bmojo_f.gdn, layer_idx)
                elif hasattr(bmojo_f, GKA):
                    gka_proj_init(attn_weights, bmojo_f.gka, layer_idx)
        
        elif LAYER_TYPE_GDN in layer_type:
            candidate_name = get_layer_name(layer_idx, GATED_DELTANET)
            if is_hybrid_layer_candidate(candidate_name, hybridization_candidates):
                gated_deltanet_proj_init(attn_weights, hybrid_layer.gdn, layer_idx)
        
        elif LAYER_TYPE_GKA in layer_type:
            candidate_name = get_layer_name(layer_idx, GKA)
            if is_hybrid_layer_candidate(candidate_name, hybridization_candidates):
                gka_proj_init(attn_weights, hybrid_layer.gka, layer_idx)
        
        elif LAYER_TYPE_SWA in layer_type:
            candidate_name = get_layer_name(layer_idx, SWA)
            if is_hybrid_layer_candidate(candidate_name, hybridization_candidates):
                swa_proj_init(attn_weights, hybrid_layer.swa, layer_idx)



# =============================================================================
# Main Construction Function
# =============================================================================

def construct_hybrid_qwen(
    base_model: nn.Module,
    config: PretrainedConfig,
    hybrid_config: HybridConfig,
    version: int,
    is_moe: bool = False,
) -> nn.Module:
    """
    Construct a hybrid model from a base Qwen model.
    
    This is the main entry point for creating hybrid models. It does the following:
        1. Determines the appropriate hybrid model class
        2. Creates the hybrid configuration
        3. Instantiates a hybrid model with random weights
        4. Copies shared weights from the base model
        5. Initializes hybrid-specific layers
    
    Args:
        base_model: Base transformer model to convert
        config: Base model configuration
        hybrid_config: Hybrid-specific configuration
        version: Qwen version (2 or 3)
        is_moe: Whether this is a Qwen3 MoE model
        
    Returns:
        Initialized hybrid model
        
    Raises:
        ValueError: If version is unsupported or model class is incompatible
    """
    if is_moe:
        logger.section(f"Constructing Hybrid Qwen{version}-MoE")
    else:
        logger.section(f"Constructing Hybrid Qwen{version}")

    # Get hybrid model class
    hybrid_qwen_class = get_qwen_class(base_model, version=version, is_moe=is_moe)
    logger.info(f"Using hybrid model class: {hybrid_qwen_class.__name__}")

    # Create hybrid configuration
    # Pop model_type to avoid overwriting the hybrid config's model_type class attribute
    config_dict = config.to_dict()
    config_dict.pop("model_type", None)
    
    if is_moe:
        if version == 3:
            hybrid_qwen_config = HybridQwen3MoeConfig.from_dict(config_dict)
        else:
            raise ValueError(f"Unsupported MoE version: {version}")
    else:
        if version == 2:
            hybrid_qwen_config = HybridQwen2Config.from_dict(config_dict)
        elif version == 3:
            hybrid_qwen_config = HybridQwen3Config.from_dict(config_dict)
        else:
            raise ValueError(f"Unsupported Qwen version: {version}")

    hybrid_qwen_config = add_hybrid_configs(
        hybrid_qwen_config, hybrid_config=hybrid_config
    )

    # Create hybrid model with random initialization
    logger.info("Creating hybrid model with random initialization (parameters will be initialized afterwards)")
    hybrid_model = hybrid_qwen_class(config=hybrid_qwen_config).to(
        dtype=config.torch_dtype
    )

    # Track parameter changes
    initial_param_sum = sum(p.sum().item() for p in hybrid_model.parameters())
    logger.info(f"Initial random parameter sum: {initial_param_sum:.6f}")

    # Copy shared weights
    hybridization_candidates = copy_shared_weights(base_model, hybrid_model)

    after_copy_param_sum = sum(p.sum().item() for p in hybrid_model.parameters())
    logger.info(
        f"Parameter sum after copying shared weights: {after_copy_param_sum:.6f}"
    )

    # Initialize hybrid layers
    logger.info("Initializing hybrid-specific layers")
    init_hybrid_layers(
        base_model, hybrid_model, hybridization_candidates, config=hybrid_qwen_config
    )

    final_param_sum = sum(p.sum().item() for p in hybrid_model.parameters())

    logger.summary(
        f"Hybrid model construction complete:\n"
        f"  Initial param sum: {initial_param_sum:.6f}\n"
        f"  After copy param sum: {after_copy_param_sum:.6f}\n"
        f"  Final param sum: {final_param_sum:.6f}\n"
    )

    return hybrid_model
