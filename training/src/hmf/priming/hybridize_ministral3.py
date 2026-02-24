"""
Hybrid Ministral3 Model Construction Module

This module provides functionality for converting base Ministral3 transformer models into hybrid models
that combine different layer types (Mamba2, B'MOJO, Gated DeltaNet, GKA, SWA).
"""

from typing import List

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

from hmf.model.hybrid_zoo.models.hybrid_ministral3.configuration_hybrid_ministral3 import (
    HybridMinistral3Config,
)
from hmf.model.hybrid_zoo.models.hybrid_ministral3.modeling_hybrid_ministral3 import (
    HybridMinistral3ForCausalLM,
    HybridMinistral3ForSequenceClassification,
    HybridMinistral3ForTokenClassification,
    HybridMinistral3ForQuestionAnswering,
    HybridMinistral3Model,
    HybridMinistral3PreTrainedModel,
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
GATED_DELTANET = "gated_deltanet"
GKA = "gka"
SWA = "swa"
MODEL_LAYERS_PREFIX = "model.layers"


# =============================================================================
# Model Class Registry
# =============================================================================

MINISTRAL3_CLASS_REGISTRY = {
    "ForCausalLM": HybridMinistral3ForCausalLM,
    "Model": HybridMinistral3Model,
    "PreTrainedModel": HybridMinistral3PreTrainedModel,
    "ForSequenceClassification": HybridMinistral3ForSequenceClassification,
    "ForQuestionAnswering": HybridMinistral3ForQuestionAnswering,
    "ForTokenClassification": HybridMinistral3ForTokenClassification,
}


# =============================================================================
# Utility Functions
# =============================================================================

def get_ministral3_class(base_model: nn.Module) -> type:
    """
    Get the appropriate hybrid model class based on the base model.
    
    Args:
        base_model: The base model to convert
        
    Returns:
        The corresponding hybrid model class
        
    Raises:
        ValueError: If the model class is unknown
    """
    model_class_name = type(base_model).__name__

    for suffix, hybrid_class in MINISTRAL3_CLASS_REGISTRY.items():
        if f"Ministral3{suffix}" in model_class_name:
            return hybrid_class

    raise ValueError(
        f"Unknown model class: {model_class_name}. "
        f"Expected one of: {[f'Ministral3{s}' for s in MINISTRAL3_CLASS_REGISTRY.keys()]}"
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
    config: HybridMinistral3Config,
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
                    gated_deltanet_proj_init(attn_weights, bmojo_f.gated_deltanet, layer_idx)
                elif hasattr(bmojo_f, GKA):
                    gka_proj_init(attn_weights, bmojo_f.gka, layer_idx)
        
        elif LAYER_TYPE_GDN in layer_type:
            candidate_name = get_layer_name(layer_idx, GATED_DELTANET)
            if is_hybrid_layer_candidate(candidate_name, hybridization_candidates):
                gated_deltanet_proj_init(attn_weights, hybrid_layer.gated_deltanet, layer_idx)
        
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

def construct_hybrid_ministral3(
    base_model: nn.Module,
    config: PretrainedConfig,
    hybrid_config: HybridConfig,
) -> nn.Module:
    """
    Construct a hybrid model from a base Ministral3 model.
    
    This is the main entry point for creating hybrid Ministral3 models. It does the following:
        1. Determines the appropriate hybrid model class
        2. Creates the hybrid configuration
        3. Instantiates a hybrid model with random weights
        4. Copies shared weights from the base model
        5. Initializes hybrid-specific layers
    
    Args:
        base_model: Base Ministral3 transformer model to convert
        config: Base model configuration
        hybrid_config: Hybrid-specific configuration
        
    Returns:
        Initialized hybrid model
        
    Raises:
        ValueError: If model class is incompatible
    """
    logger.section("Constructing Hybrid Ministral3")

    # Get hybrid model class
    hybrid_ministral3_class = get_ministral3_class(base_model)
    logger.info(f"Using hybrid model class: {hybrid_ministral3_class.__name__}")

    # Create hybrid configuration
    # Pop model_type to avoid overwriting the hybrid config's model_type class attribute
    config_dict = config.to_dict()
    config_dict.pop("model_type", None)
    hybrid_ministral3_config = HybridMinistral3Config.from_dict(config_dict)
    hybrid_ministral3_config = add_hybrid_configs(
        hybrid_ministral3_config, hybrid_config=hybrid_config
    )

    # Create hybrid model with random initialization
    logger.info("Creating hybrid model with random initialization (parameters will be initialized afterwards)")
    hybrid_model = hybrid_ministral3_class(config=hybrid_ministral3_config).to(
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
        base_model, hybrid_model, hybridization_candidates, config=hybrid_ministral3_config
    )

    final_param_sum = sum(p.sum().item() for p in hybrid_model.parameters())

    logger.summary(
        f"Hybrid model construction complete:\n"
        f"  Initial param sum: {initial_param_sum:.6f}\n"
        f"  After copy param sum: {after_copy_param_sum:.6f}\n"
        f"  Final param sum: {final_param_sum:.6f}\n"
    )

    return hybrid_model
