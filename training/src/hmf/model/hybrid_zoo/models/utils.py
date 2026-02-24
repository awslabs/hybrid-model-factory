import inspect
from typing import Dict, List, Type, Any
import logging

# Define the pattern-to-class-suffix mapping
LAYER_TYPE_PATTERNS = {
    "*": "DecoderLayer",  # Attention
    # Pure Hybrid (>= Stage 2)
    "M2": "Mamba2DecoderLayer",
    "BMF": "BMOJOFDecoderLayer",
    "GDN": "GDNDecoderLayer",
    "GKA": "GKADecoderLayer",
    "SWA": "SWADecoderLayer",
    # Fused Hybrid + Attention (Stage 1)
    "M2*": "FusedAttMamba2DecoderLayer",
    "BMF*": "FusedAttBMojoFDecoderLayer",
    "GDN*": "FusedAttGDNDecoderLayer",
    "GKA*": "FusedAttGKADecoderLayer",
    "SWA*": "FusedAttSWADecoderLayer",
    # Fused Hybrid + Hybrid (SSM→SSM distillation)
    "GDN>GKA": "FusedGDNToGKADecoderLayer",
    # Dual path Attention (Stage 1)
    "*DA": "DualAttDecoderLayer",
}


def build_decoder_layer_registry(module_or_dict, model_prefix: str) -> Dict[str, Type]:
    """
    Automatically build a decoder layer registry by finding classes in a module.
    
    This function uses the LAYER_TYPE_PATTERNS mapping to find classes by convention.
    For example, if model_prefix="Qwen3" and pattern="M2", it looks for "Qwen3Mamba2DecoderLayer".
    
    Args:
        module_or_dict: Either a module object or a dict of {name: class}
        model_prefix: Model name prefix (e.g., "Qwen3", "Qwen2", "Llama3")
        
    Returns:
        Dictionary mapping pattern strings to layer classes
        
    Raises:
        ValueError: If a required class is not found
        
    Example:
        >>> import modeling_hybrid_qwen3
        >>> registry = build_decoder_layer_registry(modeling_hybrid_qwen3, "Qwen3")
        >>> registry["M2"]
        <class 'Qwen3Mamba2DecoderLayer'>
    """
    # Get all classes from the module
    if isinstance(module_or_dict, dict):
        classes = module_or_dict
    else:
        classes = {
            name: obj
            for name, obj in inspect.getmembers(module_or_dict, inspect.isclass)
            if name.startswith(model_prefix)
        }

    registry = {}
    missing_classes = []

    for pattern, class_suffix in LAYER_TYPE_PATTERNS.items():
        class_name = f"{model_prefix}{class_suffix}"
        layer_class = classes.get(class_name)

        if layer_class is not None:
            registry[pattern] = layer_class
        else:
            missing_classes.append(f"{class_name} (for pattern '{pattern}')")

    # If there are missing classes, show warning
    if missing_classes:
        logger = logging.getLogger(__name__)
        logger.debug(
            f"Some decoder layer classes not found for {model_prefix}: "
            f"{', '.join(missing_classes)}"
        )

    if not registry:
        raise ValueError(
            f"No decoder layer classes found for model prefix '{model_prefix}'. "
            f"Expected classes like {model_prefix}DecoderLayer, {model_prefix}Mamba2DecoderLayer, etc."
        )

    return registry


def set_decoder_layers_from_pattern(config, registry: Dict[str, Type]) -> List[Any]:
    """
    Create decoder layers based on hybrid override pattern and registry.
    
    Args:
        config: Model configuration with hybrid_override_pattern and num_hidden_layers
        registry: Dictionary mapping pattern strings to layer classes
        
    Returns:
        List of instantiated decoder layer modules
        
    Raises:
        ValueError: If an unknown layer type is encountered in the pattern
    """
    hybrid_override_pattern = parse_override_pattern(config.hybrid_override_pattern)
    layers = []

    for layer_idx in range(config.num_hidden_layers):
        layer_type = hybrid_override_pattern[layer_idx]
        layer_class = registry.get(layer_type)

        if layer_class is None:
            raise ValueError(
                f"Unknown layer type '{layer_type}' at position {layer_idx}. "
                f"Valid types: {list(registry.keys())}"
            )

        layers.append(layer_class(config=config, layer_idx=layer_idx))

    return layers


def parse_override_pattern(pattern: str) -> Dict[int, str]:
    """
    Converts a string pattern like "*-*-*-M2-M2" into a dictionary mapping 
    layer index to layer type.

    Symbol definitions (see LAYER_TYPE_PATTERNS for full list):
        *    -> Standard attention layer
        M2   -> Mamba2 layer
        BMF  -> B'MOJO-F layer
        GDN  -> Gated DeltaNet layer
        GKA  -> Gated KalmaNet layer
        SWA -> Sliding Window Attention layer
        M2*  -> Fused Mamba2 + Attention layer
        BMF* -> Fused B'MOJO-F + Attention layer
        GDN* -> Fused Gated DeltaNet + Attention layer
        GKA* -> Fused GKA + Attention layer
        SWA* -> Fused SWA + Attention layer
        *DA  -> Dual Attention layer
        GDN>GKA -> Fused GDN + GKA layer (for GDN→GKA distillation)
    
    Args:
        pattern: Hyphen-separated pattern string
        
    Returns:
        Dictionary mapping layer index to pattern string
        
    Example:
        >>> parse_override_pattern("*DA-*DA-M2*-*DA-M2*-M2*")
        {0: "*DA", 1: "*DA", 2: "M2*", 3: "*DA", 4: "M2*", 5: "*DA"}
    """
    split = pattern.split("-")

    parsed_pattern = {i: split[i] for i in range(len(split))}
    return parsed_pattern
