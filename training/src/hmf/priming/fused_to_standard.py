import argparse
import copy
import os
from typing import Tuple

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from .hybridize_logger import get_logger
from .utils import copy_auxiliary_files

import hmf.model.hybrid_zoo.models.model_register  # Registers hybrid models and suppresses warnings. Must be first
from hmf.model.hybrid_zoo.models.hybrid_llama.modeling_hybrid_llama import (
    HybridLlamaForCausalLM,
)
from hmf.model.hybrid_zoo.models.hybrid_ministral3.modeling_hybrid_ministral3 import (
    HybridMinistral3ForCausalLM,
)
from hmf.model.hybrid_zoo.models.hybrid_qwen2.modeling_hybrid_qwen2 import (
    HybridQwen2ForCausalLM,
)
from hmf.model.hybrid_zoo.models.hybrid_qwen3.modeling_hybrid_qwen3 import (
    HybridQwen3ForCausalLM,
)
from hmf.model.hybrid_zoo.models.hybrid_qwen3_moe.modeling_hybrid_qwen3_moe import (
    HybridQwen3MoeForCausalLM,
)
from hmf.model.hybrid_zoo.models.qwen3_next_hmf.modeling_qwen3_next_hmf import (
    Qwen3NextHMFForCausalLM,
)
from hmf.model.hybrid_zoo.models.qwen3_5_moe_hmf.modeling_qwen3_5_moe_hmf import (
    Qwen3_5MoeHMFForCausalLM,
)
from hmf.model.hybrid_zoo.models.utils import LAYER_TYPE_PATTERNS


logger = get_logger(__name__)


def _get_text_config(config: AutoConfig) -> AutoConfig:
    """
    Extract the text config from a potentially composite config.
    
    Models like Qwen3.5-MoE use a composite config (text_config + vision_config)
    where hybrid_override_pattern and layer_types live on text_config.
    For non-composite models, returns the config as-is.
    """
    if hasattr(config, "text_config") and hasattr(config.text_config, "hybrid_override_pattern"):
        return config.text_config
    return config


def _is_composite_config(config: AutoConfig) -> bool:
    """Check if a config is a composite config wrapping a text_config."""
    return hasattr(config, "text_config") and hasattr(config.text_config, "hybrid_override_pattern")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_dir",
        type=str,
        help="The directory of the model trained with layerwise distillation.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=None,
        help="The directory to save the unfused model (auto-generated if not specified).",
    )
    parser.add_argument(
        "--save_max_shard_size",
        type=str,
        default="5GB",
        help="Max shard size for saving the unfused model (e.g., '5GB', '10GB').",
    )

    return parser.parse_args()


def get_model_type(model: nn.Module) -> type:
    """
    Get the appropriate hybrid model class based on the model type.
    
    Args:
        model: The loaded model instance.
        
    Returns:
        The corresponding hybrid ForCausalLM class for the model type.
        
    Raises:
        ValueError: If the model type is not supported.
    """
    model_type = _get_text_config(model.config).model_type
    if "qwen2" in model_type:
        return HybridQwen2ForCausalLM
    elif "qwen3_5_moe" in model_type:
        return Qwen3_5MoeHMFForCausalLM
    elif "qwen3_moe" in model_type:
        return HybridQwen3MoeForCausalLM
    elif "qwen3_next" in model_type:
        return Qwen3NextHMFForCausalLM
    elif "qwen3" in model_type:
        return HybridQwen3ForCausalLM
    elif "llama" in model_type:
        return HybridLlamaForCausalLM
    elif "ministral3" in model_type:
        return HybridMinistral3ForCausalLM
    else:
        raise ValueError(f"The current model type ({model_type}) is not supported.")


def load_model(
    model_name_or_path: str
) -> Tuple[AutoModelForCausalLM, AutoConfig, AutoTokenizer]:
    """
    Load a fused hybrid model with its configuration and tokenizer.
    
    Args:
        model_name_or_path: Path to the fused model checkpoint directory.
    
    Returns:
        Tuple containing:
            - model: The loaded fused hybrid model.
            - config: The model's configuration.
            - tokenizer: The model's tokenizer.
    """
    config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        config=config,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path, trust_remote_code=True
    )
    return model, config, tokenizer


def unfuse_layer_type(layer_type: str) -> str:
    """
    Convert a single fused layer type to its unfused equivalent.
    
    Args:
        layer_type: A single layer type string (e.g., "*DA", "GDN*", "*", "GDN>GKA").
        
    Returns:
        The unfused layer type (e.g., "*DA" -> "*", "GDN*" -> "GDN", "GDN>GKA" -> "GKA").
    """
    # Replace fused dual Attention with standard Attention
    if layer_type == "*DA":
        return "*"

    # Convert fused hybrid layers to pure layers (e.g., "GDN*" -> "GDN")
    if layer_type.endswith("*") and layer_type != "*":
        return layer_type[:-1]

    # Convert fused hybrid-to-hybrid layers to target layer (e.g., "GDN>GKA" -> "GKA")
    if ">" in layer_type:
        return layer_type.split(">")[1]

    return layer_type


def get_unfused_config(config: AutoConfig) -> AutoConfig:
    """
    Convert a fused model config to an unfused (standard) hybrid model config.
    
    Performs two types of conversions in the hybrid_override_pattern and layer_types:
    1. Fused layers (X*) -> Pure layers (X): e.g., M2* -> M2, BMF* -> BMF
    2. Dual attention (*DA, *XBP) -> Standard attention (*): e.g., *DA -> *, *BMFBP -> *
    
    Handles composite configs (e.g., Qwen3.5-MoE with text_config + vision_config)
    by unfusing the text_config and returning the full composite config.
    
    Args:
        config: Configuration of the fused model (may be composite or flat).
    
    Returns:
        New configuration with unfused hybrid pattern for standard hybrid model construction.
    """
    new_config = copy.deepcopy(config)

    # For composite configs, operate on the text_config
    target_config = _get_text_config(new_config)

    new_hybrid_pattern = target_config.hybrid_override_pattern

    # Replace fused dual Attention with standard Attention
    if "*DA" in new_hybrid_pattern:
        new_hybrid_pattern = new_hybrid_pattern.replace("*DA", "*")

    # Convert fused hybrid layers to pure layers
    # Find all patterns ending with * (except standalone * which is attention)
    for pattern in LAYER_TYPE_PATTERNS.keys():
        if pattern.endswith("*") and pattern != "*":
            fused = pattern  # e.g., "M2*"
            pure = pattern[:-1]  # e.g., "M2"
            new_hybrid_pattern = new_hybrid_pattern.replace(fused, pure)

    # Convert fused hybrid-to-hybrid layers to target layer (e.g., "GDN>GKA" -> "GKA")
    for pattern in LAYER_TYPE_PATTERNS.keys():
        if ">" in pattern:
            target = pattern.split(">")[1]  # e.g., "GDN>GKA" -> "GKA"
            new_hybrid_pattern = new_hybrid_pattern.replace(pattern, target)

    target_config.hybrid_override_pattern = new_hybrid_pattern

    # Also update layer_types if present
    if hasattr(target_config, "layer_types") and target_config.layer_types is not None:
        target_config.layer_types = [unfuse_layer_type(lt) for lt in target_config.layer_types]

    return new_config


def unfuse_model(model_dir: str, save_dir: str, save_max_shard_size: str = "5GB") -> None:
    """
    Convert a fused hybrid model to a standard hybrid model and save it.

    Loads the fused model, computes the unfused config (removing fused layer types),
    creates a new model with the unfused config, copies matching weights, and saves.

    Args:
        model_dir: Path to the fused model checkpoint directory.
        save_dir: Path to save the unfused model.
        save_max_shard_size: Max shard size for saving (e.g., '5GB', '10GB').
    """
    logger.section("Converting Fused Model to Standard Hybrid Model")
    logger.info(f"Loading model from {model_dir}")

    fused_model, fused_config, tokenizer = load_model(model_dir)
    fused_text_config = _get_text_config(fused_config)

    # Log model info
    logger.info(f"Model type: {fused_text_config.model_type}")
    logger.info(f"Number of layers: {fused_text_config.num_hidden_layers}")
    logger.info(f"Model dtype: {fused_text_config.torch_dtype}")

    logger.section("Fused Model")
    print()
    print(fused_model)

    # Get unfused config and log pattern transformation
    model_for_causal_lm = get_model_type(fused_model)
    og_pattern = fused_text_config.hybrid_override_pattern
    new_config = get_unfused_config(fused_config)
    new_text_config = _get_text_config(new_config)
    new_pattern = new_text_config.hybrid_override_pattern

    logger.info(f"Original hybrid pattern: {og_pattern}")
    logger.info(f"New hybrid pattern: {new_pattern}")

    # Count layer types in patterns
    pattern_layers = og_pattern.split("-")
    new_pattern_layers = new_pattern.split("-")

    logger.info(f"Layer conversions:")
    logger.info(
        f"  Fused layers (X*) -> Pure layers (X): {sum(1 for p in pattern_layers if p.endswith('*') and p != '*')}"
    )
    logger.info(
        f"  Dual attention (*DA/*XBP) -> Standard attention (*): {sum(1 for p in pattern_layers if 'DA' in p or 'BP' in p)}"
    )
    logger.info(f"  Standard attention layers: {new_pattern_layers.count('*')}")

    unfused_model = model_for_causal_lm(new_text_config)

    logger.section("Unfused Model")
    print()
    print(unfused_model)

    logger.info("Copying weights from fused model to unfused model")
    fused_model_state_dict = fused_model.state_dict()

    # Build filtered state dict with only matching keys
    unfused_keys = set(unfused_model.state_dict().keys())
    fused_keys = set(fused_model_state_dict.keys())

    matching_keys = unfused_keys & fused_keys
    missing_keys = unfused_keys - fused_keys

    copied_params = len(matching_keys)
    random_init_params = len(missing_keys)

    for param_name in missing_keys:
        logger.warning(
            f"{param_name} is initialized randomly! Is this intentional?"
        )

    logger.info(
        f"Copied {copied_params} parameters, {random_init_params} initialized randomly"
    )

    # Load matching weights directly (much faster than manual copy loop)
    filtered_state_dict = {k: v for k, v in fused_model_state_dict.items() if k in matching_keys}
    unfused_model.load_state_dict(filtered_state_dict, strict=False)
    unfused_model = unfused_model.to(torch.bfloat16)

    old_params = sum([p.numel() for p in fused_model.parameters()])
    new_params = sum([p.numel() for p in unfused_model.parameters()])

    logger.summary(
        f"Conversion complete:\n"
        f"  Fused model params: {old_params:,}\n"
        f"  Unfused model params: {new_params:,}\n"
        f"  Parameters copied: {copied_params}\n"
        f"  Parameters randomly initialized: {random_init_params}"
    )

    # For composite configs (e.g., Qwen3.5-MoE with vision_config), swap the full
    # composite config back onto the model so save_pretrained writes vision config
    # and token IDs alongside the text config.
    if _is_composite_config(new_config):
        unfused_model.config = new_config
        logger.info("Attached composite config (text + vision) to model")

    logger.info(f"Saving unfused model to {save_dir}")
    os.makedirs(save_dir, exist_ok=True)
    unfused_model.save_pretrained(save_dir, max_shard_size=save_max_shard_size)
    tokenizer.save_pretrained(save_dir)
    copy_auxiliary_files(model_dir, save_dir)
    logger.info(f"Saved to {save_dir}")


if __name__ == "__main__":
    args = parse_args()
    save_dir = args.save_dir or args.model_dir.rstrip("/") + "_unfused"
    unfuse_model(args.model_dir, save_dir, args.save_max_shard_size)
