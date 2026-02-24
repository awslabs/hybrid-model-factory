import os
import re
import shutil
from collections import defaultdict
from functools import partial
from typing import Dict, List

import torch
import torch.nn as nn
from .hybridize_logger import get_logger
from transformers import PreTrainedModel
from transformers.configuration_utils import PretrainedConfig

from hmf.model.hybrid_zoo.layers.hybrid_dataclasses import (
    HybridBMOJOConfig,
    HybridConfig,
    HybridGatedDeltaNetConfig,
    HybridMamba2Config,
    HybridGKAConfig,
)
from hmf.model.hybrid_zoo.layers.bmojo_f.bmojo_config import (
    construct_bmojo_config,
)
from hmf.model.hybrid_zoo.layers.gated_deltanet.gdn_config import (
    construct_gdn_config,
)
from hmf.model.hybrid_zoo.layers.mamba.mamba2_config import (
    construct_mamba2_config,
)
from hmf.model.hybrid_zoo.layers.gated_kalmanet.gka_config import (
    construct_gka_config,
)
from hmf.model.hybrid_zoo.models.utils import parse_override_pattern

logger = get_logger(__name__)

LAYER_CONFIG_REGISTRY = {
    "mamba2": (construct_mamba2_config, HybridMamba2Config),
    "gdn": (construct_gdn_config, HybridGatedDeltaNetConfig),
    "gka": (construct_gka_config, HybridGKAConfig),
    "bmojo": (construct_bmojo_config, HybridBMOJOConfig),
}


def _generalize_param_name(param_name: str) -> str:
    """
    Generalize parameter names by replacing layer indices with wildcards.
    
    Examples:
        model.layers.0.input_layernorm.weight -> model.layers.*.input_layernorm.weight
        model.layers.15.mlp.gate_proj.weight -> model.layers.*.mlp.gate_proj.weight
    """
    # Replace layer indices with *
    generalized = re.sub(r"\.layers\.\d+\.", ".layers.*.", param_name)
    # Replace any other numeric indices with *
    generalized = re.sub(r"\.\d+\.", ".*.", generalized)

    return generalized


def _group_params_by_pattern(param_names: List[str]) -> Dict[str, int]:
    """
    Group parameter names by their generalized patterns and count occurrences.
    
    Returns:
        Dictionary mapping pattern -> count
    """
    pattern_counts = defaultdict(int)
    for param_name in param_names:
        pattern = _generalize_param_name(param_name)
        pattern_counts[pattern] += 1
    return dict(pattern_counts)


def copy_shared_weights(base_model: nn.Module, hybrid_model: nn.Module) -> List[str]:
    """
    Copies over the shared parameters from the base_model to the hybrid model (e.g.,
    embedding layer, MLPs, LM-head). By shared parameters we mean all parameters except
    the hybrid layer parameters, which are handled separately.

    Arguments:
        base_model: The base model from which to copy the parameters.
        hybrid_model: The hybrid model to which the parameters will be copied.

    Returns:
        hybridization_candidates: A list of parameter names (strings) that are not
            initialized from the base model. 
    """
    logger.info("Copying shared weights from base model to hybrid model")

    base_model_state_dict = base_model.state_dict()
    hybrid_model_state_dict = hybrid_model.state_dict()

    hybridization_candidates = []
    copied_params = []

    # Track parameter sums
    param_sum_before = sum(p.sum().item() for p in hybrid_model_state_dict.values())

    for param_name in hybrid_model_state_dict.keys():
        if param_name in base_model_state_dict:
            hybrid_model_state_dict[param_name].copy_(base_model_state_dict[param_name])
            copied_params.append(param_name)
        else:
            hybridization_candidates.append(param_name)

    hybrid_model.load_state_dict(hybrid_model_state_dict)

    param_sum_after = sum(p.sum().item() for p in hybrid_model_state_dict.values())

    # Group copied parameters by pattern
    copied_patterns = _group_params_by_pattern(copied_params)

    # Log summary
    logger.weights_update(
        f"Copied {len(copied_params)} parameters, "
        f"{len(hybridization_candidates)} parameters remain randomly initialized"
    )

    # Log copied parameter patterns
    logger.info("Copied parameter patterns:")
    for pattern, count in sorted(copied_patterns.items()):
        logger.info(f"  {pattern}: {count} param{'s' if count > 1 else ''}")

    # Log randomly initialized parameter patterns
    if hybridization_candidates:
        random_patterns = _group_params_by_pattern(hybridization_candidates)
        logger.info("Randomly initialized parameter patterns:")
        for pattern, count in sorted(random_patterns.items()):
            logger.info(f"  {pattern}: {count} param{'s' if count > 1 else ''}")

    logger.weights_update(
        f"Parameter sum: {param_sum_before:.6f} -> {param_sum_after:.6f}"
    )

    return hybridization_candidates


def setup_layer_config(
    base_config: PretrainedConfig, hybrid_config: HybridConfig, layer_name: str
) -> None:
    """
    Layer config construction and assignment. This is a generic function which abstracts
    logic such as:

        mamba2_exp_config = hybrid_config.mamba2 or HybridMamba2Config()
        mamba2_config = construct_mamba2_config(config, mamba2_exp_config)
        config.mamba2_config = mamba2_config.to_dict()
    
    Args:
        base_config: The base model's config.
        hybrid_config: The hybrid configuration specified during the priming step.
        layer_name: Name of the layer (e.g., 'gdn')

    Returns:
        The Hybrid layer's config.
    """
    if layer_name not in LAYER_CONFIG_REGISTRY:
        raise ValueError(
            f"Unknown layer name '{layer_name}'. Valid options: {list(LAYER_CONFIG_REGISTRY.keys())}"
        )

    constructor_fn, default_config_class = LAYER_CONFIG_REGISTRY[layer_name]
    layer_exp_config = getattr(hybrid_config, layer_name) or default_config_class()
    layer_config = constructor_fn(base_config, layer_exp_config)
    setattr(base_config, f"{layer_name}_config", layer_config.to_dict())

    return layer_config


def add_hybrid_configs(
    config: PretrainedConfig, hybrid_config: HybridConfig
) -> PretrainedConfig:
    """
    Adds hybridization configs to the base model config. For example, if the
    hybrid_override_pattern string in hybrid_config specifies a Mamba2 layer,
    this function adds Mamba2 parameters to the given config.

    Args:
        config: The hybrid model's HF-style config.
        hybrid_config: The hybridization config, which specifies the hybridization pattern
            and specific hybrid layer parameters. This is typically specified via a yaml file.
    
    Returns:
        config: The updated config with additional hybrid parameters.
    """

    hybrid_override_pattern = parse_override_pattern(config.hybrid_override_pattern)
    layer_types = set(list(hybrid_override_pattern.values()))
    setup_layer_config_fn = partial(setup_layer_config, config, hybrid_config)
    for layer_type in layer_types:
        if "M2" in layer_type:
            setup_layer_config_fn("mamba2")
        elif "GDN" in layer_type:
            setup_layer_config_fn("gdn")
        elif "GKA" in layer_type:
            setup_layer_config_fn("gka")
        elif "BMF" in layer_type:
            if hybrid_config.bmojo is None:
                raise ValueError(
                    f"Layer type '{layer_type}' requires 'bmojo' config to be specified in the hybridization config."
                )
            # Construct configs for B'MOJO components
            bmojo_config = setup_layer_config_fn("bmojo")

            # Construct config for the specific SSM mixer being used
            ssm_mixer = bmojo_config.ssm_mixer.lower()
            if ssm_mixer == "mamba2":
                setup_layer_config_fn("mamba2")
            elif ssm_mixer in ["gated_deltanet", "gdn"]:
                setup_layer_config_fn("gdn")
            elif ssm_mixer in ["gated_kalmanet", "gka"]:
                setup_layer_config_fn("gka")
        elif "SWA" in layer_type:
            if hybrid_config.swa is None:
                raise ValueError(
                    f"Layer type '{layer_type}' requires 'swa' config to be specified in the hybridization config."
                )
            config.swa_config = hybrid_config.swa.to_dict()

    return config


def dequantize_fp8_model(model: PreTrainedModel, dtype: torch.dtype) -> PreTrainedModel:
    """
    Dequantize FP8 weights in a model to the target dtype.
    
    FP8 models use FP8Linear layers with weight_scale_inv factors. This function
    converts them to standard nn.Linear with properly scaled weights.
    
    Arguments:
        model: Model potentially containing FP8Linear layers
        dtype: Target dtype for dequantized weights
        
    Returns:
        Model with FP8 weights dequantized to target dtype
    """
    try:
        from transformers.integrations.finegrained_fp8 import FP8Linear
    except ImportError:
        logger.info("FP8Linear not available, skipping FP8 dequantization")
        return model

    dequantized_count = 0

    for name, module in model.named_modules():
        if isinstance(module, FP8Linear):
            # Get the scale factor
            scale_inv = getattr(module, "weight_scale_inv", None)
            if scale_inv is None:
                logger.warning(f"FP8Linear {name} has no weight_scale_inv, skipping")
                continue

            # Dequantize: convert FP8 to target dtype and apply scale
            fp8_weight = module.weight.data
            dequantized_weight = fp8_weight.to(dtype) * scale_inv.to(dtype)

            # Get bias if present
            bias_data = None
            if hasattr(module, "bias") and module.bias is not None:
                bias_data = module.bias.data.to(dtype)

            # Create replacement Linear layer
            new_linear = torch.nn.Linear(
                module.in_features,
                module.out_features,
                bias=(bias_data is not None),
                dtype=dtype,
                device=fp8_weight.device,
            )
            new_linear.weight.data.copy_(dequantized_weight)
            if bias_data is not None:
                new_linear.bias.data.copy_(bias_data)

            # Replace the module in the parent
            parent_name = ".".join(name.split(".")[:-1])
            child_name = name.split(".")[-1]
            if parent_name:
                parent = model.get_submodule(parent_name)
            else:
                parent = model
            setattr(parent, child_name, new_linear)
            dequantized_count += 1

    if dequantized_count > 0:
        logger.info(f"Dequantized {dequantized_count} FP8Linear layers to {dtype}")

    return model


def _resolve_model_path(model_name_or_path: str) -> str:
    """
    Resolves a model identifier to a local directory path. When copying auxiliary files,
    this function is used to determine the path to copy from. If the input is a local
    directory, returns it as-is. Otherwise, attempts to resolve it as a HuggingFace Hub
    model ID by looking up the cached snapshot.
    
    Arguments:
        model_name_or_path: Local path or HuggingFace Hub model ID.
        
    Returns:
        Local directory path, or None if resolution fails.
    """
    if os.path.isdir(model_name_or_path):
        return model_name_or_path
    
    # Try to resolve from HF cache
    try:
        from huggingface_hub import snapshot_download
        return snapshot_download(
            repo_id=model_name_or_path,
            local_files_only=True,  # Don't download, just find cached
        )
    except Exception:
        return None


def copy_auxiliary_files(
    source_dir: str,
    target_dir: str,
    skip_file_patterns: List[str] = [".bin", ".pt", ".pth", ".ckpt"],
    skip_file_contains: List[str] = ["safetensors"],
    skip_file_specific: List[str] = ["config.json", ".gitattributes", "README.md", "LICENSE"],
) -> None:
    """
    Copy auxiliary files from source model directory to target directory.
    
    Copies files like system prompts, tokenizer files, etc. that aren't handled
    by save_pretrained. Skips model weight files and standard config files.
    
    Arguments:
        source_dir: Source model directory path or HuggingFace Hub model ID.
        target_dir: Target directory to copy files to.
        skip_file_patterns: File extensions or exact filenames to skip. Files ending
            with or matching these patterns will not be copied.
        skip_file_contains: Substrings to match against filenames. Files containing
            any of these substrings will not be copied.
        skip_file_specific: List of specific filenames that should be should be skipped.
    """
    resolved_source = _resolve_model_path(source_dir)
    if resolved_source is None:
        logger.info(
            f"Could not resolve source path '{source_dir}', skipping auxiliary file copy"
        )
        return
    
    source_dir = resolved_source

    copied_files = []
    skipped_files = []

    for filename in os.listdir(source_dir):
        source_path = os.path.join(source_dir, filename)
        target_path = os.path.join(target_dir, filename)

        # Skip directories
        if os.path.isdir(source_path):
            continue

        # Skip specific files
        if filename in skip_file_specific:
            skipped_files.append(f"{filename} (specific omission)")
            continue

        # Skip model weight files and standard HF files
        should_skip = any(
            filename.endswith(pattern) or filename == pattern
            for pattern in skip_file_patterns
        ) or any(pattern in filename for pattern in skip_file_contains)
        if should_skip:
            skipped_files.append(f"{filename} (weight/config)")
            continue

        # Copy the file (overwriting if in overwrite_files)
        shutil.copy2(source_path, target_path)
        copied_files.append(filename)

    if copied_files:
        logger.info(
            f"Copied {len(copied_files)} files from base model: {', '.join(copied_files)}"
        )
