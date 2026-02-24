"""
This file contains the code used to convert a Transformer model to a Hybrid model.
It can be used via the CLI:

    hmf prime-init path/to/hybrid_model_config.yaml

Or programmatically:

    from hmf.priming import load_config, verify_hybrid_config, hybridize_model

    config = load_config("path/to/config.yaml")
    verify_hybrid_config(config)
    hybridize_model(config)

"""
import hmf.model.hybrid_zoo.models.model_register  # Registers hybrid models and suppresses warnings. Must be first

import argparse
import json
import os
from typing import Tuple, Optional

import yaml
from .hybridize_llama import construct_hybrid_llama
from .hybridize_ministral3 import construct_hybrid_ministral3
from .hybridize_qwen import construct_hybrid_qwen
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    PretrainedConfig,
)

import torch

from ..model.hybrid_zoo.layers.hybrid_dataclasses import (
    HybridBMOJOConfig,
    HybridGatedDeltaNetConfig,
    HybridMamba2Config,
    HybridGKAConfig,
    HybridSWAConfig,
    HybridConfig,
)

from .utils import copy_auxiliary_files, dequantize_fp8_model

# Mapping from string dtype names to torch dtypes
DTYPE_MAP = {
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
    "fp16": torch.float16,
    "float16": torch.float16,
    "fp32": torch.float32,
    "float32": torch.float32,
}

from hmf.model.hybrid_zoo.models.utils import LAYER_TYPE_PATTERNS

from .hybridize_logger import get_logger

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parse hybridization parameters.")

    parser.add_argument(
        "--config",
        type=str,
        help="Path to the config containing hybridization parameters.",
    )

    return parser.parse_args()


def load_config(path: str) -> HybridConfig:
    """
    Given the path to a yaml file containing arguments like those found in HybridConfig,
    loads the file and copies the arguments to an instance of HybridConfig.

    Arguments:
        path: The path to the yaml file.
    
    Returns:
        An instance of HybridConfig with the arguments specified in the @path file.
    """
    with open(path, "r") as f:
        data = yaml.safe_load(f)

    # Map of field names to their dataclass types
    nested_configs = {
        "mamba2": HybridMamba2Config,
        "bmojo": HybridBMOJOConfig,
        "gdn": HybridGatedDeltaNetConfig,
        "gka": HybridGKAConfig,
        "swa": HybridSWAConfig,
    }

    # Convert nested dicts to dataclass instances
    for field_name, dataclass_type in nested_configs.items():
        if field_name in data and data[field_name] is not None:
            data[field_name] = dataclass_type(**data[field_name])

    # Handle dtype field - convert string to torch dtype
    if "dtype" in data and data["dtype"] is not None:
        dtype_str = data["dtype"].lower()
        if dtype_str not in DTYPE_MAP:
            raise ValueError(
                f"Unknown dtype '{data['dtype']}'. Supported values: {list(DTYPE_MAP.keys())}"
            )
        data["dtype"] = DTYPE_MAP[dtype_str]

    return HybridConfig(**data)


def get_num_model_layers(model_name_or_path: str, config: Optional[PretrainedConfig] = None) -> int:
    """Returns the number of layers in the specified model."""
    if config is None:
        config = AutoConfig.from_pretrained(model_name_or_path)
    if hasattr(config, "num_hidden_layers"):
        return config.num_hidden_layers
    elif hasattr(config, "n_layer"):
        return config.n_layer
    elif hasattr(config, "text_config"):
        text_config = config.text_config
        return get_num_model_layers(model_name_or_path, config=text_config)
    else:
        raise ValueError(f"Unknown layer attribute for {config.__class__.__name__}")


def verify_hybrid_config(config: HybridConfig):
    """
    Verifies that the provided arguments are valid. In particular, the hybridization
    config needs to specify the base model's name or path (base_model_name_or_path)
    which will be loaded from HuggingFace. The config must also specify a hybridization
    pattern (hybrid_override_pattern), e.g., "*-M2-M2-..." specifying which layers will
    be attention (*) and which will be the "linear" layer (e.g., M2). Finally, the
    directory in which to save the model (output_dir) must be specified.

    Arguments:
        config: An instance of HybridConfig. Typically the output of @load_config.
    """
    if config.base_model_name_or_path is None:
        raise ValueError(
            "Please provide a model name or path for the Transformer model."
        )

    if config.hybrid_override_pattern is None:
        raise ValueError("Please provide a layer hybridization pattern.")

    # Verify that the number of layers in the hybrid pattern matches the number of layers of the model
    hybrid_layer_patterns = config.hybrid_override_pattern.split("-")
    num_layers_in_pattern = len(hybrid_layer_patterns)
    num_model_layers = get_num_model_layers(config.base_model_name_or_path)
    if num_model_layers != num_layers_in_pattern:
        raise ValueError(
            f"{config.base_model_name_or_path} has {num_model_layers} layers, but hybrid_override_pattern specified {num_layers_in_pattern}."
        )

    # Make sure the layers specified in the override pattern are supported
    for layer_symbol in hybrid_layer_patterns:
        if not layer_symbol in LAYER_TYPE_PATTERNS:
            raise ValueError(
                f"Layer pattern in config.hybrid_override_pattern must be one of {list(LAYER_TYPE_PATTERNS)}; got {layer_symbol}."
            )

    if config.output_dir is None:
        raise ValueError("Please provide an output directory for the hybridized model.")


def load_pretrained_model(
    model_name_or_path: str,
    dtype: torch.dtype = torch.bfloat16,
) -> Tuple[PreTrainedModel, PreTrainedTokenizer, PretrainedConfig]:
    """
    Loads a pre-trained Hugging Face Transformer model along with its config and tokenizer.

    Arguments:
        model_name_or_path: Path or name of the pre-trained transformer model.
        dtype: The torch dtype to load the model in. Defaults to bfloat16.
            This is important for FP8 models - specifying a dtype forces dequantization
            of FP8 weights to the target precision.

    Returns:
        model: The loaded Hugging Face model.
        tokenizer: The tokenizer associated with the model.
        config: The configuration of the model.
    """
    config = AutoConfig.from_pretrained(model_name_or_path)

    # Check if source model uses quantization (e.g., FP8)
    has_quantization = hasattr(config, "quantization_config") and config.quantization_config is not None
    if has_quantization:
        quant_method = getattr(config.quantization_config, "quant_method", "unknown")
        logger.info(f"Detected quantized model (quant_method={quant_method})")
        logger.info(f"Will dequantize weights to {dtype}")

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path, dtype=dtype, trust_remote_code=True
    )
    
    # Explicitly dequantize FP8 weights if present
    if has_quantization:
        model = dequantize_fp8_model(model, dtype)

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path, trust_remote_code=True
    )

    return model, tokenizer, config


def hybridize_model(hybrid_config: HybridConfig) -> None:
    """
    Loads a Transformer model and hybridizes it by replacing its Attention layers with layers
    specified in hybrid_config. The model is saved in the directory specified by hybrid_config.
    
    Arguments:
        hybrid_config: The config containing the arguments specified above. Typically the output
            of load_config.
    """
    logger.section("Model Hybridization")

    # Load base Transformer model
    model_name_or_path = hybrid_config.base_model_name_or_path
    dtype = hybrid_config.dtype or torch.bfloat16  # Default to bf16 if not specified
    logger.info(f"Loading base model from: {model_name_or_path}")
    logger.info(f"Target dtype: {dtype}")
    transformer_model, tokenizer, config = load_pretrained_model(model_name_or_path, dtype=dtype)

    base_num_params = sum([p.numel() for p in transformer_model.parameters()])
    logger.info(f"Base model loaded: {base_num_params:,} parameters")
    logger.info(f"Model type: {transformer_model.config.model_type}")
    logger.info(f"Hybridization pattern: {hybrid_config.hybrid_override_pattern}")

    # Add hybridization pattern from hybrid_config to the base model's config
    config.hybrid_override_pattern = hybrid_config.hybrid_override_pattern

    # Construct hybridized model
    model_type = transformer_model.config.model_type
    if model_type == "qwen3_next":
        # Qwen3-Next is a special case: it's already a hybrid model (attention + GDN)
        # and requires specialized conversion logic
        from .convert_qwen3_next import construct_hybrid_qwen3_next
        hybrid_model = construct_hybrid_qwen3_next(
            base_model=transformer_model,
            config=config,
            hybrid_config=hybrid_config,
        )
    elif model_type == "qwen3_5_moe_text":
        # Qwen3.5-MoE is also already a hybrid model (attention + GDN)
        from .convert_qwen3_5_moe import construct_hybrid_qwen3_5_moe
        hybrid_model = construct_hybrid_qwen3_5_moe(
            base_model=transformer_model,
            config=config,
            hybrid_config=hybrid_config,
        )
    elif "qwen" in model_type:
        if "qwen2" in model_type:
            version = 2
        elif "qwen3" in model_type:
            version = 3
        else:
            raise ValueError(
                f"Unsupported Qwen version in model_type '{model_type}'. "
                f"Supported versions: qwen2, qwen3"
            )

        is_moe = "moe" in model_type
        hybrid_model = construct_hybrid_qwen(
            base_model=transformer_model,
            config=config,
            hybrid_config=hybrid_config,
            version=version,
            is_moe=is_moe,
        )
    elif model_type == "llama":
        hybrid_model = construct_hybrid_llama(
            base_model=transformer_model,
            config=config,
            hybrid_config=hybrid_config,
        )
    elif model_type == "ministral3":
        hybrid_model = construct_hybrid_ministral3(
            base_model=transformer_model,
            config=config,
            hybrid_config=hybrid_config,
        )
    else:
        raise ValueError(
            f"Model type '{model_type}' is currently not supported for hybridization. "
            f"Currently supported model types: qwen2, qwen3, qwen3_next, qwen3_5_moe_text, llama, ministral3."
        )

    # Calculate parameter statistics
    hybrid_num_params = sum([p.numel() for p in hybrid_model.parameters()])
    param_diff = hybrid_num_params - base_num_params
    param_diff_pct = (param_diff / base_num_params) * 100

    # Save
    output_dir = hybrid_config.output_dir
    logger.info(f"Saving hybrid model to: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    hybrid_model.save_pretrained(output_dir, max_shard_size="5GB")
    tokenizer.save_pretrained(output_dir)

    # Remove quantization_config if present (weights are dequantized during hybridization)
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, "r") as f:
        saved_config = json.load(f)
    if "quantization_config" in saved_config:
        logger.info("Removing quantization_config from saved model (weights are no longer quantized)")
        del saved_config["quantization_config"]
        with open(config_path, "w") as f:
            json.dump(saved_config, f, indent=2)

    # Copy auxiliary files from source model (system prompts, tokenizer files, etc.)
    if hybrid_config.copy_auxiliary_files:
        copy_auxiliary_files(model_name_or_path, output_dir)

    # Final summary
    logger.summary(
        f"Hybridization complete:\n"
        f"  Base model params:   {base_num_params:,}\n"
        f"  Hybrid model params: {hybrid_num_params:,}\n"
        f"  Parameter change:    {param_diff:+,} ({param_diff_pct:+.2f}%)\n"
        f"  Saved to:            {output_dir}"
    )


if __name__ == "__main__":
    args = parse_args()
    hybrid_config = load_config(args.config)
    verify_hybrid_config(hybrid_config)
    hybridize_model(hybrid_config)
