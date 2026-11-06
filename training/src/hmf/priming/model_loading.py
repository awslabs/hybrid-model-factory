"""
Loading of the pretrained Transformer models that priming and decision-init start from.

This module imports no hybrid layer, so that decision-init can convert transformers models on
machines without the CUDA-only libraries that the hybrid layers need (e.g., a Mac).
"""

from typing import Tuple

import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    PretrainedConfig,
)

from .hybridize_logger import get_logger
from .utils import dequantize_fp8_model


logger = get_logger(__name__)

# Mapping from string dtype names to torch dtypes
DTYPE_MAP = {
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
    "fp16": torch.float16,
    "float16": torch.float16,
    "fp32": torch.float32,
    "float32": torch.float32,
}


def load_pretrained_model(
    model_name_or_path: str,
    dtype: torch.dtype = torch.bfloat16,
) -> Tuple[PreTrainedModel, PreTrainedTokenizer, PretrainedConfig]:
    """
    Loads a pre-trained Hugging Face Transformer model along with its config and tokenizer.

    Arguments:
        model_name_or_path: Path or name of the pre-trained transformer model.
        dtype: The torch dtype to load the model in. Defaults to bfloat16.
            This is important for FP8 models--specifying a dtype forces dequantization
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
