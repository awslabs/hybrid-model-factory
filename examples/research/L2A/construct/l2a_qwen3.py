"""Qwen3-specific L2A model construction: instantiates and initializes an L2A-Qwen3 model from a base Qwen3."""

from typing import List
from transformers import AutoModelForCausalLM
from model.l2a_qwen3.modeling_qwen3 import (
    L2AQwen3ForCausalLM,
    L2AQwen3v2Model,
    L2AQwen3PreTrainedModel,
)

from l2a_construct_utils import copy_shared_weights, copy_shared_weights_with_baseinit
from model.l2a_qwen3.configuration_qwen3 import (
    L2AQwen3Config,
)

def get_qwen3_class(base_model):
    model_class_name = type(base_model).__name__
    if "Qwen3ForCausalLM" in model_class_name:
        return L2AQwen3ForCausalLM
    elif "Qwen3Model" in model_class_name:
        return L2AQwen3v2Model
    elif "Qwen3PreTrainedModel" in model_class_name:
        return L2AQwen3PreTrainedModel
    else:
        raise ValueError(
            f"Unsupported Qwen3 model class: {model_class_name}. "
            f"Expected one of Qwen3ForCausalLM, Qwen3Model, or Qwen3PreTrainedModel."
        )

def construct_l2a_qwen3(base_model, config, l2a_config):
    """
    Arguments:
        - base_model: The base model loaded using AutoModelForCausalLM.from_pretrained.
        - config: The config of the base model
        - l2a_config: The config specifying which layers will be augmented with external memory
    """
    # Extract model class
    l2a_qwen3_class = get_qwen3_class(base_model)

    # Set hybrid model's config
    l2a_qwen3_config = L2AQwen3Config.from_dict(config.to_dict())

    # Load (randomly-initialized) memory augmented model
    l2a_model = l2a_qwen3_class(config=l2a_qwen3_config).to(
        dtype=config.torch_dtype
    )

    # Sum of randomly-initialized memory augmented model weights
    param_sum0 = sum([p.sum() for p in l2a_model.parameters()])

    # Copy shared weights from the base model to the memory augmented model
    if l2a_config.l2a_initfrombase:
        print("****Initialize augmented layers with corresponding base model layers!****")
        l2a_candidates = copy_shared_weights_with_baseinit(base_model, l2a_model, l2a_config)
    else:
        l2a_candidates = copy_shared_weights(base_model, l2a_model, l2a_config)

    # Sum of memory augmented model's weights after copying over common parameters.
    param_sum1 = sum([p.sum() for p in l2a_model.parameters()])

    print(
        f"Random model param sum: {param_sum0} | Copied param model sum: {param_sum1} | "
    )

    return l2a_model