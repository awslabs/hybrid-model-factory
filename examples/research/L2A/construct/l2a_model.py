"""CLI entry point for constructing an L2A model from a pre-trained base Transformer.

Usage:
    python l2a_model.py --config ../configs/construct_qwen3_8b.yaml
"""

import argparse
import os
import sys
import torch
import yaml
from dataclasses import dataclass

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

# Add L2A root to path so we can import from model/ package
_L2A_ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, _L2A_ROOT)
sys.path.insert(0, os.path.dirname(__file__))

import model.model_register  # noqa: F401 — triggers AutoModel registration

from l2a_qwen3 import construct_l2a_qwen3

def parse_args():
    parser = argparse.ArgumentParser(description="Parse memory augmentation parameters.")

    parser.add_argument(
        "--config",
        type=str,
        help="Path to the config containing memory augmentation parameters.",
    )

    return parser.parse_args()


@dataclass
class L2AConfig:
    base_model_name_or_path: str
    l2a_initfrombase: bool
    sigmoid_linear: bool
    output_dir: str
    sigmoid_linear_zero_init: bool
    sliding_window: int = 4095
    sigmoid_temp: float = 0.1
    sig_input_type: str = "hidden_linearproj"


def load_config(path: str) -> L2AConfig:
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return L2AConfig(**data)


def verify_l2a_config(config: L2AConfig):
    """
    Verifies that the provided arguments are valid. The config needs to specify
    the base model's name or path (base_model_name_or_path) which will be loaded
    from HuggingFace, and the directory in which to save the model (output_dir).
    """
    if config.base_model_name_or_path is None:
        raise ValueError(
            "Please provide a model name or path for the Transformer model."
        )

    if config.l2a_initfrombase is None:
        raise ValueError("Please provide a way to initialize the memory augmented layers.")

    if config.sigmoid_linear is None:
        raise ValueError("Please specify if you need a linear proj layer for sigmoid ste module.")

    if config.sigmoid_linear_zero_init:
        assert config.sigmoid_linear, "sigmoid_linear must be True to enable zero_init for sigmoid linear layer."

    if config.output_dir is None:
        raise ValueError("Please provide an output directory for the memory augmented model.")


def load_pretrained_model(model_name_or_path):
    """
    Loads the pre-trained transformer model along with its config and tokenizer.
    """
    config = AutoConfig.from_pretrained(model_name_or_path)

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path, torch_dtype=config.torch_dtype, trust_remote_code=True
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path, trust_remote_code=True
    )

    return model, tokenizer, config


def l2a_model(l2a_config):
    # Load model
    model_name_or_path = l2a_config.base_model_name_or_path
    print(f"Loading model from {model_name_or_path}")
    transformer_model, tokenizer, config = load_pretrained_model(model_name_or_path)

    print(f"=" * 25 + " Base Transformer Model " + "=" * 25)
    print(transformer_model)

    # Add L2A configuration
    config.l2a_initfrombase = l2a_config.l2a_initfrombase
    config.sigmoid_linear = l2a_config.sigmoid_linear
    config.sliding_window = l2a_config.sliding_window
    config.sigmoid_temp = l2a_config.sigmoid_temp
    config.sig_input_type = l2a_config.sig_input_type

    model_type = transformer_model.config.model_type

    if "qwen3" in model_type:
        l2a_model = construct_l2a_qwen3(
            base_model=transformer_model, config=config, l2a_config=l2a_config
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}. Only qwen3 is supported.")

    print(f"=" * 25 + " Memory Augmented Model " + "=" * 25)
    print(l2a_model)
    print(l2a_model.config)

    # Save
    output_dir = l2a_config.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Before save_pretrained, verify a few key weights match
    base_sd = transformer_model.state_dict()
    l2a_sd = l2a_model.state_dict()
    for key in ["model.layers.0.mlp.gate_proj.weight", "lm_head.weight"]:
        if key in base_sd and key in l2a_sd:
            match = torch.allclose(base_sd[key], l2a_sd[key].to(base_sd[key].dtype))
            print(f"PRE-SAVE CHECK: {key} match={match}, base_sum={base_sd[key].float().sum():.4f}, l2a_sum={l2a_sd[key].float().sum():.4f}")


    l2a_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"Saved to {output_dir}")


if __name__ == "__main__":
    # Example usage: python l2a_model.py --config xxx.yaml

    args = parse_args()
    l2a_config = load_config(args.config)
    verify_l2a_config(l2a_config)
    l2a_model(l2a_config)