"""
Loads the backbone of a decision model (an hmf_decision checkpoint written by hmf decision-init or by training) into
mlx-lm's text model of the same architecture, for training and running decision models on Apple silicon with MLX.

A decision checkpoint stores its backbone's weights under `model.`, which is also where mlx-lm's Qwen3 and Qwen3.5
text models keep their decoders, so the weights load by name. The pointer head's weights, under `decision_head.`,
are returned separately (see model.py). Only Qwen3 and Qwen3.5 backbones are supported. HMF's hybrid models, whose
layers mlx-lm does not have, are refused.
"""

import glob
import json
import os
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, Iterator, Optional

import mlx.core as mx
from mlx import nn
from mlx.utils import tree_flatten

from .gated_delta import install_training_kernels


__all__ = ["DTYPES", "Backbone", "load_backbone"]

DTYPES = {"bfloat16": mx.bfloat16, "float16": mx.float16, "float32": mx.float32}
# The mlx-lm module that builds each supported text model type
MLX_MODULES = {"qwen3": "qwen3", "qwen3_5_text": "qwen3_5"}
DECISION_MODEL_TYPE = "hmf_decision"
HEAD_PREFIX = "decision_head."


@dataclass
class Backbone:
    """
    A decision model's backbone in MLX, with the rest of its checkpoint.

    Args:
        decoder: mlx-lm's decoder, which maps token ids of shape [batch, seq_len] to the final normalized hidden
            states of shape [batch, seq_len, hidden_size].
        head_weights: The pointer head's weights, by name within the head (e.g. "q.weight"), in float32.
        config: The decision model's configuration (config.json).
    """

    decoder: nn.Module
    head_weights: Dict[str, mx.array]
    config: Dict[str, Any]

    @property
    def text_config(self) -> Dict[str, Any]:
        return self.config["text_config"]


def _layer_pattern(text_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Returns the text configuration with the full_attention_interval that its layer_types imply. mlx-lm builds the
    layers of a Qwen3.5 model from full_attention_interval (every k-th layer is a full attention layer), while
    transformers reads layer_types, which decision checkpoints list. A pattern that no interval describes is refused
    rather than guessed.
    """
    types = text_config.get("layer_types")
    if not types:
        return text_config
    full = [i for i, layer_type in enumerate(types) if layer_type == "full_attention"]
    interval = full[0] + 1 if full else len(types) + 1
    if [i for i in range(len(types)) if (i + 1) % interval == 0] != full:
        raise ValueError(
            f"The layer types {types} do not repeat a full attention layer every k layers, which is the only "
            "pattern mlx-lm can build."
        )
    return {**text_config, "full_attention_interval": interval}


def _rope_settings(text_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Returns the text configuration with rope_theta and rope_scaling at its top level, where mlx-lm's Qwen3 reads
    them. Configurations saved by transformers 5 keep both in rope_parameters instead, whose rope_type is "default"
    when the rotary embedding is not scaled.
    """
    rope = text_config.get("rope_parameters")
    if not rope or text_config.get("rope_theta") is not None:
        return text_config
    scaling = None if rope.get("rope_type", "default") == "default" else rope
    return {**text_config, "rope_theta": rope["rope_theta"], "rope_scaling": scaling}


@contextmanager
def _exact_norm_shift(module: Any) -> Iterator[None]:
    """
    Upcasts the 1-D weights to float32 before the sanitize methods of an mlx-lm module run, for as long as the
    context lasts. Qwen3.5's norms store w and compute with 1 + w, which mlx-lm's sanitize adds to the stored
    tensor. In bfloat16, that sum would be rounded (about 2e-3 relative error per norm), while transformers computes
    it in float32, so the 1-D weights (norms, biases, and gates, never matrices) are upcast first and rounded once
    afterwards, when the backbone is cast to its dtype.
    """
    classes = [cls for cls in vars(module).values() if isinstance(cls, type) and "sanitize" in vars(cls)]
    originals = {cls: cls.sanitize for cls in classes}

    def upcasting(original):
        def sanitize(self, weights):
            weights = {
                key: value.astype(mx.float32) if value.ndim == 1 and mx.issubdtype(value.dtype, mx.floating) else value
                for key, value in weights.items()
            }
            return original(self, weights)

        return sanitize

    for cls, original in originals.items():
        cls.sanitize = upcasting(original)
    try:
        yield
    finally:
        for cls, original in originals.items():
            cls.sanitize = original


def _load_weights(model_path: str) -> Dict[str, mx.array]:
    files = sorted(glob.glob(os.path.join(model_path, "*.safetensors")))
    if not files:
        raise FileNotFoundError(f"No safetensors files in {model_path}.")
    weights = {}
    for file in files:
        weights.update(mx.load(file))
    return weights


def load_backbone(model_path: str, dtype: str = "bfloat16", config: Optional[Dict[str, Any]] = None) -> Backbone:
    """
    Loads a decision model's backbone into mlx-lm's text model. The backbone is cast to dtype, except for the
    weights that mlx-lm keeps in float32 (e.g. Qwen3.5's A_log), and its Gated DeltaNet layers, if any, use the
    chunked rule of gated_delta.py in training mode. The decoder's parameters are frozen, so that training updates
    only what is unfrozen afterwards (e.g. LoRA adapters).

    Args:
        model_path: The folder of a decision model (an hmf_decision checkpoint).
        dtype: The dtype of the backbone, one of DTYPES. Defaults to "bfloat16".
        config: The decision model's configuration. Defaults to the folder's config.json.

    Returns:
        The backbone, with the pointer head's weights and the configuration.
    """
    import importlib

    if dtype not in DTYPES:
        raise ValueError(f"The MLX backend has no dtype {dtype!r}. Use one of {sorted(DTYPES)}.")
    if config is None:
        with open(os.path.join(model_path, "config.json"), encoding="utf-8") as f:
            config = json.load(f)
    if config.get("model_type") != DECISION_MODEL_TYPE:
        raise ValueError(
            f"{model_path} is not a decision model (its model_type is {config.get('model_type')!r}, not "
            f"{DECISION_MODEL_TYPE!r}). Convert a causal LM with hmf decision-init first."
        )
    text_config = config["text_config"]
    model_type = text_config.get("model_type")
    if model_type not in MLX_MODULES:
        raise ValueError(
            f"The MLX backend supports decision models on {sorted(MLX_MODULES)} backbones, not {model_type!r}."
        )

    module = importlib.import_module(f"mlx_lm.models.{MLX_MODULES[model_type]}")
    text_config = _layer_pattern(text_config)
    if model_type == "qwen3_5_text":
        model = module.TextModel(module.TextModelArgs.from_dict(text_config))
    else:
        model = module.Model(module.ModelArgs.from_dict(_rope_settings(text_config)))

    weights = _load_weights(model_path)
    head_weights = {
        key[len(HEAD_PREFIX):]: value.astype(mx.float32)
        for key, value in weights.items()
        if key.startswith(HEAD_PREFIX)
    }
    backbone_weights = {key: value for key, value in weights.items() if key.startswith("model.")}
    unknown = sorted(set(weights) - set(backbone_weights) - {HEAD_PREFIX + key for key in head_weights})
    if unknown:
        raise ValueError(f"{model_path} has weights that are neither the backbone's nor the head's: {unknown[:5]}")
    with _exact_norm_shift(module):
        backbone_weights = model.sanitize(backbone_weights)
    # Every parameter of the decoder must come from the checkpoint. The LM head, which a decision model has none
    # of, is the only parameter left out.
    model.model.load_weights(
        [(key[len("model."):], value) for key, value in backbone_weights.items()], strict=True
    )
    decoder = model.model

    keep_float32 = getattr(model, "cast_predicate", None)
    target = DTYPES[dtype]
    cast = [
        (path, value.astype(target))
        for path, value in tree_flatten(decoder.parameters())
        if mx.issubdtype(value.dtype, mx.floating)
        and value.dtype != target
        and (keep_float32 is None or keep_float32(f"model.{path}"))
    ]
    if cast:
        decoder.load_weights(cast, strict=False)
    install_training_kernels(decoder)
    decoder.freeze()
    mx.eval(decoder.parameters(), head_weights)
    return Backbone(decoder=decoder, head_weights=head_weights, config=config)
