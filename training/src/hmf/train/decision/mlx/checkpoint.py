"""
Saves and loads the checkpoints of decision models trained in MLX, in the formats HMF's PyTorch code reads, so that
labeling, evaluation, and serving (which run in PyTorch) use them unchanged.

    - LoRA: a PEFT adapter, with adapter_config.json, written by PEFT's own LoraConfig, and adapter_model.safetensors.
      The keys are PEFT's, base_model.model.<path>.lora_A.weight and .lora_B.weight for each adapted layer and
      base_model.model.decision_head.<name> for the pointer head, which PEFT saves with the adapter
      (modules_to_save). mlx-lm holds A as [in_features, rank] and B as [rank, out_features], while PEFT holds the
      transposes, A as [rank, in_features] and B as [out_features, rank].
    - Full fine-tuning: a decision model (an hmf_decision checkpoint), with config.json and model.safetensors, which
      holds the backbone's weights under `model.`, in transformers' layout, and the head's under `decision_head.`.

load_adapter reads PEFT adapters, whether saved by MLX or by PyTorch.
"""

import json
import os
from typing import Any, Dict, Iterable, Optional

import mlx.core as mx
from mlx.utils import tree_flatten

from .backbone import DTYPES, _load_weights, to_hf_layout
from .lora import HEAD_NAME, attach_lora, lora_layers
from .model import DecisionModelMLX


__all__ = ["ADAPTER_CONFIG", "ADAPTER_WEIGHTS", "load_adapter", "save_adapter", "save_full_model"]

ADAPTER_CONFIG = "adapter_config.json"
ADAPTER_WEIGHTS = "adapter_model.safetensors"
FULL_WEIGHTS = "model.safetensors"
PEFT_PREFIX = "base_model.model."
# The metadata transformers and PEFT expect in a safetensors file saved for PyTorch
SAFETENSORS_METADATA = {"format": "pt"}


def save_adapter(
    model: DecisionModelMLX,
    output_dir: str,
    targets: Iterable[str],
    rank: int,
    alpha: float,
    dropout: float = 0.0,
    use_rslora: bool = False,
    base_model_name_or_path: Optional[str] = None,
) -> None:
    """
    Saves a decision model's LoRA adapters and pointer head as a PEFT adapter, in float32.

    Args:
        model: The decision model, with its adapters attached.
        output_dir: The folder to save the adapter in.
        targets: The names of the adapted layers (lora_target).
        rank: The rank of the adapters.
        alpha: The scaling numerator of the adapters.
        dropout: The dropout of the adapters. Defaults to 0.
        use_rslora: Whether the adapters are scaled by alpha / sqrt(rank). Defaults to False.
        base_model_name_or_path: The decision model the adapter was trained on. Defaults to none.
    """
    from peft import LoraConfig

    layers = lora_layers(model)
    if not layers:
        raise ValueError("The model has no LoRA adapters to save.")
    os.makedirs(output_dir, exist_ok=True)
    LoraConfig(
        r=rank, lora_alpha=alpha, lora_dropout=dropout, target_modules=sorted(set(targets)), use_rslora=use_rslora,
        modules_to_save=[HEAD_NAME], task_type=None, base_model_name_or_path=base_model_name_or_path,
    ).save_pretrained(output_dir)

    tensors = {}
    for path, layer in layers:
        tensors[f"{PEFT_PREFIX}{path}.lora_A.weight"] = layer.lora_a.T.astype(mx.float32)
        tensors[f"{PEFT_PREFIX}{path}.lora_B.weight"] = layer.lora_b.T.astype(mx.float32)
    for name, value in tree_flatten(model.decision_head.parameters()):
        tensors[f"{PEFT_PREFIX}{HEAD_NAME}.{name}"] = value.astype(mx.float32)
    mx.save_safetensors(os.path.join(output_dir, ADAPTER_WEIGHTS), tensors, metadata=SAFETENSORS_METADATA)


def load_adapter(model: DecisionModelMLX, adapter_dir: str) -> Dict[str, Any]:
    """
    Loads a PEFT adapter into a decision model, saved by MLX or by PyTorch. Its LoRA layers are attached with the
    adapter's settings and loaded, in float32, and the pointer head is loaded from the adapter when it is there (as
    modules_to_save).

    Args:
        model: The decision model, without adapters.
        adapter_dir: The folder of the adapter.

    Returns:
        The adapter's configuration (adapter_config.json).
    """
    with open(os.path.join(adapter_dir, ADAPTER_CONFIG), encoding="utf-8") as f:
        config = json.load(f)
    if config.get("peft_type") != "LORA":
        raise ValueError(f"{adapter_dir} is a {config.get('peft_type')} adapter. The MLX backend reads LoRA adapters.")
    if config.get("use_dora"):
        raise ValueError(f"{adapter_dir} is a DoRA adapter, which the MLX backend does not support.")
    targets = config["target_modules"]
    if isinstance(targets, str):
        raise ValueError(
            f"{adapter_dir} selects its layers by the pattern {targets!r}. The MLX backend reads adapters whose "
            "target_modules is a list of layer names."
        )
    attach_lora(
        model, targets, rank=config["r"], alpha=config["lora_alpha"], dropout=config.get("lora_dropout", 0.0),
        use_rslora=config.get("use_rslora", False),
    )

    tensors = mx.load(os.path.join(adapter_dir, ADAPTER_WEIGHTS))
    used = set()
    for path, layer in lora_layers(model):
        keys = [f"{PEFT_PREFIX}{path}.lora_{name}.weight" for name in ("A", "B")]
        missing = [key for key in keys if key not in tensors]
        if missing:
            raise ValueError(f"{adapter_dir} has no weights for the LoRA layer {path} ({missing[0]}).")
        layer.lora_a = tensors[keys[0]].T.astype(mx.float32)
        layer.lora_b = tensors[keys[1]].T.astype(mx.float32)
        used.update(keys)

    head_prefix = f"{PEFT_PREFIX}{HEAD_NAME}."
    head = {
        key[len(head_prefix):]: value.astype(mx.float32)
        for key, value in tensors.items()
        if key.startswith(head_prefix)
    }
    if head:
        model.decision_head.load_weights(list(head.items()), strict=True)
        used.update(head_prefix + name for name in head)
    unused = sorted(set(tensors) - used)
    if unused:
        raise ValueError(f"{adapter_dir} has weights that match no layer of the model: {unused[:5]}")
    mx.eval(model.parameters())
    return config


def save_full_model(
    model: DecisionModelMLX,
    output_dir: str,
    source_dir: str,
    config: Dict[str, Any],
    dtype: str = "bfloat16",
) -> None:
    """
    Saves a fully fine-tuned decision model as a decision model (an hmf_decision checkpoint). The trained parameters
    are saved from the model, the backbone's in dtype and the head's in float32. The frozen parameters are copied
    from the source checkpoint unchanged, because the model holds some of them only in a rounded form (in bfloat16,
    Qwen3.5's norms hold 1 + w, from which w cannot be recovered exactly).

    Args:
        model: The decision model, without LoRA adapters (merge them first, see lora.merge_lora).
        output_dir: The folder to save the model in.
        source_dir: The decision model the training started from.
        config: The decision model's configuration, saved as config.json.
        dtype: The dtype of the saved backbone, one of backbone.DTYPES. Defaults to "bfloat16".
    """
    if lora_layers(model):
        raise ValueError("The model has LoRA adapters. Save them with save_adapter, or merge them first.")
    if dtype not in DTYPES:
        raise ValueError(f"The MLX backend has no dtype {dtype!r}. Use one of {sorted(DTYPES)}.")
    trained = dict(tree_flatten(model.trainable_parameters()))
    source = _load_weights(source_dir)

    backbone = {f"model.{path}": value for path, value in tree_flatten(model.model.parameters())}
    backbone = to_hf_layout(backbone, config["text_config"]["model_type"])
    tensors = {}
    for key, value in backbone.items():
        if key in trained:
            tensors[key] = value.astype(DTYPES[dtype])
        elif key in source:
            tensors[key] = source[key]
        else:
            raise ValueError(f"The frozen parameter {key} is not in the source checkpoint {source_dir}.")
    for name, value in tree_flatten(model.decision_head.parameters()):
        tensors[f"{HEAD_NAME}.{name}"] = value.astype(mx.float32)

    os.makedirs(output_dir, exist_ok=True)
    mx.save_safetensors(os.path.join(output_dir, FULL_WEIGHTS), tensors, metadata=SAFETENSORS_METADATA)
    with open(os.path.join(output_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump({**config, "dtype": dtype}, f, indent=2, sort_keys=True)
