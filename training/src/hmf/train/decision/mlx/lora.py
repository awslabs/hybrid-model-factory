"""
LoRA adapters and trainable parameters for decision models in MLX, which mirror what HMF sets up for PyTorch training
(model/adapter.py). With LoRA, the backbone's linear layers get adapters, and the pointer head is trained in full and
saved with the adapters (PEFT's modules_to_save). With full fine-tuning, every parameter is trained, or with
hybrid_learnable_params only those whose names contain one of its comma-separated parts (e.g., decision_head, to
train the head alone).

LoRA uses mlx-lm's LoRALinear layers, which hold A as [in_features, rank] and B as [rank, out_features] (PEFT holds
the transposes, see checkpoint.py) and add scale * (x A) B to the base layer's output. A starts uniform in plus or
minus 1 / sqrt(in_features), which is PEFT's default distribution, and B starts at 0, so a new adapter leaves the
model's answers unchanged. Their output is scaled by alpha / rank, or alpha / sqrt(rank) with rsLoRA, as in PEFT.
"""

import math
from typing import List, Optional, Sequence

import mlx.core as mx
from mlx import nn
from mlx.utils import tree_flatten, tree_unflatten

from .backbone import KeyedDropout
from .model import DecisionModelMLX


__all__ = ["LoRALinear", "attach_lora", "lora_layers", "lora_scale", "lora_targets", "merge_lora", "set_trainable"]

HEAD_NAME = "decision_head"
# The finetuning types the MLX backend supports
FINETUNING_TYPES = ("lora", "full")


def _lora_linear_class() -> type:
    from mlx_lm.tuner.lora import LoRALinear

    return LoRALinear


LoRALinear = _lora_linear_class()


def lora_scale(rank: int, alpha: float, use_rslora: bool = False) -> float:
    """The factor of the adapters' output, which is alpha / rank, or alpha / sqrt(rank) with rsLoRA."""
    return alpha / (math.sqrt(rank) if use_rslora else rank)


def lora_targets(model: DecisionModelMLX, lora_target: Sequence[str]) -> List[str]:
    """
    Returns the names of the layers to adapt. ["all"] stands for the names of all the backbone's linear layers, as
    find_all_linear_modules gives for PyTorch (the pointer head, which is trained in full, has no adapters). Other
    values are returned as they are.

    Args:
        model: The decision model.
        lora_target: The lora_target argument, as a list of layer names.

    Returns:
        The sorted names of the layers to adapt (the last part of each layer's path, e.g., q_proj).
    """
    if list(lora_target) != ["all"]:
        return sorted(set(lora_target))
    return sorted(
        {path.rsplit(".", 1)[-1] for path, module in model.model.named_modules() if isinstance(module, nn.Linear)}
    )


def lora_layers(model: nn.Module) -> List[tuple]:
    """The (path, layer) pairs of a model's LoRA layers."""
    return [(path, module) for path, module in model.named_modules() if isinstance(module, LoRALinear)]


def attach_lora(
    model: DecisionModelMLX,
    targets: Sequence[str],
    rank: int,
    alpha: float,
    dropout: float = 0.0,
    use_rslora: bool = False,
) -> List[str]:
    """
    Wraps the backbone's linear layers whose names are in targets with LoRA layers, whose A and B are in float32.

    Args:
        model: The decision model.
        targets: The names of the layers to adapt (the last part of each layer's path, e.g., q_proj).
        rank: The rank of the adapters.
        alpha: The scaling numerator of the adapters.
        dropout: Dropout on the adapters' input, in training mode. Defaults to 0.
        use_rslora: Whether to scale by alpha / sqrt(rank) rather than alpha / rank. Defaults to False.

    Returns:
        The paths of the adapted layers within the backbone, sorted.
    """
    wanted = set(targets)
    scale = lora_scale(rank, alpha, use_rslora)
    swaps = []
    for path, module in model.model.named_modules():
        if path.rsplit(".", 1)[-1] in wanted and isinstance(module, nn.Linear):
            layer = LoRALinear.from_base(module, r=rank, dropout=dropout, scale=scale)
            layer.dropout = KeyedDropout(dropout)  # the same masks in checkpointed layers' reruns
            layer.lora_a = layer.lora_a.astype(mx.float32)
            layer.lora_b = layer.lora_b.astype(mx.float32)
            swaps.append((path, layer))
    if not swaps:
        raise ValueError(f"No linear layer of the backbone is named one of the LoRA targets {sorted(wanted)}.")
    model.model.update_modules(tree_unflatten(swaps))
    return sorted(path for path, _ in swaps)


def merge_lora(model: nn.Module) -> int:
    """
    Folds each LoRA layer's update into its base layer's weights, in the base layer's dtype, and replaces the LoRA
    layer by the merged linear layer, as PEFT's merge_and_unload does.

    Returns:
        The number of merged layers.
    """
    layers = lora_layers(model)
    if layers:
        model.update_modules(tree_unflatten([(path, layer.fuse()) for path, layer in layers]))
    return len(layers)


def set_trainable(
    model: DecisionModelMLX,
    finetuning_type: str,
    hybrid_learnable_params: Optional[str] = None,
) -> List[str]:
    """
    Unfreezes the parameters to train, freezes the others, and casts the trainable ones to float32, as HMF keeps
    trainable parameters by default. With LoRA, the trainable parameters are the adapters' A and B and the pointer
    head. With full fine-tuning, they are all parameters, or those whose paths contain one of the comma-separated
    parts of hybrid_learnable_params.

    Args:
        model: The decision model, with its adapters attached for LoRA.
        finetuning_type: "lora" or "full".
        hybrid_learnable_params: For full fine-tuning, comma-separated parts of the names of the parameters to
            train. Defaults to all parameters.

    Returns:
        The paths of the trainable parameters, sorted.
    """
    if finetuning_type not in FINETUNING_TYPES:
        raise ValueError(
            f"The MLX backend supports finetuning_type {' and '.join(FINETUNING_TYPES)}, not {finetuning_type!r}."
        )
    if finetuning_type == "lora" and hybrid_learnable_params is not None:
        raise ValueError("hybrid_learnable_params cannot be combined with LoRA.")

    model.freeze()
    if finetuning_type == "lora":
        if not lora_layers(model):
            raise ValueError("finetuning_type lora needs adapters, but the model has none (see attach_lora).")
        for _, layer in lora_layers(model):
            layer.unfreeze(keys=["lora_a", "lora_b"], recurse=False)
        model.decision_head.unfreeze()
    elif hybrid_learnable_params is None:
        model.unfreeze()
    else:
        parts = [part.strip() for part in hybrid_learnable_params.split(",") if part.strip()]
        paths = [path for path, _ in tree_flatten(model.parameters()) if any(part in path for part in parts)]
        if not paths:
            raise ValueError(f"No parameter's name contains one of hybrid_learnable_params {parts}.")
        for path in paths:
            module_path, _, name = path.rpartition(".")
            module = model if not module_path else dict(model.named_modules())[module_path]
            module.unfreeze(keys=[name], recurse=False)

    trainable = tree_flatten(model.trainable_parameters())
    model.load_weights([(path, value.astype(mx.float32)) for path, value in trainable], strict=False)
    return sorted(path for path, _ in trainable)
