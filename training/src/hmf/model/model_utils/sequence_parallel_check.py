"""
Checks, before training starts, that a model's backbone supports sequence parallelism.

Sequence parallelism splits each sequence across the GPUs of a sequence parallel group, so every
layer that mixes tokens must exchange information between the GPUs. HMF's layers do this with their
own sequence parallel versions, and the attention layers of Hugging Face models do it through ring
attention, which sequence parallelism registers in place of their attention function (see
sequence_parallel.py). Other Hugging Face layers, e.g., Gated DeltaNet or Mamba layers, only see
their own GPU's part of each sequence, and silently compute the wrong result. Ring attention also
ignores sliding windows and softcapping (as in Gemma 2), so attention layers that use either of them
are wrong too. HMF's layers other than attention support the zigzag-ring mode only.

The SFT, pre-training, and decision trainers call check_sequence_parallel_backbone at their first
batch, and it raises an error that names the unsupported layers.
"""

from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn


# The attention implementation that sequence parallelism registers, which runs ring attention
# (see sequence_parallel.py)
SP_ATTENTION = "sequence_parallel_attention"

# The package of HMF's models (hmf.model.hybrid_zoo)
HYBRID_ZOO_PACKAGE = __name__.rsplit(".model_utils.", 1)[0] + ".hybrid_zoo"

# The reasons a layer can be unsupported
NO_SEQUENCE_PARALLELISM = "no sequence parallelism"
ZIGZAG_RING_ONLY = "zigzag-ring only"
SLIDING_WINDOW = "sliding-window attention"
SOFTCAPPING = "softcapping"

# The steps that make a causal LM trainable with sequence parallelism, for the error message
CAUSAL_LM_CONVERSION_STEPS = [
    "Convert it to an HMF model with hmf prime-init, with a hybrid_override_pattern that keeps its layers\n"
    "     as they are.",
    "Train that model.",
]

# A layer that does not support sequence parallelism, given as its index, its class name, and the reason
BackboneProblem = Tuple[Optional[int], str, str]


def unwrap_model(model: nn.Module) -> nn.Module:
    """The model inside model, without its DDP, DeepSpeed, or PEFT (LoRA) wrappers."""
    while hasattr(model, "module"):  # DDP, DeepSpeed
        model = model.module
    if hasattr(model, "get_base_model"):  # PEFT
        model = model.get_base_model()
    return model


def is_hmf_model(module: nn.Module) -> bool:
    """Whether a module's class is one of HMF's models or layers (from hmf.model.hybrid_zoo)."""
    return type(module).__module__.startswith(HYBRID_ZOO_PACKAGE + ".")


def find_backbone(model: nn.Module) -> Optional[nn.Module]:
    """
    The backbone of a model, that is, the module that holds its decoder layers as `layers`. This is
    the model's `model` for causal LMs and decision models, or its language model for multimodal
    models. Returns None if there is none.
    """
    model = unwrap_model(model)
    for path in ("", "model", "model.language_model", "language_model"):
        try:
            module = model.get_submodule(path)
        except AttributeError:
            continue
        if isinstance(getattr(module, "layers", None), nn.ModuleList):
            return module
    return None


def sequence_parallel_backbone_problems(
    model: nn.Module, run_forward: Callable[[], Any], mode: str = "zigzag-ring", always_run_forward: bool = False
) -> List[BackboneProblem]:
    """
    Checks that every layer of a model's backbone supports sequence parallelism, that is, mixes
    tokens across the whole sequence rather than only within its rank's part.

    With zigzag-ring, HMF's models support it throughout, so they pass without further checks. For
    other models, and for HMF's models with Ulysses, one forward pass records which modules call
    ring attention, and with which options. Every layer must have called it, without a sliding
    window and without softcapping, because ring attention ignores both. The sliding-window rule
    concerns Hugging Face's attention layers only, as HMF's own sliding-window layers (SWA) have their
    own sequence parallel version. The check shows that each layer has a sequence mixer that supports sequence
    parallelism, but not that it has no other one: a layer that ran, say, a Mamba mixer alongside
    its attention would still pass.

    The forward pass runs in eval mode, without gradients. With always_run_forward, it also runs
    when there is nothing to record, because the SFT and pre-training trainers rely on it as their
    first forward pass with sequence parallelism. Every rank of the sequence parallel group must call
    this at the same time, as the forward pass communicates between them.

    Args:
        model: The model, possibly wrapped (DDP, DeepSpeed, PEFT), with sequence parallelism set up.
        run_forward: Runs one forward pass of the model on a batch prepared for sequence parallelism.
        mode: The sequence parallel mode, "zigzag-ring" or "ulysses".
        always_run_forward: Whether to run the forward pass even when there is nothing to record.
            Defaults to False.

    Returns:
        One (layer index, layer class, reason) per problem, or an empty list if the backbone
        supports sequence parallelism.
    """
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, AttentionInterface

    backbone = find_backbone(model)
    hmf_backbone = backbone is not None and is_hmf_model(backbone)
    # HMF's models need no recording with zigzag-ring, and a model without layers cannot be checked
    record = backbone is not None and not (hmf_backbone and mode == "zigzag-ring")

    # Record which modules call ring attention, and which options each of them used that ring
    # attention ignores. The attention modules look the function up by name at every call, so
    # replacing it for the one forward pass is enough.
    calls: Dict[nn.Module, Set[str]] = {}
    ring_attention = ALL_ATTENTION_FUNCTIONS[SP_ATTENTION]

    def recording(module, *args, **kwargs):
        ignored = calls.setdefault(module, set())
        if (kwargs.get("sliding_window") or 0) > 0:
            ignored.add(SLIDING_WINDOW)
        if (kwargs.get("softcap") or 0) > 0:
            ignored.add(SOFTCAPPING)
        return ring_attention(module, *args, **kwargs)

    if record or always_run_forward:
        was_training = model.training
        if record:
            AttentionInterface.register(SP_ATTENTION, recording)
        try:
            model.eval()
            with torch.no_grad():
                run_forward()
        finally:
            if record:
                AttentionInterface.register(SP_ATTENTION, ring_attention)
            model.train(was_training)

    if backbone is None:
        return [(None, type(unwrap_model(model)).__name__, "has no `layers` to check")]
    if not record:
        return []
    problems = []
    for index, layer in enumerate(backbone.layers):
        callers = [module for module in layer.modules() if module in calls]
        if not callers:
            reason = ZIGZAG_RING_ONLY if hmf_backbone else NO_SEQUENCE_PARALLELISM
            problems.append((index, sequence_mixer_name(layer), reason))
        for reason in (SLIDING_WINDOW, SOFTCAPPING):
            names = sorted({type(module).__name__ for module in callers if reason in calls[module]})
            if names:
                problems.append((index, "/".join(names), reason))
    return problems


def sequence_mixer_name(layer: nn.Module) -> str:
    """
    The class name of a decoder layer's sequence mixer, for error messages. The sequence mixer is any
    direct submodule other than norms, MLPs, and MoE blocks, and the layer's own class name is used if
    there is none. Classes from the transformers library and from HMF are named as such, e.g.,
    "transformers' Qwen3_5GatedDeltaNet".
    """

    def name(module: nn.Module) -> str:
        if type(module).__module__.startswith("transformers."):
            return f"transformers' {type(module).__name__}"
        if is_hmf_model(module):
            return f"HMF's {type(module).__name__}"
        return type(module).__name__

    names = [
        name(child)
        for child in layer.children()
        if not any(part in type(child).__name__.lower() for part in ("norm", "mlp", "moe"))
    ]
    return "/".join(names) if names else name(layer)


def describe_backbone_problems(problems: List[BackboneProblem]) -> List[str]:
    """
    One sentence per kind of problem, listing its layers, e.g.,
    "layers 0, 1, 2 use transformers' Qwen3_5GatedDeltaNet, which has no sequence parallel version".
    """
    # The sentence for each reason, for one layer and for several
    sentences = {
        NO_SEQUENCE_PARALLELISM: (
            "{layers} uses {cls}, which has no sequence parallel version",
            "{layers} use {cls}, which has no sequence parallel version",
        ),
        ZIGZAG_RING_ONLY: (
            "{layers} uses {cls}, which supports sequence parallelism in the zigzag-ring mode only",
            "{layers} use {cls}, which supports sequence parallelism in the zigzag-ring mode only",
        ),
        SLIDING_WINDOW: (
            "{layers} ({cls}) uses sliding-window attention, which ring attention does not support",
            "{layers} ({cls}) use sliding-window attention, which ring attention does not support",
        ),
        SOFTCAPPING: (
            "{layers} ({cls}) uses attention softcapping, which ring attention does not support",
            "{layers} ({cls}) use attention softcapping, which ring attention does not support",
        ),
    }
    layers: Dict[Tuple[str, str], List[int]] = {}
    lines = []
    for index, layer_class, reason in problems:
        if index is None:
            lines.append(f"{layer_class} {reason}")
        else:
            layers.setdefault((layer_class, reason), []).append(index)
    for (layer_class, reason), indices in layers.items():
        one, several = sentences.get(reason, ("{layers} ({cls}) " + reason,) * 2)
        if len(indices) == 1:
            lines.append(one.format(layers=f"layer {indices[0]}", cls=layer_class))
        else:
            lines.append(several.format(layers=f"layers {', '.join(map(str, indices))}", cls=layer_class))
    return lines


def check_sequence_parallel_backbone(
    model: nn.Module,
    run_forward: Callable[[], Any],
    mode: str = "zigzag-ring",
    conversion_steps: Optional[List[str]] = None,
    always_run_forward: bool = False,
) -> None:
    """
    Raises an error that names the layers that do not support sequence parallelism, if there are
    any (see sequence_parallel_backbone_problems).

    Args:
        model: The model, possibly wrapped (DDP, DeepSpeed, PEFT), with sequence parallelism set up.
        run_forward: Runs one forward pass of the model on a batch prepared for sequence parallelism.
        mode: The sequence parallel mode, "zigzag-ring" or "ulysses".
        conversion_steps: The steps that make the model trainable with sequence parallelism, for the
            error message. Defaults to CAUSAL_LM_CONVERSION_STEPS.
        always_run_forward: Whether to run the forward pass even when there is nothing to record (see
            sequence_parallel_backbone_problems). Defaults to False.
    """
    problems = sequence_parallel_backbone_problems(model, run_forward, mode, always_run_forward)
    if not problems:
        return
    name = getattr(unwrap_model(model).config, "_name_or_path", "") or "the model"
    if any(reason == ZIGZAG_RING_ONLY for _, _, reason in problems):
        fix = "Use sequence_parallel_mode: zigzag-ring, which all of HMF's layers support."
    else:
        steps = conversion_steps or CAUSAL_LM_CONVERSION_STEPS
        fix = (
            "Sequence parallelism works with HMF's models, and with Hugging Face models whose layers all use full attention "
            "without softcapping. To use this model:\n"
            + "\n".join(f"  {number}. {step}" for number, step in enumerate(steps, start=1))
            + '\nhmf prime-init supports only the models listed under "Supported Base Models" in docs/Priming.md.'
        )
    raise ValueError(
        f"Sequence parallelism can't be used with {name}, because some of its layers do not support it "
        "and would silently compute the wrong result:\n  " + "\n  ".join(describe_backbone_problems(problems))
        + "\n" + fix
    )
