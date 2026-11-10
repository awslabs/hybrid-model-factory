"""
Hybrid Gemma 4 Model Construction Module

This module converts a base Gemma 4 model into a text-only hybrid Gemma 4 model (HybridGemma4ForCausalLM), whose
layers the hybrid override pattern chooses among Gemma 4's own attention, Mamba2, Gated DeltaNet, GKA, and SWA, or
their Stage 1 layers ('*DA' and the fused 'X*'). The base checkpoints are multimodal (Gemma4ForConditionalGeneration),
so the conversion takes their text model and language modeling head, and leaves out the vision and audio components.

Priming follows the other families (a hybrid layer starts from the attention that it replaces, and the rest of the
model is copied), with these differences, which come from Gemma 4's architecture:

Gemma 4 interleaves sliding window attention layers, whose heads have 256 dimensions and whose RoPE is the default
one, with full attention layers (one in five in E2B and E4B), whose heads have 512 dimensions and whose RoPE rotates
only a quarter of each head's dimensions, with a larger base. Its config therefore has no single head size
(per_layer_config holds each layer's), and the hybrid config saves the base model's layout as base_layer_types,
while layer_types gets transformers' names for the hybrid layers. Each hybrid layer takes the head size and RoPE of
the base model's attention kind at its layer, so a hybrid layer at a full attention layer gets heads of 512. The
settings of each hybrid layer type (such as GKA's solver) are built once, from the config of layer 0, and each layer
recomputes its sizes from its own layer's config.

Since GKA's kernels run heads of at most 128 dimensions and GDN's at most 256, priming splits larger heads into
heads of the largest size that the kernel supports, with as many more heads, and reorders the weights so that
grouped-query attention still pairs each query head with its key and value head (see split_attention_heads in
hybrid_layer_init.py), while Mamba2 runs heads of 512 as they are.

Besides its attention and MLP, a Gemma 4 decoder layer has four norms (before and after its attention and its MLP),
a per-layer embedding block after its MLP, and a scalar that scales its output. A hybrid layer keeps all of them,
copied from the base model with the MLP, so only its sequence mixer is new, and the model keeps Gemma 4's scaled
token embeddings and the final softcapping of its logits.

In E2B and E4B, the last layers have no key and value projections of their own, and read the keys and values of the
last earlier layer of their kind, their source (in E2B, layers 15 to 34 read those of layers 13, which is sliding,
and 14, which is full). Priming handles these layers and their sources by three rules:
    - A GKA, GDN, or Mamba2 layer at a sharing layer gets key and value projections of its own, which start as
      copies of its source's.
    - An SWA layer at a sharing layer keeps reading its source's keys and values, and only limits them to its window.
    - A GKA, GDN, or Mamba2 layer at a source keeps the source's key and value projections in a small module of its
      decoder layer (Gemma4SourceKeysAndValues), which computes the keys and values that the later attention layers
      of its kind still read. If no attention layer of its kind still reads them, the module is left out.

Priming currently refuses B'MOJO, since its attention scales its scores by head_dim**-0.5, while Gemma 4's attention scales
them by 1 (its queries and keys are normalized). It also refuses full attention layers whose keys double as values
(attention_k_eq_v, as in Gemma 4 31B), since they have no value projection to start from.

In Stage 1, a '*DA' layer runs Gemma 4's attention on both streams, and a fused 'X*' layer runs the hybrid mixer on
the hybrid stream and Gemma 4's frozen attention on the attention stream. Each stream keeps its own shared keys and
values, and a fused layer at a source computes the hybrid stream's keys and values with its frozen key and value
projections, which unfusing (fused_to_standard.py) copies into the unfused layer's Gemma4SourceKeysAndValues.

When a primed model is trained without sequence parallelism, it needs SDPA or eager attention if it keeps any of
Gemma 4's full attention layers, since flash attention 2 supports heads of at most 256 (transformers issue 45201).
Under sequence parallelism, Gemma 4's attention layers exchange keys and values between the GPUs themselves (see
models/hybrid_gemma4/sequence_parallel.py): the sliding layers run flash attention 2 with their window, which needs
each GPU's chunks of the sequence to be at least window - 1 tokens long, and the full layers run SDPA's
memory-efficient kernel. Gradient checkpointing must be non-reentrant, since the shared keys and values pass between
the layers outside their inputs and outputs, and prepare_model_for_training switches to it on its own.
"""

import torch.nn as nn
from transformers.configuration_utils import PretrainedConfig

from hmf.model.hybrid_zoo.layers.hybrid_dataclasses import HybridConfig
from hmf.model.hybrid_zoo.models.hybrid_gemma4.configuration_hybrid_gemma4 import HybridGemma4TextConfig
from hmf.model.hybrid_zoo.models.hybrid_gemma4.modeling_hybrid_gemma4 import HybridGemma4ForCausalLM
from hmf.model.hybrid_zoo.models.utils import parse_override_pattern

from .hybrid_layer_init import (
    AttentionWeights,
    extract_attention_weights,
    gated_deltanet_proj_init,
    gka_proj_init,
    mamba2_init,
    swa_proj_init,
)
from .hybridize_logger import get_logger
from .utils import add_hybrid_configs, copy_shared_weights

logger = get_logger(__name__)

# The pattern entries that priming supports for Gemma 4
SUPPORTED_LAYER_TYPES = ("*", "M2", "GDN", "GKA", "SWA", "*DA", "M2*", "GDN*", "GKA*", "SWA*")
# The settings of each hybrid layer type that add_hybrid_configs writes into the config
HYBRID_LAYER_CONFIGS = ("mamba2_config", "gdn_config", "gka_config", "swa_config")


class _TextWeights:
    """
    The weights of a Gemma 4 checkpoint's text model and language modeling head, under the names that
    HybridGemma4ForCausalLM gives them, for copy_shared_weights. A multimodal checkpoint keeps its text model under
    model.language_model, and a text-only one under model.
    """

    def __init__(self, base_model: nn.Module):
        text_model = getattr(base_model.model, "language_model", base_model.model)
        self._state_dict = {f"model.{name}": tensor for name, tensor in text_model.state_dict().items()}
        self._state_dict["lm_head.weight"] = base_model.lm_head.weight

    def state_dict(self) -> dict:
        return self._state_dict


def get_text_layers(base_model: nn.Module) -> nn.ModuleList:
    """Returns the decoder layers of a Gemma 4 checkpoint's text model."""
    return getattr(base_model.model, "language_model", base_model.model).layers


def check_supported(text_config: PretrainedConfig, hybrid_override_pattern: str) -> None:
    """
    Raises an error for what priming doesn't support for Gemma 4 yet: B'MOJO, whose attention scales its scores by
    head_dim**-0.5 while Gemma 4's attention scales them by 1, and full attention layers whose
    keys double as values (attention_k_eq_v, as in Gemma 4 31B), which have no value projection to start from.
    """
    unsupported = sorted(set(hybrid_override_pattern.split("-")) - set(SUPPORTED_LAYER_TYPES))
    if unsupported:
        raise NotImplementedError(
            f"Priming Gemma 4 supports the layer types {list(SUPPORTED_LAYER_TYPES)}, got {unsupported}."
        )
    if text_config.attention_k_eq_v:
        raise NotImplementedError(
            "Priming Gemma 4 does not support models whose full attention layers use their keys as values "
            "(attention_k_eq_v) yet."
        )


def build_hybrid_config(text_config: PretrainedConfig, hybrid_config: HybridConfig) -> HybridGemma4TextConfig:
    """
    Returns the hybrid model's config: the base model's text config, with the hybrid override pattern and each
    hybrid layer type's settings. Gemma 4's config has per-layer head sizes, so the settings are built from the
    config of layer 0, whose sizes they record. Each hybrid layer recomputes its sizes from its own layer's config
    when it is built, and takes only the user's settings (such as GKA's solver) from them.
    """
    config_dict = text_config.to_dict()
    # Keep the hybrid config's model_type, which the base config's would overwrite
    config_dict.pop("model_type", None)
    config_dict["hybrid_override_pattern"] = hybrid_config.hybrid_override_pattern
    config = HybridGemma4TextConfig.from_dict(config_dict)

    layer_view = config.per_layer_config[0]
    add_hybrid_configs(layer_view, hybrid_config=hybrid_config)
    for name in HYBRID_LAYER_CONFIGS:
        if getattr(layer_view, name, None) is not None:
            setattr(config, name, getattr(layer_view, name))
    return config


def attention_weights(base_layers: nn.ModuleList, layer_idx: int, config: HybridGemma4TextConfig) -> AttentionWeights:
    """
    Returns the weights of the Gemma 4 attention at layer_idx. A layer that shares keys and values has no key and
    value projections of its own, so its keys and values come from its source, the layer whose keys and values it
    reads, which gives a hybrid layer there the same keys and values to start from.
    """
    attn = base_layers[layer_idx].self_attn
    if not attn.is_kv_shared_layer:
        return extract_attention_weights(attn)
    source = base_layers[config.kv_sharing_sources()[config.base_layer_types[layer_idx]]].self_attn
    return AttentionWeights(
        q_weight=attn.q_proj.weight.data,
        k_weight=source.k_proj.weight.data,
        v_weight=(source.v_proj if source.v_proj is not None else source.k_proj).weight.data,
        o_weight=attn.o_proj.weight.data,
        q_norm=attn.q_norm,
        k_norm=source.k_norm,
        head_dim=attn.head_dim,
    )


def init_source_keys_and_values(base_layers: nn.ModuleList, hybrid_layer: nn.Module, layer_idx: int) -> None:
    """
    Copies a source layer's key and value projections and key norm into the hybrid layer that replaces it, which
    keeps them to compute the keys and values that later attention layers read (Gemma4SourceKeysAndValues).
    """
    attn = base_layers[layer_idx].self_attn
    source_kv = hybrid_layer.source_kv
    source_kv.k_proj.weight.data.copy_(attn.k_proj.weight.data)
    source_kv.k_norm.weight.data.copy_(attn.k_norm.weight.data)
    if source_kv.v_proj is not None:
        source_kv.v_proj.weight.data.copy_(attn.v_proj.weight.data)
    logger.info(f"Layer {layer_idx}: kept the source's key and value projections for the layers that share them")


def init_shared_swa(attn_weights: AttentionWeights, swa_layer: nn.Module) -> None:
    """
    Initializes an SWA layer at a layer that shares keys and values, which, like Gemma 4's attention there, has
    only query and output projections, and reads the keys and values of its source.
    """
    swa_layer.q_proj.weight.data.copy_(attn_weights.q_weight)
    swa_layer.o_proj.weight.data.copy_(attn_weights.o_weight)
    swa_layer.q_norm.weight.data.copy_(attn_weights.q_norm.weight.data)


def init_hybrid_layers(base_layers: nn.ModuleList, hybrid_model: nn.Module, config: HybridGemma4TextConfig) -> None:
    """
    Initializes each hybrid layer from the Gemma 4 attention that it replaces, whose head size it has. The rest of
    each decoder layer (its MLP, four norms, per-layer embedding block, and scalar) is copied by copy_shared_weights,
    and so is the attention of Stage 1's layers ('*DA' and the fused 'X*'), whose mixers start as those of 'X' do.
    """
    for layer_idx, entry in parse_override_pattern(config.hybrid_override_pattern).items():
        if entry in ("*", "*DA"):
            continue
        layer_type = entry.removesuffix("*")
        attn_weights = attention_weights(base_layers, layer_idx, config)
        hybrid_layer = hybrid_model.model.layers[layer_idx]
        if getattr(hybrid_layer, "source_kv", None) is not None:
            init_source_keys_and_values(base_layers, hybrid_layer, layer_idx)
        if layer_type == "SWA" and hybrid_layer.swa.is_kv_shared_layer:
            init_shared_swa(attn_weights, hybrid_layer.swa)
        elif layer_type == "M2":
            mamba2_init(attn_weights, hybrid_layer.mamba, layer_idx)
        elif layer_type == "GDN":
            gated_deltanet_proj_init(attn_weights, hybrid_layer.gdn, layer_idx)
        elif layer_type == "GKA":
            gka_proj_init(attn_weights, hybrid_layer.gka, layer_idx)
        elif layer_type == "SWA":
            swa_proj_init(attn_weights, hybrid_layer.swa, layer_idx)


def construct_hybrid_gemma4(
    base_model: nn.Module,
    config: PretrainedConfig,
    hybrid_config: HybridConfig,
) -> nn.Module:
    """
    Constructs a text-only hybrid Gemma 4 model from a base Gemma 4 model. It does the following:
        1. Builds the hybrid config from the base model's text config
        2. Instantiates a hybrid model with random weights
        3. Copies the weights that the base and hybrid models share (embeddings, MLPs, norms, per-layer
           embeddings, and the attention layers that the pattern keeps)
        4. Initializes the hybrid layers from the attention layers that they replace

    Args:
        base_model: Base Gemma 4 model (Gemma4ForConditionalGeneration or Gemma4ForCausalLM) to convert
        config: Base model configuration
        hybrid_config: Hybrid-specific configuration

    Returns:
        Initialized hybrid model
    """
    logger.section("Constructing Hybrid Gemma 4")

    text_config = config.get_text_config()
    check_supported(text_config, hybrid_config.hybrid_override_pattern)
    hybrid_text_config = build_hybrid_config(text_config, hybrid_config)

    logger.info("Creating hybrid model with random initialization (parameters will be initialized afterwards)")
    hybrid_model = HybridGemma4ForCausalLM(hybrid_text_config).to(dtype=base_model.dtype)
    initial_param_sum = sum(p.sum().item() for p in hybrid_model.parameters())

    copy_shared_weights(_TextWeights(base_model), hybrid_model)
    after_copy_param_sum = sum(p.sum().item() for p in hybrid_model.parameters())

    logger.info("Initializing hybrid-specific layers")
    init_hybrid_layers(get_text_layers(base_model), hybrid_model, hybrid_text_config)
    final_param_sum = sum(p.sum().item() for p in hybrid_model.parameters())

    logger.summary(
        f"Hybrid model construction complete:\n"
        f"  Initial param sum: {initial_param_sum:.6f}\n"
        f"  After copy param sum: {after_copy_param_sum:.6f}\n"
        f"  Final param sum: {final_param_sum:.6f}\n"
    )
    return hybrid_model
