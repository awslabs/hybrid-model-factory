# Copyright 2026 the HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""PyTorch Hybrid Gemma 4 text model.

This module implements a hybrid variant of the Gemma 4 text architecture that supports replacing its attention
layers with alternative sequence mixing mechanisms. It starts from transformers' Gemma 4 text model, whose layers
alternate between sliding window and full attention, and keeps its per-layer embeddings, its keys and values shared
across the last layers, and its four norms per decoder layer.

Supported sequence mixing layers:
    - Mamba2
    - B'MOJO
    - Gated DeltaNet (GDN)
    - Gated KalmaNet (GKA)
    - Sliding Window Attention (SWA)

Architecture:
    Each decoder layer is chosen by the hybrid override pattern in the model configuration, and can be either:
    - Standard Gemma 4 attention layer (Gemma4DecoderLayer), which keeps the base model's sliding or full attention
    - One of several hybrid layer types (subclasses of Gemma4HybridDecoderLayerBase)
    - For Stage 1 of hybridization (layerwise distillation), a '*DA' or fused 'X*' layer (subclasses of
      Gemma4Stage1DecoderLayerBase), which runs a hybrid stream and Gemma 4's own attention stream side by side,
      through HybridGemma4TextModel.stage1_layer_outputs

Each layer's head size, RoPE, and key and value sharing follow the base model's attention kind at that layer
(config.base_layer_types), so a hybrid layer that replaces a full attention layer gets its larger heads. The last
layers of Gemma 4 E2B and E4B reuse the keys and values of an earlier layer of their kind, their source. A hybrid
layer that replaces one of these sharing layers has key and value projections of its own, while an SWA layer there
keeps reading the shared keys and values. A hybrid layer that replaces a source keeps the source's key and value
projections, to compute the keys and values that later attention layers still read (Gemma4SourceKeysAndValues).
Under sequence parallelism, the attention layers exchange keys and values between the ranks themselves, since ring
attention supports neither heads of 512 nor sliding windows (see sequence_parallel.py).
"""

import copy
import sys
from abc import ABC, abstractmethod
from collections import UserDict
from collections.abc import Callable, Iterator

import torch
from torch import nn

from transformers import initialization as init
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache
from transformers.generation import GenerationMixin
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel, slice_logits_to_keep
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, auto_docstring, can_return_tuple, logging
from transformers.utils.generic import merge_with_config_defaults
from transformers.utils.output_capturing import capture_outputs

from ...layers.bmojo_f.modules import BMojoLayer
from ...layers.gated_deltanet.modules import GatedDeltaNetLayer
from ...layers.gated_kalmanet.modules import GatedKalmaNetLayer
from ...layers.mamba.modules import Mamba2Layer
from ..cache import HybridCache
from ..utils import build_decoder_layer_registry, set_decoder_layers_from_pattern
from .configuration_hybrid_gemma4 import HybridGemma4TextConfig
from .sequence_parallel import FLASH_ATTENTION_MAX_HEAD_DIM, sp_full_attention, sp_windowed_attention

logger = logging.get_logger(__name__)

# The suffix of the mask keys (see the layers' attention_type) of the attention layers that run SDPA in a model that
# runs flash attention 2, whose masks SDPA's format must have
SDPA_MASK_SUFFIX = ":sdpa"


class Gemma4RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, with_scale: bool = True):
        super().__init__()
        self.eps = eps
        self.with_scale = with_scale

        if self.with_scale:
            self.weight = nn.Parameter(torch.ones(dim), requires_grad=True)

    def _norm(self, hidden_states: torch.Tensor):
        mean_squared = hidden_states.pow(2).mean(-1, keepdim=True) + self.eps
        # Use torch.pow() (over torch.sqrt() or torch.rsqrt()) to address compiler differences between Torch and JAX
        return hidden_states * torch.pow(mean_squared, -0.5)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        normed_output = self._norm(hidden_states.float())
        if self.with_scale:
            normed_output = normed_output * self.weight.float()
        return normed_output.type_as(hidden_states)


class Gemma4TextMLP(nn.Module):
    def __init__(self, config: HybridGemma4TextConfig, layer_idx: int):
        super().__init__()
        first_kv_shared_layer_idx = config.num_hidden_layers - config.num_kv_shared_layers
        is_kv_shared_layer = layer_idx >= first_kv_shared_layer_idx > 0
        use_double_wide_mlp = config.use_double_wide_mlp and is_kv_shared_layer
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size * (2 if use_double_wide_mlp else 1)
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_activation]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


def layer_config_for_kind(config: HybridGemma4TextConfig, kind: str) -> HybridGemma4TextConfig:
    """
    Returns the per-layer config of the first layer whose base attention kind is kind, which holds the head size of
    that kind. Gemma 4's own lookup by kind reads layer_types, which no longer holds a kind once every layer of that
    kind is a hybrid layer.
    """
    return config.per_layer_config[config.base_layer_types.index(kind)]


class Gemma4TextRotaryEmbedding(nn.Module):
    def __init__(self, config: HybridGemma4TextConfig, device=None):
        super().__init__()
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        # One RoPE for each of the base model's attention kinds, which the hybrid layers at those layers also use
        self.layer_types = sorted(set(config.base_layer_types))
        self.rope_init_fns: dict[str, Callable[..., tuple[torch.Tensor, float]]] = {}
        self.rope_type: dict[str, str] = {}

        for layer_type in self.layer_types:
            rope_params = self.config.rope_parameters[layer_type]
            if rope_params is None:
                continue

            if (rope_type := rope_params["rope_type"]) != "default":
                rope_init_fn = ROPE_INIT_FUNCTIONS[rope_type]
            else:
                rope_init_fn = self.compute_default_rope_parameters

            self.rope_init_fns[layer_type] = rope_init_fn
            self.rope_type[layer_type] = rope_type

            # `inv_freq` depends on the head dim, which varies by layer type, so initialise
            # from a config resolved for this layer type rather than the global one.
            rope_config = layer_config_for_kind(config, layer_type)
            curr_inv_freq, curr_attention_scaling = rope_init_fn(rope_config, device, layer_type=layer_type)
            setattr(self, f"{layer_type}_inv_freq", nn.Buffer(curr_inv_freq, persistent=False))
            setattr(self, f"{layer_type}_original_inv_freq", nn.Buffer(curr_inv_freq.clone(), persistent=False))
            setattr(self, f"{layer_type}_attention_scaling", curr_attention_scaling)

    @staticmethod
    def compute_default_rope_parameters(
        config: HybridGemma4TextConfig, device=None, layer_type: str | None = None, **kwargs
    ) -> tuple[torch.Tensor, float]:
        """
        Computes the inverse frequencies according to the original RoPE implementation
        Args:
            config ([`~transformers.PreTrainedConfig`]):
                The model configuration.
            layer_type (`str`):
                The current layer type if the model has different RoPE parameters per type.
                Should not be used unless `config.layer_types is not None`

        Returns:
            Tuple of (`torch.Tensor`, `float`), containing the inverse frequencies for the RoPE embeddings and the
            post-processing scaling factor applied to the computed cos/sin (unused in this type of RoPE).
        """
        base = config.rope_parameters[layer_type]["rope_theta"]
        dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads

        attention_factor = 1.0  # Unused in this type of RoPE
        # Compute the inverse frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        return inv_freq.to(device), attention_factor

    @torch.no_grad()
    @dynamic_rope_update  # power user: used with advanced RoPE types (e.g. dynamic rope)
    def forward(self, x, position_ids, layer_type):
        inv_freq = getattr(self, f"{layer_type}_inv_freq")
        attention_scaling = getattr(self, f"{layer_type}_attention_scaling")

        freqs = position_ids[..., None].float() * inv_freq.to(device=x.device, dtype=torch.float)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos() * attention_scaling
        sin = emb.sin() * attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, unsqueeze_dim: int = 1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        x (`torch.Tensor`): The tensor to embed.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    return (x * cos) + (rotate_half(x) * sin)


def apply_rotary_pos_emb_qk(q, k, cos, sin, unsqueeze_dim=1):
    """
    Applies the rotary embedding to both the queries and the keys, which is the signature that Mamba2 and B'MOJO
    take, while Gemma 4's attention rotates its queries and keys separately.
    """
    return apply_rotary_pos_emb(q, cos, sin, unsqueeze_dim), apply_rotary_pos_emb(k, cos, sin, unsqueeze_dim)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    dropout: float | int = 0.0,
    scaling: float | None = None,
    softcap: float | None = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    if scaling is None:
        scaling = module.head_dim**-0.5

    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling

    if softcap is not None:
        attn_weights = attn_weights / softcap
        attn_weights = torch.tanh(attn_weights)
        attn_weights = attn_weights * softcap
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    # upcast attention to fp32
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


class Gemma4TextAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: HybridGemma4TextConfig, layer_idx: int):
        super().__init__()
        # The base model's attention kind at this layer, which is also the key under which it stores or reads the
        # keys and values that layers share
        self.layer_type = config.base_layer_types[layer_idx]
        self.config = config
        self.layer_idx = layer_idx
        self.is_sliding = self.layer_type == "sliding_attention"
        self.sliding_window = config.sliding_window if self.is_sliding else None
        self.sequence_parallel_group = None
        layer_config = config.per_layer_config[layer_idx]
        self.head_dim = layer_config.head_dim
        self.use_alternative_attention = config.attention_k_eq_v and not self.is_sliding
        self.num_key_value_groups = config.num_attention_heads // layer_config.num_key_value_heads
        self.scaling = 1.0
        self.attention_dropout = self.config.attention_dropout
        self.is_causal = config.use_bidirectional_attention != "all"

        # Shared kv cache
        self.is_kv_shared_layer = layer_idx >= config.first_kv_shared_layer
        self.store_full_length_kv = config.kv_sharing_sources().get(self.layer_type) == layer_idx

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.q_norm = Gemma4RMSNorm(dim=self.head_dim, eps=config.rms_norm_eps)

        # Layers sharing kv states don't need any weight matrices
        if not self.is_kv_shared_layer:
            self.k_norm = Gemma4RMSNorm(dim=self.head_dim, eps=config.rms_norm_eps)
            self.v_norm = Gemma4RMSNorm(self.head_dim, eps=config.rms_norm_eps, with_scale=False)

            self.k_proj = nn.Linear(
                config.hidden_size, layer_config.num_key_value_heads * self.head_dim, bias=config.attention_bias
            )
            self.v_proj = (
                nn.Linear(
                    config.hidden_size, layer_config.num_key_value_heads * self.head_dim, bias=config.attention_bias
                )
                if not self.use_alternative_attention
                else None
            )

        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )

    @property
    def runs_sdpa_for_flash_attention(self) -> bool:
        """
        Returns whether the layer runs SDPA in a model that runs flash attention 2, which supports heads of at most
        FLASH_ATTENTION_MAX_HEAD_DIM. Gemma 4's full attention layers have heads of 512, so they run SDPA, while its
        sliding window layers run flash attention 2, and an HMF SWA layer follows the layer that it replaces.
        """
        return (self.config._attn_implementation == "flash_attention_2"
                and self.head_dim > FLASH_ATTENTION_MAX_HEAD_DIM)

    def mask_key(self, kind: str) -> str:
        """Returns the key of the layer's mask of the given kind, in SDPA's format if the layer runs SDPA."""
        return kind + SDPA_MASK_SUFFIX if self.runs_sdpa_for_flash_attention else kind

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: torch.Tensor,
        attention_mask: torch.Tensor | None,
        shared_kv_states: dict[str, tuple[torch.Tensor, torch.Tensor]],
        past_key_values: Cache | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        cos, sin = position_embeddings

        query_states = self.q_proj(hidden_states).view(hidden_shape)
        query_states = self.q_norm(query_states)
        query_states = apply_rotary_pos_emb(query_states, cos, sin, unsqueeze_dim=2)
        query_states = query_states.transpose(1, 2)

        # For layers with shared KV (from kv sharing point onwards), we reuse the same keys/values states as the last
        # non-sharing layer. We cannot simply reuse the cached state if we have a Cache, as sliding layers will not
        # remember the full states in their Cache once we are past the sliding window - so we always use
        # `shared_kv_states` instead, even when past_key_values is not None
        if self.is_kv_shared_layer:
            key_states, value_states = shared_kv_states[self.layer_type]
            # Device of past layer may be different from current one
            key_states = key_states.to(query_states.device)
            value_states = value_states.to(query_states.device)
        else:
            key_states = self.k_proj(hidden_states).view(hidden_shape)
            value_states = self.v_proj(hidden_states).view(hidden_shape) if self.v_proj is not None else key_states

            key_states = self.k_norm(key_states)
            key_states = apply_rotary_pos_emb(key_states, cos, sin, unsqueeze_dim=2)
            key_states = key_states.transpose(1, 2)

            value_states = self.v_norm(value_states)
            value_states = value_states.transpose(1, 2)

        if past_key_values is not None and not self.is_kv_shared_layer:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)
        if self.store_full_length_kv:
            shared_kv_states[self.layer_type] = key_states, value_states

        if self.sequence_parallel_group is not None:
            # Under sequence parallelism, the layer exchanges keys and values between the ranks itself (see
            # sequence_parallel.py), with whichever keys and values it has, its own or those that it shares
            dropout = self.attention_dropout if self.training else 0.0
            if self.sliding_window:
                attn_output = sp_windowed_attention(
                    query_states, key_states, value_states, self.sliding_window, self.scaling,
                    self.sequence_parallel_group, dropout,
                )
            else:
                attn_output = sp_full_attention(
                    query_states, key_states, value_states, self.scaling, self.sequence_parallel_group, dropout
                )
            attn_output, attn_weights = attn_output.transpose(1, 2), None
        else:
            attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
                "sdpa" if self.runs_sdpa_for_flash_attention else self.config._attn_implementation,
                eager_attention_forward,
            )

            attn_output, attn_weights = attention_interface(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask,
                dropout=self.attention_dropout if self.training else 0.0,
                scaling=self.scaling,
                sliding_window=self.sliding_window,
                **kwargs,
            )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class Gemma4SWAttention(Gemma4TextAttention):
    """
    HMF's sliding window attention, which keeps the base model's attention at its layer (its head size, RoPE, and
    weights) and limits it to the window of swa_config.
    """

    def __init__(self, config: HybridGemma4TextConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.sliding_window = config.swa_config.get("window_size", None)
        if self.sliding_window is None:
            raise ValueError("swa_config must specify window_size.")
        self.is_sliding = True

    def forward(self, hidden_states, position_embeddings, attention_mask, shared_kv_states=None, past_key_values=None,
                **kwargs):
        attn_output, attn_weights = super().forward(
            hidden_states, position_embeddings, attention_mask, shared_kv_states, past_key_values, **kwargs
        )
        # Return 3-tuple to be consistent with all other Hybrid layer outputs
        return attn_output, attn_weights, None


def project_keys_and_values(
    module: nn.Module, hidden_states: torch.Tensor, position_embeddings: tuple[torch.Tensor, torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns the keys and values that Gemma 4's attention computes from the normalized hidden states, with the key
    and value projections and norms of module (a Gemma4TextAttention, or a Gemma4SourceKeysAndValues), in the shape
    (batch, num_key_value_heads, seq_len, head_dim).
    """
    hidden_shape = (*hidden_states.shape[:-1], -1, module.head_dim)
    cos, sin = position_embeddings
    key_states = module.k_proj(hidden_states).view(hidden_shape)
    value_states = module.v_proj(hidden_states).view(hidden_shape) if module.v_proj is not None else key_states

    key_states = module.k_norm(key_states)
    key_states = apply_rotary_pos_emb(key_states, cos, sin, unsqueeze_dim=2).transpose(1, 2)
    value_states = module.v_norm(value_states).transpose(1, 2)
    return key_states, value_states


class Gemma4SourceKeysAndValues(nn.Module):
    """
    The key and value projections of a source layer, whose keys and values later layers of its kind reuse, for a
    hybrid layer that replaces the source (see HybridGemma4TextConfig.keeps_source_keys_and_values). It computes the
    keys and values from the layer's normalized input as the source's attention does, caches them under the layer's
    index, and stores them for the later layers, while the hybrid layer's sequence mixer computes its output.
    """

    def __init__(self, config: HybridGemma4TextConfig, layer_idx: int):
        super().__init__()
        self.layer_type = config.base_layer_types[layer_idx]
        self.layer_idx = layer_idx
        layer_config = config.per_layer_config[layer_idx]
        self.head_dim = layer_config.head_dim
        self.k_norm = Gemma4RMSNorm(dim=self.head_dim, eps=config.rms_norm_eps)
        self.v_norm = Gemma4RMSNorm(self.head_dim, eps=config.rms_norm_eps, with_scale=False)
        self.k_proj = nn.Linear(
            config.hidden_size, layer_config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        use_alternative_attention = config.attention_k_eq_v and self.layer_type != "sliding_attention"
        self.v_proj = (
            None
            if use_alternative_attention
            else nn.Linear(
                config.hidden_size, layer_config.num_key_value_heads * self.head_dim, bias=config.attention_bias
            )
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        shared_kv_states: dict[str, tuple[torch.Tensor, torch.Tensor]],
        past_key_values: Cache | None = None,
    ) -> None:
        key_states, value_states = project_keys_and_values(self, hidden_states, position_embeddings)
        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)
        shared_kv_states[self.layer_type] = key_states, value_states


class Gemma4DecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: HybridGemma4TextConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self.self_attn = Gemma4TextAttention(config=config, layer_idx=layer_idx)
        self.mlp = Gemma4TextMLP(config, layer_idx)
        self.input_layernorm = Gemma4RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Gemma4RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.pre_feedforward_layernorm = Gemma4RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.post_feedforward_layernorm = Gemma4RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.layer_scalar = nn.Buffer(torch.ones(1))

        self.hidden_size_per_layer_input = config.hidden_size_per_layer_input
        if self.hidden_size_per_layer_input:
            self.act_fn = ACT2FN[config.hidden_activation]
            self.per_layer_input_gate = nn.Linear(self.hidden_size, self.hidden_size_per_layer_input, bias=False)
            self.per_layer_projection = nn.Linear(self.hidden_size_per_layer_input, self.hidden_size, bias=False)
            self.post_per_layer_input_norm = Gemma4RMSNorm(self.hidden_size, eps=config.rms_norm_eps)

    @property
    def attention_type(self) -> str:
        """Returns the key of the layer's mask, that of the base model's attention kind at this layer."""
        return self.self_attn.mask_key(self.self_attn.layer_type)

    def forward(
        self,
        hidden_states: torch.Tensor,
        per_layer_input: torch.Tensor = None,
        shared_kv_states: dict[str, tuple[torch.Tensor, torch.Tensor]] | None = None,
        position_embeddings: torch.Tensor = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            shared_kv_states=shared_kv_states,
            position_ids=position_ids,
            past_key_values=past_key_values,
            **kwargs,
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        if self.hidden_size_per_layer_input:
            residual = hidden_states
            hidden_states = self.per_layer_input_gate(hidden_states)
            hidden_states = self.act_fn(hidden_states)
            hidden_states = hidden_states * per_layer_input
            hidden_states = self.per_layer_projection(hidden_states)
            hidden_states = self.post_per_layer_input_norm(hidden_states)
            hidden_states = residual + hidden_states

        hidden_states *= self.layer_scalar
        return hidden_states


class Gemma4DecoderLayerBody:
    """
    The parts of Gemma 4's decoder layer around its sequence mixer: the MLP, the four norms around the mixer and the
    MLP, the per-layer embedding block after the MLP, and the layer's scalar, under the names that Gemma4DecoderLayer
    gives them. The hybrid decoder layers and the Stage 1 decoder layers build them with _init_body and run them
    around their sequence mixer with _body_forward.
    """

    def _init_body(self, config: HybridGemma4TextConfig, layer_idx: int):
        self.mlp = Gemma4TextMLP(config, layer_idx)
        self.input_layernorm = Gemma4RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Gemma4RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.pre_feedforward_layernorm = Gemma4RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.post_feedforward_layernorm = Gemma4RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.layer_scalar = nn.Buffer(torch.ones(1))

        self.hidden_size_per_layer_input = config.hidden_size_per_layer_input
        if self.hidden_size_per_layer_input:
            self.act_fn = ACT2FN[config.hidden_activation]
            self.per_layer_input_gate = nn.Linear(self.hidden_size, self.hidden_size_per_layer_input, bias=False)
            self.per_layer_projection = nn.Linear(self.hidden_size_per_layer_input, self.hidden_size, bias=False)
            self.post_per_layer_input_norm = Gemma4RMSNorm(self.hidden_size, eps=config.rms_norm_eps)

    def _body_forward(
        self,
        hidden_states: torch.Tensor,
        per_layer_input: torch.Tensor | None,
        mixer: Callable[[torch.Tensor], torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Runs the decoder layer on hidden_states, with mixer, a function of the normalized hidden states, as its
        sequence mixer.

        Returns:
            - hidden_states: the layer's output
            - mixer_out: the sequence mixer's output as the layer adds it to the residual stream (after its norm)
            - mlp_out: the MLP's output as the layer adds it to the residual stream (after its norm)
        """
        residual = hidden_states
        mixer_out = self.post_attention_layernorm(mixer(self.input_layernorm(hidden_states)))
        hidden_states = residual + mixer_out

        residual = hidden_states
        mlp_out = self.post_feedforward_layernorm(self.mlp(self.pre_feedforward_layernorm(hidden_states)))
        hidden_states = residual + mlp_out

        if self.hidden_size_per_layer_input:
            residual = hidden_states
            hidden_states = self.per_layer_input_gate(hidden_states)
            hidden_states = self.act_fn(hidden_states)
            hidden_states = hidden_states * per_layer_input
            hidden_states = self.per_layer_projection(hidden_states)
            hidden_states = self.post_per_layer_input_norm(hidden_states)
            hidden_states = residual + hidden_states

        hidden_states *= self.layer_scalar
        return hidden_states, mixer_out, mlp_out


class Gemma4HybridDecoderLayerBase(Gemma4DecoderLayerBody, GradientCheckpointingLayer, ABC):
    """
    Base class for hybrid decoder layers that replace Gemma 4's attention with alternative sequence mixers:
        - Mamba2
        - B'MOJO
        - Gated DeltaNet
        - Gated KalmaNet (GKA)
        - Sliding Window Attention (SWA)

    The base class keeps the rest of Gemma 4's decoder layer as it is: the MLP, the four norms around the MLP and
    the sequence mixer, the per-layer embedding block after the MLP, and the layer's scalar. Subclasses only build the
    sequence mixer. GKA, GDN, Mamba2, and B'MOJO take the config of their layer (config.per_layer_config[layer_idx]),
    whose head size is that of the base model's attention at the layer.

    Subclasses must implement:
        - `_init_seq_mixer(config, layer_idx)`: Initialize the sequence mixer (e.g., self.gka = ...)
        - `seq_mixer` property: Return the initialized sequence mixer instance
    """

    # Whether the sequence mixer takes the shared keys and values (SWA, which is Gemma 4's attention with a window)
    reads_shared_kv = False

    def __init__(self, config: HybridGemma4TextConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self._init_body(config, layer_idx)

        # All Hybrid decoder layer subclasses must implement this
        self._init_seq_mixer(config, layer_idx)

        # At a source whose keys and values later attention layers still read, the source's key and value
        # projections stay to compute them
        self.source_kv = (
            Gemma4SourceKeysAndValues(config, layer_idx) if config.keeps_source_keys_and_values(layer_idx) else None
        )

    @property
    def attention_type(self) -> str:
        """Returns the key of the layer's mask, the full attention mask, whose padding the sequence mixers read."""
        return "full_attention"

    @property
    @abstractmethod
    def seq_mixer(self) -> nn.Module:
        """Returns the instance of the decoder layer's sequence mixer."""
        pass

    @abstractmethod
    def _init_seq_mixer(self, config: HybridGemma4TextConfig, layer_idx: int):
        """Initializes the sequence mixer (e.g., Mamba2, B'MOJO, etc.)."""
        pass

    def forward(
        self,
        hidden_states: torch.Tensor,
        per_layer_input: torch.Tensor = None,
        shared_kv_states: dict[str, tuple[torch.Tensor, torch.Tensor]] | None = None,
        position_embeddings: torch.Tensor = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: HybridCache | None = None,
        use_cache: bool | None = False,
        **kwargs,
    ) -> torch.Tensor:
        """
        Does a forward pass through the hybrid decoder layer, as Gemma4DecoderLayer does, with a hybrid sequence
        mixer (such as GKA) in place of attention.

        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            per_layer_input (`torch.FloatTensor`, *optional*): this layer's slice of the per-layer embeddings, of
                shape `(batch, seq_len, hidden_size_per_layer_input)`
            shared_kv_states (`dict`, *optional*): the keys and values that the attention layers share, which a
                hybrid layer writes only when it replaces a source (see Gemma4SourceKeysAndValues)
            position_embeddings (`Tuple[torch.FloatTensor, torch.FloatTensor]`, *optional*): the cosine and sine
                positional embeddings of the base model's attention kind at this layer
            attention_mask (`torch.FloatTensor`, *optional*): the full attention mask, whose padding the sequence
                mixers read
            position_ids (`torch.LongTensor`, *optional*): Position IDs for the input.
            past_key_values (`HybridCache`, *optional*): the cache of every layer type
            use_cache (`bool`, *optional*): whether the sequence mixer keeps its state in the cache
            kwargs (`dict`, *optional*): Arbitrary kwargs to be ignored, used for FSDP and other methods that injects
                code into the model

        Returns:
            The layer's output hidden states, of shape `(batch, seq_len, hidden_size)`
        """

        def mixer(normed_hidden_states):
            if self.source_kv is not None:
                self.source_kv(normed_hidden_states, position_embeddings, shared_kv_states, past_key_values)

            # Sequence mixing (Mamba2, B'MOJO, GDN, etc.). SWA reads or writes the shared keys and values as Gemma
            # 4's attention does at its layer, while the other mixers neither read nor write them.
            if self.reads_shared_kv:
                kwargs["shared_kv_states"] = shared_kv_states
            mixer_out, _, _ = self.seq_mixer(
                hidden_states=normed_hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                position_embeddings=position_embeddings,
                **kwargs,
            )
            return mixer_out

        return self._body_forward(hidden_states, per_layer_input, mixer)[0]


class Gemma4Mamba2DecoderLayer(Gemma4HybridDecoderLayerBase):
    """Decoder layer with Mamba2 as the sequence mixer."""

    def _init_seq_mixer(self, config: HybridGemma4TextConfig, layer_idx: int):
        self.mamba = Mamba2Layer(
            config.per_layer_config[layer_idx],
            layer_idx=layer_idx,
            apply_rotary_pos_emb_fn=apply_rotary_pos_emb_qk,
            qk_norm=Gemma4RMSNorm,
        )

    @property
    def seq_mixer(self):
        return self.mamba


class Gemma4BMOJOFDecoderLayer(Gemma4HybridDecoderLayerBase):
    """Decoder layer with B'MOJO as the sequence mixer."""

    def _init_seq_mixer(self, config: HybridGemma4TextConfig, layer_idx: int):
        self.bmojo_f = BMojoLayer(
            config.per_layer_config[layer_idx],
            apply_rotary_pos_emb_fn=apply_rotary_pos_emb_qk,
            layer_idx=layer_idx,
            qk_norm=Gemma4RMSNorm,
        )

    @property
    def seq_mixer(self):
        return self.bmojo_f


class Gemma4GDNDecoderLayer(Gemma4HybridDecoderLayerBase):
    """Decoder layer with Gated DeltaNet (GDN) as the sequence mixer."""

    def _init_seq_mixer(self, config: HybridGemma4TextConfig, layer_idx: int):
        self.gdn = GatedDeltaNetLayer(config.per_layer_config[layer_idx], layer_idx=layer_idx)

    @property
    def seq_mixer(self):
        return self.gdn


class Gemma4GKADecoderLayer(Gemma4HybridDecoderLayerBase):
    """Decoder layer with Gated KalmaNet (GKA) as the sequence mixer."""

    def _init_seq_mixer(self, config: HybridGemma4TextConfig, layer_idx: int):
        self.gka = GatedKalmaNetLayer(config.per_layer_config[layer_idx], layer_idx=layer_idx)

    @property
    def seq_mixer(self):
        return self.gka


class Gemma4SWADecoderLayer(Gemma4HybridDecoderLayerBase):
    """Decoder layer with Sliding Window Attention (SWA) as the sequence mixer."""

    reads_shared_kv = True

    @property
    def attention_type(self) -> str:
        """
        Returns the key of the layer's mask, the mask with the window of swa_config, which may differ from Gemma 4's
        own window.
        """
        return self.swa.mask_key("swa")

    def _init_seq_mixer(self, config: HybridGemma4TextConfig, layer_idx: int):
        self.swa = Gemma4SWAttention(config, layer_idx=layer_idx)

    @property
    def seq_mixer(self):
        return self.swa


class Gemma4Stage1DecoderLayerBase(Gemma4DecoderLayerBody, GradientCheckpointingLayer, ABC):
    """
    Base class for the decoder layers of Stage 1 of hybridization (layerwise distillation), which take two streams
    of hidden states: the hybrid stream, which becomes the hybrid model after unfusing, and the attention stream,
    which is Gemma 4 itself and gives the targets. Both streams go through the layer's Gemma 4 attention (self_attn)
    or its hybrid sequence mixer, and then through the same MLP, norms, per-layer embedding block, and scalar. Each
    stream has its own shared keys and values, which its source layers compute from that stream. The attention
    stream runs without gradients, since its parameters are frozen in Stage 1 and its outputs are only targets.

    A model of these layers runs only through HybridGemma4TextModel.stage1_layer_outputs, which the HKD trainer
    calls. Subclasses must implement `_hybrid_mixer`, the sequence mixer of the hybrid stream.
    """

    def __init__(self, config: HybridGemma4TextConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self.self_attn = Gemma4TextAttention(config=config, layer_idx=layer_idx)
        self._init_body(config, layer_idx)

    @property
    def attention_type(self) -> str:
        """Returns the key of the attention stream's mask, that of the base model's attention kind at this layer."""
        return self.self_attn.mask_key(self.self_attn.layer_type)

    @property
    def hybrid_attention_type(self) -> str:
        """Returns the key of the hybrid stream's mask, which for '*DA' is the attention stream's."""
        return self.attention_type

    def _attention_mixer(
        self, hidden_states, position_embeddings, attention_mask, shared_kv_states, position_ids, **kwargs
    ) -> torch.Tensor:
        """Runs the layer's Gemma 4 attention on the normalized hidden states of either stream."""
        attn_output, _ = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            shared_kv_states=shared_kv_states,
            position_ids=position_ids,
            **kwargs,
        )
        return attn_output

    @abstractmethod
    def _hybrid_mixer(
        self, hidden_states, position_embeddings, attention_mask, shared_kv_states, position_ids, **kwargs
    ) -> torch.Tensor:
        """Runs the sequence mixer of the hybrid stream on its normalized hidden states."""
        pass

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_hidden_states: torch.Tensor,
        per_layer_input: torch.Tensor | None = None,
        shared_kv_states: dict[str, tuple[torch.Tensor, torch.Tensor]] | None = None,
        attention_shared_kv_states: dict[str, tuple[torch.Tensor, torch.Tensor]] | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] = None,
        attention_mask: torch.Tensor | None = None,
        hybrid_attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Runs the hybrid stream through the layer with gradients, and the attention stream without them.

        Args:
            hidden_states (`torch.FloatTensor`): the hybrid stream's input, of shape `(batch, seq_len, hidden_size)`
            attention_hidden_states (`torch.FloatTensor`): the attention stream's input, of the same shape
            per_layer_input (`torch.FloatTensor`, *optional*): this layer's slice of the per-layer embeddings, which
                both streams use
            shared_kv_states (`dict`, *optional*): the shared keys and values of the hybrid stream
            attention_shared_kv_states (`dict`, *optional*): the shared keys and values of the attention stream
            position_embeddings (`Tuple[torch.FloatTensor, torch.FloatTensor]`, *optional*): the cosine and sine
                positional embeddings of the base model's attention kind at this layer
            attention_mask (`torch.FloatTensor`, *optional*): the attention stream's mask (see attention_type)
            hybrid_attention_mask (`torch.FloatTensor`, *optional*): the hybrid stream's mask (see
                hybrid_attention_type)
            position_ids (`torch.LongTensor`, *optional*): Position IDs for the input
            kwargs (`dict`, *optional*): Arbitrary kwargs to be ignored, used for FSDP and other methods that injects
                code into the model

        Returns:
            - hybrid_out: the hybrid stream's output
            - attn_out: the attention stream's output
            - hybrid_mixer_out: the hybrid stream's sequence mixer output, as the layer adds it to the residual
            - attn_mixer_out: the attention stream's attention output, as the layer adds it to the residual
            - hybrid_mlp_out: the hybrid stream's MLP output, as the layer adds it to the residual
            - attn_mlp_out: the attention stream's MLP output, as the layer adds it to the residual
        """
        hybrid_out, hybrid_mixer_out, hybrid_mlp_out = self._body_forward(
            hidden_states,
            per_layer_input,
            lambda normed: self._hybrid_mixer(
                normed, position_embeddings, hybrid_attention_mask, shared_kv_states, position_ids, **kwargs
            ),
        )
        with torch.no_grad():
            attn_out, attn_mixer_out, attn_mlp_out = self._body_forward(
                attention_hidden_states,
                per_layer_input,
                lambda normed: self._attention_mixer(
                    normed, position_embeddings, attention_mask, attention_shared_kv_states, position_ids, **kwargs
                ),
            )
        return hybrid_out, attn_out, hybrid_mixer_out, attn_mixer_out, hybrid_mlp_out, attn_mlp_out


class Gemma4DualAttDecoderLayer(Gemma4Stage1DecoderLayerBase):
    """
    The '*DA' layer of Stage 1: Gemma 4's attention on both streams, which becomes Gemma 4's attention ('*') after
    unfusing. In the hybrid stream, a source stores the keys and values that it computes from the hybrid stream, and
    a sharing layer reads those of the hybrid stream.
    """

    def _hybrid_mixer(self, hidden_states, position_embeddings, attention_mask, shared_kv_states, position_ids,
                      **kwargs):
        return self._attention_mixer(
            hidden_states, position_embeddings, attention_mask, shared_kv_states, position_ids, **kwargs
        )


class Gemma4FusedAttHybridDecoderLayerBase(Gemma4Stage1DecoderLayerBase):
    """
    Base class for the fused layers of Stage 1 ('X*', such as 'GKA*'), which run a hybrid sequence mixer on the
    hybrid stream and Gemma 4's frozen attention on the attention stream, and become the hybrid layer 'X' after
    unfusing. The mixer gets the config of its layer, as in Gemma4HybridDecoderLayerBase.

    At a source whose keys and values later layers of the hybrid stream still read (see
    HybridGemma4TextConfig.keeps_source_keys_and_values), the layer also computes them from the hybrid stream with
    the frozen key and value projections of its attention, as the unfused layer's Gemma4SourceKeysAndValues will.

    Subclasses must implement:
        - `_init_seq_mixer(config, layer_idx)`: Initialize the sequence mixer (e.g., self.gka = ...)
        - `seq_mixer` property: Return the initialized sequence mixer instance
    """

    # Whether the sequence mixer takes the shared keys and values (SWA, which is Gemma 4's attention with a window)
    reads_shared_kv = False

    def __init__(self, config: HybridGemma4TextConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self._init_seq_mixer(config, layer_idx)
        self.computes_source_kv = config.keeps_source_keys_and_values(layer_idx)

    @property
    def hybrid_attention_type(self) -> str:
        """Returns the key of the hybrid stream's mask, the full attention mask, whose padding the mixers read."""
        return "full_attention"

    @property
    @abstractmethod
    def seq_mixer(self) -> nn.Module:
        """Returns the instance of the decoder layer's sequence mixer."""
        pass

    @abstractmethod
    def _init_seq_mixer(self, config: HybridGemma4TextConfig, layer_idx: int):
        """Initializes the sequence mixer (such as Mamba2 or GKA)."""
        pass

    def _hybrid_mixer(self, hidden_states, position_embeddings, attention_mask, shared_kv_states, position_ids,
                      **kwargs):
        if self.computes_source_kv:
            shared_kv_states[self.self_attn.layer_type] = project_keys_and_values(
                self.self_attn, hidden_states, position_embeddings
            )
        if self.reads_shared_kv:
            kwargs["shared_kv_states"] = shared_kv_states
        mixer_out, _, _ = self.seq_mixer(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=None,
            use_cache=False,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        return mixer_out


# The fused layers build their sequence mixers as the hybrid layers do, under the same names, so that unfusing keeps
# the mixers' weights
class Gemma4FusedAttMamba2DecoderLayer(Gemma4FusedAttHybridDecoderLayerBase):
    """Stage 1's fused layer with Mamba2 as the hybrid stream's sequence mixer."""

    _init_seq_mixer = Gemma4Mamba2DecoderLayer._init_seq_mixer
    seq_mixer = Gemma4Mamba2DecoderLayer.seq_mixer


class Gemma4FusedAttGDNDecoderLayer(Gemma4FusedAttHybridDecoderLayerBase):
    """Stage 1's fused layer with Gated DeltaNet (GDN) as the hybrid stream's sequence mixer."""

    _init_seq_mixer = Gemma4GDNDecoderLayer._init_seq_mixer
    seq_mixer = Gemma4GDNDecoderLayer.seq_mixer


class Gemma4FusedAttGKADecoderLayer(Gemma4FusedAttHybridDecoderLayerBase):
    """Stage 1's fused layer with Gated KalmaNet (GKA) as the hybrid stream's sequence mixer."""

    _init_seq_mixer = Gemma4GKADecoderLayer._init_seq_mixer
    seq_mixer = Gemma4GKADecoderLayer.seq_mixer


class Gemma4FusedAttSWADecoderLayer(Gemma4FusedAttHybridDecoderLayerBase):
    """Stage 1's fused layer with Sliding Window Attention (SWA) as the hybrid stream's sequence mixer."""

    reads_shared_kv = True
    _init_seq_mixer = Gemma4SWADecoderLayer._init_seq_mixer
    seq_mixer = Gemma4SWADecoderLayer.seq_mixer

    @property
    def hybrid_attention_type(self) -> str:
        """Returns the key of the hybrid stream's mask, the mask with the window of swa_config."""
        return self.swa.mask_key("swa")


class Gemma4TextScaledWordEmbedding(nn.Embedding):
    """
    This module overrides nn.Embeddings' forward by multiplying with embeddings scale.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: int, embed_scale: float = 1.0):
        super().__init__(num_embeddings, embedding_dim, padding_idx)
        self.scalar_embed_scale = embed_scale
        self.embed_scale = nn.Buffer(torch.tensor(embed_scale), persistent=False)

    def forward(self, input_ids: torch.Tensor):
        return super().forward(input_ids) * self.embed_scale.to(self.weight.dtype)


# Build registry of Hybrid Layers
HYBRID_GEMMA4_DECODER_LAYER_REGISTRY = build_decoder_layer_registry(
    module_or_dict=sys.modules[__name__],
    model_prefix="Gemma4",
)


@auto_docstring
class HybridGemma4PreTrainedModel(PreTrainedModel):
    config_class = HybridGemma4TextConfig
    config: HybridGemma4TextConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = [name for name in globals() if name.endswith("DecoderLayer") and "Gemma4" in name]
    _skip_keys_device_placement = ["past_key_values", "shared_kv_states"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True

    _can_compile_fullgraph = True
    _supports_attention_backend = True
    _can_record_outputs = {
        "hidden_states": [Gemma4DecoderLayer, Gemma4HybridDecoderLayerBase],
        "attentions": Gemma4TextAttention,
    }

    @torch.no_grad()
    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, Gemma4TextRotaryEmbedding):
            for layer_type, rope_init_fn in module.rope_init_fns.items():
                rope_config = layer_config_for_kind(module.config, layer_type)
                curr_inv_freq, _ = rope_init_fn(rope_config, layer_type=layer_type)
                init.copy_(getattr(module, f"{layer_type}_inv_freq"), curr_inv_freq)
                init.copy_(getattr(module, f"{layer_type}_original_inv_freq"), curr_inv_freq)
        elif isinstance(module, Gemma4TextScaledWordEmbedding):
            init.constant_(module.embed_scale, module.scalar_embed_scale)
        elif isinstance(module, (Gemma4DecoderLayer, Gemma4DecoderLayerBody)):
            init.ones_(module.layer_scalar)


@auto_docstring(custom_intro="The base hybrid Gemma 4 language model without a language modeling head.")
class HybridGemma4TextModel(HybridGemma4PreTrainedModel):
    def __init__(self, config: HybridGemma4TextConfig):
        super().__init__(config)
        if config.enable_moe_block:
            raise NotImplementedError("Hybrid Gemma 4 does not support the mixture-of-experts block yet.")
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        # Gemma4 downcasts the below to bfloat16, causing sqrt(3072)=55.4256 to become 55.5.
        # See https://github.com/huggingface/transformers/pull/29402
        self.embed_tokens = Gemma4TextScaledWordEmbedding(
            config.vocab_size, config.hidden_size, self.padding_idx, embed_scale=self.config.hidden_size**0.5
        )
        self.layers = nn.ModuleList(self._set_decoder_layers(config))
        self.norm = Gemma4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Gemma4TextRotaryEmbedding(config)
        self.gradient_checkpointing = False
        self.unique_layer_types = set(self.config.base_layer_types)
        self.is_stage1 = any(isinstance(layer, Gemma4Stage1DecoderLayerBase) for layer in self.layers)

        # Per-Layer Embeddings (PLE): auxiliary embedding that feeds a residual signal
        # into each decoder layer. See `get_per_layer_inputs()` and `project_per_layer_inputs()`
        # for the full pipeline. The embedding is packed: total dim = num_layers * per_layer_dim.
        self.hidden_size_per_layer_input = config.hidden_size_per_layer_input
        if self.hidden_size_per_layer_input:
            self.embed_tokens_per_layer = Gemma4TextScaledWordEmbedding(
                config.vocab_size_per_layer_input,
                config.num_hidden_layers * config.hidden_size_per_layer_input,
                self.padding_idx,
                embed_scale=config.hidden_size_per_layer_input**0.5,
            )
            self.per_layer_input_scale = 2.0**-0.5
            self.per_layer_model_projection = nn.Linear(
                config.hidden_size,
                config.num_hidden_layers * config.hidden_size_per_layer_input,
                bias=False,
            )
            self.per_layer_model_projection_scale = config.hidden_size**-0.5
            self.per_layer_projection_norm = Gemma4RMSNorm(config.hidden_size_per_layer_input, eps=config.rms_norm_eps)

        # Update `_keys_to_ignore_on_load_unexpected` to drop all k/v proj and norms for the shared layers
        self._keys_to_ignore_on_load_unexpected = []
        for i, layer in enumerate(self.layers):
            if getattr(getattr(layer, "self_attn", None), "is_kv_shared_layer", False):
                self._keys_to_ignore_on_load_unexpected.extend(
                    [f"layers.{i}.self_attn.{name}" for name in ("k_proj", "v_proj", "k_norm", "v_norm")]
                )

        # Initialize weights and apply final processing
        self.post_init()

    def _set_decoder_layers(self, config):
        """
        Sets each decoder layer of the model from the hybrid override pattern in the model config, or Gemma 4's own
        decoder layers when there is no pattern.
        """
        if config.hybrid_override_pattern is None:
            return [Gemma4DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        return set_decoder_layers_from_pattern(config, HYBRID_GEMMA4_DECODER_LAYER_REGISTRY)

    def _mask_keys(self) -> set[str]:
        """
        Returns the keys of the masks that the layers ask for (see each layer's attention_type), including those of the
        hybrid streams of Stage 1's layers, and the full attention mask. They follow the model's attention
        implementation, which may change after the model is built.
        """
        keys = {getattr(layer, kind) for layer in self.layers
                for kind in ("attention_type", "hybrid_attention_type") if hasattr(layer, kind)}
        if any(key.endswith(SDPA_MASK_SUFFIX) for key in keys):
            logger.warning_once(
                "Hybrid Gemma 4's attention layers with heads of more than "
                f"{FLASH_ATTENTION_MAX_HEAD_DIM} (its full attention layers) run SDPA, which flash attention 2 "
                "doesn't support, while its other attention layers run flash attention 2."
            )
        return keys | {"full_attention"}

    def _causal_masks(self, mask_kwargs: dict) -> dict:
        """
        Returns the masks that the layers ask for (see _mask_keys): the full attention mask, Gemma 4's sliding window
        mask, and the mask with the window of swa_config for HMF's SWA layers, each in the format of the model's
        attention implementation, or in SDPA's for the layers that run SDPA (keys ending in SDPA_MASK_SUFFIX).
        """
        masks = {}
        for key in self._mask_keys():
            kind, sdpa = key.removesuffix(SDPA_MASK_SUFFIX), key.endswith(SDPA_MASK_SUFFIX)
            config = self.config
            if kind == "swa" or sdpa:
                config = copy.copy(self.config)
                if kind == "swa":
                    config.sliding_window = self.config.swa_config["window_size"]
                if sdpa:
                    config._attn_implementation = "sdpa"
            create_mask = create_causal_mask if kind == "full_attention" else create_sliding_window_causal_mask
            masks[key] = create_mask(**{**mask_kwargs, "config": config})
        return masks

    @merge_with_config_defaults
    @capture_outputs
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        per_layer_inputs: torch.Tensor | None = None,
        use_cache: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        r"""
        per_layer_inputs (`torch.Tensor`, *optional*):
            Pre-computed per-layer input text embeddings of shape `(batch_size, sequence_length, num_hidden_layers,
            hidden_size_per_layer_input)`. When provided, these are used directly instead of being computed from
            `input_ids` via `get_per_layer_inputs()` in the text model.
        """
        if self.is_stage1:
            raise ValueError(
                "This model has the fused and '*DA' layers of Stage 1, which run only through stage1_layer_outputs "
                "(the HKD trainer). Unfuse it first (fused_to_standard.py) to run it as a model."
            )
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if input_ids is not None and per_layer_inputs is not None:
            raise ValueError("You cannot specify per_layer_inputs if input_ids is provided")

        if input_ids is not None:
            inputs_embeds = self.embed_tokens(input_ids)

        if self.hidden_size_per_layer_input:
            if per_layer_inputs is None:
                per_layer_inputs = self.get_per_layer_inputs(input_ids, inputs_embeds)
            per_layer_inputs = self.project_per_layer_inputs(inputs_embeds, per_layer_inputs)

        if not self.training and use_cache and (
            past_key_values is None or not isinstance(past_key_values, HybridCache)
        ):
            logger.warning_once(
                "You are passing `use_cache=True` and `past_key_values` is `None`. "
                "We are setting `past_key_values` to a new instance of `HybridCache`."
            )
            del past_key_values
            past_key_values = HybridCache(self.config)

        if position_ids is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
            position_ids = position_ids.unsqueeze(0)

        # It may already have been prepared by e.g. `generate`
        if not isinstance(causal_mask_mapping := attention_mask, dict):
            # When using sequence parallelism, set attention_mask to None.
            # Tested with flash attention only.
            if getattr(self, "sequence_parallel_group", None) is not None:
                causal_mask_mapping = dict.fromkeys(self._mask_keys())
            else:
                causal_mask_mapping = self._causal_masks({
                    "config": self.config,
                    "inputs_embeds": inputs_embeds,
                    "attention_mask": attention_mask,
                    "past_key_values": past_key_values,
                    "position_ids": position_ids,
                })

        # embed positions
        hidden_states = inputs_embeds
        position_embeddings = {}
        for layer_type in self.unique_layer_types:
            position_embeddings[layer_type] = self.rotary_emb(hidden_states, position_ids, layer_type)

        # Initialize as empty dict, or reuse past shared states. We use a UserDict instead of built-in dict (it behaves
        # the same) for fsdp2 support (otherwise, `_apply_to_tensors` rebuilds every dict it recurses into, and
        # `shared_kv_states` is not correctly shared, see
        # https://github.com/pytorch/pytorch/blob/v2.10.0/torch/distributed/utils.py#L223-L255)
        shared_kv_states = kwargs.pop("shared_kv_states", UserDict())

        # decoder layers
        for i, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            per_layer_input = per_layer_inputs[:, :, i, :] if per_layer_inputs is not None else None

            hidden_states = decoder_layer(
                hidden_states,
                per_layer_input,
                shared_kv_states=shared_kv_states,
                position_embeddings=position_embeddings[self.config.base_layer_types[i]],
                attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )

    def stage1_layer_outputs(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
    ) -> Iterator[tuple[torch.Tensor, ...]]:
        """
        Runs the two streams of a Stage 1 model (whose layers are all fused or '*DA' layers) for layerwise
        distillation, and yields each decoder layer's six outputs (see Gemma4Stage1DecoderLayerBase.forward). Both
        streams start from the scaled token embeddings and use the same per-layer inputs, masks, and RoPE of each
        attention kind, as forward does, while each keeps its own shared keys and values. The HKD trainer calls it
        in place of its own loop over the layers, which builds one full mask and one RoPE, and has no per-layer
        inputs or shared keys and values.
        """
        if not all(isinstance(layer, Gemma4Stage1DecoderLayerBase) for layer in self.layers):
            raise ValueError("stage1_layer_outputs needs a model whose layers are all fused or '*DA' layers.")

        inputs_embeds = self.embed_tokens(input_ids)
        per_layer_inputs = None
        if self.hidden_size_per_layer_input:
            per_layer_inputs = self.project_per_layer_inputs(
                inputs_embeds, self.get_per_layer_inputs(input_ids, inputs_embeds)
            )
        if position_ids is None:
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device).unsqueeze(0)

        if getattr(self, "sequence_parallel_group", None) is not None:
            masks = dict.fromkeys(self._mask_keys())
        else:
            masks = self._causal_masks({
                "config": self.config,
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "past_key_values": None,
                "position_ids": position_ids,
            })
        position_embeddings = {
            layer_type: self.rotary_emb(inputs_embeds, position_ids, layer_type)
            for layer_type in self.unique_layer_types
        }

        # UserDicts, as in forward, so that FSDP2 keeps them shared across the layers
        shared_kv_states, attention_shared_kv_states = UserDict(), UserDict()
        hidden_states = attention_hidden_states = inputs_embeds
        for i, decoder_layer in enumerate(self.layers):
            layer_outputs = decoder_layer(
                hidden_states,
                attention_hidden_states,
                per_layer_inputs[:, :, i, :] if per_layer_inputs is not None else None,
                shared_kv_states=shared_kv_states,
                attention_shared_kv_states=attention_shared_kv_states,
                position_embeddings=position_embeddings[self.config.base_layer_types[i]],
                attention_mask=masks[decoder_layer.attention_type],
                hybrid_attention_mask=masks[decoder_layer.hybrid_attention_type],
                position_ids=position_ids,
            )
            hidden_states, attention_hidden_states = layer_outputs[0], layer_outputs[1]
            yield layer_outputs

    def get_per_layer_inputs(
        self, input_ids: torch.Tensor | None, inputs_embeds: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Compute the token-identity component of Per-Layer Embeddings (PLE).

        Looks up `input_ids` in `embed_tokens_per_layer` (a scaled embedding that multiplies
        by `sqrt(hidden_size_per_layer_input)`) and reshapes the packed output from
        `[batch, seq, num_hidden_layers * hidden_size_per_layer_input]` to
        `[batch, seq, num_hidden_layers, hidden_size_per_layer_input]`.

        If only `inputs_embeds` is provided (no `input_ids`), reverses the main embedding
        to recover `input_ids` for the PLE lookup.
        """
        if not self.hidden_size_per_layer_input:
            raise RuntimeError(
                "Attempting to call get_per_layer_inputs() from a model initialized with a config that does not "
                f"support per-layer embeddings. {self.config}"
            )

        # If only inputs_embeds are provided, reverse main embedding to find the input_ids - this allows to `generate`
        # from `inputs_embeds` only as other models (otherwise it would need the value from both embeddings)
        if input_ids is None:
            with torch.no_grad():
                input_ids = (
                    (
                        inputs_embeds[:, :, None, :]
                        == self.embed_tokens.weight[None, None, :, :] * self.config.hidden_size**0.5
                    )
                    .all(dim=3)
                    .nonzero()[:, 2]
                )
                try:
                    input_ids = input_ids.view(inputs_embeds.shape[:2])
                except RuntimeError:
                    raise RuntimeError(
                        "It seems like you tried to call `forward` from `inputs_embeds` without providing "
                        "`input_ids`, and that the `inputs_embeds` you provided do not exactly match the embedding "
                        "weights. Since Gemma4 needs to reverse the embedding to compute another embedding, make sure "
                        "you provide exact `inputs_embeds`"
                    )

        return self.embed_tokens_per_layer(input_ids).reshape(
            *input_ids.shape,
            self.config.num_hidden_layers,
            self.hidden_size_per_layer_input,
        )

    def project_per_layer_inputs(
        self,
        inputs_embeds: torch.Tensor,
        per_layer_inputs: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute the context-aware component of PLE and combine with token-identity.

        Projects `inputs_embeds` through `per_layer_model_projection` (Linear), scales by
        `1/sqrt(hidden_size)`, reshapes to `[batch, seq, num_layers, ple_dim]`, and normalizes
        with `per_layer_projection_norm` (RMSNorm).

        If `per_layer_inputs` (the token-identity component from `get_per_layer_inputs()`)
        is provided, combines both: `(context_projection + token_identity) * (1/sqrt(2))`.
        If `per_layer_inputs` is None (e.g. for multimodal inputs where input_ids are not
        available), returns just the context projection.
        """
        if not self.hidden_size_per_layer_input:
            raise RuntimeError(
                "Attempting to call project_per_layer_inputs() from a model initialized with a config that does not"
                f" support per-layer embeddings. {self.config}"
            )

        per_layer_projection = self.per_layer_model_projection(inputs_embeds) * self.per_layer_model_projection_scale
        per_layer_projection = per_layer_projection.reshape(
            *inputs_embeds.shape[:-1],
            self.config.num_hidden_layers,
            self.hidden_size_per_layer_input,
        )
        per_layer_projection = self.per_layer_projection_norm(per_layer_projection)

        if per_layer_inputs is None:
            return per_layer_projection

        return (per_layer_projection + per_layer_inputs) * self.per_layer_input_scale


@auto_docstring(custom_intro="The base hybrid Gemma 4 language model with a language modeling head.")
class HybridGemma4ForCausalLM(HybridGemma4PreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    _tp_plan = {"lm_head": "colwise_gather_output"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config: HybridGemma4TextConfig):
        super().__init__(config)
        self.model = HybridGemma4TextModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def _prepare_cache_for_generation(self, generation_config, model_kwargs, *args, **kwargs):
        """
        Gives generate() a HybridCache, which holds the state of every layer type. Otherwise, generate() would
        build transformers' DynamicCache from config.layer_types, and a DynamicCache cannot hold the states of the
        Mamba2, B'MOJO, GDN, and GKA layers.
        """
        if model_kwargs.get("past_key_values") is None and generation_config.use_cache is not False:
            model_kwargs["past_key_values"] = HybridCache(self.config)
        return super()._prepare_cache_for_generation(generation_config, model_kwargs, *args, **kwargs)

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        per_layer_inputs: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        r"""
        per_layer_inputs (`torch.Tensor`, *optional*):
            Pre-computed per-layer input text embeddings of shape `(batch_size, sequence_length, num_hidden_layers,
            hidden_size_per_layer_input)`. When provided, these are used directly instead of being computed from
            `input_ids` via `get_per_layer_inputs()` in the text model.
        """
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            per_layer_inputs=per_layer_inputs,
            use_cache=use_cache,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        hidden_states = slice_logits_to_keep(hidden_states, logits_to_keep)
        logits = self.lm_head(hidden_states)
        if self.config.final_logit_softcapping is not None:
            logits = logits / self.config.final_logit_softcapping
            logits = torch.tanh(logits)
            logits = logits * self.config.final_logit_softcapping

        loss = None
        if labels is not None:
            loss = self.loss_function(logits, labels, self.vocab_size, **kwargs)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


__all__ = [
    "HybridGemma4ForCausalLM",
    "HybridGemma4PreTrainedModel",
    "HybridGemma4TextModel",
]
