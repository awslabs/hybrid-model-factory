# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
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

"""PyTorch Hybrid Llama model.

This module implements a hybrid variant of the Llama architecture that supports replacing standard
attention layers with alternative sequence mixing mechanisms. The hybrid architecture enables
experimentation with and deployment of models that combine attention with other efficient sequence
processing layers.

Supported sequence mixing layers:
    - Mamba2
    - B'MOJO
    - Gated DeltaNet (GDN)
    - Gated KalmaNet (GKA)
    - Sliding Window Attention (SWA)

Architecture:
    This model architecture is based on Llama, but allows flexible layer composition through a
    hybrid override pattern specified in the model configuration. Each decoder layer can be either:
    - Standard Llama attention layer (LlamaDecoderLayer)
    - One of several hybrid layer types (subclasses of LlamaHybridDecoderLayerBase)
    - Fused attention-hybrid layers for layerwise MSE distillation (subclasses of LlamaFusedAttHybridDecoderLayer)

Hybrid models can be built as outlined in hybridization/.

Note: Unlike Qwen3, Llama does NOT use QK-norm in attention layers.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Optional, Tuple
import sys

import torch
from torch import nn

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin
from transformers.integrations import use_kernel_forward_from_hub, use_kernel_func_from_hub, use_kernelized_func
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_layers import (
    GenericForQuestionAnswering,
    GenericForSequenceClassification,
    GenericForTokenClassification,
    GradientCheckpointingLayer,
)
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, auto_docstring, can_return_tuple, logging
from transformers.utils.generic import maybe_autocast, merge_with_config_defaults
from transformers.utils.output_capturing import capture_outputs

from ...layers.bmojo_f.modules import BMojoLayer
from ...layers.gated_deltanet.modules import GatedDeltaNetLayer
from ...layers.gated_kalmanet.modules import GatedKalmaNetLayer
from ...layers.mamba.modules import Mamba2Layer
from ..cache import HybridCache
from .configuration_hybrid_llama import HybridLlamaConfig
from ..utils import build_decoder_layer_registry, set_decoder_layers_from_pattern
from ...layers.swa.sequence_parallel import (
    roll_first_half,
    swa_sp_attn_override,
    swa_sp_gather,
    swa_sp_post_attn,
)

logger = logging.get_logger(__name__)


@use_kernel_forward_from_hub("RMSNorm")
class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj



class LlamaRotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor  # fix linting for `register_buffer`

    def __init__(self, config: HybridLlamaConfig, device=None):
        super().__init__()
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config

        self.rope_type = self.config.rope_parameters["rope_type"]
        rope_init_fn: Callable = self.compute_default_rope_parameters
        if self.rope_type != "default":
            rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]
        inv_freq, self.attention_scaling = rope_init_fn(self.config, device)

        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.register_buffer("original_inv_freq", inv_freq.clone(), persistent=False)

    @staticmethod
    def compute_default_rope_parameters(
        config: HybridLlamaConfig | None = None,
        device: Optional["torch.device"] = None,
        seq_len: int | None = None,
    ) -> tuple["torch.Tensor", float]:
        """
        Computes the inverse frequencies according to the original RoPE implementation
        """
        base = config.rope_parameters["rope_theta"]
        dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads

        attention_factor = 1.0  # Unused in this type of RoPE

        # Compute the inverse frequencies
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim)
        )
        return inv_freq, attention_factor

    @torch.no_grad()
    @dynamic_rope_update  # power user: used with advanced RoPE types (e.g. dynamic rope)
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with maybe_autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


@use_kernel_func_from_hub("rotary_pos_emb")
def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


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
    scaling: float,
    dropout: float = 0.0,
    **kwargs: Unpack[TransformersKwargs],
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights



@use_kernelized_func(apply_rotary_pos_emb)
class LlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: HybridLlamaConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )
        # Note: Llama does NOT use QK-norm (unlike Qwen3)
        # Added below for consistency with Qwen models
        self.layer_type = config.layer_types[layer_idx] if hasattr(config, "layer_types") else None
        self.sliding_window = config.sliding_window if self.layer_type == "sliding_attention" else None

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class LlamaSWAttention(LlamaAttention):
    """Sliding Window Attention for Llama."""

    def __init__(self, config: HybridLlamaConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.sequence_parallel_group = None

        swa_config = config.swa_config
        self.sliding_window = swa_config.get("window_size", None)

        if self.sliding_window is None:
            raise ValueError("swa_config must specify window_size.")

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[HybridCache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        
        if self.sequence_parallel_group is not None:
            return self._forward_sp(
                hidden_states, position_embeddings, attention_mask,
                past_key_values, cache_position, **kwargs,
            )
        
        attn_output, attn_weights = super().forward(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            cache_position=cache_position,
            **kwargs
        )
        
        # Return 3-tuple to be consistent with all other Hybrid layer outputs 
        return attn_output, attn_weights, None


    def _forward_sp(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[HybridCache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        assert attention_mask is None, (
            "varlen support for SWA with SP is not yet supported."
        )
        sp_group = self.sequence_parallel_group
        sp_rank = torch.distributed.get_rank(group=sp_group)

        # Gather boundary tokens from neighbors
        hidden_states, position_embeddings, orig_bsz, orig_seq_len = swa_sp_gather(
            hidden_states, position_embeddings, sp_group, self.sliding_window,
        )

        # For rank 0, the first chunk has zero-padded boundary tokens (no previous chunk).
        # Roll them to the end so they don't affect the causal attention computation.
        if sp_rank == 0:
            q_len = orig_seq_len // 2
            hidden_states = roll_first_half(hidden_states, q_len)
            cos = roll_first_half(position_embeddings[0], q_len)
            sin = roll_first_half(position_embeddings[1], q_len)
            position_embeddings = (cos, sin)

        # SWA handles its own SP via P2P gather, so we must use flash_attention_2
        # here — not the globally-patched sequence_parallel_attention (ring attention),
        # which doesn't support sliding window.
        with swa_sp_attn_override(self.config):
            attn_output, attn_weights = super().forward(
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                cache_position=cache_position,
                **kwargs,
            )

        # Trim overlap and recombine
        attn_output = swa_sp_post_attn(
            attn_output, sp_group, self.sliding_window, orig_bsz,
        )

        return attn_output, attn_weights, None

class LlamaDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: HybridLlamaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = LlamaAttention(config=config, layer_idx=layer_idx)

        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        if config.layer_types[layer_idx] in ["full_attention", "sliding_attention"]:
            self.attention_type = config.layer_types[layer_idx]
        else:
            self.attention_type = "full_attention"

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: HybridCache | None = None,
        use_cache: bool | None = False,
        cache_position: torch.LongTensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        # Self Attention
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states


class LlamaHybridDecoderLayerBase(GradientCheckpointingLayer, ABC):
    """
    Base class for hybrid decoder layers that replace standard attention with alternative sequence mixers.

    This abstract class provides a template for decoder layers where the sequence mixing mechanism
    can be one of several alternatives to standard attention:
        - Mamba2
        - B'MOJO
        - Gated DeltaNet
        - Gated KalmaNet (GKA)

    The base class handles the residual connections, layer normalization, and MLP, while subclasses
    only need to implement the specific sequence mixer initialization.

    Subclasses must implement:
        - `_init_seq_mixer(config, layer_idx)`: Initialize the sequence mixer (e.g., self.mamba = ...)
        - `seq_mixer` property: Return the initialized sequence mixer instance
    """

    def __init__(self, config: HybridLlamaConfig, layer_idx: int):
        super().__init__()
        
        self.hidden_size = config.hidden_size
        self.attention_type = "full_attention"

        # Norms
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # MLP
        self.mlp = LlamaMLP(config)

        # All Hybrid decoder layer subclasses must implement this
        self._init_seq_mixer(config, layer_idx)

    @property
    @abstractmethod
    def seq_mixer(self) -> nn.Module:
        """Returns the instance of the decoder layer's sequence mixer."""
        pass

    @abstractmethod
    def _init_seq_mixer(self, config: HybridLlamaConfig, layer_idx: int):
        """Initializes the sequence mixer (e.g., Mamba2, B'MOJO, etc.)."""
        pass

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[HybridCache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> torch.FloatTensor:
        """
        Does a forward pass through the Hybrid decoder layer.
        """
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Sequence mixing (Mamba2, B'MOJO, GDN, etc.)
        hidden_states, _, _ = self.seq_mixer(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )

        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states



class LlamaMamba2DecoderLayer(LlamaHybridDecoderLayerBase):
    """Decoder layer with Mamba2 as the sequence mixer."""

    def _init_seq_mixer(self, config: HybridLlamaConfig, layer_idx: int):
        # Llama does NOT use QK-norm, so we pass qk_norm=None
        self.mamba = Mamba2Layer(
            config,
            layer_idx=layer_idx,
            apply_rotary_pos_emb_fn=apply_rotary_pos_emb,
            qk_norm=None,
        )

    @property
    def seq_mixer(self):
        return self.mamba


class LlamaBMOJOFDecoderLayer(LlamaHybridDecoderLayerBase):
    """Decoder layer with B'MOJO as the sequence mixer."""

    def _init_seq_mixer(self, config: HybridLlamaConfig, layer_idx: int):
        # Llama does NOT use QK-norm, so we pass qk_norm=None
        self.bmojo_f = BMojoLayer(
            config,
            apply_rotary_pos_emb_fn=apply_rotary_pos_emb,
            layer_idx=layer_idx,
            qk_norm=None,
        )

    @property
    def seq_mixer(self):
        return self.bmojo_f


class LlamaGDNDecoderLayer(LlamaHybridDecoderLayerBase):
    """Decoder layer with Gated DeltaNet (GDN) as the sequence mixer."""

    def _init_seq_mixer(self, config: HybridLlamaConfig, layer_idx: int):
        self.gdn = GatedDeltaNetLayer(config, layer_idx=layer_idx)

    @property
    def seq_mixer(self):
        return self.gdn


class LlamaGKADecoderLayer(LlamaHybridDecoderLayerBase):
    """Decoder layer with Gated KalmaNet (GKA) as the sequence mixer."""

    def _init_seq_mixer(self, config: HybridLlamaConfig, layer_idx: int):
        self.gka = GatedKalmaNetLayer(config, layer_idx=layer_idx)

    @property
    def seq_mixer(self):
        return self.gka


class LlamaSWADecoderLayer(LlamaHybridDecoderLayerBase):
    """Decoder layer with Sliding Window Attention (SWA) as the sequence mixer."""

    def __init__(self, config, layer_idx):
        super().__init__(config, layer_idx)
        self.attention_type = "sliding_attention"

    def _init_seq_mixer(self, config: HybridLlamaConfig, layer_idx: int):
        self.swa = LlamaSWAttention(config, layer_idx=layer_idx)

    @property
    def seq_mixer(self):
        return self.swa



class LlamaDualAttDecoderLayer(nn.Module):
    """
    This layer is used as the Attention layer for Stage 1 of our Hybridization pipeline 
    (layerwise distillation). It is essentially a standard Attention layer, except it takes two
    hidden states instead of one: the Hybrid layer's hidden states and the Attention hidden states.
    It processes both through Attention (independently) and then returns both new hidden states.
    """

    def __init__(self, config: HybridLlamaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.attention_type = "full_attention"

        # Shared Attention for both paths
        self.self_attn = LlamaAttention(config, layer_idx)

        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def _forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Standard forward pass for Attention."""
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        attn_mixer_out, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )
        hidden_states = residual + attn_mixer_out

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states_mlp = self.mlp(hidden_states)
        hidden_states = residual + hidden_states_mlp

        return hidden_states, hidden_states_mlp, attn_mixer_out

    def forward(
        self,
        hidden_states: torch.Tensor,  # Hybrid output from previous layer
        attention_hidden_states: torch.Tensor = None,  # Attention from previous layer
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[torch.Tensor]] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, 
               torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """
        Forward pass that processes two separate hidden state streams through attention.
        """
        # Attention output given hybrid input (or initial embedding)
        hybrid_out, hybrid_mlp_out, hybrid_mixer_out = self._forward(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )

        # Attention output given attention input
        attn_out, attn_mlp_out, attn_mixer_out = None, None, None
        if attention_hidden_states is not None:
            attn_out, attn_mlp_out, attn_mixer_out = self._forward(
                hidden_states=attention_hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )
        else:
            # If the first decoder layer is a Hybrid layer, then there will be no attention input,
            # so there is no attention output. In this case, the Hybrid output is treated as the attention
            # input into the next layer.
            attn_out = hybrid_out
            attn_mlp_out = hybrid_mlp_out
            attn_mixer_out = hybrid_mixer_out

        return (
            hybrid_out,
            attn_out.detach(),
            hybrid_mixer_out,
            attn_mixer_out.detach(),
            hybrid_mlp_out,
            attn_mlp_out.detach(),
        )



class LlamaFusedAttHybridDecoderLayer(nn.Module, ABC):
    """
    This is our fused Hybrid+Attention layer used during Stage 1 of hybridization.
    It has a Hybrid (e.g., Mamba2) and an Attention module, along with (shared) norm
    and MLP layers. It takes two hidden states as inputs: Hybrid hidden states and Attention
    hidden states, both coming from the previous layer.

    During Stage 1, attention parameters are kept frozen and their outputs are detached to prevent
    gradient flow, while hybrid parameters are trained via distillation.
    """

    def __init__(self, config: HybridLlamaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.attention_type = "full_attention"

        # Attention mixer
        self.self_attn = LlamaAttention(config, layer_idx)
        
        # Norms
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # MLP
        self.mlp = LlamaMLP(config)

        # Hybrid mixer
        self._init_hybrid_seq_mixer(config, layer_idx)

    @property
    @abstractmethod
    def hybrid_seq_mixer(self) -> nn.Module:
        """Returns the instance of the Hybrid sequence mixer."""
        pass

    @abstractmethod
    def _init_hybrid_seq_mixer(self, config: HybridLlamaConfig, layer_idx: int):
        """Initializes the sequence mixer (e.g., Mamba2, B'MOJO, etc.)."""
        pass

    def _hybrid_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[torch.Tensor]] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Given the Hybrid hidden states as input, processes them through the Hybrid module."""
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Hybrid mixer
        hybrid_mixer_out, _, _ = self.hybrid_seq_mixer(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )
        
        hidden_states = residual + hybrid_mixer_out

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states_mlp = self.mlp(hidden_states)
        hidden_states = residual + hidden_states_mlp

        return hidden_states, hidden_states_mlp, hybrid_mixer_out

    def _attn_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Standard Attention forward function applied to the Attention hidden states."""
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        attn_mixer_out, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )
        hidden_states = residual + attn_mixer_out

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states_mlp = self.mlp(hidden_states)
        hidden_states = residual + hidden_states_mlp

        # Since we keep attention params frozen in stage 1, we detach these tensors
        return (
            hidden_states.detach(),
            hidden_states_mlp.detach(),
            attn_mixer_out.detach(),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,  # Hybrid output from previous layer
        attention_hidden_states: torch.Tensor = None,  # Attention output from previous layer
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward function which has two input paths, one for the previous layer's Hybrid
        output, and one for the previous layer's Attention output.
        """
        hybrid_out, hybrid_mlp_out, hybrid_mixer_out = self._hybrid_forward(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )

        attn_out, attn_mlp_out, attn_mixer_out = self._attn_forward(
            hidden_states=attention_hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )

        return (
            hybrid_out,
            attn_out,
            hybrid_mixer_out,
            attn_mixer_out,
            hybrid_mlp_out,
            attn_mlp_out,
        )



class LlamaFusedAttMamba2DecoderLayer(LlamaFusedAttHybridDecoderLayer):
    @property
    def hybrid_seq_mixer(self) -> nn.Module:
        return self.mamba

    def _init_hybrid_seq_mixer(self, config: HybridLlamaConfig, layer_idx: int):
        # Llama does NOT use QK-norm, so we pass qk_norm=None
        self.mamba = Mamba2Layer(
            config,
            layer_idx,
            apply_rotary_pos_emb_fn=apply_rotary_pos_emb,
            qk_norm=None,
        )


class LlamaFusedAttBMojoFDecoderLayer(LlamaFusedAttHybridDecoderLayer):
    @property
    def hybrid_seq_mixer(self) -> nn.Module:
        return self.bmojo_f

    def _init_hybrid_seq_mixer(self, config: HybridLlamaConfig, layer_idx: int):
        # Llama does NOT use QK-norm, so we pass qk_norm=None
        self.bmojo_f = BMojoLayer(
            config,
            apply_rotary_pos_emb_fn=apply_rotary_pos_emb,
            layer_idx=layer_idx,
            qk_norm=None,
        )


class LlamaFusedAttGDNDecoderLayer(LlamaFusedAttHybridDecoderLayer):
    @property
    def hybrid_seq_mixer(self) -> nn.Module:
        return self.gdn

    def _init_hybrid_seq_mixer(self, config: HybridLlamaConfig, layer_idx: int):
        self.gdn = GatedDeltaNetLayer(config, layer_idx=layer_idx)


class LlamaFusedAttGKADecoderLayer(LlamaFusedAttHybridDecoderLayer):
    @property
    def hybrid_seq_mixer(self) -> nn.Module:
        return self.gka

    def _init_hybrid_seq_mixer(self, config: HybridLlamaConfig, layer_idx: int):
        self.gka = GatedKalmaNetLayer(config, layer_idx=layer_idx)


class LlamaFusedAttSWADecoderLayer(LlamaFusedAttHybridDecoderLayer):
    @property
    def hybrid_seq_mixer(self) -> nn.Module:
        return self.swa

    def _init_hybrid_seq_mixer(self, config: HybridLlamaConfig, layer_idx: int):
        self.swa = LlamaSWAttention(config, layer_idx=layer_idx)


# Build registry of Hybrid Layers
HYBRID_LLAMA_DECODER_LAYER_REGISTRY = build_decoder_layer_registry(
    module_or_dict=sys.modules[__name__],
    model_prefix="Llama"
)



@auto_docstring
class HybridLlamaPreTrainedModel(PreTrainedModel):
    config: HybridLlamaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = [
        name for name in globals() 
        if name.endswith("DecoderLayer") and "Llama" in name
    ]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _can_compile_fullgraph = True
    _supports_attention_backend = True
    # For hidden_states, we include both the standard decoder layer and the hybrid base class.
    _can_record_outputs = {
        "hidden_states": [LlamaDecoderLayer, LlamaHybridDecoderLayerBase],
        "attentions": LlamaAttention,
    }


@auto_docstring
class HybridLlamaModel(HybridLlamaPreTrainedModel):
    def __init__(self, config: HybridLlamaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(self._set_decoder_layers(config))
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = LlamaRotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self.has_sliding_layers = "sliding_attention" in self.config.layer_types or "SWA" in self.config.layer_types

        # Initialize weights and apply final processing
        self.post_init()

    def _set_decoder_layers(self, config):
        """
        Sets each decoder layer of the model. Each decoder layer is determined by the
        hybrid override pattern in the model config.
        """
        return set_decoder_layers_from_pattern(config, HYBRID_LLAMA_DECODER_LAYER_REGISTRY)

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
        use_cache: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if not self.training and use_cache and (
            past_key_values is None or not isinstance(past_key_values, HybridCache)
        ):
            logger.warning_once(
                "You are passing `use_cache=True` and `past_key_values` is `None`. "
                "We are setting `past_key_values` to a new instance of `HybridCache`."
            )
            del past_key_values
            past_key_values = HybridCache(self.config)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # When using sequence parallelism, set attention_mask to None.
        # Tested with flash attention only.
        sp_group = getattr(self, "sequence_parallel_group", None)
        if sp_group is not None:
            causal_mask_mapping = {
                "full_attention": None,
            }
            if self.has_sliding_layers:
                causal_mask_mapping["sliding_attention"] = None
        else:
            # Prepare mask arguments
            if getattr(self.config, "swa_config", None) is not None:
                setattr(self.config, "sliding_window", self.config.swa_config["window_size"])
                
            # Prepare mask arguments
            mask_kwargs = {
                "config": self.config,
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            # Create the masks
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
            }
            # The sliding window alternating layers are not always activated depending on the config
            if self.has_sliding_layers:
                causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids=position_ids)

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            attention_type = getattr(decoder_layer, "attention_type", "full_attention")
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[attention_type],
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )



@auto_docstring
class HybridLlamaForCausalLM(HybridLlamaPreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    _tp_plan = {"lm_head": "colwise_gather_output"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config):
        super().__init__(config)
        self.model = HybridLlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

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
        cache_position: torch.LongTensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        """
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        # Keep track of sequence offset (e.g., # tokens seen) for Hybrid layers
        past_key_values_ = outputs.past_key_values
        if use_cache and past_key_values_ is not None:
            if hasattr(past_key_values_, "update_offset"):
                seq_len = hidden_states.shape[1]
                past_key_values_.update_offset(seq_len)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class HybridLlamaForSequenceClassification(GenericForSequenceClassification, HybridLlamaPreTrainedModel):
    pass


class HybridLlamaForQuestionAnswering(GenericForQuestionAnswering, HybridLlamaPreTrainedModel):
    base_model_prefix = "transformer"  # For BC, where `transformer` was used instead of `model`


class HybridLlamaForTokenClassification(GenericForTokenClassification, HybridLlamaPreTrainedModel):
    pass


__all__ = [
    "HybridLlamaForCausalLM",
    "HybridLlamaForQuestionAnswering",
    "HybridLlamaModel",
    "HybridLlamaPreTrainedModel",
    "HybridLlamaForSequenceClassification",
    "HybridLlamaForTokenClassification",
]
