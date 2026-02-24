# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Copyright 2024 The Qwen team.
# Copyright 2023 The vLLM team.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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
"""Inference-only Qwen3MoE model compatible with HuggingFace weights."""
import re
import typing
from collections.abc import Callable, Iterable
from typing import Any, Optional, Union

import torch
import torch.nn.functional as F
from torch import nn
from transformers import Qwen3MoeConfig

from vllm.v1.attention.backend import AttentionMetadata, AttentionType
from vllm.attention.layer import Attention
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig, get_current_vllm_config, ModelConfig
from vllm.distributed import (get_ep_group, get_pp_group,
                              get_tensor_model_parallel_world_size,
                              tensor_model_parallel_all_gather)
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.fused_moe import SharedFusedMoE
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (MergedColumnParallelLinear,
                                               QKVParallelLinear,
                                               ReplicatedLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader, maybe_remap_kv_scale_name)
from vllm.sequence import IntermediateTensors

from vllm.model_executor.models.interfaces import (
    HasInnerState, IsHybrid, SupportsPP, SupportsLoRA, SupportsMambaPrefixCaching, MixtureOfExperts
)

from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateDtypeCalculator,)

from vllm.model_executor.models.utils import (
    AutoWeightsLoader, PPMissingLayer, extract_layer_index,
    is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory, make_layers,
    maybe_prefix, sequence_parallel_chunk
)

from .utils import parse_override_pattern, pattern_has_symbols, Symbols
from .hybridqwen2 import Qwen2MLP as Qwen3MLP
from .primed_mamba2_layer import PrimedMamba2Layer, extra_groups_for_head_shards
from .primed_utils import PrimedStateShapeCalculator
from .primed_gdn_layer import PrimedGDNLayer
from .primed_gka_layer import PrimedGKALayer
from .bmojof.bmojof_layer import BMOJOFLayer


from primed_vllm.hf_configs import HybridQwen3MoeConfig



logger = init_logger(__name__)


class Qwen3MoeMLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: Optional[QuantizationConfig] = None,
        reduce_results: bool = True,
        expert_gate: torch.nn.Linear | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size, [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj")
        self.down_proj = RowParallelLinear(intermediate_size,
                                           hidden_size,
                                           bias=False,
                                           quant_config=quant_config,
                                           reduce_results=reduce_results,
                                           prefix=f"{prefix}.down_proj")
        if hidden_act != "silu":
            raise ValueError(f"Unsupported activation: {hidden_act}. "
                             "Only silu is supported for now.")
        self.act_fn = SiluAndMul()
        self.expert_gate = expert_gate

    def forward(self, x):
        gate_up, _ = self.gate_up_proj(x)
        out = self.act_fn(gate_up)
        out, _ = self.down_proj(out)

        if self.expert_gate is not None:
            out = F.sigmoid(self.expert_gate(x)[0]) * out

        return out


class Qwen3MoeSparseMoeBlock(nn.Module):

    def __init__(
        self,
        config: Qwen3MoeConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        enable_eplb: bool = False,
    ):
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()

        self.ep_group = get_ep_group().device_group
        self.ep_rank = get_ep_group().rank_in_group
        self.ep_size = self.ep_group.size()
        self.n_routed_experts = config.num_experts

        # Load balancing settings.
        vllm_config = get_current_vllm_config()
        parallel_config = vllm_config.parallel_config
        eplb_config = parallel_config.eplb_config
        self.enable_eplb = enable_eplb

        self.is_sequence_parallel = parallel_config.use_sequence_parallel_moe

        if self.tp_size > config.num_experts:
            raise ValueError(
                f"Tensor parallel size {self.tp_size} is greater than "
                f"the number of experts {config.num_experts}.")

        self.n_logical_experts = self.n_routed_experts
        self.n_redundant_experts = eplb_config.num_redundant_experts
        self.n_physical_experts = (self.n_logical_experts +
                                   self.n_redundant_experts)
        self.n_local_physical_experts = self.n_physical_experts // self.ep_size

        self.physical_expert_start = (self.ep_rank *
                                      self.n_local_physical_experts)
        self.physical_expert_end = (self.physical_expert_start +
                                    self.n_local_physical_experts)

        self.gate = ReplicatedLinear(config.hidden_size,
                                     config.num_experts,
                                     bias=False,
                                     quant_config=quant_config,
                                     prefix=f"{prefix}.gate")

        shared_expert_intermediate_size = getattr(
            config, "shared_expert_intermediate_size", 0
        )
        if shared_expert_intermediate_size > 0:
            self.shared_expert_gate = ReplicatedLinear(
                config.hidden_size,
                1,
                bias=False,
                quant_config=None,
                prefix=f"{prefix}.shared_expert_gate",
            )
            self.shared_expert = Qwen3MoeMLP(
                hidden_size=config.hidden_size,
                intermediate_size=shared_expert_intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                reduce_results=False,
                expert_gate=self.shared_expert_gate,
                prefix=f"{prefix}.shared_expert",
            )
        else:
            self.shared_expert_gate = None
            self.shared_expert = None

        self.experts = SharedFusedMoE(
            shared_experts=self.shared_expert,
            gate=self.gate,
            num_experts=self.n_routed_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            reduce_results=False,
            renormalize=config.norm_topk_prob,
            quant_config=quant_config,
            prefix=f"{prefix}.experts",
            enable_eplb=self.enable_eplb,
            num_redundant_experts=self.n_redundant_experts,
            is_sequence_parallel=self.is_sequence_parallel)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        assert hidden_states.dim() <= 2, (
            "Qwen3MoeSparseMoeBlock only supports 1D or 2D inputs"
        )
        is_input_1d = hidden_states.dim() == 1
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)

        if self.is_sequence_parallel:
            logger.warning_once(
                "Sequence parallel support for HybridQwen3MoE has not been "
                "fully tested. We recommend not using this feature."
            )
            hidden_states = sequence_parallel_chunk(hidden_states)

        # router_logits: (num_tokens, n_experts)
        router_logits, _ = self.gate(hidden_states)
        shared_out, fused_out = self.experts(
            hidden_states=hidden_states, router_logits=router_logits
        )
        final_hidden_states = (
            shared_out + fused_out if shared_out is not None else fused_out
        )

        if self.is_sequence_parallel:
            final_hidden_states = tensor_model_parallel_all_gather(
                final_hidden_states, 0
            )
            final_hidden_states = final_hidden_states[:num_tokens]
        elif self.tp_size > 1:
            final_hidden_states = self.experts.maybe_all_reduce_tensor_model_parallel(
                final_hidden_states
            )

        # return to 1d if input is 1d
        return final_hidden_states.squeeze(0) if is_input_1d else final_hidden_states


class Qwen3MoeAttention(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        rope_parameters: dict[str, Any],
        max_position_embeddings: int = 8192,
        head_dim: Optional[int] = None,
        rms_norm_eps: float = 1e-06,
        qkv_bias: bool = False,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        dual_chunk_attention_config: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = head_dim or (hidden_size // self.total_num_heads)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.max_position_embeddings = max_position_embeddings
        self.dual_chunk_attention_config = dual_chunk_attention_config

        self.qkv_proj = QKVParallelLinear(hidden_size,
                                          self.head_dim,
                                          self.total_num_heads,
                                          self.total_num_kv_heads,
                                          bias=qkv_bias,
                                          quant_config=quant_config,
                                          prefix=f"{prefix}.qkv_proj")

        self.o_proj = RowParallelLinear(self.total_num_heads * self.head_dim,
                                        hidden_size,
                                        bias=False,
                                        quant_config=quant_config,
                                        prefix=f"{prefix}.o_proj")

        self.rotary_emb = get_rope(
            self.head_dim,
            max_position=max_position_embeddings,
            rope_parameters=rope_parameters,
            dual_chunk_attention_config=dual_chunk_attention_config,
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
            **{
                "layer_idx": extract_layer_index(prefix),
                "dual_chunk_attention_config": dual_chunk_attention_config,
            } if dual_chunk_attention_config else {},
        )

        self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        # Add qk-norm
        q_by_head = q.view(*q.shape[:-1], q.shape[-1] // self.head_dim,
                           self.head_dim)
        q_by_head = self.q_norm(q_by_head)
        q = q_by_head.view(q.shape)

        k_by_head = k.view(*k.shape[:-1], k.shape[-1] // self.head_dim,
                           self.head_dim)
        k_by_head = self.k_norm(k_by_head)
        k = k_by_head.view(k.shape)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output


class Qwen3MoeSWAttention(nn.Module):
    """Sliding Window Attention layer for HybridQwen3MoE."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        rope_parameters: dict[str, Any],
        max_position_embeddings: int = 8192,
        head_dim: Optional[int] = None,
        rms_norm_eps: float = 1e-06,
        qkv_bias: bool = False,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        swa_config: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__()

        if swa_config is None or "window_size" not in swa_config:
            raise ValueError("swa_config must specify window_size for SWA layers")

        sliding_window = swa_config["window_size"]

        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0
        else:
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = head_dim or (hidden_size // self.total_num_heads)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5

        self.qkv_proj = QKVParallelLinear(
            hidden_size, self.head_dim, self.total_num_heads, self.total_num_kv_heads,
            bias=qkv_bias, quant_config=quant_config, prefix=f"{prefix}.qkv_proj")
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim, hidden_size, bias=False,
            quant_config=quant_config, prefix=f"{prefix}.o_proj")

        self.rotary_emb = get_rope(
            self.head_dim,  max_position=max_position_embeddings,
            rope_parameters=rope_parameters,
        )

        self.attn = Attention(
            self.num_heads, self.head_dim, self.scaling, num_kv_heads=self.num_kv_heads,
            cache_config=cache_config, quant_config=quant_config, prefix=f"{prefix}.attn",
            per_layer_sliding_window=sliding_window)

        self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q_by_head = q.view(*q.shape[:-1], q.shape[-1] // self.head_dim, self.head_dim)
        q_by_head = self.q_norm(q_by_head)
        q = q_by_head.view(q.shape)
        k_by_head = k.view(*k.shape[:-1], k.shape[-1] // self.head_dim, self.head_dim)
        k_by_head = self.k_norm(k_by_head)
        k = k_by_head.view(k.shape)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output


class HybridQwen3MoeBMOJOFDecoderLayer(nn.Module):
    """BMOJO-F decoder layer for HybridQwen3MoE.

    This layer uses the V1 architecture with:
    - BMOJOFLayer containing SSM + dual-stream attention
    - No V0 cache parameters
    - Simplified forward signature matching V1 patterns
    """

    def __init__(
        self,
        config: HybridQwen3MoeConfig,
        layer_idx: int,
        model_config: Optional[ModelConfig] = None,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        enable_eplb: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size

        # BMOJO-F layer (Mamba2 + dual-stream attention)
        self.bmojo_f = BMOJOFLayer(
            config=config,
            layer_idx=layer_idx,
            prefix=f"{prefix}.bmojo_f",
            model_config=model_config,
            cache_config=cache_config,
            quant_config=quant_config,
        )

        # `mlp_only_layers` in the config.
        layer_idx = extract_layer_index(prefix)
        mlp_only_layers = ([] if not hasattr(config, "mlp_only_layers") else
                           config.mlp_only_layers)
        if (layer_idx not in mlp_only_layers) and (
                config.num_experts > 0 and
            (layer_idx + 1) % config.decoder_sparse_step == 0):
            self.mlp = Qwen3MoeSparseMoeBlock(config=config,
                                              quant_config=quant_config,
                                              prefix=f"{prefix}.mlp",
                                              enable_eplb=enable_eplb)
        else:
            self.mlp = Qwen3MoeMLP(hidden_size=config.hidden_size,
                                   intermediate_size=config.intermediate_size,
                                   hidden_act=config.hidden_act,
                                   quant_config=quant_config,
                                   prefix=f"{prefix}.mlp")

        self.input_layernorm = RMSNorm(config.hidden_size,
                                       eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size,
                                                eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for BMOJO-F decoder layer.

        Args:
            positions: Position indices for RoPE
            hidden_states: Input hidden states
            residual: Residual connection tensor (optional)
            **kwargs: Additional arguments (ignored for V1 compatibility)

        Returns:
            Tuple of (hidden_states, residual)
        """
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(
                hidden_states, residual)

        # BMOJO-F layer (Mamba2 + dual-stream attention)
        # No cache params needed - layers access cache via forward_context
        hidden_states = self.bmojo_f(
            positions=positions,
            hidden_states=hidden_states,
        )

        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class HybridQwen3MoeMamba2DecoderLayer(nn.Module):
    def __init__(
        self,
        config: HybridQwen3MoeConfig,
        layer_idx: int,
        model_config: Optional[ModelConfig] = None,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        enable_eplb: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size

        self.mamba = PrimedMamba2Layer(
            config=config,
            model_config=model_config,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.mixer",
        )

        # `mlp_only_layers` in the config.
        layer_idx = extract_layer_index(prefix)
        mlp_only_layers = ([] if not hasattr(config, "mlp_only_layers") else
                           config.mlp_only_layers)
        if (layer_idx not in mlp_only_layers) and (
                config.num_experts > 0 and
            (layer_idx + 1) % config.decoder_sparse_step == 0):
            self.mlp = Qwen3MoeSparseMoeBlock(config=config,
                                              quant_config=quant_config,
                                              prefix=f"{prefix}.mlp",
                                              enable_eplb=enable_eplb)
        else:
            self.mlp = Qwen3MoeMLP(hidden_size=config.hidden_size,
                                   intermediate_size=config.intermediate_size,
                                   hidden_act=config.hidden_act,
                                   quant_config=quant_config,
                                   prefix=f"{prefix}.mlp")

        self.input_layernorm = RMSNorm(config.hidden_size,
                                       eps=config.mamba2_config.get("rms_norm_eps", 1e-6))
        self.post_attention_layernorm = RMSNorm(config.hidden_size,
                                                eps=config.mamba2_config.get("rms_norm_eps", 1e-6))

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
        positions: torch.Tensor,
        **kwargs,
    ):
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        output = self.mamba(
            hidden_states=hidden_states,
            positions=positions
        )

        hidden_states, residual = self.post_attention_layernorm(
            output, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class HybridQwen3MoeAttnDecoderLayer(nn.Module):

    def __init__(
        self,
        config: HybridQwen3MoeConfig,
        layer_idx: int,
        model_config: Optional[ModelConfig] = None,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        enable_eplb: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        max_position_embeddings = getattr(config, "max_position_embeddings",
                                          8192)
        dual_chunk_attention_config = getattr(config,
                                              "dual_chunk_attention_config",
                                              None)
        self.self_attn = Qwen3MoeAttention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            rope_parameters=config.rope_parameters,
            max_position_embeddings=max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, 'attention_bias', False),
            head_dim=getattr(config, 'head_dim', None),
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
            dual_chunk_attention_config=dual_chunk_attention_config,
        )

        # `mlp_only_layers` in the config.
        layer_idx = extract_layer_index(prefix)
        mlp_only_layers = ([] if not hasattr(config, "mlp_only_layers") else
                           config.mlp_only_layers)
        if (layer_idx not in mlp_only_layers) and (
                config.num_experts > 0 and
            (layer_idx + 1) % config.decoder_sparse_step == 0):
            self.mlp = Qwen3MoeSparseMoeBlock(config=config,
                                              quant_config=quant_config,
                                              prefix=f"{prefix}.mlp",
                                              enable_eplb=enable_eplb)
        else:
            self.mlp = Qwen3MoeMLP(hidden_size=config.hidden_size,
                                   intermediate_size=config.intermediate_size,
                                   hidden_act=config.hidden_act,
                                   quant_config=quant_config,
                                   prefix=f"{prefix}.mlp")

        self.input_layernorm = RMSNorm(config.hidden_size,
                                       eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size,
                                                eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Self Attention
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(
                hidden_states, residual)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
        )

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class HybridQwen3MoeGDNDecoderLayer(nn.Module):
    """GatedDeltaNet decoder layer for HybridQwen3MoE.

    Uses PrimedGDNLayer which provides:
    - vLLM V1 optimizations (custom ops, cache management)
    - Checkpoint compatibility with Hybrid Model Factory
    - Beta scaling support (allow_neg_eigval)
    """

    def __init__(
        self,
        config: HybridQwen3MoeConfig,
        layer_idx: int,
        model_config: Optional[ModelConfig] = None,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        enable_eplb: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size

        self.gdn = PrimedGDNLayer(
            config=config,
            model_config=model_config,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.gdn",
        )

        # `mlp_only_layers` in the config.
        layer_idx = extract_layer_index(prefix)
        mlp_only_layers = ([] if not hasattr(config, "mlp_only_layers") else
                           config.mlp_only_layers)
        if (layer_idx not in mlp_only_layers) and (
                config.num_experts > 0 and
            (layer_idx + 1) % config.decoder_sparse_step == 0):
            self.mlp = Qwen3MoeSparseMoeBlock(config=config,
                                              quant_config=quant_config,
                                              prefix=f"{prefix}.mlp",
                                              enable_eplb=enable_eplb)
        else:
            self.mlp = Qwen3MoeMLP(hidden_size=config.hidden_size,
                                   intermediate_size=config.intermediate_size,
                                   hidden_act=config.hidden_act,
                                   quant_config=quant_config,
                                   prefix=f"{prefix}.mlp")

        # Get norm eps from gdn_config or use default
        gdn_config = getattr(config, "gdn_config", {})
        norm_eps = gdn_config.get("norm_eps", config.rms_norm_eps)

        self.input_layernorm = RMSNorm(config.hidden_size, eps=norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
        positions: torch.Tensor,
        **kwargs,
    ):
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        output = torch.empty_like(hidden_states)
        self.gdn(
            hidden_states=hidden_states,
            output=output,
        )

        hidden_states, residual = self.post_attention_layernorm(output, residual)
        hidden_states = self.mlp(hidden_states)

        return hidden_states, residual


class HybridQwen3MoeGKADecoderLayer(nn.Module):
    """GKA decoder layer for HybridQwen3MoE."""

    def __init__(
        self,
        config: HybridQwen3MoeConfig,
        layer_idx: int,
        model_config: Optional[ModelConfig] = None,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        enable_eplb: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size

        self.gka = PrimedGKALayer(
            config=config,
            model_config=model_config,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.gka",
        )

        layer_idx = extract_layer_index(prefix)
        mlp_only_layers = ([] if not hasattr(config, "mlp_only_layers") else
                           config.mlp_only_layers)
        if (layer_idx not in mlp_only_layers) and (
                config.num_experts > 0 and
            (layer_idx + 1) % config.decoder_sparse_step == 0):
            self.mlp = Qwen3MoeSparseMoeBlock(config=config,
                                              quant_config=quant_config,
                                              prefix=f"{prefix}.mlp",
                                              enable_eplb=enable_eplb)
        else:
            self.mlp = Qwen3MoeMLP(hidden_size=config.hidden_size,
                                   intermediate_size=config.intermediate_size,
                                   hidden_act=config.hidden_act,
                                   quant_config=quant_config,
                                   prefix=f"{prefix}.mlp")

        gka_config = getattr(config, "gka_config", {})
        norm_eps = gka_config.get("norm_eps", config.rms_norm_eps)

        self.input_layernorm = RMSNorm(config.hidden_size, eps=norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
        positions: torch.Tensor,
        **kwargs,
    ):
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        output = torch.empty_like(hidden_states)
        self.gka(hidden_states=hidden_states, output=output)

        hidden_states, residual = self.post_attention_layernorm(output, residual)
        hidden_states = self.mlp(hidden_states)

        return hidden_states, residual


class HybridQwen3MoeSWADecoderLayer(nn.Module):
    """Sliding Window Attention decoder layer for HybridQwen3MoE."""

    def __init__(
        self,
        config: HybridQwen3MoeConfig,
        layer_idx: int,
        model_config: Optional[ModelConfig] = None,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        enable_eplb: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size

        swa_config = getattr(config, "swa_config", None)

        self.swa = Qwen3MoeSWAttention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position_embeddings=getattr(config, "max_position_embeddings", 8192),
            head_dim=getattr(config, 'head_dim', None),
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, 'attention_bias', False),
            rope_parameters=config.rope_parameters,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.swa",
            swa_config=swa_config,
        )

        layer_idx = extract_layer_index(prefix)
        mlp_only_layers = ([] if not hasattr(config, "mlp_only_layers") else
                           config.mlp_only_layers)
        if (layer_idx not in mlp_only_layers) and (
                config.num_experts > 0 and
            (layer_idx + 1) % config.decoder_sparse_step == 0):
            self.mlp = Qwen3MoeSparseMoeBlock(config=config,
                                              quant_config=quant_config,
                                              prefix=f"{prefix}.mlp",
                                              enable_eplb=enable_eplb)
        else:
            self.mlp = Qwen3MoeMLP(hidden_size=config.hidden_size,
                                   intermediate_size=config.intermediate_size,
                                   hidden_act=config.hidden_act,
                                   quant_config=quant_config,
                                   prefix=f"{prefix}.mlp")

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        hidden_states = self.swa(positions=positions, hidden_states=hidden_states)

        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


ALL_DECODER_LAYER_TYPES = {
    "attention": HybridQwen3MoeAttnDecoderLayer,
    "mamba": HybridQwen3MoeMamba2DecoderLayer,
    "bmojo-f": HybridQwen3MoeBMOJOFDecoderLayer,
    "gdn": HybridQwen3MoeGDNDecoderLayer,
    "gka": HybridQwen3MoeGKADecoderLayer,
    "swa": HybridQwen3MoeSWADecoderLayer,
}

@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        "positions": 0,
        "intermediate_tensors": 0,
        "inputs_embeds": 0,
    }
)
class HybridQwen3MoeModel(nn.Module):

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_config.get_text_config()
        model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config


        parallel_config = vllm_config.parallel_config
        enable_eplb = parallel_config.enable_eplb
        eplb_config = parallel_config.eplb_config
        self.num_redundant_experts = eplb_config.num_redundant_experts

        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.config = config
        self.quant_config = quant_config
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=f"{prefix}.embed_tokens")

        self.layer_allocation = parse_override_pattern(config.hybrid_override_pattern)
        self.config.layers_block_type = []

        # Validate: cannot mix incompatible state shapes
        has_gdn = any(layer_type == Symbols.GDN for layer_type in self.layer_allocation.values())
        has_gka = any(layer_type == Symbols.GKA for layer_type in self.layer_allocation.values())
        has_mamba = any(layer_type == Symbols.MAMBA for layer_type in self.layer_allocation.values())
        if has_gdn and has_mamba:
            raise ValueError(
                f"Cannot mix GDN and Mamba2 layers in the same model. "
                f"They have incompatible state shapes (GDN uses 2D recurrent state, "
                f"Mamba2 uses 3D temporal state). "
                f"Pattern: {config.hybrid_override_pattern}\n"
                f"Please use either GDN or Mamba2, not both."
            )
        if has_gdn and has_gka:
            raise ValueError(
                f"Cannot mix GDN and GKA layers in the same model. "
                f"They have incompatible state shapes (2-tuple vs 3-tuple). "
                f"Pattern: {config.hybrid_override_pattern}"
            )
        if has_gka and has_mamba:
            raise ValueError(
                f"Cannot mix GKA and Mamba2 layers in the same model. "
                f"They have incompatible state shapes. "
                f"Pattern: {config.hybrid_override_pattern}"
            )

        def get_layer(prefix: str):
            layer_idx = int(prefix.rsplit(".", 1)[1])
            layer_symbol = self.layer_allocation[layer_idx]
            match layer_symbol:
                case Symbols.MAMBA:
                    self.config.layers_block_type.append("mamba")
                    return HybridQwen3MoeMamba2DecoderLayer(
                        config, layer_idx, model_config, cache_config,
                        quant_config=quant_config, prefix=prefix, enable_eplb=enable_eplb)
                case Symbols.ATTENTION:
                    self.config.layers_block_type.append("attention")
                    return HybridQwen3MoeAttnDecoderLayer(
                        config, layer_idx, model_config, cache_config,
                        quant_config=quant_config, prefix=prefix, enable_eplb=enable_eplb)
                case Symbols.BMOJO_F:
                    self.config.layers_block_type.append("bmojo-f")
                    return HybridQwen3MoeBMOJOFDecoderLayer(
                        config, layer_idx, model_config, cache_config,
                        quant_config=quant_config, prefix=prefix, enable_eplb=enable_eplb)
                case Symbols.GDN:
                    self.config.layers_block_type.append("gdn")
                    return HybridQwen3MoeGDNDecoderLayer(
                        config, layer_idx, model_config, cache_config,
                        quant_config=quant_config, prefix=prefix, enable_eplb=enable_eplb)
                case Symbols.GKA:
                    self.config.layers_block_type.append("gka")
                    return HybridQwen3MoeGKADecoderLayer(
                        config, layer_idx, model_config, cache_config,
                        quant_config=quant_config, prefix=prefix, enable_eplb=enable_eplb)
                case Symbols.SWA:
                    self.config.layers_block_type.append("swa")
                    return HybridQwen3MoeSWADecoderLayer(
                        config, layer_idx, model_config, cache_config,
                        quant_config=quant_config, prefix=prefix, enable_eplb=enable_eplb)

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers, get_layer, prefix=f"{prefix}.layers")

        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.make_empty_intermediate_tensors = (
            make_empty_intermediate_tensors_factory(
                ["hidden_states", "residual"], config.hidden_size))

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        """Forward pass for HybridQwen3MoeModel.

        Args:
            input_ids: Input token IDs
            positions: Position indices
            intermediate_tensors: Intermediate tensors for pipeline parallelism
            inputs_embeds: Input embeddings (optional)

        Returns:
            Hidden states tensor
        """
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_input_ids(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        for i, layer in enumerate(self.layers[self.start_layer:self.end_layer]):
            # All layer types (Attention, Mamba2, BMOJO-F, GDN) use the same interface
            hidden_states, residual = layer(
                positions=positions,
                hidden_states=hidden_states,
                residual=residual,
            )

        if not get_pp_group().is_last_rank:
            return IntermediateTensors({"hidden_states": hidden_states, "residual": residual})

        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        # Params for weights, fp8 weight scales, fp8 activation scales
        # (param_name, weight_name, expert_id, shard_id)
        return SharedFusedMoE.make_expert_params_mapping(
            self,
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.num_experts,
            num_redundant_experts=self.num_redundant_experts)

    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            # Use fully qualified names to avoid matching gdn.q_proj etc.
            ("self_attn.qkv_proj", "self_attn.q_proj", "q"),
            ("self_attn.qkv_proj", "self_attn.k_proj", "k"),
            ("self_attn.qkv_proj", "self_attn.v_proj", "v"),
            ("swa.qkv_proj", "swa.q_proj", "q"),
            ("swa.qkv_proj", "swa.k_proj", "k"),
            ("swa.qkv_proj", "swa.v_proj", "v"),
            ("bmojo_attn.kv_proj", "bmojo_attn.k_proj", 0),
            ("bmojo_attn.kv_proj", "bmojo_attn.v_proj", 1),
            ("bmojo_attn.kv_proj_", "bmojo_attn.k_proj_", 0),
            ("bmojo_attn.kv_proj_", "bmojo_attn.v_proj_", 1),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
            # GDN weight remapping: separate → fused
            # Conv weights: q_conv1d, k_conv1d, v_conv1d → conv1d (merged at load time)
            ("gdn.conv1d", "gdn.q_conv1d", 0),
            ("gdn.conv1d", "gdn.k_conv1d", 1),
            ("gdn.conv1d", "gdn.v_conv1d", 2),
            # Linear projections: in_proj_all contains q, k, v, g, b, gk
            ("gdn.in_proj_all", "gdn.q_proj", 0),
            ("gdn.in_proj_all", "gdn.k_proj", 1),
            ("gdn.in_proj_all", "gdn.v_proj", 2),
            ("gdn.in_proj_all", "gdn.g_proj", 3),
            ("gdn.in_proj_all", "gdn.b_proj", 4),
            ("gdn.in_proj_all", "gdn.gk_proj", 5),
            # GKA conv weights: q_conv1d, k_conv1d, v_conv1d → conv1d (merged at load time)
            ("gka.conv1d", "gka.q_conv1d", 0),
            ("gka.conv1d", "gka.k_conv1d", 1),
            ("gka.conv1d", "gka.v_conv1d", 2),
            # GKA merged in_proj: q, k, v, a, [alpha], [b], [g]
            ("gka.in_proj", "gka.q_proj", "q_proj"),
            ("gka.in_proj", "gka.k_proj", "k_proj"),
            ("gka.in_proj", "gka.v_proj", "v_proj"),
            ("gka.in_proj", "gka.a_proj", "a_proj"),
            ("gka.in_proj", "gka.alpha_proj", "alpha_proj"),
            ("gka.in_proj", "gka.b_proj", "b_proj"),
            ("gka.in_proj", "gka.g_proj", "g_proj"),
        ]

        # Skip loading extra parameters for GPTQ/modelopt models.
        ignore_suffixes = (".bias", "_bias", ".k_scale", "_k_scale",
                           ".v_scale", "_v_scale", ".weight_scale",
                           "_weight_scale", ".input_scale", "_input_scale")

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        expert_params_mapping = self.get_expert_mapping()
        for name, loaded_weight in weights:
            # Handle self_attn -> swa renaming for SWA layers.
            # Original Qwen3MoE checkpoints use "self_attn" for all layers,
            # but our SWA decoder layers store the module as "swa".
            if "self_attn" in name:
                match = re.search(r"layers\.(\d+)\.", name)
                if match:
                    layer_idx = int(match.group(1))
                    if self.layer_allocation.get(layer_idx) == Symbols.SWA:
                        name = name.replace("self_attn", "swa")

            # Handle Mamba2/GDN/GKA A_log -> A renaming
            if "A_log" in name:
                name = name.replace("A_log", "A")

            # Handle GKA alpha_proj.bias and b_proj.bias -> alpha_bias, b_bias
            if "gka.alpha_proj.bias" in name:
                name = name.replace("gka.alpha_proj.bias", "gka.alpha_bias")
            elif "gka.b_proj.bias" in name:
                name = name.replace("gka.b_proj.bias", "gka.b_bias")

            if self.quant_config is not None and (
                scale_name := self.quant_config.get_cache_scale(name)
            ):
                # Loading kv cache quantization scales
                param = params_dict[scale_name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                assert loaded_weight.numel() == 1, (
                    f"KV scale numel {loaded_weight.numel()} != 1"
                )
                loaded_weight = loaded_weight.squeeze()
                weight_loader(param, loaded_weight)
                loaded_params.add(scale_name)
                continue

            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                # Skip non-stacked layers and experts (experts handled below).
                if weight_name not in name:
                    continue
                # We have mlp.experts[0].gate_proj in the checkpoint.
                # Since we handle the experts below in expert_params_mapping,
                # we need to skip here BEFORE we update the name, otherwise
                # name will be updated to mlp.experts[0].gate_up_proj, which
                # will then be updated below in expert_params_mapping
                # for mlp.experts[0].gate_gate_up_proj, which breaks load.
                if "mlp.experts" in name:
                    continue
                name = name.replace(weight_name, param_name)

                # Skip loading extra parameters for GPTQ/modelopt models.
                if name.endswith(ignore_suffixes) and name not in params_dict:
                    continue

                # Skip layers on other devices.
                if is_pp_missing_parameter(name, self):
                    continue
                if name.endswith("scale"):
                    # Remapping the name of FP8 kv-scale.
                    name = maybe_remap_kv_scale_name(name, params_dict)
                    if name is None:
                        continue
                if name not in params_dict:
                    continue

                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                if weight_loader == default_weight_loader:
                    weight_loader(param, loaded_weight)
                else:
                    weight_loader(param, loaded_weight, shard_id)
                break
            else:
                is_expert_weight = False
                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping
                    if weight_name not in name:
                        continue

                    # Anyway, this is an expert weight and should not be
                    # attempted to load as other weights later
                    is_expert_weight = True

                    # Do not modify `name` since the loop may continue here
                    # Instead, create a new variable
                    name_mapped = name.replace(weight_name, param_name)

                    if is_pp_missing_parameter(name_mapped, self):
                        continue

                    # Skip loading extra parameters for GPTQ/modelopt models.
                    if name_mapped.endswith(
                            ignore_suffixes
                    ) and name_mapped not in params_dict:
                        continue

                    param = params_dict[name_mapped]
                    # We should ask the weight loader to return success or not
                    # here since otherwise we may skip experts with other
                    # available replicas.
                    weight_loader = typing.cast(Callable[..., bool],
                                                param.weight_loader)
                    success = weight_loader(param,
                                            loaded_weight,
                                            name_mapped,
                                            shard_id=shard_id,
                                            expert_id=expert_id,
                                            return_success=True)
                    if success:
                        name = name_mapped
                        break
                else:
                    if is_expert_weight:
                        # We've checked that this is an expert weight
                        # However it's not mapped locally to this rank
                        # So we simply skip it
                        continue

                    # Skip loading extra parameters for GPTQ/modelopt models.
                    if name.endswith(
                            ignore_suffixes) and name not in params_dict:
                        continue
                    # Skip layers on other devices.
                    if is_pp_missing_parameter(name, self):
                        continue
                    # Remapping the name of FP8 kv-scale.
                    if name.endswith("kv_scale"):
                        remapped_kv_scale_name = name.replace(
                            ".kv_scale", ".attn.kv_scale")
                        if remapped_kv_scale_name not in params_dict:
                            logger.warning_once(
                                "Found kv scale in the checkpoint (e.g. %s), but not found the expected name in the model (e.g. %s). kv-scale is not loaded.",  # noqa: E501
                                name,
                                remapped_kv_scale_name,
                            )
                            continue
                        else:
                            name = remapped_kv_scale_name
                    param = params_dict[name]
                    weight_loader = getattr(param, "weight_loader",
                                            default_weight_loader)
                    weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


class HybridQwen3MoeForCausalLM(nn.Module, SupportsPP, SupportsLoRA, HasInnerState, SupportsMambaPrefixCaching, IsHybrid, MixtureOfExperts):
    packed_modules_mapping = {
        "self_attn.qkv_proj": ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
        "swa.qkv_proj": ["swa.q_proj", "swa.k_proj", "swa.v_proj"],
        "bmojo_attn.kv_proj": ["bmojo_attn.k_proj", "bmojo_attn.v_proj"],
        "bmojo_attn.kv_proj_": ["bmojo_attn.k_proj_", "bmojo_attn.v_proj_"],
        "in_proj": ["in_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
        # GDN packed modules (all merged at load time)
        "gdn.conv1d": [
            "gdn.q_conv1d",
            "gdn.k_conv1d",
            "gdn.v_conv1d",
        ],
        "gdn.in_proj_all": [
            "gdn.q_proj",
            "gdn.k_proj",
            "gdn.v_proj",
            "gdn.g_proj",
            "gdn.b_proj",
            "gdn.gk_proj",
        ],
        # GKA packed modules (conv and in_proj merged at load time)
        "gka.conv1d": [
            "gka.q_conv1d",
            "gka.k_conv1d",
            "gka.v_conv1d",
        ],
        "gka.in_proj": [
            "gka.q_proj",
            "gka.k_proj",
            "gka.v_proj",
            "gka.a_proj",
            "gka.alpha_proj",
            "gka.b_proj",
            "gka.g_proj",
        ],
    }


    embedding_modules = {
        "embed_tokens": "input_embeddings",
        "lm_head": "output_embeddings",
    }

    fall_back_to_pt_during_load = False

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        config = vllm_config.model_config.hf_config.get_text_config()
        if hasattr(config, "bmojo_config"):
            if "window_size" in config.bmojo_config:
                if config.bmojo_config["window_size"] % 2:
                    config.bmojo_config["window_size"] = config.bmojo_config["window_size"] + 1
        quant_config = vllm_config.quant_config
        cache_config = vllm_config.cache_config
        scheduler_config = vllm_config.scheduler_config
        if cache_config.mamba_cache_mode == "all":
            raise NotImplementedError(
                "HybridQwen3Moe currently does not support 'all' prefix caching, "
                "please use '--mamba-cache-mode=align' instead"
            )

        super().__init__()
        self.config = config
        self.vllm_config = vllm_config
        self.scheduler_config = scheduler_config
        self.model_config = vllm_config.model_config
        self.quant_config = quant_config
        self.model = HybridQwen3MoeModel(vllm_config=vllm_config,
                                   prefix=maybe_prefix(prefix, "model"))

        self.lm_head = ParallelLMHead(
            config.vocab_size, config.hidden_size)

        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = self.model.make_empty_intermediate_tensors

        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

        # Set MoE hyperparameters
        self.expert_weights = []

        self.moe_layers: list[SharedFusedMoE] = []
        example_layer = None
        for layer in self.model.layers:
            if isinstance(layer, PPMissingLayer):
                continue

            if isinstance(layer.mlp, Qwen3MoeSparseMoeBlock):
                example_layer = layer.mlp
                self.moe_layers.append(layer.mlp.experts)

        if example_layer is None:
            raise RuntimeError("No Qwen3MoE layer found in the model.layers.")

        self.num_moe_layers = len(self.moe_layers)
        self.num_expert_groups = 1
        self.num_shared_experts = 0
        self.num_logical_experts = example_layer.n_logical_experts
        self.num_physical_experts = example_layer.n_physical_experts
        self.num_local_physical_experts = example_layer.n_local_physical_experts
        self.num_routed_experts = example_layer.n_routed_experts
        self.num_redundant_experts = example_layer.n_redundant_experts

    def set_eplb_state(
        self,
        expert_load_view: torch.Tensor,
        logical_to_physical_map: torch.Tensor,
        logical_replica_count: torch.Tensor,
    ) -> None:
        for layer_idx, layer in enumerate(self.moe_layers):
            # Register the expert weights.
            self.expert_weights.append(layer.get_expert_weights())
            layer.set_eplb_state(
                moe_layer_idx=layer_idx,
                expert_load_view=expert_load_view,
                logical_to_physical_map=logical_to_physical_map,
                logical_replica_count=logical_replica_count,
            )

    def update_physical_experts_metadata(
        self,
        num_physical_experts: int,
        num_local_physical_experts: int,
    ) -> None:
        assert self.num_local_physical_experts == num_local_physical_experts
        self.num_physical_experts = num_physical_experts
        self.num_local_physical_experts = num_local_physical_experts
        self.num_redundant_experts = (num_physical_experts -
                                      self.num_logical_experts)
        for layer in self.model.layers:
            if isinstance(layer.mlp, Qwen3MoeSparseMoeBlock):
                moe = layer.mlp
                moe.n_local_physical_experts = num_local_physical_experts
                moe.n_physical_experts = num_physical_experts
                moe.n_redundant_experts = self.num_redundant_experts
                moe.experts.update_expert_map()

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Apply token embeddings to input_ids (V1 required method)."""
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        """Forward pass for HybridQwen3MoE model.

        Args:
            input_ids: Input token IDs
            positions: Position indices
            intermediate_tensors: Intermediate tensors for pipeline parallelism
            inputs_embeds: Input embeddings (optional)
            **kwargs: Additional arguments

        Returns:
            Hidden states tensor
        """
        hidden_states = self.model(
            input_ids,
            positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )
        return hidden_states

    @classmethod
    def get_mamba_state_shape_from_config(
        cls,
        vllm_config: "VllmConfig",
    ) -> tuple[tuple[int, int], tuple[int, int, int]] | tuple[tuple[int, int], tuple[int, int]]:
        """Calculate shapes for inner state caches (Mamba2, GDN, or GKA).

        Args:
            vllm_config: vLLM config

        Returns:
            For Mamba2:
                - conv_state_shape: Shape for convolutional state cache
                - temporal_state_shape: Shape for state space model cache (3D)
            For GDN:
                - conv_state_shape: Shape for convolutional state cache
                - ssm_state_shape: Shape for recurrent state cache (2D)
            For GKA:
                - conv_state_shape, h_kk_shape, h_kv_shape (3-tuple)
        """
        parallel_config = vllm_config.parallel_config
        hf_config = vllm_config.model_config.hf_config.get_text_config()

        pattern = getattr(hf_config, "hybrid_override_pattern", "")
        bmojo_config = getattr(hf_config, "bmojo_config", {})
        syms = pattern_has_symbols(pattern, bmojo_config)
        has_gdn = Symbols.GDN in syms
        has_gka = Symbols.GKA in syms
        has_mamba = Symbols.MAMBA in syms

        # Check for attention-only models
        has_inner_state_layers = has_gdn or has_gka or has_mamba
        if not has_inner_state_layers:
            return (0,), (0,)

        # Validate: cannot mix incompatible state shapes
        bmojo_ssm_mixer = bmojo_config.get("ssm_mixer", "").lower() if bmojo_config else ""
        if has_gdn and has_gka:
            raise ValueError(
                f"Cannot mix GDN and GKA layers in the same model. "
                f"They have incompatible state shapes (2-tuple vs 3-tuple). "
                f"Pattern: {pattern}"
                + (f", BMOJO-F SSM mixer: {bmojo_ssm_mixer}" if bmojo_ssm_mixer else "")
            )
        if (has_gdn or has_gka) and has_mamba:
            raise ValueError(
                f"Cannot mix GDN/GKA and Mamba2 layers in the same model. "
                f"They have incompatible state shapes. "
                f"Pattern: {pattern}"
                + (f", BMOJO-F SSM mixer: {bmojo_ssm_mixer}" if bmojo_ssm_mixer else "")
                + "\nPlease use either GDN, GKA, or Mamba2, not mixed."
            )

        # If we have GKA layers, return GKA state shape (3-tuple)
        if has_gka:
            tp_size = parallel_config.tensor_parallel_size
            num_spec = (
                vllm_config.speculative_config.num_speculative_tokens
                if vllm_config.speculative_config
                else 0
            )

            hidden_size = hf_config.hidden_size
            gka_config = getattr(hf_config, "gka_config", {}) or {}
            if "num_q_heads" in gka_config:
                num_heads = gka_config["num_q_heads"]
                num_kv_heads = gka_config["num_k_heads"]
                head_dim = gka_config["head_dim"]
            else:
                num_heads = hf_config.num_attention_heads
                num_kv_heads = hf_config.num_key_value_heads
                head_dim = getattr(hf_config, "head_dim", hidden_size // num_heads)

            conv_kernel_size = gka_config.get("conv_size", 4)
            num_sketches_per_head = gka_config.get("num_sketches_per_head", 1)

            return PrimedStateShapeCalculator.gka_state_shape(
                tp_world_size=tp_size,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                hidden_size=hidden_size,
                head_dim=head_dim,
                conv_kernel_size=conv_kernel_size,
                num_spec=num_spec,
                num_sketches_per_head=num_sketches_per_head,
            )

        # If we have GDN layers, return GDN state shape
        if has_gdn:
            tp_size = parallel_config.tensor_parallel_size
            num_spec = (
                vllm_config.speculative_config.num_speculative_tokens
                if vllm_config.speculative_config
                else 0
            )

            hidden_size = hf_config.hidden_size
            layer_config = getattr(hf_config, "gdn_config", {}) or {}

            if "num_q_heads" in layer_config:
                num_heads = layer_config["num_q_heads"]
                num_kv_heads = layer_config["num_k_heads"]
                head_dim = layer_config["head_dim"]
            else:
                num_heads = hf_config.num_attention_heads
                num_kv_heads = hf_config.num_key_value_heads
                head_dim = getattr(hf_config, "head_dim", hidden_size // num_heads)

            conv_kernel_size = layer_config.get("conv_size", 4)

            return PrimedStateShapeCalculator.gated_delta_net_state_shape(
                tp_world_size=tp_size,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                hidden_size=hidden_size,
                head_dim=head_dim,
                conv_kernel_size=conv_kernel_size,
                num_spec=num_spec,
            )

        # Otherwise, return Mamba2 state shape
        if has_mamba:
            intermediate_size = hf_config.mamba2_config.get("d_inner", 128)
            state_size = hf_config.mamba2_config.get("ssm_cfg", {}).get("d_state", 64)
            d_conv = hf_config.mamba2_config.get("d_conv", 4)
            n_groups = hf_config.mamba2_config.get("ssm_cfg", {}).get("ngroups", 1)
            num_heads = hf_config.mamba2_config.get("ssm_cfg", {}).get("ngroups", 1)
            head_dim = hf_config.mamba2_config.get("ssm_cfg", {}).get("d_state", 64)
            repeat_kv_before_conv = hf_config.mamba2_config.get("ssm_cfg", {}).get("repeat_kv_before_conv", False)
            xb_size = hf_config.mamba2_config.get("d_xb", 128)
            if repeat_kv_before_conv:
                conv_dim = intermediate_size + intermediate_size + intermediate_size
            else:
                conv_dim = intermediate_size + xb_size + xb_size

            return PrimedStateShapeCalculator.mamba2_state_shape(
                intermediate_size=intermediate_size,
                tp_world_size=parallel_config.tensor_parallel_size,
                n_groups=n_groups,
                num_heads=num_heads,
                head_dim=head_dim,
                state_size=state_size,
                conv_kernel=d_conv,
                conv_dim=conv_dim,
            )

        raise RuntimeError(f"Unexpected state in get_mamba_state_shape_from_config for pattern: {pattern}")

    @classmethod
    def get_mamba_state_dtype_from_config(cls, vllm_config: "VllmConfig"):
        hf_config = vllm_config.model_config.hf_config.get_text_config()
        pattern = getattr(hf_config, "hybrid_override_pattern", "")
        bmojo_config = getattr(hf_config, "bmojo_config", {})
        syms = pattern_has_symbols(pattern, bmojo_config)
        has_gdn = Symbols.GDN in syms
        has_gka = Symbols.GKA in syms
        has_mamba = Symbols.MAMBA in syms

        if has_gka:
            # GKA needs 3 dtypes: (conv_dtype, h_kk_dtype, h_kv_dtype)
            conv_dtype, ssm_dtype = MambaStateDtypeCalculator.gated_delta_net_state_dtype(
                vllm_config.model_config.dtype,
                vllm_config.cache_config.mamba_cache_dtype,
            )
            return conv_dtype, torch.float32, ssm_dtype

        if has_gdn:
            # GDN uses 2 dtypes
            return MambaStateDtypeCalculator.gated_delta_net_state_dtype(
                vllm_config.model_config.dtype,
                vllm_config.cache_config.mamba_cache_dtype,
            )

        if has_mamba:
            # Mamba2 uses 2 dtypes
            return MambaStateDtypeCalculator.mamba2_state_dtype(
                vllm_config.model_config.dtype,
                vllm_config.cache_config.mamba_cache_dtype,
                vllm_config.cache_config.mamba_ssm_cache_dtype,
            )

        # Attention-only model
        return (vllm_config.model_config.dtype,)

    def get_mamba_state_copy_func(self) -> tuple:
        """
        Returns copy functions for each mamba state to support prefix caching.

        Returns a tuple of MambaStateCopyFunc callables, one per state tensor.
        The order must match get_mamba_state_shape_from_config.
        """
        from vllm.model_executor.layers.mamba.mamba_utils import (
            MambaStateCopyFuncCalculator, get_conv_copy_spec,
            get_temporal_copy_spec,
        )

        pattern = getattr(self.config, "hybrid_override_pattern", "")
        bmojo_config = getattr(self.config, "bmojo_config", {})
        syms = pattern_has_symbols(pattern, bmojo_config)

        has_mamba2 = Symbols.MAMBA in syms
        has_gdn = Symbols.GDN in syms
        has_gka = Symbols.GKA in syms

        if has_gka:
            # 3 states: conv_state, h_kk, h_kv
            return (get_conv_copy_spec, get_temporal_copy_spec, get_temporal_copy_spec)
        if has_gdn:
            # 2 states: conv_state, ssm_state
            return MambaStateCopyFuncCalculator.gated_delta_net_state_copy_func()
        if has_mamba2:
            # 2 states: conv_state, ssm_state
            return MambaStateCopyFuncCalculator.mamba2_state_copy_func()

        # Attention-only model — no mamba states
        return ()

    # V1: BMOJO-F cache shape and dtype methods removed
    # Cache management is now handled automatically by V1's cache system
    # via get_kv_cache_spec() in BMOJOFLayer

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        logits = self.logits_processor(self.lm_head, hidden_states)
        return logits

    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        return self.model.get_expert_mapping()
