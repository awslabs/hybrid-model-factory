# Copyright 2025 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
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
"""PyTorch L2A Qwen3 model.

Architecture:
  Each L2A layer has one unified attention module that runs:
    1. SWA (Local Attention) via P2P gather of sliding window tokens
    2. Router (sigmoid gate decides per-token whether to invoke Global Attention)
    3. Conditional Global Attention (SP-gather + qsparse/flash kernel)
  Output = SWA_output + (mask * GlobalAttn_output)
"""

import gc
import math
from typing import List, Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch.utils.checkpoint
from torch import nn

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache
from transformers.generation import GenerationMixin
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.utils import (
    auto_docstring,
    can_return_tuple,
    is_flash_attn_greater_or_equal_2_10,
    is_torchdynamo_compiling,
    logging,
)

from hmf.model.hybrid_zoo.layers.sp_p2p_utils import reorder_for_ssm_p2p
from hmf.model.model_utils.allgather_sp_helper import ZigZagGatherScatter, ZigZagScatter

from .cache import L2ACache

from .configuration_qwen3 import L2AQwen3Config

try:
    from ..layers.qsparse_prefill import attention_prefill as qsparse_attention_prefill
    QSPARSE_AVAILABLE = True
except ImportError:
    QSPARSE_AVAILABLE = False

try:
    from ring_flash_attn.zigzag_ring_flash_attn import zigzag_ring_flash_attn_func
    RING_ATTN_AVAILABLE = True
except ImportError:
    RING_ATTN_AVAILABLE = False

logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "Qwen/Qwen3-8B"
_CONFIG_FOR_DOC = "L2AQwen3Config"


def sample_bernoulli_sync_cpu(p):
    """Sample on CPU rank 0, broadcast to all."""
    if dist.get_rank() == 0:
        sample = torch.bernoulli(torch.tensor([p])).to(device=torch.cuda.current_device())
    else:
        sample = torch.zeros(1, device=torch.cuda.current_device())
    dist.broadcast(sample, src=0)
    return sample.item()


def roll_first_half_oneliner(tensor: torch.tensor, k: int) -> torch.tensor:
    """Roll first half of batch by k positions (needed for zigzag SP rank 0)."""
    half_bs = tensor.shape[0] // 2
    return torch.cat([
        torch.roll(tensor[:half_bs], shifts=k, dims=1),
        tensor[half_bs:],
    ], dim=0)


def straight_through_sigmoid(x, temp=1, threshold=0.5):
    sig = torch.sigmoid(x / float(temp))
    hard = (sig >= threshold).float()
    return hard + (sig - sig.detach()), sig

class Qwen3RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps: float = 1e-6) -> None:
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


class Qwen3MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class Qwen3RotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor

    def __init__(self, config: L2AQwen3Config, device=None):
        super().__init__()
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings
        self.config = config

        rope_parameters = getattr(config, "rope_parameters", None) or {}
        self.rope_type = rope_parameters.get("rope_type", "default")
        rope_init_fn = self.compute_default_rope_parameters
        if self.rope_type != "default":
            rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]
        inv_freq, self.attention_scaling = rope_init_fn(self.config, device)

        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.register_buffer("original_inv_freq", inv_freq.clone(), persistent=False)

    @staticmethod
    def compute_default_rope_parameters(config=None, device=None, seq_len=None):
        rope_parameters = getattr(config, "rope_parameters", None) or {}
        base = rope_parameters.get("rope_theta", getattr(config, "rope_theta", 10000.0))
        dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim))
        return inv_freq, 1.0

    @torch.no_grad()
    @dynamic_rope_update
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()
        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class LayerwiseSparsityTracker:
    """Tracks per-layer sparsity of L2A layers."""
    def __init__(self, layer_ids):
        self.layer_ids = layer_ids
        self.running_sparsity = {lid: 0 for lid in layer_ids}
        self.total_sparsity = {lid: 0.0 for lid in layer_ids}
        self.count = {lid: 0 for lid in layer_ids}
        self.avg_sparsity = {lid: 0 for lid in layer_ids}
        self.current_avg_sparsity = 0

    def reset_sparsity(self):
        self.running_sparsity = {lid: 0 for lid in self.layer_ids}
        self.total_sparsity = {lid: 0.0 for lid in self.layer_ids}
        self.count = {lid: 0 for lid in self.layer_ids}
        self.avg_sparsity = {lid: 0 for lid in self.layer_ids}
        self.current_avg_sparsity = 0

    def update(self, mask, layer_idx):
        sparsity = round((mask == 0).float().mean().item() * 100, 2)
        self.running_sparsity[layer_idx] = sparsity
        self.total_sparsity[layer_idx] += sparsity
        self.count[layer_idx] += 1
        self.avg_sparsity[layer_idx] = round(self.total_sparsity[layer_idx] / self.count[layer_idx], 2)

    def get_current_avg_sparsity(self):
        if len(self.running_sparsity) != 0:
            self.current_avg_sparsity = round(sum(self.running_sparsity.values()) / len(self.running_sparsity), 2)
        else:
            self.current_avg_sparsity = 0


class Qwen3L2AAttention(nn.Module):
    """
    Unified L2A attention module consisting of:
      1. Local Attention (SWA), which uses P2P sliding-window gather when SP is enabled
      2. Router: sigmoid gate on post-SWA hidden states
      3. Conditional Global Attention: SP-gather + qsparse/flash kernel (only when router fires)

    Uses two sets of Q/K/V/O projections for SWA and Global Attention.
    """

    def __init__(self, config: L2AQwen3Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim ** -0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True
        self.sliding_window = config.sliding_window
        self.sigmoid_threshold = config.sigmoid_threshold

        # SWA (Local Attention) projections
        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=config.attention_bias)
        # Qwen3 QK-norm (SWA)
        self.q_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        # Global Attention projections (separate set, initialized from base model)
        self.q_proj_global = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj_global = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj_global = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj_global = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=config.attention_bias)
        # Qwen3 QK-norm (Global)
        self.q_norm_global = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm_global = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        # Router
        if config.sigmoid_linear:
            self.sigmoid_linear = nn.Linear(config.hidden_size, 1, bias=False)

        # SP group (set externally by the SP monkey-patch)
        self.sequence_parallel_group = None

        # Flash attention utils
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()
        from transformers.modeling_flash_attention_utils import _flash_attention_forward
        self._flash_attention_forward = _flash_attention_forward
        # Direct flash_attn call for SWA — bypasses transformers wrapper entirely
        from flash_attn import flash_attn_func
        self._raw_flash_attn_func = flash_attn_func

    # SWA/Local Attention
    def forward_swa(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        """Compute SWA (local attention) and return s_t.

        Args:
            hidden_states: layernorm'd input [batch, seq_len, hidden_size]
            position_embeddings: (cos, sin) for RoPE
            past_key_values: L2ACache for inference
            cache_position: position indices for cache

        Returns:
            swa_output: the local attention output s_t [batch, seq_len, hidden_size]
        """
        bsz, q_len, _ = hidden_states.shape
        hidden_shape = (bsz, q_len, -1, self.head_dim)

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update_swa(key_states, value_states, self.layer_idx, cache_kwargs)

        # Repeat KV for GQA (prefill only; flash_attn handles GQA natively during decode)
        if query_states.shape[2] > 1:
            key_states = repeat_kv(key_states, self.num_key_value_groups)
            value_states = repeat_kv(value_states, self.num_key_value_groups)

        dropout_rate = 0.0 if not self.training else self.attention_dropout

        # Cast to target dtype if needed
        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype
            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        # Reshape to [bsz, seq_len, num_heads, head_dim] for flash attention
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        sliding_window = self.sliding_window

        if self.sequence_parallel_group is not None:
            sp_size = dist.get_world_size(group=self.sequence_parallel_group)
            sp_rank = dist.get_rank(group=self.sequence_parallel_group)

            # P2P gather: get sliding_window tokens from neighbor
            query_swa = reorder_for_ssm_p2p(
                query_states.reshape(bsz, q_len, -1), self.sequence_parallel_group,
                torch.cuda.Stream(), sp_size, sp_rank, (sliding_window + 1)
            )
            key_swa = reorder_for_ssm_p2p(
                key_states.reshape(bsz, q_len, -1), self.sequence_parallel_group,
                torch.cuda.Stream(), sp_size, sp_rank, (sliding_window + 1)
            )
            value_swa = reorder_for_ssm_p2p(
                value_states.reshape(bsz, q_len, -1), self.sequence_parallel_group,
                torch.cuda.Stream(), sp_size, sp_rank, (sliding_window + 1)
            )

            bsz_sp = 2 * bsz
            q_len_sp = q_len // 2

            query_swa = query_swa.view(bsz_sp, q_len_sp + sliding_window, self.num_heads, self.head_dim)
            key_swa = key_swa.view(bsz_sp, q_len_sp + sliding_window, self.num_heads, self.head_dim)
            value_swa = value_swa.view(bsz_sp, q_len_sp + sliding_window, self.num_heads, self.head_dim)

            if sp_rank == 0:
                query_swa = roll_first_half_oneliner(query_swa, q_len_sp)
                key_swa = roll_first_half_oneliner(key_swa, q_len_sp)
                value_swa = roll_first_half_oneliner(value_swa, q_len_sp)
        else:
            query_swa = query_states
            key_swa = key_states
            value_swa = value_states
            bsz_sp = bsz
            q_len_sp = q_len

        # Run SWA flash attention
        with torch.cuda.device(query_swa.device):
            swa_attn_output = self._raw_flash_attn_func(
                query_swa, key_swa, value_swa,
                dropout_p=dropout_rate,
                softmax_scale=self.scaling,
                causal=self.is_causal,
                window_size=(sliding_window - 1, 0),
            )

        # Undo SP reordering for SWA output
        if self.sequence_parallel_group is not None:
            if sp_rank > 0:
                swa_attn_output = swa_attn_output[:, sliding_window:].contiguous()
            else:
                drop_tokens = sliding_window
                half_bs = swa_attn_output.shape[0] // 2
                swa_attn_output = torch.cat([
                    swa_attn_output[:half_bs, :-drop_tokens],
                    swa_attn_output[half_bs:, drop_tokens:],
                ], dim=0).contiguous()

            if bsz == 2:
                swa_attn_output = swa_attn_output.view(bsz_sp // 2, -1, self.num_heads, self.head_dim)
            else:
                swa_attn_output = torch.cat([swa_attn_output[:bsz_sp // 2], swa_attn_output[bsz_sp // 2:]], dim=1).contiguous()

        bsz, q_len_out, _, _ = swa_attn_output.shape
        swa_attn_output = swa_attn_output.reshape(bsz, q_len_out, self.hidden_size).contiguous()
        swa_output = self.o_proj(swa_attn_output)

        return swa_output

    # Router + Global Attention
    def forward_global(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        token_counter: Optional[int] = 0,
        sparsity_tracker: Optional[LayerwiseSparsityTracker] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute router decision + conditional global attention.

        This receives the post-SWA representation (after residual + layernorm_global) as input

        Args:
            hidden_states: layernorm_global(x + swa_output) [batch, seq_len, hidden_size]
            position_embeddings: (cos, sin) for RoPE
            past_key_values: L2ACache for inference
            cache_position: position indices for cache
            token_counter: total tokens seen (for logging)
            sparsity_tracker: tracks per-layer sparsity

        Returns:
            global_output: masked global attention output (d_t * a_t)
            soft_decision_batched_sums: regularization term for training
        """
        bsz, q_len, _ = hidden_states.shape
        soft_decision_batched_sums = torch.zeros(bsz, device=hidden_states.device, requires_grad=True)

        # Router: d_t = sigma(W * s_t)
        sig_input_type = self.config.sig_input_type
        sigmoid_temp = self.config.sigmoid_temp

        if sig_input_type == "hidden_firstdim":
            mask, soft_dec = straight_through_sigmoid(
                hidden_states[:, :, 0], temp=sigmoid_temp, threshold=self.sigmoid_threshold
            )
        else:
            if self.config.sigmoid_linear:
                sig_inputs = self.sigmoid_linear(hidden_states)
                if self.training:
                    mask, soft_dec = straight_through_sigmoid(
                        sig_inputs.squeeze(-1), temp=sigmoid_temp, threshold=self.sigmoid_threshold
                    )
                else:
                    logit_threshold = math.log(self.sigmoid_threshold / (1.0 - self.sigmoid_threshold))
                    mask = (sig_inputs.squeeze(-1) >= sigmoid_temp * logit_threshold).float()
                    soft_dec = mask
            else:
                sig_inputs = hidden_states.mean(dim=-1)
                mask, soft_dec = straight_through_sigmoid(
                    sig_inputs, temp=sigmoid_temp, threshold=self.sigmoid_threshold
                )

        mask = mask.unsqueeze(-1).unsqueeze(-1)  # [batch, seq_len, 1, 1]

        if self.training or q_len > 1:
            if sparsity_tracker is not None:
                sparsity_tracker.update(mask, self.layer_idx)
            if self.training:
                soft_decision_batched_sums = torch.sum(soft_dec ** 2, dim=1)  # [batch]

        # Global Attention: a_t = GlobalAttn(s_t, s_{t-1}, ..., s_1)
        hidden_shape = (bsz, q_len, -1, self.head_dim)
        cos, sin = position_embeddings

        query_states_global = self.q_norm_global(self.q_proj_global(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states_global = self.k_norm_global(self.k_proj_global(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states_global = self.v_proj_global(hidden_states).view(hidden_shape).transpose(1, 2)

        query_states_global, key_states_global = apply_rotary_pos_emb(query_states_global, key_states_global, cos, sin)

        # Cache global KV states unconditionally during inference
        if past_key_values is not None and hasattr(past_key_values, "update_global"):
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states_global, value_states_global = past_key_values.update_global(
                key_states_global, value_states_global, self.layer_idx, cache_kwargs
            )

        # Repeat KV for GQA
        if query_states_global.shape[2] > 1:
            key_states_global = repeat_kv(key_states_global, self.num_key_value_groups)
            value_states_global = repeat_kv(value_states_global, self.num_key_value_groups)

        dropout_rate = 0.0 if not self.training else self.attention_dropout

        # Cast to target dtype if needed
        input_dtype = query_states_global.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype
            query_states_global = query_states_global.to(target_dtype)
            key_states_global = key_states_global.to(target_dtype)
            value_states_global = value_states_global.to(target_dtype)

        # Reshape to [bsz, seq_len, num_heads, head_dim] for flash attention
        query_states_global = query_states_global.transpose(1, 2)
        key_states_global = key_states_global.transpose(1, 2)
        value_states_global = value_states_global.transpose(1, 2)

        if self.training:
            coin_flip = sample_bernoulli_sync_cpu(p=0.9)
        else:
            coin_flip = 1.0

        # SP gather is CONDITIONAL on coin_flip > 0.5
        if self.sequence_parallel_group is not None and coin_flip > 0.5:
            sp_size = dist.get_world_size(group=self.sequence_parallel_group)
            sp_rank = dist.get_rank(group=self.sequence_parallel_group)

            BATCH, N_CTX, H, D_HEAD = query_states_global.shape
            q_flat = query_states_global.reshape(BATCH, N_CTX, -1)
            k_flat = key_states_global.reshape(BATCH, N_CTX, -1)
            v_flat = value_states_global.reshape(BATCH, N_CTX, -1)

            combined = torch.cat([q_flat, k_flat, v_flat], dim=-1)
            combined_gathered = ZigZagGatherScatter.apply(
                combined, self.sequence_parallel_group, sp_rank, sp_size, sp_size * 2
            )
            q_g, k_g, v_g = torch.split(combined_gathered, [H * D_HEAD, H * D_HEAD, H * D_HEAD], dim=-1)

            query_states_global = q_g.view(BATCH, N_CTX * sp_size, H, D_HEAD).contiguous()
            key_states_global = k_g.view(BATCH, N_CTX * sp_size, H, D_HEAD).contiguous()
            value_states_global = v_g.view(BATCH, N_CTX * sp_size, H, D_HEAD).contiguous()

            if mask is not None:
                mask = ZigZagGatherScatter.apply(
                    mask.view(BATCH, N_CTX, -1), self.sequence_parallel_group, sp_rank, sp_size, sp_size * 2
                )
                mask = mask.view(BATCH, N_CTX * sp_size, 1, 1).contiguous()

            q_len_global = N_CTX * sp_size
        else:
            q_len_global = q_len

        # Dispatch: qsparse kernel or flash attention
        # NOTE: torch.cuda.device() ensures the CUDA context targets the correct GPU
        # when the model is split across devices.
        with torch.cuda.device(query_states_global.device):
            if (query_states_global.shape[1] > 1 and mask.sum() != 0) and QSPARSE_AVAILABLE and coin_flip > 0.5:
                attn_output = qsparse_attention_prefill(
                    query_states_global,
                    key_states_global,
                    value_states_global,
                    mask.squeeze(-1).expand(-1, -1, self.num_heads).contiguous(),
                )
            elif mask.sum() == 0 and coin_flip > 0.5:
                # All tokens masked. Skips global attention entirely
                if not self.training:
                    return torch.zeros_like(hidden_states), soft_decision_batched_sums
                attn_output = query_states_global
            else:
                # coin_flip <= 0.5 (10%): run full attention for ALL tokens
                if RING_ATTN_AVAILABLE and self.sequence_parallel_group is not None:
                    attn_output = zigzag_ring_flash_attn_func(
                        query_states_global,
                        key_states_global,
                        value_states_global,
                        dropout_rate,
                        causal=self.is_causal,
                        group=self.sequence_parallel_group,
                    )
                else:
                    if self.training:
                        logger.warning_once(
                            "Ring attention not available or no SP group. Falling back to full gather for stochastic leakage path. "
                            "This is expensive. Install ring_flash_attn for efficient training."
                        )
                    # Gather full sequence and run dense flash attention
                    if self.sequence_parallel_group is not None:
                        sp_size = dist.get_world_size(group=self.sequence_parallel_group)
                        sp_rank = dist.get_rank(group=self.sequence_parallel_group)
                        BATCH, N_CTX, H, D_HEAD = query_states_global.shape
                        q_flat = query_states_global.reshape(BATCH, N_CTX, -1)
                        k_flat = key_states_global.reshape(BATCH, N_CTX, -1)
                        v_flat = value_states_global.reshape(BATCH, N_CTX, -1)
                        combined = torch.cat([q_flat, k_flat, v_flat], dim=-1)
                        combined_gathered = ZigZagGatherScatter.apply(
                            combined, self.sequence_parallel_group, sp_rank, sp_size, sp_size * 2
                        )
                        q_g, k_g, v_g = torch.split(combined_gathered, [H * D_HEAD, H * D_HEAD, H * D_HEAD], dim=-1)
                        query_states_global = q_g.view(BATCH, N_CTX * sp_size, H, D_HEAD).contiguous()
                        key_states_global = k_g.view(BATCH, N_CTX * sp_size, H, D_HEAD).contiguous()
                        value_states_global = v_g.view(BATCH, N_CTX * sp_size, H, D_HEAD).contiguous()
                        if mask is not None:
                            mask = ZigZagGatherScatter.apply(
                                mask.view(BATCH, N_CTX, -1), self.sequence_parallel_group, sp_rank, sp_size, sp_size * 2
                            )
                            mask = mask.view(BATCH, N_CTX * sp_size, 1, 1).contiguous()
                        q_len_global = N_CTX * sp_size

                    attn_output = self._raw_flash_attn_func(
                        query_states_global, key_states_global, value_states_global,
                        dropout_p=dropout_rate,
                        softmax_scale=self.scaling,
                        causal=self.is_causal,
                    )

        # Apply router mask: d_t * a_t
        if mask is not None:
            attn_output = (attn_output * mask).to(input_dtype)
        attn_output = attn_output.reshape(bsz, q_len_global, self.hidden_size).contiguous()
        attn_output = self.o_proj_global(attn_output)

        # SP scatter (conditional, matching the gather)
        _did_fallback_gather = (coin_flip <= 0.5 and not RING_ATTN_AVAILABLE and self.sequence_parallel_group is not None)
        if self.sequence_parallel_group is not None and (coin_flip > 0.5 or _did_fallback_gather):
            sp_size = dist.get_world_size(group=self.sequence_parallel_group)
            sp_rank = dist.get_rank(group=self.sequence_parallel_group)
            attn_output = ZigZagScatter.apply(
                attn_output, self.sequence_parallel_group, sp_rank, sp_size, sp_size * 2
            )

        return attn_output, soft_decision_batched_sums


class Qwen3L2ADecoderLayer(nn.Module):
    """Single L2A decoder layer.

    x → LayerNorm → SWA → +residual → LayerNorm_global → Router + GlobalAttn → +residual → LayerNorm → MLP → +residual
    """

    def __init__(self, config: L2AQwen3Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx

        self.self_attn = Qwen3L2AAttention(config, layer_idx)
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.input_layernorm_global = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        token_counter: Optional[int] = 0,
        sparsity_tracker: Optional[LayerwiseSparsityTracker] = None,
        sigmoid_reg: Optional[list] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, ...]:
        """
        Forward pass through a single L2A decoder layer.

        Executes:
          1. SWA (Local Attention) on the layernorm'd input
          2. Router + conditional Global Attention on the post-SWA residual stream
          3. MLP on the post-attention residual stream

        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, sequence_length)` where padding elements are indicated by 0.
            position_ids (`torch.LongTensor`, *optional*): position IDs for the input.
            past_key_values (`Cache`, *optional*): cached past key and value projection states.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors. Not used by L2A layers (returns None).
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` are returned and can be used to speed up decoding.
            cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
                Indices depicting the position of the input sequence tokens in the sequence.
            position_embeddings (`Tuple[torch.FloatTensor, torch.FloatTensor]`, *optional*):
                Tuple containing the cosine and sine positional embeddings of shape
                `(batch_size, seq_len, head_dim)`.
            token_counter (`int`, *optional*): total tokens seen so far (for logging).
            sparsity_tracker (`LayerwiseSparsityTracker`, *optional*): tracks per-layer router sparsity.
            sigmoid_reg (`list`, *optional*): unused, kept for interface compatibility.
            kwargs (`dict`, *optional*): arbitrary kwargs to be ignored.

        Returns:
            A tuple containing:
              - hidden_states (`torch.FloatTensor`): output of shape `(batch, seq_len, hidden_size)`
              - attentions (None): placeholder for API compatibility
              - past_key_values (`Cache`, optional): returned if `use_cache=True`
              - soft_decision_batched_sums (`torch.FloatTensor`): router regularization term
        """
        # Step 1: SWA (Local Attention)
        # s_t = LocalAttn(layernorm(x))
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        swa_output = self.self_attn.forward_swa(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            past_key_values=past_key_values,
            cache_position=cache_position,
        )
        # x = x + s_t
        hidden_states = residual + swa_output

        # Step 2: Router + Global Attention
        # Router and global attention operate on s_t (post-SWA stream)
        # d_t = sigma(W * layernorm_global(x + s_t))
        # a_t = GlobalAttn(layernorm_global(x + s_t)) if d_t == 1
        # x = x + d_t * a_t
        residual = hidden_states
        hidden_states_for_global = self.input_layernorm_global(hidden_states)

        global_output, soft_decision_batched_sums = self.self_attn.forward_global(
            hidden_states=hidden_states_for_global,
            position_embeddings=position_embeddings,
            past_key_values=past_key_values,
            cache_position=cache_position,
            token_counter=token_counter,
            sparsity_tracker=sparsity_tracker,
        )
        hidden_states = residual + global_output

        # Step 3: MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (None,)
        if use_cache:
            outputs += (past_key_values,)
        outputs += (soft_decision_batched_sums,)
        return outputs


@auto_docstring
class L2AQwen3PreTrainedModel(PreTrainedModel):
    config_class = L2AQwen3Config
    config: L2AQwen3Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Qwen3L2ADecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = False
    _supports_flex_attn = False
    _can_compile_fullgraph = False
    _supports_attention_backend = True


@auto_docstring
class L2AQwen3v2Model(L2AQwen3PreTrainedModel):
    """L2A Qwen3 transformer decoder."""

    def __init__(self, config: L2AQwen3Config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Qwen3L2ADecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(config=config)

        self.gradient_checkpointing = False
        self.post_init()

        self.token_counter = 0
        self.sparsity_tracker = LayerwiseSparsityTracker(list(range(config.num_hidden_layers)))

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    @can_return_tuple
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        sigmoid_reg = []

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once("`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`...")
                use_cache = False

        if not isinstance(past_key_values, (type(None), Cache)):
            raise ValueError("The `past_key_values` should be either a `Cache` object or `None`.")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if not self.training and use_cache and (
            past_key_values is None or not isinstance(past_key_values, L2ACache)
        ):
            del past_key_values
            gc.collect()
            torch.cuda.empty_cache()
            past_key_values = L2ACache(self.config)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions)

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states, causal_mask, position_ids, past_key_values,
                    output_attentions, use_cache, cache_position, position_embeddings,
                    self.token_counter, self.sparsity_tracker, None,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    token_counter=self.token_counter,
                    sparsity_tracker=self.sparsity_tracker,
                    sigmoid_reg=None,
                )

            hidden_states = layer_outputs[0]

            if self.training:
                index = 1
                if output_attentions:
                    index += 1
                if use_cache:
                    index += 1
                soft_decision_batched_sums = layer_outputs[index]
                batch_reg_loss = torch.mean(soft_decision_batched_sums) / len(self.layers)
                sigmoid_reg.append(batch_reg_loss)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if not self.training:
            self.token_counter += hidden_states.shape[1]

        self.sparsity_tracker.get_current_avg_sparsity()

        if len(sigmoid_reg) == 0:
            self.reg_loss_intermediate = torch.tensor(0.0, device=inputs_embeds.device, requires_grad=True)
        else:
            self.reg_loss_intermediate = torch.stack(sigmoid_reg).sum()

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    def _update_causal_mask(self, attention_mask, input_tensor, cache_position, past_key_values, output_attentions):
        """Flash attention handles causality internally--return None or raw padding mask."""
        if self.config._attn_implementation in ["flash_attention_2", "sequence_parallel_attention"]:
            if attention_mask is not None and past_key_values is not None:
                is_padding_right = attention_mask[:, -1].sum().item() != input_tensor.size()[0]
                if is_padding_right:
                    raise ValueError(
                        "You are attempting to perform batched generation with padding_side='right'. "
                        "Use `tokenizer.padding_side = 'left'` before tokenizing."
                    )
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None

        # Fallback: build 4D causal mask
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        dtype, device = input_tensor.dtype, input_tensor.device
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]
        target_length = (
            attention_mask.shape[-1] if isinstance(attention_mask, torch.Tensor)
            else past_seen_tokens + sequence_length + 1
        )
        causal_mask = torch.full((sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device)
        diagonal_attend_mask = torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
        causal_mask *= diagonal_attend_mask
        causal_mask = causal_mask[None, None, :, :].expand(input_tensor.shape[0], 1, -1, -1)
        if attention_mask is not None:
            causal_mask = causal_mask.clone()
            mask_length = attention_mask.shape[-1]
            padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :].to(causal_mask.device)
            causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(padding_mask == 0, min_dtype)
        return causal_mask


@auto_docstring
class L2AQwen3ForCausalLM(L2AQwen3PreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    _tp_plan = {"lm_head": "colwise_gather_output"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config):
        super().__init__(config)
        self.model = L2AQwen3v2Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @can_return_tuple
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
        )

        hidden_states = outputs.last_hidden_state
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size)

        # Update offset for L2ACache
        past_key_values_ = outputs.past_key_values
        if use_cache and past_key_values_ is not None:
            if hasattr(past_key_values_, "update_offset"):
                past_key_values_.update_offset(hidden_states.shape[1])

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


__all__ = [
    "L2AQwen3ForCausalLM",
    "L2AQwen3v2Model",
    "L2AQwen3PreTrainedModel",
]
