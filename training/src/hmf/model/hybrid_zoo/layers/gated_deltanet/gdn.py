"""
Gated DeltaNet layer implementation with Grouped Query Attention (GQA) support.

This module implements the Gated Delta Networks layer from the paper:
"Gated Delta Networks: Improving Mamba2 with Delta Rule" (https://arxiv.org/abs/2412.06464).

Overview:
    Gated DeltaNet combines the delta rule with gating mechanisms to improve upon Mamba2's
    architecture. The layer supports GQA-style head grouping where the number of key/value
    heads can be different from the number of query heads. The hyperparameters in this module
    are set so that we can initialize the module's weights from existing Attention layers.

Key Variables:
    - head_dim: Head dimension (same for q/k/v)
    - num_q_heads: Number of query heads
    - num_k_heads: Number of key heads
    - num_v_heads: Number of value heads
    - expanded_num_heads: Maximum of num_q_heads, num_k_heads, num_v_heads
    - num_q_groups: Repetition factor for Q heads, computed as expanded_num_heads // num_q_heads
    - num_k_groups: Repetition factor for K heads, computed as expanded_num_heads // num_k_heads
    - num_v_groups: Repetition factor for V heads, computed as expanded_num_heads // num_v_heads
    - query_dim: Total query dimension, num_q_heads * head_dim
    - key_dim: Total key dimension, num_k_heads * head_dim
    - value_dim: Total value dimension, num_v_heads * head_dim
    - expanded_dim: Expanded dimension, expanded_num_heads * head_dim

Extended from: https://github.com/sustcsonglin/flash-linear-attention
"""

from __future__ import annotations

import math
import warnings
from typing import TYPE_CHECKING, Optional, Tuple

import torch
import torch.nn as nn
from einops import rearrange, repeat
from torch import Tensor
from torch.nn import functional as F

from fla.layers.utils import get_unpad_data, index_first_axis, pad_input
from fla.modules import FusedRMSNormGated, RMSNorm, ShortConvolution
from fla.ops.gated_delta_rule import (
    chunk_gated_delta_rule,
    fused_recurrent_gated_delta_rule,
)

from ..low_rank_linear import LinearLowRank
from hmf.model.model_utils.allgather_sp_helper import (
    DifferentiableAllGather,
    ZigZagGatherScatter,
    ZigZagScatter,
)

if TYPE_CHECKING:
    from hmf.model.hybrid_zoo.models.cache import GatedDeltaNetCache


def _get_preceding_gpu(sp_rank, sp_size, chunk_idx):
    """
    Get the GPU rank and chunk type (0=fwd, 1=bwd) that holds the chunk
    preceding `chunk_idx` in the global zigzag sequence.

    Zigzag assignment: GPU i holds fwd=position i, bwd=position 2*SP-1-i.
    """
    prev_pos = chunk_idx - 1
    if prev_pos < 0:
        return None, None
    if prev_pos < sp_size:
        return prev_pos, 0
    else:
        return 2 * sp_size - 1 - prev_pos, 1


def _compute_received_states(all_states_fwd, all_states_bwd, all_M_fwd, all_M_bwd, sp_rank, sp_size):
    """
    Compute received initial states using exact matrix chain.

    The actual state after chunk j: actual[j] = M_j @ actual[j-1] + state_0[j]
    where M_j is the linear operator for chunk j, and state_0[j] is
    the final state when running with initial_state=0.

    Args:
        all_states_fwd: [SP, B, H, K, V] - final states with init=0 (forward-sweep chunks).
        all_states_bwd: [SP, B, H, K, V] - final states with init=0 (backward-sweep chunks).
        all_M_fwd: [SP, B, H, K, V] - linear operators M (forward-sweep chunks).
        all_M_bwd: [SP, B, H, K, V] - linear operators M (backward-sweep chunks).
        sp_rank: This GPU's rank in the SP group.
        sp_size: Number of GPUs in the SP group.

    Returns:
        recv_state1: Initial state for this GPU's forward-sweep chunk, or None for rank 0.
        recv_state2: Initial state for this GPU's backward-sweep chunk.
    """
    if sp_rank == 0:
        recv_state1 = None
    else:
        actual = all_states_fwd[0]
        for j in range(1, sp_rank):
            actual = torch.matmul(all_M_fwd[j], actual) + all_states_fwd[j]
        recv_state1 = actual

    actual = all_states_fwd[0]
    for j in range(1, sp_size):
        actual = torch.matmul(all_M_fwd[j], actual) + all_states_fwd[j]

    target = 2 * sp_size - 1 - sp_rank
    for pos in range(sp_size, target):
        gpu = 2 * sp_size - 1 - pos
        actual = torch.matmul(all_M_bwd[gpu], actual) + all_states_bwd[gpu]

    recv_state2 = actual
    return recv_state1, recv_state2


class GatedDeltaNet(nn.Module):
    """
    Gated DeltaNet layer with support for Grouped Query Attention (GQA).

    This layer implements the Gated Delta Networks architecture which improves upon Mamba2
    by incorporating the delta rule with gating mechanisms. It supports both chunk-based
    and fused recurrent computation modes.

    Args:
        hidden_size: The hidden size of the input. Default: 2048.
        head_dim: The dimension of each head. Assumes q/k/v have the same head dimension.
            Default: 128.
        num_q_heads: The number of query heads. Default: 16.
        num_k_heads: The number of key heads for GQA. If None, defaults to num_q_heads.
            Default: None.
        num_v_heads: The number of value heads for GQA. If None, defaults to num_q_heads.
            Default: None.
        mode: Which Gated DeltaNet kernel to use. Options: 'chunk' or 'fused_recurrent'.
            Default: 'chunk'.
        use_gate: Whether to use output gate. Default: True.
        use_short_conv: Whether to use short convolutions. Default: True.
        allow_neg_eigval: Allow negative eigenvalues. If True, beta is multiplied by 2.
            See: "Unlocking State-Tracking in Linear RNNs Through Negative Eigenvalues"
            (https://arxiv.org/abs/2411.12537). Default: False.
        conv_size: The kernel size of the short convolution. Only used when use_short_conv is True.
            Default: 4.
        conv_bias: Whether to use bias in the short convolution. Only used when use_short_conv is True.
            Default: False.
        layer_idx: The index of the layer. Used for caching during inference. Default: None.
        norm_eps: The epsilon value for the normalization layer. Default: 1e-5.
        kv_proj_rank: Rank for low-rank KV projection to expand K/V heads. If None, uses
            standard GQA repeat. Default: None.
        kv_learnable_residual: Whether to use a learnable residual for the KV head expansion
            instead of a fixed repeat. Only used when kv_proj_rank is set. Default: False.
        use_state_passing_sp: Whether to use state-passing sequence parallelism instead of
            gather-based SP. State passing keeps each GPU's chunk local and propagates a
            small state matrix between GPUs, giving ~1.6x full-model speedup at 128k
            context. Falls back to gather-based SP for short chunks or masked inputs.
            Default: False.
    """

    def __init__(
        self,
        hidden_size: int = 2048,
        head_dim: int = 128,
        num_q_heads: int = 16,
        num_k_heads: Optional[int] = None,
        num_v_heads: Optional[int] = None,
        mode: str = "chunk",
        use_gate: bool = True,
        use_short_conv: bool = True,
        allow_neg_eigval: bool = False,
        conv_size: int = 4,
        conv_bias: bool = False,
        layer_idx: Optional[int] = None,
        norm_eps: float = 1e-5,
        kv_proj_rank: Optional[int] = None,
        kv_learnable_residual: bool = False,
        use_state_passing_sp: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()

        self.mode = mode
        self.allow_neg_eigval = allow_neg_eigval
        self.hidden_size = hidden_size

        self.use_gate = use_gate
        self.use_short_conv = use_short_conv
        self.use_state_passing_sp = use_state_passing_sp
        self.conv_size = conv_size
        self.conv_bias = conv_bias
        self.sequence_parallel_group = None
        self.register_buffer("_eye_cache", torch.eye(head_dim), persistent=False)

        self.head_dim = head_dim
        self.num_q_heads = num_q_heads
        self.num_k_heads = num_k_heads if num_k_heads is not None else num_q_heads
        self.num_v_heads = num_v_heads if num_v_heads is not None else num_q_heads
        self.expanded_num_heads = max(self.num_q_heads, self.num_k_heads, self.num_v_heads)
        self.num_q_groups = self.expanded_num_heads // self.num_q_heads
        self.num_k_groups = self.expanded_num_heads // self.num_k_heads
        self.num_v_groups = self.expanded_num_heads // self.num_v_heads

        self.query_dim = self.num_q_heads * self.head_dim
        self.key_dim = self.num_k_heads * self.head_dim
        self.value_dim = self.num_v_heads * self.head_dim
        self.expanded_dim = self.expanded_num_heads * self.head_dim
        
        self.layer_idx = layer_idx

        assert mode in ["chunk", "fused_recurrent"], f"Not supported mode `{mode}`."

        # Projection layers
        self.q_proj = nn.Linear(hidden_size, self.query_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
        self.b_proj = nn.Linear(hidden_size, self.expanded_num_heads, bias=False)

        # NOTE: gk_proj is called a_proj in FLA implementation, named gk_proj for BC.
        self.gk_proj = nn.Linear(hidden_size, self.expanded_num_heads, bias=False)

        # Initialize A parameter (log scale)
        A = torch.empty(self.expanded_num_heads, dtype=torch.float32).uniform_(0, 16)
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True

        # Initialize dt bias parameter
        dt_min = 0.001
        dt_max = 0.1
        dt_init_floor = 1e-4
        dt = torch.exp(
            torch.rand(self.expanded_num_heads) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        dt = torch.clamp(dt, min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        self.dt_bias._no_weight_decay = True

        # Short convolution layers
        if use_short_conv:
            self.conv_size = conv_size
            self.q_conv1d = ShortConvolution(
                hidden_size=self.query_dim,
                kernel_size=conv_size,
                bias=self.conv_bias,
                activation="silu",
            )
            self.k_conv1d = ShortConvolution(
                hidden_size=self.key_dim,
                kernel_size=conv_size,
                bias=self.conv_bias,
                activation="silu",
            )
            self.v_conv1d = ShortConvolution(
                hidden_size=self.value_dim,
                kernel_size=conv_size,
                bias=self.conv_bias,
                activation="silu",
            )
        else:
            warnings.warn(
                "ShortConvolution is crucial to model performance. "
                "Turning it off (i.e., by setting `use_short_conv=False`) is not recommended."
            )

        # Output gating and normalization
        if use_gate:
            self.g_proj = nn.Linear(hidden_size, self.expanded_dim, bias=False)
            self.o_norm = FusedRMSNormGated(self.head_dim, eps=norm_eps)
        else:
            self.o_norm = RMSNorm(self.head_dim, eps=norm_eps)

        self.o_proj = nn.Linear(self.expanded_dim, hidden_size, bias=False)

        # Optional KV projection for GQA
        self.kv_proj_rank = kv_proj_rank
        self.kv_learnable_residual = kv_learnable_residual
        if self.kv_proj_rank is not None:
            act_fn = nn.SiLU()
            self.proj_k = LinearLowRank(
                in_features=self.key_dim,
                out_features=self.expanded_num_heads * self.head_dim,
                rank=self.kv_proj_rank,
                act_fn=act_fn,
            )
            self.proj_v = LinearLowRank(
                in_features=self.value_dim,
                out_features=self.expanded_num_heads * self.head_dim,
                rank=self.kv_proj_rank,
                act_fn=act_fn,
            )

            if self.kv_learnable_residual:
                # Learnable mixing matrices initialized to mimic repeat_kv
                eye_k = torch.eye(self.num_k_heads)
                ones_k = torch.ones((self.num_k_groups, 1))
                self.k_rep = nn.Parameter(
                    torch.kron(eye_k, ones_k)
                )  # [expanded_num_heads, num_k_heads]

                eye_v = torch.eye(self.num_v_heads)
                ones_v = torch.ones((self.num_v_groups, 1))
                self.v_rep = nn.Parameter(
                    torch.kron(eye_v, ones_v)
                )  # [expanded_num_heads, num_v_heads]


    def _expand_kv(self, k: Tensor, v: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Expand k and v from their native head counts to expanded_num_heads.

        Three modes:
        1. kv_proj_rank + kv_learnable_residual: learnable mixing residual + low-rank projection
        2. kv_proj_rank without learnable residual: plain repeat residual + low-rank projection
        3. No projection: simple repeat (or rearrange if groups == 1). This is the standard GQA repeat.

        Args:
            k: [B, L, Dk] raw key tensor (flat, not yet reshaped to heads)
            v: [B, L, Dv] raw value tensor (flat, not yet reshaped to heads)

        Returns:
            k: [B, L, H, D] expanded keys
            v: [B, L, H, D] expanded values
        """
        if self.kv_proj_rank is not None and (
            self.num_k_groups > 1 or self.num_v_groups > 1
        ):
            # Build residual
            if self.kv_learnable_residual:
                k_heads = rearrange(k, "... (h d) -> ... h d", h=self.num_k_heads)
                k_res = torch.einsum("eh,blhd->bled", self.k_rep, k_heads)
                v_heads = rearrange(v, "... (h d) -> ... h d", h=self.num_v_heads)
                v_res = torch.einsum("eh,blhd->bled", self.v_rep, v_heads)
            else:
                k_res = repeat(
                    rearrange(k, "... (h d) -> ... h d", h=self.num_k_heads),
                    "... h d -> ... (h g) d",
                    g=self.num_k_groups,
                )
                v_res = repeat(
                    rearrange(v, "... (h d) -> ... h d", h=self.num_v_heads),
                    "... h d -> ... (h g) d",
                    g=self.num_v_groups,
                )

            # Low-rank projection to expanded dim + add residual
            k = (
                rearrange(
                    self.proj_k(k), "... (h d) -> ... h d", h=self.expanded_num_heads
                )
                + k_res
            )
            v = (
                rearrange(
                    self.proj_v(v), "... (h d) -> ... h d", h=self.expanded_num_heads
                )
                + v_res
            )
        else:
            # Plain rearrange + repeat
            k = rearrange(k, "... (h d) -> ... h d", h=self.num_k_heads)
            v = rearrange(v, "... (h d) -> ... h d", h=self.num_v_heads)
            if self.num_k_groups > 1:
                k = repeat(k, "... h d -> ... (h g) d", g=self.num_k_groups)
            if self.num_v_groups > 1:
                v = repeat(v, "... h d -> ... (h g) d", g=self.num_v_groups)

        return k, v

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        past_key_values: Optional["GatedDeltaNetCache"] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[Tensor, None, Optional["GatedDeltaNetCache"]]:
        """
        Gated DeltaNet forward pass with optional caching.

        NOTE: We use the following variables for tensor shape hints:
            - B: Input batch size
            - L: Sequence length
            - SP: Sequence parallel size. 1 if not using sequence parallel
            - Dh: Hidden dimension (self.hidden_size)
            - H: Expanded number of heads (self.expanded_num_heads)
            - Hq: Number of query heads (self.num_q_heads)
            - Hk: Number of key heads (self.num_k_heads)
            - Hv: Number of value heads (self.num_v_heads)
            - D: Head dimension (self.head_dim)
            - Dq: Query dimension (self.query_dim)
            - Dk: Key dimension (self.key_dim)
            - Dv: Value dimension (self.value_dim)
            - Dexp: Expanded dimension (self.expanded_dim)
            - C: Convolution size (self.conv_size if self.use_short_conv else 0)

        Args:
            hidden_states: Input tensor with shape [B, L // SP, Dh].
            attention_mask: Optional 0-1 mask with shape [B, L] for padding purposes.
                0 indicates padding positions. Arbitrary attention masks of shape
                [B, L, L] are not allowed.
            past_key_values: Optional cache containing conv_state and recurrent_state
                for inference. When provided, performs incremental decoding.
            use_cache: Whether to return updated cache states. Default: False.
            output_attentions: Whether to return attention weights (not used). Default: False.
            **kwargs: Additional keyword arguments, may include cu_seqlens for packed sequences.

        Returns:
            Tuple of (output, None, past_key_values) where:
                - output: Output tensor with same shape as hidden_states [B, L, Dh]
                - None: Placeholder for consistency with attention layers
                - past_key_values: Updated cache (same as input if provided, else None)
        """
        if attention_mask is not None:
            assert len(attention_mask.shape) == 2, (
                "Expected attention_mask as a 0-1 matrix with shape [batch_size, seq_len] "
                "for padding purposes (0 indicating padding). "
                "Arbitrary attention masks of shape [batch_size, seq_len, seq_len] are not allowed."
            )

        batch_size, q_len, _ = hidden_states.shape

        # Change to inference mode for short sequences
        mode = "fused_recurrent" if (q_len <= 64 and not self.training) else self.mode

        if self.training:
            assert mode == "chunk", "Only chunk mode is supported in training."

        last_state = None
        if not self.training and past_key_values is not None:
            last_state = past_key_values[self.layer_idx]

        cu_seqlens = kwargs.get("cu_seqlens")

        # Sequence parallel setup
        use_state_passing = False
        sp_size = None
        sp_rank = None
        if self.sequence_parallel_group is not None:
            assert (
                cu_seqlens is None
            ), "SP not implemented for cu_seqlens (variable length samples in a batch)."
            sp_size = torch.distributed.get_world_size(
                group=self.sequence_parallel_group
            )
            sp_rank = torch.distributed.get_rank(group=self.sequence_parallel_group)

            chunk_size = q_len // 2
            if self.use_state_passing_sp and chunk_size >= 256 and attention_mask is None and mode == "chunk":
                use_state_passing = True
            else:
                # Fallback: gather full sequence (for small chunks or special modes)
                hidden_states = ZigZagGatherScatter.apply(
                    hidden_states,
                    self.sequence_parallel_group,
                    sp_rank,
                    sp_size,
                    sp_size * 2,
                )  # [B, L, Dh]
                q_len = sp_size * q_len

        # Handle variable-length sequences with padding
        if attention_mask is not None:
            indices, cu_seqlens, _ = get_unpad_data(attention_mask[:, -q_len:])
            hidden_states = index_first_axis(
                rearrange(hidden_states, "b s ... -> (b s) ..."), indices
            ).unsqueeze(0)

        # Apply projections and convolutions
        if self.use_short_conv:
            conv_state_q, conv_state_k, conv_state_v = None, None, None
            if last_state is not None:
                conv_state_q, conv_state_k, conv_state_v = last_state["conv_state"]

            q_proj_out = self.q_proj(hidden_states)  # [B, L, Din]
            k_proj_out = self.k_proj(hidden_states)  # [B, L, Dkv]
            v_proj_out = self.v_proj(hidden_states)  # [B, L, Dkv]

            if use_state_passing:
                chunk_len = q_len // 2
                W = self.conv_size

                q_fwd, q_bwd = q_proj_out.split(chunk_len, dim=1)
                k_fwd, k_bwd = k_proj_out.split(chunk_len, dim=1)
                v_fwd, v_bwd = v_proj_out.split(chunk_len, dim=1)

                fwd_bnd = torch.cat([q_fwd[:, -W:], k_fwd[:, -W:], v_fwd[:, -W:]], dim=-1)
                bwd_bnd = torch.cat([q_bwd[:, -W:], k_bwd[:, -W:], v_bwd[:, -W:]], dim=-1)
                packed_bnd = torch.stack([fwd_bnd, bwd_bnd], dim=1)  # [B, 2, W, Dtotal]

                all_bnd = DifferentiableAllGather.apply(
                    packed_bnd, self.sequence_parallel_group
                )  # [SP, B, 2, W, Dtotal]

                Dq = self.query_dim
                Dk = self.key_dim

                fwd_gpu, fwd_type = _get_preceding_gpu(sp_rank, sp_size, sp_rank)

                bwd_pos = 2 * sp_size - 1 - sp_rank
                bwd_gpu, bwd_type = _get_preceding_gpu(sp_rank, sp_size, bwd_pos)
                bwd_prev = all_bnd[bwd_gpu, :, bwd_type]  # [B, W, Dtotal]

                def _conv_with_boundary(conv_fn, chunk, prev_bnd):
                    if prev_bnd is not None:
                        extended = torch.cat([prev_bnd, chunk], dim=1)
                        out, _ = conv_fn(x=extended, cache=None, output_final_state=False)
                        return out[:, W:]
                    else:
                        out, _ = conv_fn(x=chunk, cache=None, output_final_state=False)
                        return out

                if fwd_gpu is not None:
                    fwd_prev = all_bnd[fwd_gpu, :, fwd_type]  # [B, W, Dtotal]
                    fwd_bnd_q = fwd_prev[:, :, :Dq]
                    fwd_bnd_k = fwd_prev[:, :, Dq:Dq+Dk]
                    fwd_bnd_v = fwd_prev[:, :, Dq+Dk:]
                else:
                    fwd_bnd_q, fwd_bnd_k, fwd_bnd_v = None, None, None

                bwd_bnd_q = bwd_prev[:, :, :Dq]
                bwd_bnd_k = bwd_prev[:, :, Dq:Dq+Dk]
                bwd_bnd_v = bwd_prev[:, :, Dq+Dk:]

                q1 = _conv_with_boundary(self.q_conv1d, q_fwd, fwd_bnd_q)
                q2 = _conv_with_boundary(self.q_conv1d, q_bwd, bwd_bnd_q)
                q = torch.cat([q1, q2], dim=1)
                k1 = _conv_with_boundary(self.k_conv1d, k_fwd, fwd_bnd_k)
                k2 = _conv_with_boundary(self.k_conv1d, k_bwd, bwd_bnd_k)
                k = torch.cat([k1, k2], dim=1)
                v1 = _conv_with_boundary(self.v_conv1d, v_fwd, fwd_bnd_v)
                v2 = _conv_with_boundary(self.v_conv1d, v_bwd, bwd_bnd_v)
                v = torch.cat([v1, v2], dim=1)
            else:
                q, conv_state_q = self.q_conv1d(
                    x=q_proj_out,
                    cache=conv_state_q,
                    output_final_state=use_cache,
                    cu_seqlens=cu_seqlens,
                )
                k, conv_state_k = self.k_conv1d(
                    x=k_proj_out,
                    cache=conv_state_k,
                    output_final_state=use_cache,
                    cu_seqlens=cu_seqlens,
                )
                v, conv_state_v = self.v_conv1d(
                    x=v_proj_out,
                    cache=conv_state_v,
                    output_final_state=use_cache,
                    cu_seqlens=cu_seqlens,
                )
        else:
            q = F.silu(self.q_proj(hidden_states))  # [B, L, Din]
            k = F.silu(self.k_proj(hidden_states))  # [B, L, Dkv]
            v = F.silu(self.v_proj(hidden_states))  # [B, L, Dkv]

        # Reshape q to heads and expand k/v (handles projection + residual if configured)
        q = rearrange(q, "... (h d) -> ... h d", h=self.num_q_heads)
        k, v = self._expand_kv(k, v)

        # Apply GQA/head repeats for q to match expanded_num_heads
        if self.num_q_groups > 1:
            q = repeat(q, "... h d -> ... (h g) d", g=self.num_q_groups)

        # Compute beta (gating parameter)
        beta = self.b_proj(hidden_states).sigmoid()  # [B, L, H]

        if self.allow_neg_eigval:
            beta = beta * 2.0  # [B, L, H]

        # Compute g (time-varying decay parameter)
        g = -self.A_log.float().exp() * F.softplus(
            self.gk_proj(hidden_states).float() + self.dt_bias
        )  # [B, L, H]

        # Get recurrent state from cache
        recurrent_state = (
            last_state["recurrent_state"] if last_state is not None else None
        )  # [B, H, D, D] or None

        # Apply gated delta rule
        if use_state_passing:
            o = self._state_passing_forward(q, k, v, g, beta, sp_rank, sp_size)
            recurrent_state = None
        elif mode == "chunk":
            o, recurrent_state = chunk_gated_delta_rule(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                initial_state=recurrent_state,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
                use_qk_l2norm_in_kernel=True,
            )
        elif mode == "fused_recurrent":
            o, recurrent_state = fused_recurrent_gated_delta_rule(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                initial_state=recurrent_state,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
                use_qk_l2norm_in_kernel=True,
            )
        else:
            raise NotImplementedError(f"Not supported mode `{mode}`.")

        # Update cache
        if past_key_values is not None:
            past_key_values.update(
                recurrent_state=recurrent_state,
                conv_state=(conv_state_q, conv_state_k, conv_state_v)
                if self.use_short_conv
                else None,
                layer_idx=self.layer_idx,
                offset=q_len,
            )

        # Apply output gating and normalization
        if self.use_gate:
            g = rearrange(
                self.g_proj(hidden_states), "... (h d) -> ... h d", d=self.head_dim
            )  # [B, L, H, D]
            o = self.o_norm(o, g)  # [B, L, H, D]
        else:
            o = self.o_norm(o)  # [B, L, H, D]

        # Reshape and project output
        o = rearrange(o, "b t h d -> b t (h d)")  # [B, L, Din]
        o = self.o_proj(o)  # [B, L, Dh]

        # Restore padding if attention mask was used
        if attention_mask is not None:
            o = pad_input(o.squeeze(0), indices, batch_size, q_len)  # [B, q_len, Dh]

        # Scatter sequence if using sequence parallel (only for non-state-passing path)
        if self.sequence_parallel_group is not None and not use_state_passing:
            o = ZigZagScatter.apply(
                o, self.sequence_parallel_group, sp_rank, sp_size, sp_size * 2
            )  # [B, L_local, Dh]

        return o, None, past_key_values

    def _state_passing_forward(self, q, k, v, g, beta, sp_rank, sp_size):
        """
        Run delta rule with exact state passing for sequence parallelism.

        Uses the batched 4B approach: runs the kernel once with batch_size=4B
        (base runs + identity runs) to compute o_base, state_base, C, and M.
        Then uses matrix chain to propagate states and correct outputs.

        Args:
            q: [B, L_local, H, D] query tensor (after GQA expand)
            k: [B, L_local, H, D] key tensor (after GQA expand)
            v: [B, L_local, H, D] value tensor (after GQA expand)
            g: [B, L_local, H] log-space decay (float32)
            beta: [B, L_local, H] beta gating parameter
            sp_rank: This GPU's rank in the SP group
            sp_size: Number of GPUs in the SP group

        Returns:
            o: [B, L_local, H, D] output tensor
        """
        B, L, H, D = q.shape[0], q.shape[1], q.shape[2], q.shape[3]
        chunk_len = L // 2
        B2 = 2 * B

        q_batch = q.reshape(B2, chunk_len, H, D)
        k_batch = k.reshape(B2, chunk_len, H, D)
        v_batch = v.reshape(B2, chunk_len, H, D)
        g_batch = g.reshape(B2, chunk_len, H)
        beta_batch = beta.reshape(B2, chunk_len, H)

        o_base, state_base, C, M_batch = self._batched_4B_compute(
            q_batch, k_batch, v_batch, g_batch, beta_batch, B, B2, chunk_len, H, D
        )

        packed_all = torch.stack([
            state_base[:B].float(), state_base[B:].float(),
            M_batch[:B].float(), M_batch[B:].float(),
        ], dim=1)  # [B, 4, H, D, D]

        all_packed = DifferentiableAllGather.apply(
            packed_all, self.sequence_parallel_group
        )  # [SP, B, 4, H, D, D]

        recv_state1, recv_state2 = _compute_received_states(
            all_packed[:, :, 0], all_packed[:, :, 1],
            all_packed[:, :, 2], all_packed[:, :, 3],
            sp_rank, sp_size,
        )

        if recv_state1 is None:
            recv_state1 = torch.zeros(B, H, D, D, device=q.device, dtype=q.dtype)
        recv_state = torch.cat([recv_state1, recv_state2], dim=0)  # [2B, H, D, D]
        recv_state = recv_state.to(C.dtype)

        o_corr = torch.einsum('bthi,bhij->bthj', C, recv_state)

        return (o_base + o_corr).reshape(B, L, H, D)

    def _batched_4B_compute(self, q_batch, k_batch, v_batch, g_batch, beta_batch, B, B2, chunk_len, H, D):
        """Single 4B kernel: computes base and identity runs in one call."""
        B4 = 4 * B
        q_4B = torch.cat([q_batch, q_batch], dim=0)
        k_4B = torch.cat([k_batch, k_batch], dim=0)
        v_4B = torch.cat([v_batch, v_batch], dim=0)
        g_4B = torch.cat([g_batch, g_batch], dim=0)
        beta_4B = torch.cat([beta_batch, beta_batch], dim=0)

        init_4B = torch.zeros(B4, H, D, D, device=q_batch.device, dtype=q_batch.dtype)
        init_4B[B2:] = self._eye_cache.to(dtype=q_batch.dtype)

        o_4B, state_4B = chunk_gated_delta_rule(
            q=q_4B, k=k_4B, v=v_4B, g=g_4B, beta=beta_4B,
            initial_state=init_4B, output_final_state=True,
            use_qk_l2norm_in_kernel=True,
        )

        o_base = o_4B[:B2]
        state_base = state_4B[:B2]
        C = o_4B[B2:] - o_base
        M_batch = state_4B[B2:].float() - state_base.float()

        return o_base, state_base, C, M_batch
