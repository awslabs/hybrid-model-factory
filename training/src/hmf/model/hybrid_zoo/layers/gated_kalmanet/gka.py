"""
Gated KalmaNet (GKA) layer implementation with Grouped Query Attention (GQA) support.

This module implements the Gated KalmaNet layer from the paper:
"Gated KalmaNet: A Fading Memory Layer Through Test-Time Ridge Regression" (https://arxiv.org/abs/2511.21016).

Overview:
    GKA is a state space model (SSM) that updates its state based on the entire history
    of previously seen tokens, rather than just the current token. At each timestep t,
    the optimal state S_t minimizes a ridge regression objective over all past tokens:

        S_t = argmin_S { λ·||S||²_F + Σ(i=1 to t) η_i·||S·k_i - v_i||² }

    where λ is the ridge regularization strength and η_i are exponentially decaying weights
    controlled by forgetting gates. This has a closed-form solution:

        S_t = U_t · (H_t + λI)^(-1)

    where:
        U_t = Σ(i=1 to t) η_i v_i k_i^T    (cumulative value-key outer products)
        H_t = Σ(i=1 to t) η_i k_i k_i^T    (cumulative key-key covariance)

    The output is computed as y_t = S_t · q_t, which reduces to solving the ridge
    regression problem (H_t + λI)·x_t = q_t for x_t, then computing y_t = U_t · x_t.
    This is solved numerically via the Chebyshev iteration algorithm.

    The layer supports GQA-style head grouping where the number of key/value heads
    can be different from the number of query heads. The hyperparameters in this module
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
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Optional, Tuple

import torch
import torch.nn as nn
from einops import rearrange, repeat
from torch import Tensor
from torch.nn import functional as F

from fla.modules import FusedRMSNormGated, RMSNorm, ShortConvolution
from fla.ops.simple_gla import chunk_simple_gla

from ..sp_p2p_utils import reorder_for_ssm_p2p
from ..low_rank_linear import LinearLowRank
from .ops.chebyshev.chunk_dk_backward_gating_sp import compute_kk_final_with_gating
from .sequence_parallel.gka_sp_utils import (
    state_passing_gka_p2p,
    recompute_with_initial_states,
)

from .ops.chebyshev.gka_chebyshev_solve import (
    gka_chebyshev_gla,
    torch_decoding_one_step,
)

if TYPE_CHECKING:
    from hmf.model.hybrid_zoo.models.cache import GatedKalmaNetCache


class GatedKalmaNet(nn.Module):
    """
    GKA layer with Chebyshev solver and GQA support.

    Args:
        hidden_size: The hidden size of the input. Defaults to 2048.
        head_dim: The dimension of each head. Assumes q/k/v have the same head dimension.
            Defaults to 128.
        num_q_heads: The number of query heads. Defaults to 16.
        num_k_heads: The number of key heads for GQA. If None, defaults to num_q_heads.
            Defaults to None.
        num_v_heads: The number of value heads for GQA. If None, defaults to num_q_heads.
            Defaults to None.
        use_beta_gate: Whether to use beta gating mechanism. Defaults to True.
        use_alpha_connection: Whether to use alpha connection for residual paths. Defaults to True.
        use_v_conv: Whether to apply convolution to value vectors. Defaults to True.
        use_forgetting_gate: Whether to use forgetting gate mechanism. Defaults to True.
        use_forgetting_gate_kk: Whether to use forgetting gate for key-key interactions. Defaults to True.
        gla_rescale: Whether to apply GLA-style rescaling. Defaults to True.
        solver_type: Solver method. Currently, we only support "chebyshev". Defaults to "chebyshev".
        bp_lambda: Whether to backpropagate through lambda parameters. Defaults to True.
        num_iter: Number of iterations for iterative solvers. Defaults to 30.
        ridge_strength: Regularization strength for ridge regression. Defaults to 0.02.
        use_gate: Whether to use output gating mechanism. Defaults to True.
        conv_size: Size of the convolution kernel. Defaults to 4.
        layer_idx: The index of the layer. Used for caching during inference. Defaults to None.
        norm_eps: The epsilon value for the normalization layer. Defaults to 1e-6.
        chunk_size: Size of chunks for parallel processing in Triton kernels. Defaults to 64.
        kv_proj_rank: Rank for low-rank KV projection to expand K/V heads. If None, uses
            standard GQA repeat. Defaults to None.
        kv_learnable_residual: Whether to use a learnable residual for the KV head expansion
            instead of a fixed repeat. Only used when kv_proj_rank is set. Defaults to False.
    """

    def __init__(
        self,
        hidden_size: int = 2048,
        head_dim: int = 128,
        num_q_heads: int = 16,
        num_k_heads: Optional[int] = None,
        num_v_heads: Optional[int] = None,
        use_beta_gate: bool = True,
        use_alpha_connection: bool = True,
        use_v_conv: bool = True,
        use_forgetting_gate: bool = True,
        use_forgetting_gate_kk: bool = True,
        gla_rescale: bool = True,
        solver_type: str = "chebyshev",
        bp_lambda: bool = True,
        num_iter: int = 30,
        ridge_strength: float = 0.02,
        use_gate: bool = True,
        conv_size: int = 4,
        layer_idx: Optional[int] = None,
        norm_eps: float = 1e-6,
        chunk_size: int = 64,
        kv_proj_rank: Optional[int] = None,
        kv_learnable_residual: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()

        assert solver_type == "chebyshev", f"Received {solver_type=}. Currently, GKA only supports 'chebyshev' solver."

        self.hidden_size = hidden_size

        self.use_gate = use_gate
        self.conv_size = conv_size
        self.sequence_parallel_group = None
        self.chunk_size = chunk_size

        self.head_dim = head_dim
        self.num_q_heads = num_q_heads
        self.num_k_heads = num_k_heads if num_k_heads is not None else num_q_heads
        self.num_v_heads = num_v_heads if num_v_heads is not None else num_q_heads
        self.expanded_num_heads = max(
            self.num_q_heads, self.num_k_heads, self.num_v_heads
        )
        self.num_q_groups = self.expanded_num_heads // self.num_q_heads
        self.num_k_groups = self.expanded_num_heads // self.num_k_heads
        self.num_v_groups = self.expanded_num_heads // self.num_v_heads

        self.query_dim = self.num_q_heads * self.head_dim
        self.key_dim = self.num_k_heads * self.head_dim
        self.value_dim = self.num_v_heads * self.head_dim
        self.expanded_dim = self.expanded_num_heads * self.head_dim

        self.layer_idx = layer_idx

        self.ridge_strength = ridge_strength
        self.num_iter = num_iter
        self.use_beta_gate = use_beta_gate
        self.use_alpha_connection = use_alpha_connection
        self.solver_type = solver_type
        self.bp_lambda = bp_lambda
        self.use_v_conv = use_v_conv
        self.use_forgetting_gate = use_forgetting_gate
        self.use_forgetting_gate_kk = use_forgetting_gate_kk

        if gla_rescale:
            self.gla_scale = self.head_dim ** -0.5
        else:
            self.gla_scale = 1.0

        # Projection layers
        self.q_proj = nn.Linear(hidden_size, self.query_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)

        self.a_proj = nn.Linear(hidden_size, self.expanded_num_heads, bias=False)

        self.b_proj = (
            nn.Linear(hidden_size, self.expanded_num_heads, bias=True)
            if self.use_beta_gate
            else None
        )

        if self.use_alpha_connection:
            self.alpha_proj = nn.Linear(hidden_size, self.expanded_num_heads, bias=True)

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
        self.conv_size = conv_size
        self.q_conv1d = ShortConvolution(
            hidden_size=self.query_dim, kernel_size=conv_size, activation="silu"
        )
        self.k_conv1d = ShortConvolution(
            hidden_size=self.key_dim, kernel_size=conv_size, activation="silu"
        )
        if self.use_v_conv:
            self.v_conv1d = ShortConvolution(
                hidden_size=self.value_dim, kernel_size=conv_size, activation="silu"
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
        past_key_values: Optional["GatedKalmaNetCache"] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[Tensor, Optional[Tensor], Optional["GatedKalmaNetCache"]]:
        """
        GKA forward pass with optional caching.

        NOTE: We use the following variables for tensor shape hints:
            - B: Input batch size
            - L: Sequence length (full sequence length before SP distribution)
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
            - C: Convolution size (self.conv_size)
            - SP: Sequence parallel size. 1 if not using sequence parallel

            - B_sp: Batch size after SP reordering. B_sp = B*2 if self.sequence_parallel_group else B_sp = B
            - L_sp: Sequence length after SP reordering. L_sp = L/(SP*2) + C-1 if self.sequence_parallel_group else L_sp = L/SP
            - Lp: Sequence length after SP reordering and conv padding removal. Lp = L/(SP*2) if self.sequence_parallel_group else Lp = L/SP

        Args:
            hidden_states: Input tensor with shape [B, L/SP, Dh].
            attention_mask: Optional 0-1 mask. Not used.
            past_key_values: Optional cache containing conv_state and recurrent_state
                for inference. When provided, performs incremental decoding.
            use_cache: Whether to return updated cache states. Defaults to False.
            output_attentions: Whether to return attention weights (not used). Defaults to False.
            **kwargs: Additional keyword arguments, may include cu_seqlens for packed sequences.

        Returns:
            Tuple of (output, attention_weights, past_key_values) where:
                - output: Output tensor with same shape as hidden_states [B, L, Dh]
                - attention_weights: None (not computed for GKA)
                - past_key_values: Updated cache (same as input if provided, else None)
        """
        # During inference, if attention_mask is not None, throw an error
        if attention_mask is not None and not self.training:
            raise NotImplementedError(
                "GKA kernels currently do not support varlen inputs."
            )

        batch_size, q_len, _ = hidden_states.shape

        last_state = None
        if past_key_values is not None:
            last_state = past_key_values[self.layer_idx]

        cu_seqlens = kwargs.get("cu_seqlens", None)

        # Apply P2P reordering to hidden_states first for custom zigzag SP
        if self.sequence_parallel_group is not None:
            assert (
                cu_seqlens is None
            ), "SP not implemented for cu_seqlens (variable length samples in a batch)."
            sp_size = torch.distributed.get_world_size(
                group=self.sequence_parallel_group
            )
            sp_rank = torch.distributed.get_rank(group=self.sequence_parallel_group)

            hidden_states = reorder_for_ssm_p2p(
                hidden_states,
                self.sequence_parallel_group,
                torch.cuda.Stream(),
                sp_size,
                sp_rank,
                self.conv_size,
            )
            # hidden_states: [B_sp, L_sp, Dh]

        # Apply projections and convolutions
        conv_state_q, conv_state_k, conv_state_v = None, None, None
        if last_state is not None:
            conv_state_q, conv_state_k, conv_state_v = last_state["conv_state"]

        q, conv_state_q = self.q_conv1d(
            x=self.q_proj(hidden_states),
            cache=conv_state_q,
            output_final_state=use_cache,
            cu_seqlens=cu_seqlens,
        )
        # q: [B_sp, L_sp, Dq]
        # conv_state_q: [B, Dq, C] or None

        k, conv_state_k = self.k_conv1d(
            x=self.k_proj(hidden_states),
            cache=conv_state_k,
            output_final_state=use_cache,
            cu_seqlens=cu_seqlens,
        )
        # k: [B_sp, L_sp, Dk]
        # conv_state_k: [B, Dk, C] or None

        if self.use_v_conv:
            v, conv_state_v = self.v_conv1d(
                x=self.v_proj(hidden_states),
                cache=conv_state_v,
                output_final_state=use_cache,
                cu_seqlens=cu_seqlens,
            )
            # v: [B_sp, L_sp, Dv]
            # conv_state_v: [B, Dv, C] or None
        else:
            v = self.v_proj(hidden_states)  # [B_sp, L_sp, Dv]
            conv_state_v = None

        # Remove padding for custom zigzag SP (first conv_size-1 tokens added by P2P)
        if self.sequence_parallel_group is not None:
            q = q[:, self.conv_size - 1 :, :].contiguous()  # [B_sp, Lp, Dq]
            k = k[:, self.conv_size - 1 :, :].contiguous()  # [B_sp, Lp, Dk]
            if self.use_v_conv:
                v = v[:, self.conv_size - 1 :, :].contiguous()  # [B_sp, Lp, Dv]
            # Also remove padding from hidden_states so beta/alpha/g/gk match seq length
            hidden_states = hidden_states[
                :, self.conv_size - 1 :, :
            ].contiguous()  # [B_sp, Lp, Dh]

        # Reshape q to heads and expand k/v (handles projection + residual if configured)
        q = rearrange(
            q, "... (h d) -> ... h d", h=self.num_q_heads
        )  # [B_sp, Lp, Hq, D]
        k, v = self._expand_kv(k, v)  # [B_sp, Lp, H, D] each

        # Apply GQA/head repeats for q to match expanded_num_heads
        if self.num_q_groups > 1:
            q = repeat(q, "... h d -> ... (h g) d", g=self.num_q_groups)
        # q: [B_sp, Lp, H, D]

        # Get recurrent state from cache
        last_h_kk, last_h_kv = (
            last_state["recurrent_state"] if last_state is not None else (None, None)
        )

        # Normalize queries and keys
        qk_norm_eps = 1e-6
        q = q / (
            torch.linalg.vector_norm(q, dim=-1, keepdim=True).to(q) + qk_norm_eps
        )  # [B_sp, Lp, H, D]
        k = k / (
            torch.linalg.vector_norm(k, dim=-1, keepdim=True).to(k) + qk_norm_eps
        )  # [B_sp, Lp, H, D]

        # Apply beta gating if enabled
        if self.use_beta_gate:
            beta = self.b_proj(hidden_states).sigmoid()  # [B_sp, Lp, H]
            eps = 1e-6
            k = ((beta[..., None] + eps) * k).to(k.dtype)  # [B_sp, Lp, H, D]
            v = ((beta[..., None] + eps) * v).to(v.dtype)  # [B_sp, Lp, H, D]

        # Compute alpha connection parameter
        if self.use_alpha_connection:
            alpha = self.alpha_proj(hidden_states).sigmoid()  # [B_sp, Lp, H]
        else:
            alpha = torch.ones_like(q[..., 0])  # [B_sp, Lp, H]

        # Compute forgetting gate
        if self.use_forgetting_gate:
            g = -self.A_log.float().exp() * F.softplus(
                self.a_proj(hidden_states).float() + self.dt_bias
            )  # [B_sp, Lp, H]
        else:
            g = None

        gk = g if self.use_forgetting_gate_kk else None

        # Apply GKA Chebyshev solver
        if last_state is None:
            if self.sequence_parallel_group is not None:
                # Custom zigzag SP: Compute h_kk first, pass states, then compute rest

                # 1. Compute h_kk locally (just k @ k^T cumsum) with zero initial state
                k_proj = k  # [B_sp, Lp, H, D]

                if gk is not None:
                    # For state passing, we only need the sum (equivalent to cumsum[-1])
                    gkk_final = torch.sum(gk, dim=1)  # [B_sp, H]
                else:
                    gkk_final = torch.zeros(
                        q.shape[0],
                        self.expanded_num_heads,
                        device=q.device,
                        dtype=q.dtype,
                    )

                # Extract g sum (total decay across chunk)
                if g is not None:
                    g_cumsum_last = torch.sum(g, dim=1)  # [B_sp, H]
                else:
                    g_cumsum_last = None

                # Compute h_kk: cumsum of k @ k^T with gating
                with torch.cuda.device(q.device):
                    if self.num_iter > 0:
                        kk_final = compute_kk_final_with_gating(
                            k_proj, gk, chunk_size=self.chunk_size
                        )
                    else:
                        kk_final = torch.zeros(
                            k_proj.shape[0],
                            self.expanded_num_heads,
                            k_proj.shape[3],
                            k_proj.shape[3],
                            device=q.device,
                            dtype=q.dtype,
                        )
                    # kk_final: [B_sp, H, D, D]

                    # Compute h_kv locally to get final state
                    _, h_kv_final = chunk_simple_gla(
                        q=q,
                        k=k,
                        v=v,
                        scale=self.gla_scale,
                        g=g,
                        output_final_state=True,
                    )
                    # h_kv_final: [B_sp, H, D, D]

                # 2. Pass states via P2P in zigzag pattern
                bs = q.shape[0]  # Batch size (B_sp, includes both chunks per GPU)

                # Ensure all inputs to state_passing are on the same device
                with torch.cuda.device(q.device):
                    prev_kk, prev_h_kv = state_passing_gka_p2p(
                        kk_final.to(q.device),
                        gkk_final.to(q.device),  # Local only, used for decay
                        h_kv_final.to(q.device),
                        g_cumsum_last.to(q.device)
                        if g_cumsum_last is not None
                        else None,
                        self.sequence_parallel_group,
                        sp_rank,
                        sp_size,
                        bs,
                    )
                    # prev_kk: [B_sp, H, D, D]
                    # prev_h_kv: [B_sp, H, D, D]

                # 3. Now recompute with initial states from previous chunk
                # Ensure all tensors are on the same device before calling recompute
                with torch.cuda.device(q.device):
                    o, h_kk, h_kv = recompute_with_initial_states(
                        q=q,
                        k=k,
                        v=v,
                        alpha=alpha,
                        g=g,
                        gk=gk,
                        prev_kk=prev_kk.to(q.device),
                        prev_h_kv=prev_h_kv.to(q.device),
                        gla_scale=self.gla_scale,
                        ridge_strength=self.ridge_strength,
                        solver_type=self.solver_type,
                        num_iter=self.num_iter,
                        bp_lambda=self.bp_lambda,
                        chunk_size=self.chunk_size,
                    )
                    # o: [B_sp, Lp, H, D]
                    # h_kk: [B_sp, H, D, D]
                    # h_kv: [B_sp, H, D, D]
            else:
                with torch.cuda.device(q.device):
                    o, h_kk, h_kv = gka_chebyshev_gla(
                        q=q,
                        k=k,
                        v=v,
                        alpha=alpha,
                        g=g,
                        gk=gk,
                        gla_scale=self.gla_scale,
                        ridge_strength=self.ridge_strength,
                        solver_type=self.solver_type,
                        num_iter=self.num_iter,
                        bp_lambda=self.bp_lambda,
                        chunk_size=self.chunk_size,
                    )
                    # o: [B_sp, Lp, H, D]
                    # h_kk: [B_sp, H, D, D]
                    # h_kv: [B_sp, H, D, D]
        else:
            with torch.cuda.device(q.device):
                o, h_kk, h_kv = torch_decoding_one_step(
                    q=q.squeeze(0),
                    k=k.squeeze(0),
                    v=v.squeeze(0),
                    alpha=alpha.squeeze(0),
                    g=g,
                    gk=gk,
                    gla_scale=self.gla_scale,
                    ridge_strength=self.ridge_strength,
                    solver_type=self.solver_type,
                    num_iter=self.num_iter,
                    prev_h_kk=last_h_kk,
                    prev_h_kv=last_h_kv,
                )
                # o: [Lp, H, D] (batch squeezed)
                # h_kk: [H, D, D]
                # h_kv: [H, D, D]
            if len(o.shape) == 3:
                o = o.unsqueeze(0)  # [1, Lp, H, D]

            o = o.to(q)

        # Update cache
        if past_key_values is not None:
            past_key_values.update(
                recurrent_state=(h_kk, h_kv),
                conv_state=(conv_state_q, conv_state_k, conv_state_v),
                layer_idx=self.layer_idx,
                offset=q_len,
            )

        # Apply output gating and normalization
        if self.use_gate:
            g = rearrange(
                self.g_proj(hidden_states), "... (h d) -> ... h d", d=self.head_dim
            )  # [B_sp, Lp, H, D]
            o = self.o_norm(o, g)  # [B_sp, Lp, H, D]
        else:
            o = self.o_norm(o)  # [B_sp, Lp, H, D]

        # Reshape and project output
        o = rearrange(o, "b t h d -> b t (h d)")  # [B_sp, Lp, Dexp]
        o = self.o_proj(o)  # [B_sp, Lp, Dh]

        # Scatter sequence if using sequence parallel
        if self.sequence_parallel_group is not None:
            # Recombine zigzag chunks to contiguous
            bz, len_o, dim = o.shape
            if bz == 2:
                o = o.view(bz // 2, len_o * 2, dim)
            else:
                o = torch.cat([o[: bz // 2], o[bz // 2 :]], dim=1).contiguous()
            # o: [B, L/SP, Dh]

        return o, None, past_key_values
