# Copyright (c) 2024, Tri Dao, Albert Gu.

"""
Mamba2 layer implementation with Grouped Query Attention (GQA) support.

This module implements a custom Mamba2 layer that maps attention-style QKV projections to
Mamba2's xBC formulation, enabling GQA-style head grouping.

Overview:
    In standard Mamba2, B and C have shape [batch, seqlen, ngroups, d_state], while x
    has shape [batch, seqlen, nheads, headdim]. This implementation bridges attention
    and SSM paradigms by mapping:
        - x <-> V (value)
        - B <-> K (key)
        - C <-> Q (query)

    The head dimension remains constant across QKV, but the number of heads in Q and KV
    can vary (as in GQA). To accommodate this:

    1. We require headdim == d_state
    2. ngroups is set equal to the number of Q heads
    3. x and B initially share shape [batch, seqlen, nheads_kv, headdim] but are then
        repeated along the head dimension to achieve
        [batch, seqlen, ngroups, headdim] == [batch, seqlen, ngroups, d_state]

Key Variables:
    - ngroups: Number of attention head groups (equals number of Q heads)
    - nheads: Number of heads (equals ngroups in this implementation)
    - headdim: Head dimension (set equal to d_state)
    - d_state: State dimension in Mamba2 SSM
    - d_xb: Key/value dimension, computed as nheads_kv * headdim
    - repeat_group: Repetition factor for GQA, computed as nheads // nheads_kv

Extended from: https://github.com/jxiw/MambaInLlama/blob/main/mamba2/hybrid_mamba_layer.py
"""

import math
from typing import Any, Callable, Optional, Tuple, TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import Tensor

try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = None, None

try:
    from causal_conv1d.causal_conv1d_varlen import causal_conv1d_varlen_states
except ImportError:
    causal_conv1d_varlen_states = None

try:
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
except ImportError:
    selective_state_update = None


import deepspeed
import torch.nn.init as init
from huggingface_hub import PyTorchModelHubMixin
from mamba_ssm.ops.triton.layernorm_gated import RMSNorm as RMSNormGated
from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined

if TYPE_CHECKING:
    from hmf.model.hybrid_zoo.models.cache import Mamba2Cache

from ..sp_p2p_utils import reorder_for_ssm_p2p
from .sequence_parallel.mamba2.mamba_mixer_cp_utils import state_passing_p2p


def repeat_kv(hidden_states: Tensor, n_rep: int) -> Tensor:
    """
    Repeats the heads in hidden_states n_rep times. Equivalent to
    torch.repeat_interleave(hidden_states, dim=1, repeats=n_rep).

    Args:
        hidden_states: Tensor with shape [B, Hkv, L, D].
        n_rep: Number of times to repeat the hidden_states tensor along the Hkv dimension.

    Returns:
        A tensor with shape [B, Hkv * n_rep, L, D]. For example, if hidden_states has heads
        [H1, H2, H3, H4] and n_rep = 2, the new hidden_states will have 8 heads:
        [H1, H1, H2, H2, H3, H3, H4, H4].
    """
    if n_rep == 1:
        return hidden_states
    batch, nheads_kv, seqlen, headdim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, nheads_kv, n_rep, seqlen, headdim
    )
    return hidden_states.reshape(batch, nheads_kv * n_rep, seqlen, headdim)


class Mamba2(nn.Module, PyTorchModelHubMixin):
    def __init__(
        self,
        d_model: int,
        d_xb: int,
        d_inner: int,
        d_state: int = 128,
        d_conv: int = 4,
        expand: int = 1,
        conv_init: Optional[float] = None,
        ngroups: int = 1,
        A_init_range: Tuple[float, float] = (0, 16),
        D_has_hdim: bool = False,
        rmsnorm: bool = True,
        norm_before_gate: bool = False,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init_floor: float = 1e-4,
        dt_limit: Tuple[float, float] = (0.0, float("inf")),
        bias: bool = False,
        conv_bias: bool = True,
        # Fused kernel and sharding options
        chunk_size: int = 256,
        use_mem_eff_path: bool = True,
        layer_idx: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        apply_rotary_pos_emb_fn: Optional[
            Callable[[Tensor, Tensor, Tensor, Tensor], Tuple[Tensor, Tensor]]
        ] = None,
        use_pos_emb: bool = False,
        headdim: int = 64,  # Set to d_state
        d_ssm: Optional[int] = None,
        use_qk_norm: bool = True,
        qk_norm: Optional[nn.Module] = None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        assert expand == 1, "We currently only support Mamba2 with expand == 1."
        self.expand = expand

        self.d_inner = (
            d_inner if d_inner is not None else (self.expand * self.d_model)
        )  # Equivalent to (number of heads) * headdim
        self.d_model = d_model  # Equivalent to hidden size
        self.d_state = d_state  # Equivalent to head dimension
        self.ngroups = ngroups  # Equivalent to number of Attention heads
        self.headdim = self.d_state  # We set head dimension = state dimension
        self.nheads = (
            self.ngroups
        )  # We set nheads to number of Attention heads (ngroups)
        self.d_xb = d_xb  # Equivalent to (number of kv heads) * headdim
        self.num_kv_groups = (
            self.d_inner // self.d_xb
        )  # (number of heads) // (number of kv heads)
        self.d_ssm = self.d_inner if d_ssm is None else d_ssm

        self.d_conv = d_conv
        self.conv_init = conv_init
        self.D_has_hdim = D_has_hdim
        self.rmsnorm = rmsnorm
        self.norm_before_gate = norm_before_gate
        self.dt_limit = dt_limit
        self.activation = "silu"
        self.chunk_size = chunk_size
        self.use_mem_eff_path = use_mem_eff_path
        self.layer_idx = layer_idx

        self.sequence_parallel_group = None

        assert self.d_ssm % self.headdim == 0
        assert self.d_inner == self.ngroups * self.d_state
        assert self.d_inner == self.d_ssm

        # Order: [z, x, B, C, dt]
        d_in_proj = self.d_inner + self.d_xb + self.d_xb + self.d_inner + self.nheads
        self.in_proj = nn.Linear(self.d_model, d_in_proj, bias=bias, **factory_kwargs)

        conv_dim = self.d_inner + self.d_xb + self.d_xb
        self.conv1d = nn.Conv1d(
            in_channels=conv_dim,
            out_channels=conv_dim,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=conv_dim,
            padding=d_conv - 1,
            **factory_kwargs,
        )

        if self.conv_init is not None:
            nn.init.uniform_(self.conv1d.weight, -self.conv_init, self.conv_init)

        self.act = nn.SiLU()

        # Initialize log dt bias
        dt = torch.exp(
            torch.rand(self.nheads, **factory_kwargs)
            * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        dt = torch.clamp(dt, min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)

        assert A_init_range[0] >= 0 and A_init_range[1] >= A_init_range[0]
        A = torch.empty(self.nheads, dtype=torch.float32, device=device).uniform_(
            *A_init_range
        )
        A_log = torch.log(A).to(dtype=dtype)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        # D "skip" parameter
        self.D = nn.Parameter(
            torch.ones(self.d_ssm if self.D_has_hdim else self.nheads, device=device)
        )
        self.D._no_weight_decay = True

        if self.rmsnorm:
            assert RMSNormGated is not None
            self.norm = RMSNormGated(
                self.d_ssm,
                eps=1e-5,
                norm_before_gate=self.norm_before_gate,
                group_size=self.d_ssm // ngroups,
                **factory_kwargs,
            )

        self.out_proj = nn.Linear(
            self.d_inner, self.d_model, bias=bias, **factory_kwargs
        )

        # (Optional) positional embedding
        if use_pos_emb:
            assert apply_rotary_pos_emb_fn is not None
        self.apply_rotary_pos_emb_fn = apply_rotary_pos_emb_fn
        self.use_pos_emb = use_pos_emb

        # (Optional) CB norm (similar to QK norm in Attention)
        self.C_norm = None
        self.B_norm = None
        if use_qk_norm and qk_norm is not None:
            self.C_norm = qk_norm(self.headdim, 1e-6)  # Q norm equivalent
            self.B_norm = qk_norm(self.headdim, 1e-6)  # K norm equivalent

        # Added this due to issues with Conv1d when using deepsped stage 3.
        for param in self.parameters():
            deepspeed.zero.register_external_parameter(self, param)

    def _apply_pos_embeddings(
        self, xBC: Tensor, position_embeddings: Optional[Tuple[Tensor, Tensor]]
    ) -> Tensor:
        """
        Applies rotary positional embeddings to B and C (analogous to K and Q, respectively, in Attention)

        Args:
            xBC: Concatenated tensor of x, B, and C.
                Training shape: [B, L, Dxbc].
                Decoding shape: [B, Dxbc].
            position_embeddings: A tuple of (cos, sin) RoPE tensors, each with shape [B, L, D].

        Returns:
            A modified version of xBC with RoPE applied to B and C.
        """

        if self.use_pos_emb and position_embeddings is not None:
            # Handle 2D input (inference) by adding sequence dimension
            is_2d = xBC.ndim == 2
            if is_2d:
                xBC = xBC.unsqueeze(1)  # [B, Dxbc] -> [B, 1, Dxbc]

            x, B, C = torch.split(
                xBC, [self.d_xb, self.d_xb, self.ngroups * self.d_state], dim=-1
            )

            if self.sequence_parallel_group is not None:
                sp_size = torch.distributed.get_world_size(
                    group=self.sequence_parallel_group
                )
                sp_rank = torch.distributed.get_rank(group=self.sequence_parallel_group)

                position_embeddings_0 = reorder_for_ssm_p2p(
                    position_embeddings[0],
                    self.sequence_parallel_group,
                    torch.cuda.Stream(),
                    sp_size,
                    sp_rank,
                    self.d_conv,
                )
                position_embeddings_1 = reorder_for_ssm_p2p(
                    position_embeddings[1],
                    self.sequence_parallel_group,
                    torch.cuda.Stream(),
                    sp_size,
                    sp_rank,
                    self.d_conv,
                )
                cos, sin = position_embeddings_0, position_embeddings_1
            else:
                cos, sin = position_embeddings
            # Q -> C, K -> B
            # C shape: [B, L, H * D]
            # B shape: [B, L, Hkv * D]
            # B, C dimensions need to be in [B, H, L, D] format to add pos emb
            C = rearrange(C, "b l (h d) -> b h l d", d=self.headdim)
            B = rearrange(B, "b l (hkv d) -> b hkv l d", d=self.headdim)

            C, B = self.apply_rotary_pos_emb_fn(C, B, cos, sin)

            C = rearrange(C, "b h l d -> b l (h d)", d=self.d_state)
            B = rearrange(B, "b hkv l d -> b l (hkv d)", d=self.headdim)

            xBC = torch.cat([x, B, C], dim=-1)

            # Remove sequence dimension if input was 2D
            if is_2d:
                xBC = xBC.squeeze(1)  # [B, 1, Dxbc] -> [B, Dxbc]

        return xBC

    def _apply_qk_norm(self, xBC: Tensor) -> Tensor:
        """
        Applies norm to B and C when self.B_norm and self.C_norm are not None. Intended to be used
        for Qwen3-style models which replace biases in the Q/K projections with norms.

        Args:
            xBC: Concatenated tensor of x, B, and C.
                Training shape: [B, L, Dxbc].
                Decoding shape: [B, Dxbc].

        Returns:
            xBC: The concatenated x, B, and C tensors after passing B and C through their normalization
                layers.
        """
        if self.B_norm is None or self.C_norm is None:
            return xBC

        is_2d = xBC.ndim == 2

        if is_2d:
            xBC = xBC.unsqueeze(1)  # [B, Dxbc] -> [B, 1, Dxbc]

        def apply_norm(x: Tensor, norm: Any):
            x = rearrange(x, "b l (h d) -> b l h d", d=self.headdim)
            x = norm(x)
            x = rearrange(x, "b l h d -> b l (h d)", d=self.headdim)
            return x

        x_, B, C = torch.split(
            xBC, [self.d_xb, self.d_xb, self.ngroups * self.d_state], dim=-1
        )

        B = apply_norm(B, norm=self.B_norm)
        C = apply_norm(C, norm=self.C_norm)

        xBC = torch.cat([x_, B, C], dim=-1)

        if is_2d:
            xBC = xBC.squeeze(1)  # [B, 1, Dxbc] -> [B, Dxbc]

        return xBC

    def forward(
        self,
        hidden_states: Tensor,
        position_embeddings: Optional[Tuple[Tensor, Tensor]] = None,
        seqlen: Optional[int] = None,
        seq_idx: Optional[Tensor] = None,
        cu_seqlens: Optional[Tensor] = None,
        past_key_values: Optional["Mamba2Cache"] = None,
        **kwargs,
    ) -> Tuple[Tensor, None, "Mamba2Cache"]:
        """
        Mamba2 forward pass with optional caching.

        NOTE: We use the following variables for tensor shape hints:
            - B: Input batch size
            - L: Input sequence length
            - H: Number of heads (self.nheads)
            - Hkv: Number of KV heads (self.nheads // self.num_kv_groups)
            - D: Head dimension (self.headdim)
            - N: State dimension (self.d_state). Equivalent to D in this implementation
            - Din: Inner dimension, H * D (self.d_inner)
            - Dxb: Key/value dimension, Hkv * D (self.d_xb)
            - Dxbc: Combined xBC dimension, 2*Dxb + Din (self.d_xb * 2 + self.d_inner)
            - Dmlp: MLP dimension (computed dynamically, may be 0)
            - Dm: Model dimension (self.d_model)

            - B_sp: Batch size after SP reordering. B_sp = B*2 if self.sequence_parallel_group else B_sp = B
            - L_sp: Sequence length after SP reordering. L_sp = L//2 + d_conv - 1 if self.sequence_parallel_group else L_sp = L
            - Lp: Sequence length after SP reordering and conv padding removal. Lp = L//2 if self.sequence_parallel_group else Lp = L

        Args:
            hidden_states: Input tensor with shape [B, L, Dm]. When SP is enabled, L represents the
                training sequence length divided by the SP size.
            position_embeddings: Optional tuple of (cos, sin) RoPE tensors, each with shape [B, L, D].
            seqlen: Optional sequence length. When provided, hidden_states is expected in flattened
                format [B * L, Dm] for sequence parallel processing.
            seq_idx: Optional tensor of sequence indices for variable-length (packed) sequences.
                Used with causal_conv1d for efficient varlen processing.
            cu_seqlens: Optional tensor of cumulative sequence lengths for packed sequences.
                Shape [num_sequences + 1,] where cu_seqlens[i] is the start position of sequence i.
            past_key_values: Optional Mamba2Cache containing conv_state and ssm_state for inference.
                When provided with seqlen_offset > 0, performs incremental decoding via step().

        Returns:
            Tuple of (output, None, past_key_values) where:
                - output: Same shape as hidden_states input
                - None: Placeholder for consistency with other SSM layers
                - past_key_values: Updated cache (same as input if provided, else None)
        """
        if self.sequence_parallel_group:
            assert cu_seqlens is None, "SP not implemented for varlen Mamba2."
        seqlen_og = seqlen
        if seqlen is None:
            batch, seqlen, dim = hidden_states.shape
        else:
            batch_seqlen, dim = hidden_states.shape
            batch = batch_seqlen // seqlen

        conv_state, ssm_state = None, None
        if past_key_values is not None:
            inference_batch = (
                cu_seqlens.shape[0] - 1 if cu_seqlens is not None else batch
            )
            conv_state, ssm_state = self._get_states_from_cache(
                past_key_values, inference_batch
            )
            log_aprods = self._get_log_aprods_from_cache(
                past_key_values, inference_batch
            )
            if past_key_values.seqlen_offset > 0:
                # Single-token decoding - states are updated inplace
                out, _, _ = self.step(
                    hidden_states,
                    conv_state,
                    ssm_state,
                    position_embeddings=position_embeddings,
                    log_aprods=log_aprods,
                )
                return out, None, past_key_values

        zxbcdt = self.in_proj(hidden_states)  # [B, L, Din + Dxbc + H]

        if seqlen_og is not None:
            zxbcdt = rearrange(zxbcdt, "(b l) d -> b l d", l=seqlen)
        # If the model is loaded in fp16, without the .float() here, A might be -inf
        A = -torch.exp(self.A_log.float())  # [H,]
        dt_limit_kwargs = (
            {} if self.dt_limit == (0.0, float("inf")) else dict(dt_limit=self.dt_limit)
        )

        if self.sequence_parallel_group:
            sp_size = torch.distributed.get_world_size(
                group=self.sequence_parallel_group
            )
            sp_rank = torch.distributed.get_rank(group=self.sequence_parallel_group)
            zxbcdt = reorder_for_ssm_p2p(
                zxbcdt,
                self.sequence_parallel_group,
                torch.cuda.Stream(),
                sp_size,
                sp_rank,
                self.d_conv,
            )  # [B_sp, L_sp, Din + Dxbc + H]

        d_mlp = (zxbcdt.shape[-1] - 2 * self.d_inner - 2 * self.d_xb - self.nheads) // 2
        z0, x0, z, xBC, dt = torch.split(
            zxbcdt,
            [d_mlp, d_mlp, self.d_ssm, 2 * self.d_xb + self.d_ssm, self.nheads],
            dim=-1,
        )
        # z0: [B_sp, L_sp, Dmlp]
        # x0: [B_sp, L_sp, Dmlp]
        # z: [B_sp, L_sp, Din]
        # xBC: [B_sp, L_sp, Dxbc]
        # dt: [B_sp, L_sp, H]

        # (Optionally) Normalize B and C.
        xBC = self._apply_qk_norm(xBC)  # [B_sp, L_sp, Dxbc]

        # (Optionally) Apply position embeddings to B and C
        xBC = self._apply_pos_embeddings(
            xBC, position_embeddings=position_embeddings
        )  # [B_sp, L_sp, Dxbc]

        if conv_state is not None:
            assert (
                self.sequence_parallel_group is None
            ), "SP not implemented for when conv_state is not None"
            if cu_seqlens is None:
                # If we just take xBC[:, :, -self.d_conv :], it will error if seqlen < self.d_conv
                # Instead F.pad will pad with zeros if seqlen < self.d_conv, and truncate otherwise.
                xBC_t = rearrange(xBC, "b l d -> b d l")  # [B, Dxbc, L]
                conv_state.copy_(
                    F.pad(xBC_t, (self.d_conv - xBC_t.shape[-1], 0))
                )  # [B, Dxbc, d_conv]
            else:
                assert (
                    causal_conv1d_varlen_states is not None
                ), "varlen inference requires causal_conv1d package"
                assert batch == 1, "varlen inference only supports batch dimension 1"
                conv_varlen_states = causal_conv1d_varlen_states(
                    xBC.squeeze(0), cu_seqlens, state_len=conv_state.shape[-1]
                )
                conv_state.copy_(conv_varlen_states)  # [B, Dxbc, d_conv]
        assert self.activation in ["silu", "swish"]

        if causal_conv1d_fn is None or self.activation not in ["silu", "swish"]:
            assert seq_idx is None, "varlen conv1d requires the causal_conv1d package"
            xBC = self.act(
                self.conv1d(xBC.transpose(1, 2)).transpose(1, 2)[
                    :, : -(self.d_conv - 1)
                ]
            )
        else:
            xBC = causal_conv1d_fn(
                xBC.transpose(1, 2).contiguous(),
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                bias=self.conv1d.bias,
                activation=self.activation,
                seq_idx=seq_idx,
            ).transpose(
                1, 2
            )  # [B_sp, L_sp, Dxbc]

        if self.sequence_parallel_group is not None:
            # When SP is activated, we need to remove the first tokens since they were just needed for the conv operation.
            z0 = z0[:, self.d_conv - 1 :, :].contiguous()  # [B_sp, Lp, Dmlp]
            x0 = x0[:, self.d_conv - 1 :, :].contiguous()  # [B_sp, Lp, Dmlp]
            z = z[:, self.d_conv - 1 :, :].contiguous()  # [B_sp, Lp, Din]
            dt = dt[:, self.d_conv - 1 :, :].contiguous()  # [B_sp, Lp, H]
            xBC = xBC[:, self.d_conv - 1 :, :].contiguous()  # [B_sp, Lp, Dxbc]

        return_final_states = ssm_state is not None
        if self.sequence_parallel_group is not None:
            assert (
                return_final_states is False
            ), "SP not implemented for training with ssm state"
            return_final_states = True

        x, B, C = torch.split(
            xBC, [self.d_xb, self.d_xb, self.ngroups * self.d_state], dim=-1
        )

        # GQA: Repeat KV (Bx) heads to match Q (C) heads
        x = rearrange(x, "b l (h d) -> b h l d", d=self.headdim)  # [B_sp, Hkv, Lp, D]
        x = repeat_kv(x, self.num_kv_groups)  # [B_sp, H, Lp, D]
        x = rearrange(x, "b h l d -> b l h d")  # [B_sp, Lp, H, D]
        B = rearrange(B, "b l (g n) -> b g l n", n=self.d_state)  # [B_sp, Hkv, Lp, D]
        B = repeat_kv(B, self.num_kv_groups)  # [B_sp, H, Lp, D]
        B = rearrange(B, "b g l n -> b l g n")  # [B_sp, Lp, H, D]

        C = rearrange(C, "b l (g n) -> b l g n", n=self.d_state)  # [B_sp, Lp, H, D]
        with torch.cuda.device(x.device):
            y = mamba_chunk_scan_combined(
                x,
                dt,
                A,
                B,
                C,
                chunk_size=self.chunk_size,
                D=(
                    rearrange(self.D, "(h d) -> h d", d=self.headdim)
                    if self.D_has_hdim
                    else self.D
                ),
                z=(
                    rearrange(z, "b l (h d) -> b l h d", d=self.headdim)
                    if not self.rmsnorm
                    else None
                ),
                dt_bias=self.dt_bias,
                dt_softplus=True,
                seq_idx=seq_idx,
                cu_seqlens=cu_seqlens,
                **dt_limit_kwargs,
                return_final_states=return_final_states,
                return_varlen_states=cu_seqlens is not None
                and past_key_values is not None,
            )  # [B_sp, Lp, H, D]
        # During training, if SP=1, y shape is [B, L, H, D]
        # If SP>1, it is a tuple since return_final_states is True in this case.

        if self.sequence_parallel_group:
            y, curr_last_state, *rest = y
            # y: [B_sp, Lp, H, D]
            # curr_last_state: [B_sp, H, D, N]

            # Each GPU produces the correct SSM output assuming initial SSM state was 0.
            # Here we use the SSM state from the previous chunk to update the SSM output of the current chunk.
            dt_old = dt  # [B_sp, Lp, H]
            A_old = A
            # Process dt--only expand to head dimension, not headdim
            assert dt.dim() == 3, "Untied dt parameters across heads not yet supported."

            dt = rearrange(dt, "b l h -> b h l")  # [B_sp, H, Lp]

            # Process A--only use first state column and first element per head
            assert A.dim() == 1, "Untied A parameters across heads not yet supported."
            A = A.to(dtype=torch.float32)  # [H,]

            # Add bias only for the head dimension
            dt_bias = self.dt_bias  # [H,]
            dt = dt + dt_bias.unsqueeze(1).to(dtype=dt.dtype)  # [B_sp, H, Lp]
            dt = F.softplus(dt)
            dt = dt.to(x.dtype)

            # Compute directly without extra dimensions
            log_deltaA = torch.einsum("bhl,h -> bhl", dt, A)  # [B_sp, H, Lp]
            deltaA_cumprod = torch.exp(log_deltaA.cumsum(dim=2))  # [B_sp, H, Lp]

            b, l, _, _ = y.shape

            # Final result already in correct shape, just add the singleton dimension
            deltaA_cumprod = deltaA_cumprod.unsqueeze(2)  # [B_sp, H, 1, Lp]

            sp_size = torch.distributed.get_world_size(
                group=self.sequence_parallel_group
            )
            sp_rank = torch.distributed.get_rank(group=self.sequence_parallel_group)
            prev_last_state = state_passing_p2p(
                curr_last_state,
                deltaA_cumprod,
                sp_rank,
                sp_size,
                self.sequence_parallel_group,
                b,
            )  # [B_sp, H, D, N]
            dt = dt_old
            A = A_old

            with torch.cuda.device(x.device):
                contrib_prev_chunk_to_output = mamba_chunk_scan_combined(
                    torch.zeros_like(x),
                    dt,
                    A,
                    B,
                    C,
                    self.chunk_size,
                    D=None,
                    z=None,
                    initial_states=prev_last_state,
                    dt_bias=self.dt_bias.float(),
                    dt_softplus=True,
                    return_final_states=False,
                )  # [B_sp, Lp, H, D]
            y = y + contrib_prev_chunk_to_output  # [B_sp, Lp, H, D]

        # NOTE: Since we raise assertion when SP>1 and ssm_state is not None, this block will only be active if SP=1
        if ssm_state is not None:
            y, last_state, *rest = y
            if cu_seqlens is None:
                ssm_state.copy_(last_state)
            else:
                varlen_states = rest[0]
                ssm_state.copy_(varlen_states)

        y = rearrange(y, "b l h p -> b l (h p)")  # [B_sp, Lp, Din]

        if self.rmsnorm:
            y = self.norm(y, z.contiguous())  # [B_sp, Lp, Din]

        if self.sequence_parallel_group:
            batch_sp, y_len, dim = y.shape
            if batch_sp == 2:
                y = y.view(batch_sp // 2, y_len * 2, dim)  # [B, L, Din]
            else:
                y = torch.cat(
                    [y[: batch_sp // 2], y[batch_sp // 2 :]], dim=1
                ).contiguous()  # [B, L, Din]

        if d_mlp > 0:
            y = torch.cat([F.silu(z0) * x0, y], dim=-1)  # [B, L, Dmlp + Din]
        if seqlen_og is not None:
            assert self.sequence_parallel_group is None
            y = rearrange(y, "b l d -> (b l) d")  # [B*L, Din]
        out = self.out_proj(y)  # [B, L, Dm]

        # Cache cumulative A products for state composition
        if past_key_values and past_key_values.cache_aprods:
            log_aprods = self._get_log_aprods_from_cache(past_key_values, inference_batch)
            dt = F.softplus(dt + repeat(self.dt_bias.to(dtype=dt.dtype), 
                                       "h -> b l h", b=dt.shape[0], l=dt.shape[1]))
            
            log_dA = dt * A
            log_next_aprods = torch.sum(log_dA, dim=1)
            
            if past_key_values.seqlen_offset <= 0:
                log_aprods.copy_(log_next_aprods)
            else:
                log_aprods.copy_(log_aprods + log_next_aprods)

        return out, None, past_key_values

    def step(
        self,
        hidden_states: Tensor,
        conv_state: Tensor,
        ssm_state: Tensor,
        position_embeddings: Optional[Tuple[Tensor, Tensor]] = None,
        log_aprods: Optional[Tensor] = None,
    ):
        """
        Performs a Mamba2 decoding step during generation.

        Args:
            hidden_states: Input tensor with shape [B, 1, Dm].
            conv_state: Convolution state tensor with shape [B, Dxbc, d_conv].
            ssm_state: Mamba2 state tensor with shape [B, H, D, N].
            position_embeddings: Optional tuple of (cos, sin) RoPE tensors, each with shape [B, 1, D].
        """
        dtype = hidden_states.dtype
        assert (
            hidden_states.shape[1] == 1
        ), "Only support decoding with 1 token at a time for now"
        zxbcdt = self.in_proj(hidden_states.squeeze(1))  # [B, Din + Dxbc + H]

        d_mlp = (zxbcdt.shape[-1] - 2 * self.d_inner - 2 * self.d_xb - self.nheads) // 2
        z0, x0, z, xBC, dt = torch.split(
            zxbcdt,
            [d_mlp, d_mlp, self.d_ssm, self.d_ssm + 2 * self.d_xb, self.nheads],
            dim=-1,
        )
        # z0: [B, Dmlp]
        # x0: [B, Dmlp]
        # z: [B, Din]
        # xBC: [B, Dxbc]
        # dt: [B, H]

        # (Optionally) Apply QK normalization to B and C
        xBC = self._apply_qk_norm(xBC)  # [B, Dxbc]

        # (Optionally) Apply position embeddings to B and C
        xBC = self._apply_pos_embeddings(
            xBC, position_embeddings=position_embeddings
        )  # [B, Dxbc]

        # Conv step
        if causal_conv1d_update is None:
            conv_state.copy_(
                torch.roll(conv_state, shifts=-1, dims=-1)
            )  # [B, Dxbc,  d_conv]
            conv_state[:, :, -1] = xBC
            xBC = torch.sum(
                conv_state * rearrange(self.conv1d.weight, "d 1 w -> d w"), dim=-1
            )  # [B, Dxbc]
            if self.conv1d.bias is not None:
                xBC = xBC + self.conv1d.bias
            xBC = self.act(xBC).to(dtype=dtype)
        else:
            xBC = causal_conv1d_update(
                xBC,
                conv_state,
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                self.activation,
            )  # [B, Dxbc]

        A = -torch.exp(self.A_log.float())  # [H,]

        x, B, C = torch.split(
            xBC, [self.d_xb, self.d_xb, self.ngroups * self.d_state], dim=-1
        )
        # x: [B, Dxb]
        # B: [B, Dxb]
        # C: [B, Din]

        # GQA: Repeat KV (Bx) heads to match Q (C) heads
        x = rearrange(x, "b (h d) -> b h d", d=self.headdim)  # [B, Hkv, D]
        x = torch.repeat_interleave(x, dim=1, repeats=self.num_kv_groups)  # [B, H, D]

        B = rearrange(B, "b (g n) -> b g n", n=self.d_state)  # [B, Hkv, D]
        B = torch.repeat_interleave(B, dim=1, repeats=self.num_kv_groups)  # [B, H, D]

        # SSM step
        assert selective_state_update is not None

        A = repeat(A, "h -> h d n", d=self.headdim, n=self.d_state).to(
            dtype=torch.float32
        )  # [H, D, N]
        dt = repeat(dt, "b h -> b h d", d=self.headdim)  # [B, H, D]
        dt_bias = repeat(self.dt_bias, "h -> h p", p=self.headdim)  # [H, D]
        D = repeat(self.D, "h -> h d", d=self.headdim)  # [H, D]
        C = rearrange(C, "b (g n) -> b g n", n=self.d_state)  # [B, H, D]

        if not self.rmsnorm:
            z = rearrange(z, "b (h d) -> b h d", d=self.headdim)  # [B, H, D]

        y = selective_state_update(
            ssm_state,
            x,
            dt,
            A,
            B,
            C,
            D,
            z=z if not self.rmsnorm else None,
            dt_bias=dt_bias,
            dt_softplus=True,
        )
        y = rearrange(y, "b h d -> b (h d)")  # [B, H * D]

        if self.rmsnorm:
            y = self.norm(y, z)  # [B, H * D]
        if d_mlp > 0:
            y = torch.cat([F.silu(z0) * x0, y], dim=-1)  # [B, Dmlp + Din]
        out = self.out_proj(y)  # [B, Dm]

        # Update cumulative A products in log space
        if log_aprods is not None:
            A_orig = -torch.exp(self.A_log.float())
            dt_orig = dt[:, :, 0]
            dt_step = F.softplus(dt_orig + self.dt_bias.to(dtype=dt.dtype))
            log_dA = dt_step * A_orig
            log_aprods.copy_(log_aprods + log_dA)

        return out.unsqueeze(1), conv_state, ssm_state

    def allocate_inference_cache(
        self,
        batch_size: int,
        max_seqlen: int,
        dtype: Optional[torch.dtype] = None,
        **kwargs,
    ):
        """
        Allocates inference cache for convolution and SSM states.

        Args:
            batch_size: Batch size for inference.
            max_seqlen: Maximum sequence length (currently unused).
            dtype: Optional data type for cache tensors. If None, uses layer weight dtypes.

        Returns:
            Tuple of (conv_state, ssm_state) where:
                - conv_state: Tensor with shape [B, Dxbc, d_conv]
                - ssm_state: Tensor with shape [B, H, D, N]
        """
        device = self.out_proj.weight.device
        conv_dtype = self.conv1d.weight.dtype if dtype is None else dtype
        conv_state = torch.zeros(
            batch_size,
            self.d_conv,
            self.conv1d.weight.shape[0],
            device=device,
            dtype=conv_dtype,
        ).transpose(1, 2)
        ssm_dtype = self.in_proj.weight.dtype if dtype is None else dtype
        ssm_state = torch.zeros(
            batch_size,
            self.nheads,
            self.headdim,
            self.d_state,
            device=device,
            dtype=ssm_dtype,
        )
        return conv_state, ssm_state

    def _get_states_from_cache(
        self, past_key_values, batch_size, initialize_states=False
    ):
        """
        Retrieves or initializes conv and SSM states from the cache.

        Args:
            past_key_values: Mamba2Cache object containing cached states.
            batch_size: Batch size for state initialization.
            initialize_states: If True, zeros out existing states.

        Returns:
            Tuple of (conv_state, ssm_state) retrieved from or stored in the cache.
        """
        assert self.layer_idx is not None
        if self.layer_idx not in past_key_values.key_value_memory_dict:
            batch_shape = (batch_size,)
            conv_state = torch.zeros(
                batch_size,
                self.d_conv,
                self.conv1d.weight.shape[0],
                device=self.conv1d.weight.device,
                dtype=self.conv1d.weight.dtype,
            ).transpose(1, 2)
            ssm_state = torch.zeros(
                batch_size,
                self.nheads,
                self.headdim,
                self.d_state,
                device=self.in_proj.weight.device,
                dtype=self.in_proj.weight.dtype,
            )
            past_key_values.key_value_memory_dict[self.layer_idx] = (
                conv_state,
                ssm_state,
            )
        else:
            conv_state, ssm_state = past_key_values.key_value_memory_dict[
                self.layer_idx
            ]
            # TODO: What if batch size changes between generation, and we reuse the same states?
            if initialize_states:
                conv_state.zero_()
                ssm_state.zero_()
        return conv_state, ssm_state

    def _get_log_aprods_from_cache(self, past_key_values, batch_size, reset_aprods=False):
        """Retrieve or initialize log(A_prod) from cache for state composition."""
        assert self.layer_idx is not None
        if not past_key_values.cache_aprods:
            return None
        
        if self.layer_idx not in past_key_values.log_aprods:
            log_aprods = torch.zeros(
                batch_size,
                self.nheads,
                device=self.in_proj.weight.device,
                dtype=self.in_proj.weight.dtype,
            )
            past_key_values.log_aprods[self.layer_idx] = log_aprods
        else:
            log_aprods = past_key_values.log_aprods[self.layer_idx]
            if reset_aprods:
                log_aprods.zero_()

        return log_aprods
