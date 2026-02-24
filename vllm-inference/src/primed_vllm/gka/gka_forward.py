# -*- coding: utf-8 -*-
"""
GKA forward functions for vLLM inference.

This module provides the high-level API for GKA:
- latent_chebyshev_prefill: Chebyshev query refinement for prefill
- gka_chebyshev_gla_prefill: Full GKA prefill (Chebyshev + Simple GLA)
- gka_chebyshev_gla_decode: Single-token decode (Chebyshev + fused recurrent GLA)
"""

from typing import Optional, Tuple
from vllm.triton_utils import tl, triton
import logging
import torch
from einops import rearrange

from .ops.chebyshev_iteration_gqa import chebyshev_iteration_gqa as chebyshev_iteration
from .ops.chebyshev_prefill_gqa import chebyshev_iteration_prefill_gqa_fwd as chebyshev_iteration_prefill_fwd
from .ops.chunk_simple_gla_gqa import chunk_simple_gla_gqa as chunk_simple_gla



def _check_sketching_constraints(gka: Optional[torch.nn.Linear], num_sketches_per_head: int, func_name: str):
    """Check that sketching is disabled (known limitation)."""
    if gka is not None:
        raise NotImplementedError(
            f"{func_name}: gka projection is not supported. "
            "Set parallel_latent_sketch=False in config. "
        )
    if num_sketches_per_head != 1:
        raise NotImplementedError(
            f"{func_name}: num_sketches_per_head={num_sketches_per_head} is not supported. "
            "Must be 1 due to numerical instability in gating kernel. "
        )


def latent_chebyshev_prefill(
    k: torch.Tensor,
    q: torch.Tensor,
    gk: Optional[torch.Tensor] = None,
    beta: Optional[torch.Tensor] = None,
    ridge_ratio: float = 0.1,
    num_sketches_per_head: int = 1,
    num_iter: int = 30,
    gka: Optional[torch.nn.Linear] = None,
    h0: Optional[torch.Tensor] = None,
    output_final_state: bool = True,
    cu_seqlens: Optional[torch.Tensor] = None,
    chunk_size: int = 64,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Chebyshev query refinement for GKA prefill.

    Args:
        k: Keys [B, T, H, D] (B=1 for varlen)
        q: Queries [B, T, H, D]
        gk: Decay gates for h_kk [B, T, H] (optional, log-space)
        beta: Input selectivity for GKA [B, T, H] (optional)
        ridge_ratio: Ridge regularization ratio
        num_sketches_per_head: Number of sketches per head
        num_iter: Number of Chebyshev iterations
        gka: GKA projection layer (nn.Linear)
        h0: Initial h_kk state [N, H*num_sketches, sketch_dim, sketch_dim]
        output_final_state: Whether to output final h_kk state
        cu_seqlens: Cumulative sequence lengths [N+1] for variable-length
        chunk_size: Chunk size for processing

    Returns:
        x: Refined queries [B, T, H, D]
        h_kk_final: Final h_kk state if output_final_state else None
    """
    if cu_seqlens is not None and q.shape[0] != 1:
        raise ValueError(f"Batch size must be 1 when using cu_seqlens, got {q.shape[0]}")

    _check_sketching_constraints(gka, num_sketches_per_head, "latent_chebyshev_prefill")

    # Project through GKA if provided
    if gka is not None:
        q_proj = gka(q)
        k_proj = gka(k)
    else:
        q_proj = q
        k_proj = k

    # Reshape for num_sketches_per_head
    if num_sketches_per_head > 1:
        q_proj = rearrange(q_proj, 'b t h (r p) -> b t (h r) p', r=num_sketches_per_head)
        k_proj = rearrange(k_proj, 'b t h (r p) -> b t (h r) p', r=num_sketches_per_head)

    # Run Chebyshev iteration
    x, h_kk_final = chebyshev_iteration_prefill_fwd(
        k=k_proj.contiguous(),
        q=q_proj.contiguous(),
        gk=gk.contiguous() if gk is not None else None,
        beta=beta.contiguous() if beta is not None else None,
        h0=h0,
        num_iter=num_iter,
        ridge_ratio=ridge_ratio,
        chunk_size=chunk_size,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
    )

    # Reshape back
    if num_sketches_per_head > 1:
        x = rearrange(x, 'b t (h r) p -> b t h (r p)', r=num_sketches_per_head)

    # Project back through GKA weight
    if gka is not None:
        x = x @ gka.weight

    return x, h_kk_final

@triton.jit
def _normalize_qk_kernel(
    q, k,
    B: tl.constexpr, T: tl.int64,
    H: tl.constexpr, HKV: tl.constexpr,
    GQA_GROUP_SIZE: tl.constexpr,
    D: tl.constexpr, BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    b = pid // (T * H)
    remainder = pid % (T * H)
    t, h = remainder // H, remainder % H

    h_kv = h // GQA_GROUP_SIZE
    is_group_leader = (h % GQA_GROUP_SIZE == 0)  # ← Only first in group writes K

    d_offsets = tl.arange(0, BLOCK_D)
    mask = d_offsets < D

    q_offset = b * (T*H*D) + t * (H*D) + h * D
    p_q = q + q_offset + d_offsets
    b_q = tl.load(p_q, mask=mask, other=0.0).to(tl.float32)
    b_q_normalized = b_q / tl.sqrt(tl.sum(b_q * b_q, axis=0))

    if is_group_leader:
        k_offset = b * (T * HKV * D) + t * (HKV * D) + h_kv * D  # ← Different offset!
        p_k = k + k_offset + d_offsets
        b_k = tl.load(p_k, mask=mask, other=0.0).to(tl.float32)
        b_k_normalized = b_k / tl.sqrt(tl.sum(b_k * b_k, axis=0))
        tl.store(p_k, b_k_normalized.to(p_k.dtype.element_ty, fp_downcast_rounding="rtne"), mask=mask)

    tl.store(p_q, b_q_normalized.to(p_q.dtype.element_ty, fp_downcast_rounding="rtne"), mask=mask)



def fused_l2_norm(q: torch.Tensor, k: torch.Tensor):
    """Normalize Q and K by L2 norm (in-place). GQA-aware: only group leaders write K."""
    q, k = q.contiguous(), k.contiguous()
    HKV = k.shape[2]
    batch, num_tokens, num_heads, head_dim = q.shape
    GQA_GROUP_SIZE = num_heads // HKV
    _normalize_qk_kernel[(batch * num_tokens * num_heads,)](
        q, k, B=batch, T=num_tokens, H=num_heads, HKV=HKV, D=head_dim, GQA_GROUP_SIZE=GQA_GROUP_SIZE,
        BLOCK_D=triton.next_power_of_2(head_dim),
    )
    return q, k

def gka_chebyshev_gla_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    alpha: torch.Tensor,
    g: torch.Tensor,
    gk: Optional[torch.Tensor],
    beta: Optional[torch.Tensor],
    gla_scale: float,
    ridge_ratio: float = 0.02,
    num_sketches_per_head: int = 1,
    num_iter: int = 30,
    gka: Optional[torch.nn.Linear] = None,
    h_kk_initial: Optional[torch.Tensor] = None,
    h_kv_initial: Optional[torch.Tensor] = None,
    output_final_state: bool = True,
    cu_seqlens: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Full GKA prefill: Chebyshev query refinement + Simple GLA attention.

    Args:
        q: Queries [B, T, H, D] (B=1 for varlen)
        k: Keys [B, T, H, D]
        v: Values [B, T, H, V]
        alpha: Blending factor [B, T, H]
        g: Decay gates for GLA [B, T, H] (log-space)
        gk: Decay gates for h_kk [B, T, H] (optional, log-space)
        beta: Input selectivity for GKA [B, T, H] (optional)
        gla_scale: Scale factor for GLA (typically head_dim ** -0.5)
        ridge_ratio: Ridge regularization ratio
        num_sketches_per_head: Number of sketches per head
        num_iter: Number of Chebyshev iterations
        gka: GKA projection layer
        h_kk_initial: Initial h_kk state [N, H*num_sketches, D, D]
        h_kv_initial: Initial h_kv state [N, H, K, V]
        output_final_state: Whether to output final states
        cu_seqlens: Cumulative sequence lengths [N+1]

    Returns:
        o: Output [B, T, H, V]
        h_kk_final: Final h_kk state if output_final_state
        h_kv_final: Final h_kv state if output_final_state
    """
    _check_sketching_constraints(gka, num_sketches_per_head, "gka_chebyshev_gla_prefill")

    q, k = fused_l2_norm(q, k)

    # 1. Chebyshev query refinement
    if num_iter > 0:
        q_refined, h_kk_final = latent_chebyshev_prefill(
            k=k,
            q=q,
            gk=gk,
            beta=beta,
            ridge_ratio=ridge_ratio,
            num_sketches_per_head=num_sketches_per_head,
            num_iter=num_iter,
            gka=gka,
            h0=h_kk_initial,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
        )
        # Alpha blending: q = q + alpha * (q_refined - q)
        q = q + alpha[..., None] * (q_refined - q)
    else:
        h_kk_final = None

    # 2. Simple GLA attention
    o, h_kv_final = chunk_simple_gla(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=gla_scale,
        initial_state=h_kv_initial,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
    )

    return o, h_kk_final, h_kv_final


_warned_num_iter_zero = False


def gka_chebyshev_gla_decode(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    alpha: torch.Tensor,
    g: torch.Tensor,
    gk: Optional[torch.Tensor],
    beta: Optional[torch.Tensor],
    gla_scale: float,
    ridge_ratio: float = 0.02,
    num_sketches_per_head: int = 1,
    num_iter: int = 30,
    gka: Optional[torch.nn.Linear] = None,
    h_kk: Optional[torch.Tensor] = None,
    h_kv: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
    ssm_state_indices: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Single-token GKA decode: Chebyshev iteration + fused recurrent GLA.

    Args:
        q: Queries [B, T, H, D] or [1, N, H, D] for continuous batching
        k: Keys [B, T, H, D]
        v: Values [B, T, H, V]
        alpha: Blending factor [B, T, H]
        g: Decay gates for GLA [B, T, H] (log-space)
        gk: Decay gates for h_kk [B, T, H] (optional, log-space)
        beta: Input selectivity for GKA [B, T, H] (optional)
        gla_scale: Scale factor for GLA
        ridge_ratio: Ridge regularization ratio
        num_sketches_per_head: Number of sketches per head
        num_iter: Number of Chebyshev iterations
        gka: GKA projection layer
        h_kk: h_kk state [B, H*num_sketches, D, D] or [N, H, D, D] for continuous batching
        h_kv: h_kv state [B, H, K, V] or [N, H, K, V] for continuous batching
        cu_seqlens: Cumulative sequence lengths [N+1] for continuous batching
        ssm_state_indices: [N] maps sequences to state cache slots (for continuous batching)

    Returns:
        o: Output [B, T, H, V]
        h_kk_new: Updated h_kk state
        h_kv_new: Updated h_kv state
    """
    _check_sketching_constraints(gka, num_sketches_per_head, "gka_chebyshev_gla_decode")
    if num_iter == 0:
        # num_iter=0 means no Chebyshev correction. Caller must zero out alpha
        # before calling this function. We still run 1 iteration to update h_kk state.
        global _warned_num_iter_zero
        if not _warned_num_iter_zero:
            logging.warning(
                "gka_chebyshev_gla_decode: num_iter=0, running 1 Chebyshev iteration. "
                "Ensure alpha is zeroed out by the caller to skip query correction."
            )
            _warned_num_iter_zero = True
        num_iter = 1

    assert g is gk, "GLA gate g not equal to GKA gate gk is not supported for decoding."
    # 1. Chebyshev query refinement (single token)
    if num_iter > 0:
        o, h_kk_new, h_kv_new = chebyshev_iteration(
            k=k,
            q=q,
            v=v,
            alpha=alpha,
            gk=gk,
            beta=beta,
            num_iter=num_iter,
            scale=gla_scale,
            ridge_ratio=ridge_ratio,
            initial_state_kk = h_kk,
            initial_state_kv = h_kv,
            output_final_state=True,
            inplace_final_state=True,
            cu_seqlens=cu_seqlens,
            ssm_state_indices=ssm_state_indices,
        )

    return o, h_kk_new, h_kv_new
