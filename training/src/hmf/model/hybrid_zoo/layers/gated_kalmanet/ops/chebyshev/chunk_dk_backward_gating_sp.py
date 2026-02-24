from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from fla.ops.common.chunk_h import chunk_fwd_h
from fla.ops.utils import chunk_local_cumsum, chunk_global_cumsum
from fla.ops.utils.op import exp

@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [1, 2, 4, 8]
        for num_stages in [1, 2, 3, 4]
    ],
    key=['D']
)
@triton.jit(do_not_specialize=['T'])
def chunk_dh_kk_to_dk_gating_kernel_sp(
    # input
    k_ptr, d_kk_ptr, gkk_ptr, gkk_last_ptr,
    # output:
    dk_ptr, dgk_ptr,
    # constants:
    T, H: tl.constexpr, D: tl.constexpr, chunk_size: tl.constexpr, GK: tl.constexpr,
):

    chunk_id, bh_id = tl.program_id(0), tl.program_id(1)
    b_id, h_id = bh_id // H, bh_id % H

    # offset calculation
    scalar_off = b_id * T * H + h_id

    if GK:
        gkk_ptr += scalar_off
        dgk_ptr += scalar_off

    vec_off = scalar_off * D
    k_ptr += vec_off
    dk_ptr += vec_off

    d_kk_ptr += ((b_id) * H + h_id).to(tl.int64) * D * D

    if GK:
        # Load the global last gkk value for this batch/head
        gkk_last_ptr += (b_id * H + h_id).to(tl.int64)
        gkk_last = tl.load(gkk_last_ptr).to(tl.float32)

        # load
        p_gkk = tl.make_block_ptr(gkk_ptr, (T,), (H,), (chunk_id * chunk_size,), (chunk_size,), (0,))

        gkk = tl.load(p_gkk, boundary_check=(0,)).to(tl.float32)

    p_d_kk = tl.make_block_ptr(d_kk_ptr, (D, D), (D, 1), (0, 0), (D, D), (1, 0))

    k_ptr = tl.make_block_ptr(k_ptr, (T, D), (H*D, 1), (chunk_id * chunk_size, 0), (chunk_size, D), (1, 0))
    k = tl.load(k_ptr, boundary_check=(0, 1)).to(tl.float32) 


    dk = tl.zeros([chunk_size, D], dtype=tl.float32)
    d_kk = tl.load(p_d_kk, boundary_check=(0, 1)).to(tl.float32)
    d_kk_sym = d_kk + tl.trans(d_kk)

    if GK:
        dgkk = tl.zeros([chunk_size], dtype=tl.float32)
        mask1 = exp(gkk_last - gkk)[:, None]
        # Keep all computations in fp32 for precision
        dk += tl.dot(k * mask1, d_kk_sym)

    else:
        dk += tl.dot(k, d_kk_sym)

    dk = -dk

    if GK:
        # Gradient w.r.t. gk (original gate values, not cumsum)
        # The formula already accounts for the cumsum in the forward pass
        dk2 = tl.dot(k * mask1, d_kk)
        contrib = -tl.sum(dk2 * k, axis=1)

        dgkk -= contrib #we do not negate this since the upstream gradient dkk should be negative.

    if GK:
        p_dgkk = tl.make_block_ptr(dgk_ptr, (T,), (H,), (chunk_id * chunk_size,), (chunk_size,), (0,))
        tl.store(p_dgkk, dgkk.to(p_dgkk.dtype.element_ty), boundary_check=(0,))

    p_dk = tl.make_block_ptr(dk_ptr, (T, D), (H*D, 1), (chunk_id * chunk_size, 0), (chunk_size, D), (1, 0))
    tl.store(p_dk, dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))

def chunk_dh_kk_to_dk_gating_sp(
    k: torch.Tensor,
    d_kk: torch.Tensor,
    gkk: torch.Tensor,
    chunk_size: int = 64,
):

    B, T, H, D = k.shape

    dk = torch.empty_like(k)
    grid = (triton.cdiv(T, chunk_size), B * H)

    if gkk is not None:
        gkk_last = gkk[:,-1,:].contiguous()
        gkk = gkk.contiguous()
        dgk = torch.empty([B, T, H]).to(k.device)

    else:
        gkk_last = None
        dgk = None

    chunk_dh_kk_to_dk_gating_kernel_sp[grid](
        # inputs:
        k.contiguous(), d_kk.contiguous(), gkk, gkk_last,
        # output:
        dk, dgk,
        # constants:
        T, H, D, chunk_size, gkk is not None
    )

    return dk, dgk

class ChunkFwdHAutograd(torch.autograd.Function):
    """
    Autograd wrapper for chunk_fwd_h to enable gradient flow through kk_final.

    Forward: Computes h_kk states using chunk_fwd_h
    Backward: Uses chunk_final_state_bwd to compute gradients w.r.t. k and g
    """

    @staticmethod
    def forward(ctx, k_proj, gk, chunk_size=64):
        """
        Args:
            k_proj: [B, T, H, D] - key projections
            gkk: [B, T, H] - cumulative gating values (cumsum of gk)
            chunk_size: int - chunk size for processing

        Returns:
            h_kk_local: [B, T, H, D, D] - local hidden states (unused in SP path)
            kk_final: [B, H, D, D] - final state for state passing
        """
        if gk is not None:
            gkk = chunk_local_cumsum(gk, chunk_size=chunk_size)  # Local cumsum (resets every 64 tokens)
            gkk_global = chunk_global_cumsum(gk)  #
        else:
            gkk = None
            gkk_global = None

        _, kk_final = chunk_fwd_h(
            k=k_proj,
            v=k_proj,
            g=gkk,
            gk=None,
            gv=None,
            h0=None,
            output_final_state=True,
            states_in_fp32=False,
            cu_seqlens=None,
            chunk_size=chunk_size,
        )

        # Save tensors needed for backward
        ctx.save_for_backward(k_proj, gkk_global)
        ctx.chunk_size = chunk_size

        return kk_final

    @staticmethod
    def backward(ctx, grad_kk_final):
        """
        Args:
            grad_h_kk_local: [B, T, H, D, D] - gradients w.r.t. h_kk_local (unused, should be None)
            grad_kk_final: [B, H, D, D] - gradients w.r.t. kk_final

        Returns:
            grad_k_proj: [B, T, H, D] - gradients w.r.t. k
            grad_gkk: [B, T, H] - gradients w.r.t. g (cumsum)
            None - for chunk_size (no gradient)
        """
        k_proj, gkk_global = ctx.saved_tensors
        chunk_size = ctx.chunk_size

        if grad_kk_final is None:
            # No gradient flows back
            return None, None, None

        # Compute gradients using backward recurrence
        grad_k_proj, grad_gkk = chunk_dh_kk_to_dk_gating_sp(
            k=k_proj,
            d_kk=grad_kk_final,
            gkk=gkk_global,
            chunk_size = chunk_size,
        )

        # Exclusive forward cumsum, for details see Eq. 32 and 33 from the GKA paper)
        grad_gk = grad_gkk - chunk_global_cumsum(grad_gkk)

        return grad_k_proj, grad_gk, None


def compute_kk_final_with_gating(k_proj, gk, chunk_size=64):
    """
    Wrapper function for ChunkFwdHAutograd.

    Args:
        k_proj: [B, T, H, D] - key projections
        gk: [B, T, H] - gating values (differential)
        chunk_size: int - chunk size (default 64)

    Returns:
        h_kk_local: [B, T, H, D, D] - local hidden states
        kk_final: [B, H, D, D] - final state
    """
    return ChunkFwdHAutograd.apply(k_proj, gk, chunk_size)