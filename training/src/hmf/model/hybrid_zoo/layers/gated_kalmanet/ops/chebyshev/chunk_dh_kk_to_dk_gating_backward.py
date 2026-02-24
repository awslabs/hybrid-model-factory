# Adapted from https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/mesa_net/chunk_h_kk_intra_bwd.py

from typing import Tuple

from torch import Tensor
import torch
import triton
import triton.language as tl

from fla.ops.utils.op import exp


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [1, 2, 4, 8]
        for num_stages in [1, 2, 3, 4]
    ],
    key=["D"],
)
@triton.jit(do_not_specialize=["T"])
def chunk_dh_kk_to_dk_gating_kernel(
    # input
    k_ptr,
    x_out_ptr,
    kk_ptr,
    d_kk_ptr,
    dq_ptr,
    gkk_ptr,
    # output:
    dk_ptr,
    dgkk_ptr,
    # constants:
    T,
    H: tl.constexpr,
    D: tl.constexpr,
    chunk_size: tl.constexpr,
):
    """
    Computes key and gating gradients (dk, dgkk) from covariance gradient d_kk.

    Gated version that backpropagates through exponential gating applied to
    key-key covariance. Computes gradients for both keys and gating values.

    Grid: (num_chunks, B*H) - one thread block per chunk and batch-head pair
    Block size: Processes [chunk_size, D] vectors per block
    """

    chunk_id, bh_id = tl.program_id(0), tl.program_id(1)
    b_id, h_id = bh_id // H, bh_id % H

    num_chunks = tl.cdiv(T, chunk_size)

    # offset calculation
    scalar_off = b_id * T * H + h_id
    gkk_ptr += scalar_off
    dgkk_ptr += scalar_off

    vec_off = scalar_off * D
    k_ptr += vec_off
    x_out_ptr += vec_off
    dk_ptr += vec_off
    dq_ptr += vec_off

    d_kk_ptr += ((b_id * num_chunks + chunk_id) * H + h_id).to(tl.int64) * D * D
    kk_ptr += ((b_id * num_chunks + chunk_id) * H + h_id).to(tl.int64) * D * D

    # load
    x_out_ptr = tl.make_block_ptr(
        x_out_ptr,
        (T, D),
        (H * D, 1),
        (chunk_id * chunk_size, 0),
        (chunk_size, D),
        (1, 0),
    )
    x_out = tl.load(x_out_ptr, boundary_check=(0, 1))  # .to(tl.float32)
    dq_ptr = tl.make_block_ptr(
        dq_ptr, (T, D), (H * D, 1), (chunk_id * chunk_size, 0), (chunk_size, D), (1, 0)
    )
    dq = tl.load(dq_ptr, boundary_check=(0, 1))

    p_gkk = tl.make_block_ptr(
        gkk_ptr, (T,), (H,), (chunk_id * chunk_size,), (chunk_size,), (0,)
    )

    gkk = tl.load(p_gkk, boundary_check=(0,)).to(tl.float32)

    p_d_kk = tl.make_block_ptr(d_kk_ptr, (D, D), (D, 1), (0, 0), (D, D), (1, 0))
    p_kk = tl.make_block_ptr(kk_ptr, (D, D), (D, 1), (0, 0), (D, D), (1, 0))

    k_ptr = tl.make_block_ptr(
        k_ptr, (T, D), (H * D, 1), (chunk_id * chunk_size, 0), (chunk_size, D), (1, 0)
    )
    k = tl.load(k_ptr, boundary_check=(0, 1))

    chunk_range = tl.arange(0, chunk_size)
    bool_mask = chunk_range[:, None] >= chunk_range[None, :]
    mask = tl.where(bool_mask, exp(gkk[:, None] - gkk[None, :]), 0)

    dk = tl.zeros([chunk_size, D], dtype=tl.float32)

    s = tl.dot(x_out, tl.trans(k)) * mask
    ds = tl.dot(dq, tl.trans(k))
    dk += tl.dot(tl.trans(s).to(dq.dtype), dq)

    dm = tl.where(bool_mask, s * ds, 0)

    dgkk = tl.zeros([chunk_size], dtype=tl.float32)
    dgkk += tl.sum(dm, axis=1)
    dgkk -= tl.sum(dm, axis=0)
    dk += tl.dot(tl.trans(ds * mask).to(x_out.dtype), x_out)

    gkk_last = tl.sum(tl.where(chunk_range == chunk_size - 1, gkk, 0))
    mask1 = exp(gkk_last - gkk)[:, None]

    d_kk = tl.load(p_d_kk, boundary_check=(0, 1))

    dk += tl.dot((k * mask1).to(k.dtype), (d_kk + tl.trans(d_kk)).to(k.dtype))
    dk = -dk

    kk = tl.load(p_kk, boundary_check=(0, 1))
    dgkk += tl.sum(tl.dot(dq, kk) * tl.exp(gkk)[:, None] * x_out, axis=1)

    dk2 = tl.dot((k * mask1).to(k.dtype), d_kk.to(k.dtype))
    dgkk_last = tl.sum(dk2 * k, axis=1)
    dgkk -= dgkk_last
    dgkk_last = tl.sum(dgkk_last)
    dgkk_last += tl.sum(d_kk * kk) * exp(gkk_last)

    dgkk = tl.where(chunk_range < chunk_size - 1, dgkk, dgkk + dgkk_last)

    dgkk = tl.cumsum(-dgkk, reverse=True)

    p_dgkk = tl.make_block_ptr(
        dgkk_ptr, (T,), (H,), (chunk_id * chunk_size,), (chunk_size,), (0,)
    )
    tl.store(p_dgkk, dgkk.to(p_dgkk.dtype.element_ty), boundary_check=(0,))

    p_dk = tl.make_block_ptr(
        dk_ptr, (T, D), (H * D, 1), (chunk_id * chunk_size, 0), (chunk_size, D), (1, 0)
    )
    tl.store(p_dk, dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))


def chunk_dh_kk_to_dk_gating(
    k: Tensor,
    x_out: Tensor,
    h_kk: Tensor,
    d_kk: Tensor,
    dq: Tensor,
    gkk: Tensor,
    chunk_size: int = 64,
) -> Tuple[Tensor, Tensor]:
    """
    Computes key and gating gradients (dk, dgkk) from covariance gradient d_kk.

    Wrapper function that launches the Triton kernel for gated gradient computation.

    Args:
        k: Key tensor with shape [B, L, H, D]
        x_out: Output from forward pass with shape [B, L, H, D]
        h_kk: Chunk-level key-key covariance with shape [B, num_chunks, H, D, D]
        d_kk: Gradient of chunk-level covariance with shape [B, num_chunks, H, D, D]
        dq: Gradient of query with shape [B, L, H, D]
        gkk: Cumulative gating values with shape [B, L, H]
        chunk_size: Chunk size for parallel processing. Defaults to 64

    Returns:
        dk: Key gradient with shape [B, L, H, D]
        dgkk: Gating gradient with shape [B, L, H]
    """

    B, T, H, D = k.shape

    dk = torch.empty_like(k)
    grid = (triton.cdiv(T, chunk_size), B * H)

    dgkk = torch.empty([B, T, H]).to(k.device)
    chunk_dh_kk_to_dk_gating_kernel[grid](
        # inputs:
        k,
        x_out,
        h_kk,
        d_kk,
        dq,
        gkk,
        # output:
        dk,
        dgkk,
        # constants:
        T,
        H,
        D,
        chunk_size,
    )

    return dk, dgkk
