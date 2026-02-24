# Adapted from https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/mesa_net/chunk_h_kk_intra_bwd.py

from torch import Tensor

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [1, 2, 4, 8]
        for num_stages in [1, 2, 3, 4]
    ],
    key=["D"],
)
@triton.jit(do_not_specialize=["T"])
def chunk_dh_kk_to_dk_kernel(
    # input
    k_ptr,
    x_out_ptr,
    dh_kk_ptr,
    dq_ptr,
    # output:
    dk_ptr,
    # constants:
    T,
    H: tl.constexpr,
    D: tl.constexpr,
    chunk_size: tl.constexpr,
):
    """
    Computes key gradient dk from chunk-level covariance gradient dh_kk.

    Grid: (num_chunks, B*H) - one thread block per chunk and batch-head pair
    Block size: Processes [chunk_size, D] vectors per block
    """
    chunk_id, bh_id = tl.program_id(0), tl.program_id(1)
    b_id, h_id = bh_id // H, bh_id % H

    num_chunks = tl.cdiv(T, chunk_size)

    # offset calculation
    vec_off = (b_id * T * H + h_id) * D
    k_ptr += vec_off
    x_out_ptr += vec_off
    dk_ptr += vec_off
    dq_ptr += vec_off

    dh_kk_ptr += ((b_id * num_chunks + chunk_id) * H + h_id).to(tl.int64) * D * D

    # load
    x_out_ptr = tl.make_block_ptr(
        x_out_ptr,
        (T, D),
        (H * D, 1),
        (chunk_id * chunk_size, 0),
        (chunk_size, D),
        (1, 0),
    )
    x_out = tl.load(x_out_ptr, boundary_check=(0, 1))
    dq_ptr = tl.make_block_ptr(
        dq_ptr, (T, D), (H * D, 1), (chunk_id * chunk_size, 0), (chunk_size, D), (1, 0)
    )
    dq = tl.load(dq_ptr, boundary_check=(0, 1))

    dh_kk_ptr = tl.make_block_ptr(dh_kk_ptr, (D, D), (D, 1), (0, 0), (D, D), (1, 0))
    dh_kk = tl.load(dh_kk_ptr, boundary_check=(0, 1))

    k_ptr = tl.make_block_ptr(
        k_ptr, (T, D), (H * D, 1), (chunk_id * chunk_size, 0), (chunk_size, D), (1, 0)
    )
    k = tl.load(k_ptr, boundary_check=(0, 1))

    mask = tl.trans(
        tl.where(
            tl.arange(0, chunk_size)[:, None] >= tl.arange(0, chunk_size)[None, :], 1, 0
        )
    )

    dk = tl.zeros([chunk_size, D], dtype=tl.float32)

    dk += tl.dot(k, dh_kk + tl.trans(dh_kk)) + tl.dot(
        (tl.dot(k, tl.trans(x_out)) * mask).to(dq.dtype), dq
    )
    dk += tl.dot((tl.dot(k, tl.trans(dq)) * mask).to(x_out.dtype), x_out)
    dk = -dk

    p_dk = tl.make_block_ptr(
        dk_ptr, (T, D), (H * D, 1), (chunk_id * chunk_size, 0), (chunk_size, D), (1, 0)
    )
    tl.store(p_dk, dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))


def chunk_dh_kk_to_dk(
    k: Tensor, x_out: Tensor, dh_kk: Tensor, dq: Tensor, chunk_size: int = 64
) -> Tensor:
    """
    Computes key gradient dk from chunk-level covariance gradient dh_kk.

    Wrapper function that launches the Triton kernel for parallel gradient computation.

    Args:
        k: Key tensor with shape [B, L, H, D]
        x_out: Output from forward pass with shape [B, L, H, D]
        dh_kk: Gradient of chunk-level covariance with shape [B, num_chunks, H, D, D]
        dq: Gradient of query with shape [B, L, H, D]
        chunk_size: Chunk size for parallel processing. Defaults to 64

    Returns:
        dk: Key gradient with shape [B, L, H, D]
    """

    B, T, H, D = k.shape

    dk = torch.empty_like(k)
    grid = (triton.cdiv(T, chunk_size), B * H)

    chunk_dh_kk_to_dk_kernel[grid](
        # inputs:
        k,
        x_out,
        dh_kk,
        dq,
        # output:
        dk,
        # constants:
        T,
        H,
        D,
        chunk_size,
    )

    return dk
