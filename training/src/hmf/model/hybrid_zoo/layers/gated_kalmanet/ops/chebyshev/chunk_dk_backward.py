# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

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
def chunk_bwd_dk_kernel(
    # inputs:
    h_kk_ptr,
    d_kk_ptr,
    x_out_ptr,
    dq_ptr,
    k_ptr,
    fro_ptr,
    # outputs:
    h_kk_suffix_sum_partial_ptr,
    dk_ptr,
    # constants:
    T,
    H: tl.constexpr,
    D: tl.constexpr,
    chunk_size: tl.constexpr,
    # algorithm parameters:
    ridge_ratio: tl.constexpr,
):
    """
    Computes key gradients with adaptive ridge strength (no gating).

    First pass of two-pass backward: computes per-chunk gradients and partial suffix sums
    for inter-chunk gradient propagation. Non-gated version of chunk_bwd_dk_gating_kernel.

    Grid: (num_chunks, B*H) - one thread block per chunk and batch-head pair
    Block size: Processes [chunk_size, D] vectors per block
    """

    chunk_id, bh_id = tl.program_id(0), tl.program_id(1)
    b_id, h_id = bh_id // H, bh_id % H

    num_chunks = tl.cdiv(T, chunk_size)

    scalar_off = b_id * T * H + h_id
    fro_ptr += scalar_off

    vec_off = scalar_off * D
    x_out_ptr += vec_off
    dq_ptr += vec_off
    k_ptr += vec_off
    dk_ptr += vec_off

    d_kk_ptr += ((b_id * num_chunks + chunk_id) * H + h_id).to(tl.int64) * D * D
    h_kk_ptr += ((b_id * num_chunks + chunk_id) * H + h_id).to(tl.int64) * D * D
    h_kk_suffix_sum_partial_ptr += (
        ((b_id * num_chunks + chunk_id) * H + h_id).to(tl.int64) * D * D
    )

    p_dq = tl.make_block_ptr(
        dq_ptr, (T, D), (H * D, 1), (chunk_id * chunk_size, 0), (chunk_size, D), (1, 0)
    )
    p_x_out = tl.make_block_ptr(
        x_out_ptr,
        (T, D),
        (H * D, 1),
        (chunk_id * chunk_size, 0),
        (chunk_size, D),
        (1, 0),
    )
    p_k = tl.make_block_ptr(
        k_ptr, (T, D), (H * D, 1), (chunk_id * chunk_size, 0), (chunk_size, D), (1, 0)
    )
    p_fro = tl.make_block_ptr(
        fro_ptr, (T,), (H,), (chunk_id * chunk_size,), (chunk_size,), (0,)
    )

    p_d_kk = tl.make_block_ptr(d_kk_ptr, (D, D), (D, 1), (0, 0), (D, D), (1, 0))
    p_h_kk = tl.make_block_ptr(h_kk_ptr, (D, D), (D, 1), (0, 0), (D, D), (1, 0))

    d_kk = tl.load(p_d_kk, boundary_check=(0, 1))
    h_kk = tl.load(p_h_kk, boundary_check=(0, 1))

    k = tl.load(p_k, boundary_check=(0, 1))
    dq = tl.load(p_dq, boundary_check=(0, 1))
    x_out = tl.load(p_x_out, boundary_check=(0, 1))
    fro = tl.load(p_fro, boundary_check=(0,)).to(tl.float32)

    weights = tl.cumsum(ridge_ratio * tl.sum(dq * x_out, axis=1) / fro, reverse=True)
    k_weight = (weights[:, None] * k).to(k.dtype)

    bool_mask = tl.arange(0, chunk_size)[:, None] <= tl.arange(0, chunk_size)[None, :]
    mask = tl.where(bool_mask, 1, 0)
    dk = tl.zeros([chunk_size, D], dtype=tl.float32)
    dk += tl.dot(k, (d_kk + tl.trans(d_kk)).to(k.dtype)) + tl.dot(
        (tl.dot(k, tl.trans(x_out)) * mask).to(dq.dtype), dq
    )
    dk += tl.dot((tl.dot(k, tl.trans(dq)) * mask).to(x_out.dtype), x_out)

    dk += 2 * (
        tl.dot(
            (
                tl.dot(k, tl.trans(k))
                * tl.where(bool_mask, weights[None, :], weights[:, None])
            ).to(k.dtype),
            k,
        )
        + tl.dot(k_weight, h_kk.to(k.dtype))
    )

    h_kk_suffix_sum_partial = tl.sum(
        tl.where(tl.arange(0, chunk_size) == 0, 1, 0) * weights
    ) * h_kk + tl.dot(tl.trans(k_weight), k)

    p_dk = tl.make_block_ptr(
        dk_ptr, (T, D), (H * D, 1), (chunk_id * chunk_size, 0), (chunk_size, D), (1, 0)
    )
    p_h_kk_suffix_sum_partial = tl.make_block_ptr(
        h_kk_suffix_sum_partial_ptr, (D, D), (D, 1), (0, 0), (D, D), (1, 0)
    )

    tl.store(
        p_h_kk_suffix_sum_partial,
        h_kk_suffix_sum_partial.to(p_h_kk_suffix_sum_partial.dtype.element_ty),
        boundary_check=(0, 1),
    )
    tl.store(p_dk, dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [1, 2, 4, 8]
        for num_stages in [1, 2, 3, 4]
    ],
    key=["D"],
)
@triton.jit(do_not_specialize=["T"])
def chunk_bwd_dk_kernel_forloop(
    # inputs:
    h_kk_suffix_sum_partial_ptr,
    k_ptr,
    dk_ptr,
    # outputs:
    dk_final_ptr,
    # constants:
    T,
    H: tl.constexpr,
    D: tl.constexpr,
    chunk_size: tl.constexpr,
):
    """
    Accumulates suffix sums and computes final key gradients.

    Second pass of two-pass backward: processes chunks sequentially in reverse order,
    accumulating h_kk_suffix_sum to compute final gradients from later timesteps.

    Grid: (B, H) - one thread block per batch-head pair (sequential over chunks)
    Block size: Processes [chunk_size, D] vectors per block
    """
    b_id, h_id = tl.program_id(0), tl.program_id(1)

    num_chunks = tl.cdiv(T, chunk_size)

    scalar_off = b_id * T * H + h_id

    vec_off = scalar_off * D
    k_ptr += vec_off
    dk_ptr += vec_off
    dk_final_ptr += vec_off

    h_kk_suffix_sum = tl.zeros((D, D), dtype=tl.float32)
    for chunk_id in range(num_chunks - 1, -1, -1):
        # load dq and x_out
        p_k = tl.make_block_ptr(
            k_ptr,
            (T, D),
            (H * D, 1),
            (chunk_id * chunk_size, 0),
            (chunk_size, D),
            (1, 0),
        )
        p_dk = tl.make_block_ptr(
            dk_ptr,
            (T, D),
            (H * D, 1),
            (chunk_id * chunk_size, 0),
            (chunk_size, D),
            (1, 0),
        )
        p_h_kk_suffix_sum_partial = tl.make_block_ptr(
            h_kk_suffix_sum_partial_ptr
            + ((b_id * num_chunks + chunk_id) * H + h_id).to(tl.int64) * D * D,
            (D, D),
            (D, 1),
            (0, 0),
            (D, D),
            (1, 0),
        )
        h_kk_suffix_sum_partial = tl.load(
            p_h_kk_suffix_sum_partial, boundary_check=(0, 1)
        )

        k = tl.load(p_k, boundary_check=(0, 1))

        dk = tl.load(p_dk, boundary_check=(0, 1)).to(tl.float32)
        dk += 2 * tl.dot(k, h_kk_suffix_sum.to(k.dtype))
        dk = -dk

        h_kk_suffix_sum += h_kk_suffix_sum_partial

        p_dk_final = tl.make_block_ptr(
            dk_final_ptr,
            (T, D),
            (H * D, 1),
            (chunk_id * chunk_size, 0),
            (chunk_size, D),
            (1, 0),
        )

        tl.store(p_dk_final, dk.to(p_dk_final.dtype.element_ty), boundary_check=(0, 1))


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [1, 2, 4, 8]
        for num_stages in [1, 2, 3, 4]
    ],
    key=["D"],
)
@triton.jit(do_not_specialize=["T"])
def chunk_bwd_dk_kernel_single_pass(
    # inputs:
    h_kk_final_ptr,
    k_ptr,
    x_out_ptr,
    dq_ptr,
    fro_ptr,
    # outputs:
    dk_ptr,
    # constants:
    T,
    H: tl.constexpr,
    D: tl.constexpr,
    chunk_size: tl.constexpr,
    # algorithm parameters:
    ridge_ratio: tl.constexpr,
):
    """
    Single-pass backward for key gradients when final h_kk state is available.

    Processes chunks sequentially in reverse order, avoiding the need for a two-pass
    algorithm by using the final key-key covariance state.

    Grid: (B, H) - one thread block per batch-head pair (sequential over chunks)
    Block size: Processes [chunk_size, D] vectors per block
    """
    b_id, h_id = tl.program_id(0), tl.program_id(1)

    num_chunks = tl.cdiv(T, chunk_size)

    scalar_off = b_id * T * H + h_id
    fro_ptr += scalar_off

    vec_off = scalar_off * D
    k_ptr += vec_off
    x_out_ptr += vec_off
    dq_ptr += vec_off
    dk_ptr += vec_off

    h_kk_final_ptr += (b_id * H + h_id).to(tl.int64) * D * D
    p_h_kk_final = tl.make_block_ptr(
        h_kk_final_ptr, (D, D), (D, 1), (0, 0), (D, D), (1, 0)
    )
    h_kk = tl.load(p_h_kk_final, boundary_check=(0, 1)).to(tl.float32)

    d_kk = tl.zeros((D, D), dtype=tl.float32)

    bool_mask = tl.arange(0, chunk_size)[:, None] <= tl.arange(0, chunk_size)[None, :]
    mask = tl.where(bool_mask, 1, 0)

    h_kk_suffix_sum = tl.zeros_like(h_kk)

    for chunk_id in range(num_chunks - 1, -1, -1):
        # load dq and x_out
        p_dq = tl.make_block_ptr(
            dq_ptr,
            (T, D),
            (H * D, 1),
            (chunk_id * chunk_size, 0),
            (chunk_size, D),
            (1, 0),
        )
        p_x_out = tl.make_block_ptr(
            x_out_ptr,
            (T, D),
            (H * D, 1),
            (chunk_id * chunk_size, 0),
            (chunk_size, D),
            (1, 0),
        )
        p_k = tl.make_block_ptr(
            k_ptr,
            (T, D),
            (H * D, 1),
            (chunk_id * chunk_size, 0),
            (chunk_size, D),
            (1, 0),
        )
        p_fro = tl.make_block_ptr(
            fro_ptr, (T, 1), (H, 1), (chunk_id * chunk_size, 0), (chunk_size, 1), (1, 0)
        )

        dq = tl.load(p_dq, boundary_check=(0, 1))
        x_out = tl.load(p_x_out, boundary_check=(0, 1))
        k = tl.load(p_k, boundary_check=(0, 1))
        fro = tl.reshape(
            tl.load(p_fro, boundary_check=(0, 1)).to(tl.float32), (chunk_size,)
        )

        h_kk -= tl.dot(tl.trans(k), k)

        dk = tl.zeros([chunk_size, D], dtype=tl.float32)
        dk += tl.dot(k, (d_kk + tl.trans(d_kk)).to(k.dtype)) + tl.dot(
            (tl.dot(k, tl.trans(x_out)) * mask).to(dq.dtype), dq
        )
        dk += tl.dot((tl.dot(k, tl.trans(dq)) * mask).to(x_out.dtype), x_out)

        # fro norm
        dk += 2 * tl.dot(k, h_kk_suffix_sum.to(k.dtype))

        weights = ridge_ratio * tl.sum(dq * x_out, axis=1) / fro
        h_kk_suffix_sum += tl.sum(weights) * h_kk

        weights = tl.cumsum(weights, reverse=True)

        k_weight = (weights[:, None] * k).to(k.dtype)
        dk += 2 * (
            tl.dot(
                (
                    tl.dot(k, tl.trans(k))
                    * tl.where(bool_mask, weights[None, :], weights[:, None])
                ).to(k.dtype),
                k,
            )
            + tl.dot(k_weight, h_kk.to(k.dtype))
        )

        h_kk_suffix_sum += tl.dot(tl.trans(k_weight), k)

        d_kk += tl.dot(tl.trans(dq), x_out)

        p_dk = tl.make_block_ptr(
            dk_ptr,
            (T, D),
            (H * D, 1),
            (chunk_id * chunk_size, 0),
            (chunk_size, D),
            (1, 0),
        )
        tl.store(p_dk, -dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))


def chunk_bwd_dk(
    h_kk: Tensor,
    d_kk: Tensor,
    k: Tensor,
    x_out: Tensor,
    dq: Tensor,
    fro: Tensor,
    h_kk_final=None,
    chunk_size: int = 64,
    ridge_ratio: float = 0.02,
) -> Tensor:
    B, T, H, D = k.shape

    num_chunks = triton.cdiv(T, chunk_size)

    dk = torch.empty_like(k)
    if h_kk_final is None:
        grid = (num_chunks, B * H)

        h_kk_suffix_sum_partial = torch.empty_like(h_kk)
        chunk_bwd_dk_kernel[grid](
            # inputs:
            h_kk,
            d_kk,
            x_out,
            dq,
            k,
            fro,
            # outputs:
            h_kk_suffix_sum_partial,
            dk,
            # constants:
            T,
            H,
            D,
            chunk_size,
            ridge_ratio,
        )

        dk_final = torch.empty_like(k)
        chunk_bwd_dk_kernel_forloop[(B, H)](
            # inputs:
            h_kk_suffix_sum_partial,
            k,
            dk,
            # output:
            dk_final,
            # constants:
            T,
            H,
            D,
            chunk_size,
        )
        return dk_final
    else:
        grid = (B, H)
        chunk_bwd_dk_kernel_single_pass[grid](
            # inputs:
            h_kk_final,
            k,
            x_out,
            dq,
            fro,
            # outputs:
            dk,
            # constants:
            T,
            H,
            D,
            chunk_size,
            # algorithm parameters:
            ridge_ratio,
        )
        return dk
