from typing import Optional, Tuple

import torch
import triton
import triton.language as tl


from fla.ops.common.chunk_h import chunk_bwd_dh, chunk_fwd_h
from fla.ops.utils import chunk_local_cumsum
from fla.ops.utils.op import exp

from .chunk_dh_kk_to_dk_backward import chunk_dh_kk_to_dk
from .chunk_dh_kk_to_dk_gating_backward import chunk_dh_kk_to_dk_gating
from .chunk_dk_backward_gating import chunk_bwd_dk_gating
from .chunk_dk_backward import chunk_bwd_dk


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [1, 2, 4, 8]
        for num_stages in [1, 2, 3, 4]
    ],
    key=["D"],
)
@triton.jit(do_not_specialize=["T"])
def chebyshev_iteration_forward_kernel(
    # inputs:
    h_kk_ptr,
    k_ptr,
    q_ptr,
    # output:
    q_output_ptr,
    ridge_strength_ptr,
    rho_sq_4_ptr,
    stepsize_ptr,
    # constants:
    T,
    H: tl.constexpr,
    D: tl.constexpr,
    chunk_size: tl.constexpr,
    # algorithm parameters:
    num_iter: tl.constexpr,
):
    """
    Triton kernel for Chebyshev iteration solving (H_t + λI)·x = q in chunks.

    Grid: (num_chunks, B*H) - one thread block per chunk and batch-head pair
    Block size: Processes [chunk_size, D] vectors per block
    """

    chunk_id, bh_id = tl.program_id(0), tl.program_id(1)
    b_id, h_id = bh_id // H, bh_id % H

    num_chunks = tl.cdiv(T, chunk_size)

    # offsets
    scalar_off = b_id * T * H + h_id
    ridge_strength_ptr += scalar_off
    rho_sq_4_ptr += scalar_off
    stepsize_ptr += scalar_off

    vec_off = scalar_off * D
    q_ptr += vec_off
    q_output_ptr += vec_off
    k_ptr += vec_off

    h_kk_ptr += ((b_id * num_chunks + chunk_id) * H + h_id).to(tl.int64) * D * D

    p_q = tl.make_block_ptr(
        q_ptr, (T, D), (H * D, 1), (chunk_id * chunk_size, 0), (chunk_size, D), (1, 0)
    )
    p_k = tl.make_block_ptr(
        k_ptr, (T, D), (H * D, 1), (chunk_id * chunk_size, 0), (chunk_size, D), (1, 0)
    )
    p_h_kk = tl.make_block_ptr(h_kk_ptr, (D, D), (D, 1), (0, 0), (D, D), (1, 0))

    p_ridge_strength = tl.make_block_ptr(
        ridge_strength_ptr,
        (T, 1),
        (H, 1),
        (chunk_id * chunk_size, 0),
        (chunk_size, 1),
        (1, 0),
    )
    p_rho_sq_4 = tl.make_block_ptr(
        rho_sq_4_ptr,
        (T, 1),
        (H, 1),
        (chunk_id * chunk_size, 0),
        (chunk_size, 1),
        (1, 0),
    )
    p_stepsize = tl.make_block_ptr(
        stepsize_ptr,
        (T, 1),
        (H, 1),
        (chunk_id * chunk_size, 0),
        (chunk_size, 1),
        (1, 0),
    )

    q = tl.load(p_q, boundary_check=(0, 1)).to(tl.float32)
    k = tl.load(p_k, boundary_check=(0, 1)).to(q.dtype)
    h_kk = tl.load(p_h_kk, boundary_check=(0, 1)).to(q.dtype)
    ridge_strength = tl.load(p_ridge_strength, boundary_check=(0, 1)).to(tl.float32)
    rho_sq_4 = tl.load(p_rho_sq_4, boundary_check=(0, 1)).to(tl.float32)
    stepsize = tl.load(p_stepsize, boundary_check=(0, 1)).to(tl.float32)

    normalizer = (
        1 / (tl.arange(1, chunk_size + 1) + chunk_id * chunk_size).to(q.dtype)
    )[
        :, None
    ]  # [chunk_size, 1]

    mask = tl.where(
        tl.arange(0, chunk_size)[:, None] >= tl.arange(0, chunk_size)[None, :],
        normalizer,
        0,
    ).to(q.dtype)

    w = tl.full((chunk_size, 1), 1, dtype=tl.float32)
    x_prev, x = tl.zeros_like(q), stepsize * q
    for i in range(num_iter):
        w = 1.0 / (1.0 - rho_sq_4 * w)

        grad_and_x_prev = (
            stepsize
            * w
            * (
                tl.dot(x * normalizer, h_kk)
                + tl.dot((tl.dot(x, tl.trans(k)) * mask), k)
                + ridge_strength * x
                - q
            )
            + (w - 1) * x_prev
        )  # [C, D]

        x_prev = x

        x = w * x - grad_and_x_prev

    p_q_output = tl.make_block_ptr(
        q_output_ptr,
        (T, D),
        (H * D, 1),
        (chunk_id * chunk_size, 0),
        (chunk_size, D),
        (1, 0),
    )
    tl.store(
        p_q_output,
        x.to(p_q_output.dtype.element_ty, fp_downcast_rounding="rtne"),
        boundary_check=(0, 1),
    )


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [1, 2, 4, 8]
        for num_stages in [1, 2, 3, 4]
    ],
    key=["D"],
)
@triton.jit(do_not_specialize=["T"])
def chebyshev_iteration_fro_forward_kernel(
    # inputs:
    h_kk_ptr,
    k_ptr,
    q_ptr,
    # output:
    x_out_ptr,
    fro_ptr,
    # constants:
    T,
    H: tl.constexpr,
    D: tl.constexpr,
    chunk_size: tl.constexpr,
    # algorithm parameters:
    num_iter: tl.constexpr,
    ridge_ratio: tl.constexpr,
    load_fro: tl.constexpr,
):
    """
    Chebyshev iteration with adaptive ridge strength based on Frobenius norm.

    Computes or loads ||H_t||_F, sets λ = ridge_ratio * ||H_t||_F, then solves
    (H_t + λI)·x = q. Optionally saves the Frobenius norm for reuse.

    Grid: (num_chunks, B*H) - one thread block per chunk and batch-head pair
    Block size: Processes [chunk_size, D] vectors per block
    """
    chunk_id, bh_id = tl.program_id(0), tl.program_id(1)
    b_id, h_id = bh_id // H, bh_id % H

    num_chunks = tl.cdiv(T, chunk_size)

    scalar_off = b_id * T * H + h_id

    vec_off = scalar_off * D
    q_ptr += vec_off
    x_out_ptr += vec_off
    k_ptr += vec_off

    h_kk_ptr += ((b_id * num_chunks + chunk_id) * H + h_id).to(tl.int64) * D * D

    p_q = tl.make_block_ptr(
        q_ptr, (T, D), (H * D, 1), (chunk_id * chunk_size, 0), (chunk_size, D), (1, 0)
    )
    p_k = tl.make_block_ptr(
        k_ptr, (T, D), (H * D, 1), (chunk_id * chunk_size, 0), (chunk_size, D), (1, 0)
    )
    p_h_kk = tl.make_block_ptr(h_kk_ptr, (D, D), (D, 1), (0, 0), (D, D), (1, 0))

    q = tl.load(p_q, boundary_check=(0, 1))
    k = tl.load(p_k, boundary_check=(0, 1))
    h_kk = tl.load(p_h_kk, boundary_check=(0, 1))

    mask = tl.where(
        tl.arange(0, chunk_size)[:, None] >= tl.arange(0, chunk_size)[None, :], 1, 0
    ).to(q.dtype)

    if load_fro == 1:
        fro_ptr += scalar_off
        p_fro = tl.make_block_ptr(
            fro_ptr, (T, 1), (H, 1), (chunk_id * chunk_size, 0), (chunk_size, 1), (1, 0)
        )
        fro = tl.load(p_fro, boundary_check=(0, 1))
    else:
        fro = tl.zeros((chunk_size, 1), dtype=tl.float32)
        h_kk_fro = tl.dot(k, tl.trans(k))  # CxC
        fro += tl.sum(
            tl.dot(mask, (h_kk_fro * h_kk_fro).to(q.dtype)) * mask,
            axis=1,
            keep_dims=True,
        )
        fro = tl.sqrt(
            tl.sum(h_kk * h_kk)
            + 2 * tl.cumsum(tl.sum(k * tl.dot(k.to(h_kk.dtype), h_kk), axis=1))[:, None]
            + fro
        )

    ridge_strength = ridge_ratio * fro

    stepsize = 2 / (2 * ridge_strength + fro)
    rho_sq_4 = fro / (2 * ridge_strength + fro) / 2.0
    rho_sq_4 = rho_sq_4 * rho_sq_4

    x_prev = tl.zeros_like(q).to(tl.float32)
    x = stepsize * q.to(tl.float32)
    omega = tl.full((chunk_size, 1), 2.0, dtype=tl.float32)

    for _ in range(num_iter):
        omega = 1.0 / (1.0 - rho_sq_4 * omega)

        grad_and_x_prev = (
            stepsize
            * omega
            * (
                tl.dot(x.to(h_kk.dtype), h_kk)
                + tl.dot((tl.dot(x.to(k.dtype), tl.trans(k)) * mask).to(k.dtype), k)
                + ridge_strength * x
                - q
            )
            + (omega - 1) * x_prev
        )  # [C, D]
        x_prev = x

        x = omega * x - grad_and_x_prev

    # store results
    p_x_out = tl.make_block_ptr(
        x_out_ptr,
        (T, D),
        (H * D, 1),
        (chunk_id * chunk_size, 0),
        (chunk_size, D),
        (1, 0),
    )
    tl.store(
        p_x_out,
        x.to(p_x_out.dtype.element_ty, fp_downcast_rounding="rtne"),
        boundary_check=(0, 1),
    )

    if load_fro == 0:
        fro_ptr += scalar_off
        p_fro = tl.make_block_ptr(
            fro_ptr, (T, 1), (H, 1), (chunk_id * chunk_size, 0), (chunk_size, 1), (1, 0)
        )
        tl.store(
            p_fro,
            fro.to(p_fro.dtype.element_ty, fp_downcast_rounding="rtne"),
            boundary_check=(0, 1),
        )


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [1, 2, 4, 8]
        for num_stages in [1, 2, 3, 4]
    ],
    key=["D"],
)
@triton.jit(do_not_specialize=["T"])
def chebyshev_iteration_fro_gating_forward_kernel(
    # inputs:
    h_kk_ptr,
    k_ptr,
    q_ptr,
    gkk_ptr,
    # output:
    x_out_ptr,
    fro_ptr,
    # constants:
    T,
    H: tl.constexpr,
    D: tl.constexpr,
    chunk_size: tl.constexpr,
    # algorithm parameters:
    num_iter: tl.constexpr,
    ridge_ratio: tl.constexpr,
    load_fro: tl.constexpr,
):
    """
    Chebyshev iteration with gating and adaptive ridge strength.

    Like chebyshev_iteration_fro_forward_kernel but applies exponential gating
    (exp(gkk)) to key interactions for temporal decay. Solves (H_t + λI)·x = q
    with λ = ridge_ratio * ||H_t||_F.

    Grid: (num_chunks, B*H) - one thread block per chunk and batch-head pair
    Block size: Processes [chunk_size, D] vectors per block
    """
    # h_kk: [b, nc, h, d, d], kqx: [b, t, h, d]
    chunk_id, bh_id = tl.program_id(0), tl.program_id(1)
    b_id, h_id = bh_id // H, bh_id % H

    num_chunks = tl.cdiv(T, chunk_size)

    scalar_off = b_id * T * H + h_id
    gkk_ptr += scalar_off

    vec_off = scalar_off * D
    q_ptr += vec_off
    x_out_ptr += vec_off
    k_ptr += vec_off

    h_kk_ptr += ((b_id * num_chunks + chunk_id) * H + h_id).to(tl.int64) * D * D

    p_gkk = tl.make_block_ptr(
        gkk_ptr, (T,), (H,), (chunk_id * chunk_size,), (chunk_size,), (0,)
    )
    gkk = tl.load(p_gkk, boundary_check=(0,)).to(tl.float32)

    p_q = tl.make_block_ptr(
        q_ptr, (T, D), (H * D, 1), (chunk_id * chunk_size, 0), (chunk_size, D), (1, 0)
    )
    p_k = tl.make_block_ptr(
        k_ptr, (T, D), (H * D, 1), (chunk_id * chunk_size, 0), (chunk_size, D), (1, 0)
    )
    p_h_kk = tl.make_block_ptr(h_kk_ptr, (D, D), (D, 1), (0, 0), (D, D), (1, 0))

    q = tl.load(p_q, boundary_check=(0, 1))
    k = tl.load(p_k, boundary_check=(0, 1))
    h_kk = tl.load(p_h_kk, boundary_check=(0, 1))

    o_t = chunk_id * chunk_size + tl.arange(0, chunk_size)
    m_t = o_t < T

    mask = tl.where(
        (o_t[:, None] >= o_t[None, :]) & (m_t[:, None] & m_t[None, :]),
        exp(gkk[:, None] - gkk[None, :]),
        0,
    ).to(q.dtype)

    exp_gkk = exp(gkk[:, None])

    if load_fro == 1:
        fro_ptr += scalar_off
        p_fro = tl.make_block_ptr(
            fro_ptr, (T, 1), (H, 1), (chunk_id * chunk_size, 0), (chunk_size, 1), (1, 0)
        )
        fro = tl.load(p_fro, boundary_check=(0, 1))
    else:
        fro = tl.zeros((chunk_size, 1), dtype=tl.float32)
        h_kk_fro = tl.dot(k, tl.trans(k))  # CxC
        fro += tl.sum(
            tl.dot(mask, (h_kk_fro * h_kk_fro).to(q.dtype)) * mask,
            axis=1,
            keep_dims=True,
        )

        fro += exp_gkk * (exp_gkk * tl.sum(h_kk * h_kk))
        fro = tl.sqrt(
            fro
            + 2
            * exp_gkk
            * tl.sum(
                mask * tl.sum(k * tl.dot(k.to(h_kk.dtype), h_kk), axis=1)[None, :],
                axis=1,
                keep_dims=True,
            )
        )

    ridge_strength = ridge_ratio * fro

    stepsize = 2 / (2 * ridge_strength + fro)
    rho_sq_4 = fro / (2 * ridge_strength + fro) / 2.0
    rho_sq_4 = rho_sq_4 * rho_sq_4

    x_prev = tl.zeros_like(q).to(tl.float32)
    x = stepsize * q.to(tl.float32)
    omega = tl.full((chunk_size, 1), 2.0, dtype=tl.float32)

    for _ in range(num_iter):
        omega = 1.0 / (1.0 - rho_sq_4 * omega)

        grad_and_x_prev = (
            stepsize
            * omega
            * (
                tl.dot((x * exp_gkk).to(h_kk.dtype), h_kk)
                + tl.dot((tl.dot(x.to(k.dtype), tl.trans(k)) * mask).to(k.dtype), k)
                + ridge_strength * x
                - q
            )
            + (omega - 1) * x_prev
        )
        x_prev = x

        x = omega * x - grad_and_x_prev
    # store results
    p_x_out = tl.make_block_ptr(
        x_out_ptr,
        (T, D),
        (H * D, 1),
        (chunk_id * chunk_size, 0),
        (chunk_size, D),
        (1, 0),
    )
    tl.store(
        p_x_out,
        x.to(p_x_out.dtype.element_ty, fp_downcast_rounding="rtne"),
        boundary_check=(0, 1),
    )

    if load_fro == 0:
        fro_ptr += scalar_off
        p_fro = tl.make_block_ptr(
            fro_ptr, (T, 1), (H, 1), (chunk_id * chunk_size, 0), (chunk_size, 1), (1, 0)
        )
        tl.store(
            p_fro,
            fro.to(p_fro.dtype.element_ty, fp_downcast_rounding="rtne"),
            boundary_check=(0, 1),
        )


def chebyshev_iteration_forward_triton(
    k: torch.Tensor,
    q: torch.Tensor,
    gkk: torch.Tensor,
    fro: torch.Tensor,
    num_iter: int = 30,
    ridge_ratio: float = 0.02,
    chunk_size: int = 64,
    load_fro: int = 0,
    output_final_state: bool = False,
    h0=None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Solves (H_t + λI)·x = q using Chebyshev iteration with chunked processing.

    Computes chunk-level key-key covariance h_kk, then dispatches to appropriate
    Triton kernel (with or without gating) to solve the ridge regression problem.

    Args:
        k: Key tensor with shape [B, L, H, D]
        q: Query tensor with shape [B, L, H, D]
        gkk: Cumulative gating for key-key covariance with shape [B, L, H], or None
        fro: Frobenius norm tensor with shape [B, L, H] (computed if load_fro=0)
        num_iter: Number of Chebyshev iterations. Defaults to 30
        ridge_ratio: Base regularization parameter. Defaults to 0.02
        chunk_size: Chunk size for parallel processing. Defaults to 64
        load_fro: Whether to load (1) or compute (0) Frobenius norm. Defaults to 0
        output_final_state: Whether to return final h_kk state. Defaults to False

    Returns:
        x_out: Solution tensor with shape [B, L, H, D]
        fro: Frobenius norm tensor with shape [B, L, H]
        h_kk: Chunk-level key-key covariance with shape [B, num_chunks, H, D, D]
        h_kk_final: Final state with shape [B, H, D, D], or None if output_final_state=False
    """
    h_kk, h_kk_final = chunk_fwd_h(
        k=k,
        v=k,
        g=gkk,
        gk=None,
        gv=None,
        h0=h0,  # initial_state
        output_final_state=output_final_state,
        states_in_fp32=False,
        cu_seqlens=None,
        chunk_size=chunk_size,
    )

    B, T, H, D = q.shape

    num_chunks = triton.cdiv(T, chunk_size)
    grid = (num_chunks, H * B)

    x_out = torch.empty_like(q)

    if gkk is None:
        chebyshev_iteration_fro_forward_kernel[grid](
            # inputs:
            h_kk,
            k,
            q,
            # output:
            x_out,
            fro,  # omega_last,
            # constants:
            T,
            H,
            D,
            chunk_size,
            # algorithm parameters:
            num_iter,
            ridge_ratio,
            load_fro,
        )
    else:
        chebyshev_iteration_fro_gating_forward_kernel[grid](
            # inputs:
            h_kk,
            k,
            q,
            gkk,
            # output:
            x_out,
            fro,  # omega_last,
            # constants:
            T,
            H,
            D,
            chunk_size,
            # algorithm parameters:
            num_iter,
            ridge_ratio,
            load_fro,
        )

    return x_out, fro, h_kk, h_kk_final


class ChebyshevIteration(torch.autograd.Function):
    """
    Autograd function for Chebyshev iteration solving (H_t + λI)·x = q.

    The forward pass solves the ridge regression problem and computes final key-key state.
    The backward pass reuses the forward solver on gradients for efficient backpropagation.
    """

    @staticmethod
    def forward(
        ctx,
        k: torch.Tensor,
        q: torch.Tensor,
        gk: Optional[torch.Tensor] = None,
        num_iter: int = 30,
        ridge_ratio: float = 0.02,
        bp_lambda: bool = True,
        h0=None,
        chunk_size: int = 64,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, H, D = q.shape
        fro = torch.empty((B, T, H), dtype=torch.float32).to(q.device)
        gkk = chunk_local_cumsum(gk, chunk_size=chunk_size) if gk is not None else None

        x_out, fro, _, h_kk_final = chebyshev_iteration_forward_triton(
            k,
            q,
            gkk,
            fro,
            num_iter=num_iter,
            ridge_ratio=ridge_ratio,
            chunk_size=chunk_size,
            load_fro=0,
            output_final_state=True,
            h0=h0,
        )

        ctx.save_for_backward(k, x_out, fro, gkk, h0)
        ctx.num_iter = num_iter
        ctx.chunk_size = chunk_size
        ctx.ridge_ratio = ridge_ratio
        ctx.bp_lambda = bp_lambda

        return x_out, h_kk_final

    @staticmethod
    def backward(
        ctx, dx_out: torch.Tensor, d_kk_final: Optional[torch.Tensor] = None
    ) -> Tuple[
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        None,
        None,
        None,
        Optional[torch.Tensor],
        None,
    ]:
        k, x_out, fro, gkk, h0 = ctx.saved_tensors
        num_iter = ctx.num_iter
        ridge_ratio = ctx.ridge_ratio
        chunk_size = ctx.chunk_size
        bp_lambda = ctx.bp_lambda

        dq, fro, h_kk, _ = chebyshev_iteration_forward_triton(
            k,
            dx_out,
            gkk,
            fro,
            num_iter=num_iter,
            ridge_ratio=ridge_ratio,
            chunk_size=chunk_size,
            load_fro=1,
            output_final_state=False,
            h0=h0
        )

        h_kk_final = None

        d_kk, dh0 = chunk_bwd_dh(
            q=dq,
            k=k,
            v=k,
            do=x_out,
            h0=h0,
            dht=None,
            g=gkk,
            gk=None,
            gv=None,
            chunk_size=chunk_size,
            scale=1,
        )

        if gkk is None:
            if bp_lambda:
                dk = chunk_bwd_dk(
                    h_kk,
                    d_kk,
                    k,
                    x_out,
                    dq,
                    fro,
                    h_kk_final,
                    chunk_size=chunk_size,
                    ridge_ratio=ridge_ratio,
                )
            else:
                dk = chunk_dh_kk_to_dk(k, x_out, d_kk, dq, ctx.chunk_size)

            dgk = None
        else:  # use gating
            if bp_lambda:
                dk, dgk = chunk_bwd_dk_gating(
                    h_kk,
                    d_kk,
                    k,
                    x_out,
                    dq,
                    fro,
                    gkk,
                    chunk_size=chunk_size,
                    ridge_ratio=ridge_ratio,
                )
            else:
                dk, dgk = chunk_dh_kk_to_dk_gating(
                    k, x_out, h_kk, d_kk, dq, gkk, ctx.chunk_size
                )

        return dk, dq, dgk, None, None, None, dh0, None
