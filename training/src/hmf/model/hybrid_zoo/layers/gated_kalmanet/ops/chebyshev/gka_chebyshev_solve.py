"""
GKA Chebyshev Solver Implementation.

This module implements the core solving routines for the GKA layer, which uses
ridge regression over the entire sequence history to compute optimal states.

Core Algorithm:
    The GKA layer finds an optimal state matrix, S_t, at each timestep that minimizes
    a ridge regression objective over the entire history:

        S_t = argmin_S { λ·||S||²_F + Σ(i=1 to t) η_i·||S·k_i - v_i||² }

    where:
    - S ∈ ℝ^(D×D) is a linear map that should satisfy S·k_i ≈ v_i for all past tokens
    - λ is the ridge regularization strength (prevents overfitting, ensures stability)
    - η_i are exponentially decaying weights controlled by forgetting gates

    Unlike standard SSMs that update their state based only on the current token, this
    formulation considers the entire past, making it optimal in the Kalman Filter sense.

Mathematical Derivation:
    The optimal state has closed form:

        S_t = U_t · (H_t + λI)^(-1)

    where:
        U_t = Σ(i=1 to t) η_i v_i k_i^T    (cumulative value-key outer products)
        H_t = Σ(i=1 to t) η_i k_i k_i^T    (cumulative key-key covariance, Gram matrix)

    The output is computed as:

        y_t = S_t · q_t = U_t · (H_t + λI)^(-1) · q_t = U_t · x_t

    where x_t solves the ridge regression problem:

        (H_t + λI) · x = q_t

    This x_t represents q_t transformed by the inverse covariance of all past keys,
    accounting for correlations between keys rather than just individual similarities.

Ridge Regression Solver:
    The module uses Chebyshev Iteration as its solver, an iterative method using
    accelerated gradient descent with optimal convergence properties. Preferred
    for numerical stability in low-precision (bfloat16) arithmetic.

    Two modes are provided:
    - gka_chebyshev_gla: Parallel processing for entire sequences (training)
    - torch_decoding_one_step: Single-step processing with state management (inference)

Shape Notation:
    We use the following variables for shape hints:
        B: Batch size
        L: Sequence length
        H: Number of attention heads
        D: Head dimension
"""

from typing import Tuple

import torch
from torch import Tensor

import triton
import triton.language as tl

from fla.ops.simple_gla import (
    chunk_simple_gla,
    fused_recurrent_simple_gla,
)
from fla.ops.utils.op import exp

from .chebyshev_iteration import ChebyshevIteration

@torch.compiler.disable
def latent_chebyshev(
    k, q, ridge_strength, num_iter, solver_type, bp_lambda, gk, chunk_size
):
    """
    Dispatcher for ridge regression solvers: (H_t + λI)·x_t = q_t.

    Routes to different solver implementations based on solver_type. All solvers
    aim to solve the same problem but with different speed/accuracy tradeoffs.

    Args:
        k: Key tensor with shape [B, L, H, D]
        q: Query tensor with shape [B, L, H, D]
        ridge_strength: Regularization parameter λ
        num_iter: Number of iterations for iterative solvers
        solver_type: Solver method. Currently only supports 'chebyshev'.
        bp_lambda: Whether to backprop through λ (for Chebyshev solvers)
        gk: Forgetting gate for key-key covariance with shape [B, L, H]

    Returns:
        x: Solution tensor with shape [B, L, H, D]
        kk_final: Final key-key covariance H_T with shape [B, H, D, D]
    """
    if solver_type == "chebyshev":
        x, kk_final = ChebyshevIteration.apply(
            k, q, gk, num_iter, ridge_strength, bp_lambda, None, chunk_size
        )
    else:
        raise ValueError(f"Unknown GKA solver type: {solver_type}")

    return x, kk_final


def gka_chebyshev_gla(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    alpha: Tensor,
    g: Tensor,
    gk: Tensor,
    gla_scale,
    ridge_strength: float = 0.02,
    solver_type: str = "chebyshev",
    num_iter: int = 30,
    bp_lambda: bool = True,
    chunk_size: int = 64,
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    GKA Chebyshev solver for parallel training.

    Solves the ridge regression problem (H_t + λI)·x_t = q_t at each timestep,
    then computes output via GLA recurrence. Processes entire sequences in parallel.

    Args:
        q: Query tensor with shape [B, L, H, D]
        k: Key tensor with shape [B, L, H, D]
        v: Value tensor with shape [B, L, H, D]
        alpha: Mixing coefficient with shape  [B, L, H]. Controls interpolation: q_new = q + α·(x_t - q)
        g: Forgetting gate for key-value state with shape [B, L, H]
        gk: Forgetting gate for key-key covariance with shape [B, L, H]
        gla_scale: Scaling factor for GLA, typically head_dim^(-0.5)
        ridge_strength: Regularization parameter λ. Defaults to 0.02
        solver_type: Solver method. Currently only supports 'chebyshev' (Triton). Defaults to 'chebyshev'.
        num_iter: Iterations for iterative solvers. Set to 0 to skip ridge regression. Defaults to 30.
        bp_lambda: Whether to backprop through λ. Default True

    Returns:
        o: Output tensor with shape [B, L, H, D]
        kk_final: Final key-key covariance (H_T in the regression problem) with shape
            [B, H, D, D], or None if num_iter=0.
        kv_final: Final key-value state from GLA with shape [B, H, D, D]

    Notes:
        For single-step autoregressive inference, use torch_decoding_one_step instead.
    """

    if num_iter > 0:
        # q_ solves (H_t + λI) · q_ = q_t
        q_, kk_final = latent_chebyshev(
            k, q, ridge_strength, num_iter, solver_type, bp_lambda, gk, chunk_size
        )

        q = q + alpha[..., None] * (q_ - q)  # Equivalent to ⍺q_ + (1-⍺)q
    else:
        kk_final = None

    o, kv_final = chunk_simple_gla(
        q=q, k=k, v=v, scale=gla_scale, g=g, output_final_state=True
    )

    return o, kk_final, kv_final


@triton.heuristics({"USE_GK": lambda args: args["gk_ptr"] is not None})
@triton.jit(do_not_specialize=["T"])
def chebyshev_iteration_fro_inference_forward_kernel(
    # inputs:
    prev_h_kk_ptr,
    k_ptr,
    q_ptr,
    gk_ptr,
    # output:
    x_out_ptr,
    h_kk_ptr,
    # constants:
    T,
    H: tl.constexpr,
    D: tl.constexpr,
    # algorithm parameters:
    num_iter: tl.constexpr,
    ridge_ratio: tl.constexpr,
    USE_GK: tl.constexpr,
):
    """
    Triton kernel for single-step Chebyshev iteration during inference.

    Updates key-key covariance h_kk from previous state, then solves
    (h_kk + λI)·x = q using Chebyshev iteration. Optimized for autoregressive
    decoding where one token is processed at a time.

    Grid: (B, H) - one thread block per batch element and head
    Block size: Processes full D-dimensional vectors per block
    """
    b_id, h_id = tl.program_id(0), tl.program_id(1)

    scalar_off = b_id * T * H + h_id

    range_D = tl.arange(0, D)

    vec_off = scalar_off * D
    q_ptr += vec_off + range_D
    k_ptr += vec_off + range_D
    x_out_ptr += vec_off + range_D

    q = tl.load(q_ptr)
    k = tl.load(k_ptr)

    prev_h_kk_ptr += (b_id * H + h_id).to(tl.int64) * D * D

    p_prev_h_kk = tl.make_block_ptr(
        prev_h_kk_ptr, (D, D), (D, 1), (0, 0), (D, D), (1, 0)
    )
    h_kk = tl.load(p_prev_h_kk, boundary_check=(0, 1)).to(tl.float32)

    if USE_GK:
        gk_ptr += scalar_off
        gk = tl.load(gk_ptr).to(tl.float32)
        gk_exp = exp(gk)
    else:
        gk_exp = 1.0

    h_kk_ = h_kk * gk_exp + k[:, None] * k[None, :]
    fro = tl.sqrt(tl.sum(h_kk_ * h_kk_))

    ridge_strength = ridge_ratio * fro

    stepsize = 2 / (2 * ridge_strength + fro)
    rho_sq_4 = fro / (2 * ridge_strength + fro) / 2.0
    rho_sq_4 = rho_sq_4 * rho_sq_4

    x_prev = tl.zeros_like(q).to(tl.float32)
    x = stepsize * q.to(tl.float32)
    omega = 2.0

    for _ in range(num_iter):
        omega = 1.0 / (1.0 - rho_sq_4 * omega)

        grad_and_x_prev = (
            stepsize
            * omega
            * (
                tl.sum(h_kk * (x * gk_exp)[:, None], axis=0)
                + tl.sum(k[:, None] * k[None, :] * x[:, None], axis=0)
                + ridge_strength * x
                - q
            )
            + (omega - 1) * x_prev
        )

        x_prev = x

        x = omega * x - grad_and_x_prev

    tl.store(x_out_ptr, x.to(x_out_ptr.dtype.element_ty))

    h_kk_ptr += (b_id * H + h_id).to(tl.int64) * D * D

    p_h_kk = tl.make_block_ptr(h_kk_ptr, (D, D), (D, 1), (0, 0), (D, D), (1, 0))
    tl.store(
        p_h_kk,
        h_kk_.to(p_h_kk.dtype.element_ty, fp_downcast_rounding="rtne"),
        boundary_check=(0, 1),
    )


def chebyshev_iteration_forward_triton_inference(
    prev_h_kk: torch.Tensor,
    k: torch.Tensor,
    q: torch.Tensor,
    gk: torch.Tensor,
    num_iter: int = 30,
    ridge_ratio: float = 0.1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Solves (h_kk + λI)·x = q for single-step inference using Triton-optimized Chebyshev iteration.

    Updates the key-key covariance state with the current key, then solves the ridge
    regression problem. Designed for autoregressive decoding (one token at a time).

    Args:
        prev_h_kk: Previous key-key covariance state with shape [B, H, D, D]
        k: Current key vector with shape [B, 1, H, D]
        q: Current query vector with shape [B, 1, H, D]
        gk: Forgetting gate for key-key covariance with shape [B, 1, H]
        num_iter: Number of Chebyshev iterations. Defaults to 30
        ridge_ratio: Base regularization parameter (scaled by ||h_kk||_F). Defaults to 0.1

    Returns:
        x_out: Solution vector with shape [B, 1, H, D]
        h_kk: Updated key-key covariance state with shape [B, H, D, D]
    """
    B, L, H, D = q.shape

    h_kk = torch.empty_like(prev_h_kk)

    grid = (B, H)

    x_out = torch.empty_like(q)

    chebyshev_iteration_fro_inference_forward_kernel[grid](
        # inputs:
        prev_h_kk,
        k,
        q,
        gk,
        # output:
        x_out,
        h_kk,
        # constants:
        L,
        H,
        D,
        # algorithm parameters:
        num_iter,
        ridge_ratio,
    )

    return x_out, h_kk


def torch_decoding_one_step(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    alpha: torch.Tensor,
    g: torch.Tensor,
    gk: torch.Tensor,
    gla_scale: float,
    ridge_strength: float,
    solver_type: str,
    num_iter: int,
    prev_h_kk: torch.Tensor,
    prev_h_kv: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Single-step GKA inference with state management for autoregressive decoding.

    Updates key-key covariance state, solves ridge regression, applies alpha mixing,
    and computes output via GLA. Processes one token at a time with explicit state passing.

    Args:
        q: Query tensor with shape [B, 1, H, D] or [1, H, D]
        k: Key tensor with shape [B, 1, H, D] or [1, H, D]
        v: Value tensor with shape [B, 1, H, D] or [1, H, D]
        alpha: Mixing coefficient with shape [B, 1, H] or [1, H]
        g: Forgetting gate for key-value state with shape [B, 1, H] or [1, H]
        gk: Forgetting gate for key-key covariance with shape [B, 1, H] or [1, H]
        gla_scale: Scaling factor for GLA, typically head_dim^(-0.5)
        ridge_strength: Regularization parameter λ
        solver_type: Solver method. Currently only supports 'chebyshev'.
        num_iter: Iterations for iterative solvers. Set to 0 to skip ridge regression
        prev_h_kk: Previous key-key covariance state with shape [B, H, D, D] or None
        prev_h_kv: Previous key-value state with shape [B, H, D, D] or None

    Returns:
        o: Output tensor with shape [B, 1, H, D]
        h_kk: Updated key-key covariance state with shape [B, H, D, D]
        h_kv: Updated key-value state with shape [B, H, D, D]

    Notes:
        Handles both batched [B, 1, H, D] and unbatched [1, H, D] inputs.
        For parallel training, use gka_chebyshev_gla instead.
    """

    if len(q.shape) < 4:
        q = q[None, ...]
        k = k[None, ...]
        v = v[None, ...]

    q_ = q

    if num_iter > 0:
        if solver_type == "chebyshev":
            q_, h_kk = chebyshev_iteration_forward_triton_inference(
                prev_h_kk, k, q_, gk, num_iter=num_iter, ridge_ratio=ridge_strength
            )
        else:
            raise ValueError(f"Unknown GKA solver type: {solver_type}")

        q = q + alpha[..., None] * (q_ - q)
    else:
        h_kk = None

    o, h_kv = fused_recurrent_simple_gla(
        q=q,
        k=k,
        v=v,
        scale=gla_scale,
        g=g,
        initial_state=prev_h_kv,
        output_final_state=True,
    )

    return o, h_kk, h_kv
