# chebyshev_prefill with GQA support - K uses HKV heads, Q/output/state use H heads.

from typing import Optional, Tuple
import torch
from vllm.triton_utils import tl, triton
from vllm.model_executor.layers.fla.ops.cumsum import chunk_local_cumsum
from vllm.model_executor.layers.fla.ops.index import prepare_chunk_indices, prepare_chunk_offsets
from vllm.model_executor.layers.fla.ops.op import exp


@triton.heuristics({
    'USE_INITIAL_STATE': lambda args: args['h0'] is not None,
    'STORE_FINAL_STATE': lambda args: args['ht'] is not None,
    'USE_BETA': lambda args: args['beta'] is not None,
    'USE_G': lambda args: args['g'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None
})
@triton.autotune(
    configs=[
        triton.Config({'BK': BK}, num_warps=num_warps, num_stages=num_stages)
        for BK in [32, 64]
        for num_warps in [2, 4, 8]
        for num_stages in [2, 3, 4]
    ],
    key=['BT', 'USE_G']
)
@triton.jit(do_not_specialize=['T'])
def chunk_fwd_h_kk_gqa_kernel(
    k,              # [B, T, HKV, D] - K uses HKV heads
    h,              # [total_chunks, H, D, D] - h_kk uses H heads
    g,              # [B, T, H] - cumulative gating uses H heads
    beta,           # [B, T, H] - input selectivity parameter
    h0,             # [N, H, D, D]
    ht,             # [N, H, D, D]
    cu_seqlens,
    chunk_indices,
    chunk_offsets,
    T,
    H: tl.constexpr,
    HKV: tl.constexpr,
    D: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    GQA_GROUP_SIZE: tl.constexpr,
    USE_BETA: tl.constexpr,
    USE_G: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    """Compute h_kk states at chunk boundaries with GQA support."""
    i_k, i_v, i_nh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_n, i_h = i_nh // H, i_nh % H
    i_kv_h = i_h // GQA_GROUP_SIZE  # Map query head to KV head

    if IS_VARLEN:
        bos = tl.load(cu_seqlens + i_n).to(tl.int32)
        eos = tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
        NT = tl.cdiv(T, BT)
        boh = tl.load(chunk_offsets + i_n).to(tl.int32)
    else:
        bos, eos = i_n * T, i_n * T + T
        NT = tl.cdiv(T, BT)
        boh = i_n * NT

    b_h = tl.zeros([BK, BK], dtype=tl.float32)

    if USE_INITIAL_STATE:
        p_h0 = tl.make_block_ptr(h0 + i_nh * D * D, (D, D), (D, 1), (i_k * BK, i_v * BK), (BK, BK), (1, 0))
        b_h = tl.load(p_h0, boundary_check=(0, 1)).to(tl.float32)

    for i_t in range(NT):
        o_h = ((boh + i_t) * H + i_h).to(tl.int64) * D * D
        p_h = tl.make_block_ptr(h + o_h, (D, D), (D, 1), (i_k * BK, i_v * BK), (BK, BK), (1, 0))
        tl.store(p_h, b_h.to(p_h.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))

        # K uses HKV heads
        p_k1 = tl.make_block_ptr(k + (bos * HKV + i_kv_h) * D, (D, T), (1, HKV * D), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
        p_k2 = tl.make_block_ptr(k + (bos * HKV + i_kv_h) * D, (T, D), (HKV * D, 1), (i_t * BT, i_v * BK), (BT, BK), (1, 0))

        b_k1 = tl.load(p_k1, boundary_check=(0, 1))
        b_k2 = tl.load(p_k2, boundary_check=(0, 1))

        last_idx = min((i_t + 1) * BT, T) - 1

        if USE_BETA:
            p_beta = beta + bos * H + (i_t * BT + tl.arange(0, BT)) * H + i_h
            b_beta = tl.load(p_beta, mask=(i_t * BT + tl.arange(0, BT) < T), other=0.).to(tl.float32)
            b_beta = (b_beta + 1e-6)
            b_beta = b_beta * b_beta
            b_k2 = (b_k2 * b_beta[:, None]).to(b_k2.dtype)

        if USE_G:
            # g uses H heads
            b_g_last = tl.load(g + bos * H + last_idx * H + i_h)
            p_g = g + bos * H + (i_t * BT + tl.arange(0, BT)) * H + i_h
            b_g = tl.load(p_g, mask=(i_t * BT + tl.arange(0, BT) < T), other=0.)
            b_h *= exp(b_g_last)
            b_k2 = (b_k2 * exp(b_g_last - b_g)[:, None]).to(b_k2.dtype)

        b_h += tl.dot(b_k1, b_k2)

    if STORE_FINAL_STATE:
        p_ht = tl.make_block_ptr(ht + i_nh * D * D, (D, D), (D, 1), (i_k * BK, i_v * BK), (BK, BK), (1, 0))
        tl.store(p_ht, b_h.to(p_ht.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))


def chunk_fwd_h_kk_gqa(
    k: torch.Tensor,  # [B, T, HKV, D]
    num_heads: int,   # H (query heads)
    g: Optional[torch.Tensor] = None,  # [B, T, H]
    beta: Optional[torch.Tensor] = None,  # [B, T, H]
    h0: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    cu_seqlens: Optional[torch.Tensor] = None,
    chunk_size: int = 64,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Compute h_kk states with GQA support."""
    B, T, HKV, D = k.shape
    H = num_heads
    GQA_GROUP_SIZE = H // HKV
    BT = min(chunk_size, max(16, triton.next_power_of_2(T)))

    if cu_seqlens is None:
        N = B
        NT = triton.cdiv(T, BT)
        total_chunks = B * NT
        chunk_indices = None
        chunk_offsets = None
    else:
        N = len(cu_seqlens) - 1
        chunk_offsets = prepare_chunk_offsets(cu_seqlens, BT)
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
        total_chunks = len(chunk_indices)

    h = k.new_empty(total_chunks, H, D, D)
    ht = k.new_empty(N, H, D, D, dtype=torch.float32) if output_final_state else None

    def grid(meta):
        return (triton.cdiv(D, meta['BK']), triton.cdiv(D, meta['BK']), N * H)

    chunk_fwd_h_kk_gqa_kernel[grid](
        k=k, h=h, g=g, beta=beta, h0=h0, ht=ht,
        cu_seqlens=cu_seqlens, chunk_indices=chunk_indices, chunk_offsets=chunk_offsets,
        T=T, H=H, HKV=HKV, D=D, BT=BT, GQA_GROUP_SIZE=GQA_GROUP_SIZE,
    )
    return h, ht, chunk_offsets

@triton.heuristics({'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
                   'USE_G': lambda args: args['gkk_ptr'] is not None,
                   'USE_BETA': lambda args: args['beta_ptr'] is not None})
@triton.autotune(
    configs=[triton.Config({}, num_warps=nw, num_stages=ns) for nw in [1, 2, 4, 8] for ns in [1, 2, 3, 4]],
    key=['D']
)
@triton.jit(do_not_specialize=['T'])
def chebyshev_iteration_fro_gating_gqa_forward_kernel(
    h_kk_ptr, k_ptr, q_ptr, gkk_ptr, beta_ptr, x_out_ptr, fro_ptr,
    cu_seqlens, chunk_indices, chunk_offsets,
    T, H: tl.constexpr, HKV: tl.constexpr, D: tl.constexpr, BT: tl.constexpr,
    GQA_GROUP_SIZE: tl.constexpr, num_iter: tl.constexpr, ridge_ratio: tl.constexpr,
    IS_VARLEN: tl.constexpr, USE_G: tl.constexpr, USE_BETA: tl.constexpr,
):
    """Chebyshev iteration with GQA (with gating)."""
    i_chunk, i_h = tl.program_id(0), tl.program_id(1)
    i_kv_h = i_h // GQA_GROUP_SIZE

    if IS_VARLEN:
        i_n = tl.load(chunk_indices + i_chunk * 2).to(tl.int32)
        i_t = tl.load(chunk_indices + i_chunk * 2 + 1).to(tl.int32)
        bos = tl.load(cu_seqlens + i_n).to(tl.int32)
        eos = tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
        boh = tl.load(chunk_offsets + i_n).to(tl.int32)
    else:
        num_chunks = tl.cdiv(T, BT)
        i_n = i_chunk // num_chunks
        i_t = i_chunk % num_chunks
        bos = i_n * T
        boh = i_n * num_chunks

    if USE_G:
        # gkk uses H heads
        gkk_ptr += bos * H + i_h

    if USE_BETA:
        # beta uses H heads
        beta_ptr += bos * H + i_h

    # Q/output use H heads
    q_ptr += (bos * H + i_h) * D
    x_out_ptr += (bos * H + i_h) * D
    # K uses HKV heads
    k_ptr += (bos * HKV + i_kv_h) * D
    h_kk_ptr += ((boh + i_t) * H + i_h).to(tl.int64) * D * D

    if USE_G:
        p_gkk = tl.make_block_ptr(gkk_ptr, (T,), (H,), (i_t * BT,), (BT,), (0,))
        gkk = tl.load(p_gkk, boundary_check=(0,)).to(tl.float32)

    if USE_BETA:
        p_beta = tl.make_block_ptr(beta_ptr, (T,), (H,), (i_t * BT,), (BT,), (0,))
        beta = tl.load(p_beta, boundary_check=(0,)).to(tl.float32)

    p_q = tl.make_block_ptr(q_ptr, (T, D), (H * D, 1), (i_t * BT, 0), (BT, D), (1, 0))
    p_k = tl.make_block_ptr(k_ptr, (T, D), (HKV * D, 1), (i_t * BT, 0), (BT, D), (1, 0))
    p_h_kk = tl.make_block_ptr(h_kk_ptr, (D, D), (D, 1), (0, 0), (D, D), (1, 0))

    q = tl.load(p_q, boundary_check=(0, 1))
    k = tl.load(p_k, boundary_check=(0, 1))

    if USE_BETA:
        k = k*(beta[:, None] + 1e-6)

    h_kk = tl.load(p_h_kk, boundary_check=(0, 1))

    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T

    if USE_G:
        mask = tl.where((o_t[:, None] >= o_t[None, :]) & (m_t[:, None] & m_t[None, :]), exp(gkk[:, None] - gkk[None, :]), 0).to(q.dtype)
        exp_gkk = exp(gkk[:, None])
    else:
        mask = tl.where((tl.arange(0, BT)[:, None] >= tl.arange(0, BT)[None, :]) & (m_t[:, None] & m_t[None, :]), 1, 0).to(q.dtype)

    fro = tl.zeros((BT, 1), dtype=tl.float32)
    h_kk_fro = tl.dot(k, tl.trans(k))
    fro += tl.sum(tl.dot(mask, (h_kk_fro * h_kk_fro).to(q.dtype)) * mask, axis=1, keep_dims=True)

    if USE_G:
        fro += exp_gkk * (exp_gkk * tl.sum(h_kk * h_kk))
        fro = tl.sqrt(fro + 2 * exp_gkk * tl.sum(mask * tl.sum(k * tl.dot(k.to(h_kk.dtype), h_kk), axis=1)[None, :], axis=1, keep_dims=True))
    else:
        fro = tl.sqrt(tl.sum(h_kk * h_kk) + 2 * tl.cumsum(tl.sum(k * tl.dot(k.to(h_kk.dtype), h_kk), axis=1))[:, None] + fro)


    ridge_strength = ridge_ratio * fro
    stepsize = 2 / (2 * ridge_strength + fro)
    rho_sq_4 = fro / (2 * ridge_strength + fro) / 2.0
    rho_sq_4 = rho_sq_4 * rho_sq_4

    x_prev = tl.zeros_like(q).to(tl.float32)
    x = stepsize * q.to(tl.float32)
    omega = tl.full((BT, 1), 2.0, dtype=tl.float32)

    for _ in range(num_iter):
        omega = 1.0 / (1.0 - rho_sq_4 * omega)

        if USE_G:
            grad_and_x_prev = stepsize * omega * (
                tl.dot((x * exp_gkk).to(h_kk.dtype), h_kk) +
                tl.dot((tl.dot(x.to(k.dtype), tl.trans(k)) * mask).to(k.dtype), k) +
                ridge_strength * x - q
            ) + (omega - 1) * x_prev
        else:
            grad_and_x_prev = stepsize * omega * (
                tl.dot(x.to(h_kk.dtype), h_kk) +
                tl.dot((tl.dot(x.to(k.dtype), tl.trans(k)) * mask).to(k.dtype), k) +
                ridge_strength * x - q
            ) + (omega - 1) * x_prev
        x_prev = x
        x = omega * x - grad_and_x_prev

    p_x_out = tl.make_block_ptr(x_out_ptr, (T, D), (H * D, 1), (i_t * BT, 0), (BT, D), (1, 0))
    tl.store(p_x_out, x.to(p_x_out.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))

    fro_ptr += bos * H + i_h
    p_fro = tl.make_block_ptr(fro_ptr, (T, 1), (H, 1), (i_t * BT, 0), (BT, 1), (1, 0))
    tl.store(p_fro, fro.to(p_fro.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))


def chebyshev_iteration_prefill_gqa_fwd(
    k: torch.Tensor,  # [B, T, HKV, D]
    q: torch.Tensor,  # [B, T, H, D]
    gk: Optional[torch.Tensor] = None,  # [B, T, H]
    beta: Optional[torch.Tensor] = None,  # [B, T, H]
    h0: Optional[torch.Tensor] = None,
    num_iter: int = 15,
    ridge_ratio: float = 0.1,
    chunk_size: int = 64,
    output_final_state: bool = True,
    cu_seqlens: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Forward pass for Chebyshev iteration with GQA support."""
    B, T, H, D = q.shape
    HKV = k.shape[2]
    GQA_GROUP_SIZE = H // HKV
    BT = min(chunk_size, max(16, triton.next_power_of_2(T)))

    gkk = chunk_local_cumsum(gk, chunk_size=BT, cu_seqlens=cu_seqlens) if gk is not None else None

    h_kk, h_kk_final, chunk_offsets = chunk_fwd_h_kk_gqa(
        k=k, num_heads=H, g=gkk, beta=beta, h0=h0,
        output_final_state=output_final_state, cu_seqlens=cu_seqlens, chunk_size=BT
    )

    if cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
        total_chunks = len(chunk_indices)
    else:
        chunk_indices = None
        total_chunks = B * triton.cdiv(T, BT)

    x_out = torch.empty_like(q)
    fro = torch.empty((B, T, H), dtype=torch.float32, device=q.device)
    grid = (total_chunks, H)

    chebyshev_iteration_fro_gating_gqa_forward_kernel[grid](
            h_kk_ptr=h_kk, k_ptr=k, q_ptr=q, gkk_ptr=gkk, beta_ptr=beta, x_out_ptr=x_out, fro_ptr=fro,
            cu_seqlens=cu_seqlens, chunk_indices=chunk_indices, chunk_offsets=chunk_offsets,
            T=T, H=H, HKV=HKV, D=D, BT=BT, GQA_GROUP_SIZE=GQA_GROUP_SIZE,
            num_iter=num_iter, ridge_ratio=ridge_ratio,
        )

    return x_out, h_kk_final
