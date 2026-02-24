# Adapted from vLLM's chunk_o.py with native GQA support and beta gating for Simple GLA.

import torch
from vllm.triton_utils import tl, triton
from vllm.model_executor.layers.fla.ops.index import prepare_chunk_indices
from vllm.model_executor.layers.fla.ops.op import exp


@triton.heuristics({
    'USE_G': lambda args: args['g'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
    'USE_BETA': lambda args: args['beta'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({'BK': BK, 'BV': BV}, num_warps=num_warps, num_stages=num_stages)
        for BK in [32, 64]
        for BV in [32, 64]
        for num_warps in [2, 4, 8]
        for num_stages in [2, 3, 4]
    ],
    key=['H', 'K', 'V', 'BT'],
)
@triton.jit(do_not_specialize=['T'])
def chunk_fwd_kernel_o_gqa(
    q, k, v, h, g, beta, o,
    cu_seqlens, chunk_indices,
    scale,
    T,
    H: tl.constexpr,
    HKV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    GQA_GROUP_SIZE: tl.constexpr,
    USE_G: tl.constexpr,
    USE_BETA: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H
    i_kv_h = i_h // GQA_GROUP_SIZE

    if IS_VARLEN:
        i_tg = i_t
        i_n, i_t = (
            tl.load(chunk_indices + i_t * 2).to(tl.int32),
            tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32),
        )
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int32),
            tl.load(cu_seqlens + i_n + 1).to(tl.int32),
        )
        T = eos - bos
        NT = tl.cdiv(T, BT)
    else:
        NT = tl.cdiv(T, BT)
        i_tg = i_b * NT + i_t
        bos, eos = i_b * T, i_b * T + T

    # Q uses H heads
    q += (bos * H + i_h) * K
    # K uses HKV heads
    k += (bos * HKV + i_kv_h) * K
    # V uses HKV heads
    v += (bos * HKV + i_kv_h) * V
    # o uses H heads
    o += (bos * H + i_h) * V
    # h uses H heads
    h += (i_tg * H + i_h).to(tl.int64) * K * V

    b_o = tl.zeros([BT, BV], dtype=tl.float32)
    b_A = tl.zeros([BT, BT], dtype=tl.float32)

    for i_k in range(tl.cdiv(K, BK)):
        p_q = tl.make_block_ptr(q, (T, K), (H * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_k = tl.make_block_ptr(k, (K, T), (1, HKV * K), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
        p_h = tl.make_block_ptr(h, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))

        b_q = tl.load(p_q, boundary_check=(0, 1))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_h = tl.load(p_h, boundary_check=(0, 1))

        b_o += tl.dot(b_q, b_h)
        b_A += tl.dot(b_q, b_k)

    if USE_G:
        g += bos * H + i_h
        p_g = tl.make_block_ptr(g, (T,), (H,), (i_t * BT,), (BT,), (0,))
        b_g = tl.load(p_g, boundary_check=(0,))
        b_o = b_o * exp(b_g)[:, None]
        b_A = b_A * exp(b_g[:, None] - b_g[None, :])

    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T
    m_A = (o_t[:, None] >= o_t[None, :]) & (m_t[:, None] & m_t)
    b_A = tl.where(m_A, b_A, 0)

    p_v = tl.make_block_ptr(v, (T, V), (HKV * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    p_o = tl.make_block_ptr(o, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
    b_v = tl.load(p_v, boundary_check=(0, 1))

    if USE_BETA:
        beta += bos * H + i_h
        p_beta = tl.make_block_ptr(beta, (T,), (H,), (i_t * BT,), (BT,), (0,))
        b_beta = tl.load(p_beta, boundary_check=(0,)).to(tl.float32)
        b_beta = b_beta + 1e-6
        b_beta = b_beta * b_beta  # β²
        b_v = (b_v * b_beta[:, None]).to(b_v.dtype)

    b_o = b_o * scale + tl.dot(b_A.to(b_v.dtype), b_v) * scale
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))


def chunk_fwd_o_gqa(
    q: torch.Tensor,  # [B, T, H, K]
    k: torch.Tensor,  # [B, T, HKV, K]
    v: torch.Tensor,  # [B, T, HKV, V]
    h: torch.Tensor,  # [B, NT, H, K, V]
    g: torch.Tensor | None = None,
    beta: torch.Tensor | None = None,
    scale: float | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
) -> torch.Tensor:
    """
    GQA-native chunked output computation for gated linear attention.

    Combines inter-chunk (Q @ h) and intra-chunk (causal gated Q @ K^T @ V)
    contributions. GQA broadcasting is handled in the kernel (no repeat_interleave).

    Args:
        q: [B, T, H, K] - queries with full head count
        k: [B, T, HKV, K] - keys with KV head count
        v: [B, T, HKV, V] - values with KV head count
        h: [B, NT, H, K, V] - recurrent states per chunk
        g: [B, T, H] - cumulative gating (log-space), optional
        beta: [B, T, H] - input selectivity (squared internally), optional
        scale: attention scale factor (defaults to K^-0.5)
        cu_seqlens: [N+1] - cumulative sequence lengths for variable-length batching
        chunk_size: chunk size for processing (default 64)

    Returns:
        o: [B, T, H, V] - output with full head count
    """
    B, T, H, K = q.shape
    HKV = k.shape[2]
    V = v.shape[-1]
    GQA_GROUP_SIZE = H // HKV
    BT = min(chunk_size, max(16, triton.next_power_of_2(T)))

    chunk_indices = prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    if scale is None:
        scale = K ** -0.5

    o = torch.empty(B, T, H, V, dtype=q.dtype, device=q.device)

    def grid(meta):
        return (triton.cdiv(V, meta['BV']), NT, B * H)

    chunk_fwd_kernel_o_gqa[grid](
        q, k, v, h, g, beta, o,
        cu_seqlens, chunk_indices,
        scale,
        T=T, H=H, HKV=HKV, K=K, V=V, BT=BT, GQA_GROUP_SIZE=GQA_GROUP_SIZE,
    )
    return o
