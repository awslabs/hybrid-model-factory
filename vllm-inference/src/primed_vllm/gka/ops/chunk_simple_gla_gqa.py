# Adapted from FLA's chunk_simple_gla to vLLM, using vLLM's chunk_delta_h.py as reference.
# vLLM does not have a chunk_simple_gla implementation; added native GQA support.

from typing import Optional, Tuple
import torch
from vllm.triton_utils import tl, triton
from vllm.model_executor.layers.fla.ops.cumsum import chunk_local_cumsum
from vllm.model_executor.layers.fla.ops.index import prepare_chunk_offsets
from vllm.model_executor.layers.fla.ops.utils import input_guard
from vllm.model_executor.layers.fla.ops.op import exp

from .chunk_o_gqa import chunk_fwd_o_gqa


@triton.heuristics({
    'USE_INITIAL_STATE': lambda args: args['h0'] is not None,
    'STORE_FINAL_STATE': lambda args: args['ht'] is not None,
    'USE_G': lambda args: args['g'] is not None,
    'USE_BETA': lambda args: args['beta'] is not None,
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None
})
@triton.autotune(
    configs=[
        triton.Config({'BK': BK, 'BV': BV}, num_warps=num_warps, num_stages=num_stages)
        for BK in [32, 64]
        for BV in [32, 64]
        for num_warps in [2, 4, 8]
        for num_stages in [2, 3, 4]
    ],
    key=['BT', 'USE_G']
)
@triton.jit(do_not_specialize=['T'])
def chunk_simple_gla_gqa_fwd_kernel_h(
    k, v, h, g, beta, h0, ht,
    cu_seqlens, chunk_offsets,
    T, H: tl.constexpr, HKV: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
    BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
    GQA_GROUP_SIZE: tl.constexpr,
    USE_G: tl.constexpr, USE_BETA: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr, IS_VARLEN: tl.constexpr,
):
    """Compute hidden states for Simple GLA with GQA support."""
    i_k, i_v, i_nh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_n, i_h = i_nh // H, i_nh % H
    i_kv_h = i_h // GQA_GROUP_SIZE

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

    b_h = tl.zeros([BK, BV], dtype=tl.float32)

    if USE_INITIAL_STATE:
        p_h0 = tl.make_block_ptr(h0 + i_nh * K * V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        b_h = tl.load(p_h0, boundary_check=(0, 1)).to(tl.float32)

    for i_t in range(NT):
        # Store h at chunk boundary (h uses H heads)
        o_h = ((boh + i_t) * H + i_h).to(tl.int64) * K * V
        p_h = tl.make_block_ptr(h + o_h, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        tl.store(p_h, b_h.to(p_h.dtype.element_ty), boundary_check=(0, 1))

        # K uses HKV heads
        p_k = tl.make_block_ptr(k + (bos * HKV + i_kv_h) * K, (K, T), (1, HKV * K), (i_k * BK, i_t * BT), (BK, BT), (0, 1))
        # V uses HKV heads
        p_v = tl.make_block_ptr(v + (bos * HKV + i_kv_h) * V, (T, V), (HKV * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))

        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_v = tl.load(p_v, boundary_check=(0, 1))

        last_idx = min((i_t + 1) * BT, T) - 1

        if USE_BETA:
            # beta uses H heads
            p_beta = beta + bos * H + (i_t * BT + tl.arange(0, BT)) * H + i_h
            b_beta = tl.load(p_beta, mask=(i_t * BT + tl.arange(0, BT) < T), other=0.).to(tl.float32)
            b_beta = (b_beta + 1e-6)
            b_beta = b_beta * b_beta
            b_v = (b_v * b_beta[:, None]).to(b_v.dtype)

        if USE_G:
            # g uses H heads
            b_g_last = tl.load(g + bos * H + last_idx * H + i_h)
            p_g = g + bos * H + (i_t * BT + tl.arange(0, BT)) * H + i_h
            b_g = tl.load(p_g, mask=(i_t * BT + tl.arange(0, BT) < T), other=0.)
            b_h *= exp(b_g_last)
            b_v = (b_v * exp(b_g_last - b_g)[:, None]).to(b_v.dtype)

        b_h += tl.dot(b_k, b_v)

    if STORE_FINAL_STATE:
        p_ht = tl.make_block_ptr(ht + i_nh * K * V, (K, V), (V, 1), (i_k * BK, i_v * BV), (BK, BV), (1, 0))
        tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), boundary_check=(0, 1))


def chunk_simple_gla_gqa_fwd(
    q: torch.Tensor,  # [B, T, H, K]
    k: torch.Tensor,  # [B, T, HKV, K]
    v: torch.Tensor,  # [B, T, HKV, V]
    g: Optional[torch.Tensor] = None,  # [B, T, H]
    beta: Optional[torch.Tensor] = None,  # [B, T, H]
    scale: Optional[float] = None,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    cu_seqlens: Optional[torch.Tensor] = None,
    chunk_size: int = 64,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Forward pass for chunked Simple GLA with GQA support."""
    B, T, H, K = q.shape
    HKV = k.shape[2]
    V = v.shape[-1]
    GQA_GROUP_SIZE = H // HKV
    BT = min(chunk_size, max(16, triton.next_power_of_2(T)))

    if scale is None:
        scale = K ** -0.5

    if cu_seqlens is None:
        N, NT = B, triton.cdiv(T, BT)
        chunk_offsets = None
    else:
        chunk_offsets = prepare_chunk_offsets(cu_seqlens, BT)
        N = len(cu_seqlens) - 1
        NT = chunk_offsets[-1].item()

    # Compute h states
    h = k.new_empty(B, NT, H, K, V)
    ht = k.new_empty(N, H, K, V, dtype=torch.float32) if output_final_state else None

    def grid_h(meta):
        return (triton.cdiv(K, meta['BK']), triton.cdiv(V, meta['BV']), N * H)

    chunk_simple_gla_gqa_fwd_kernel_h[grid_h](
        k=k, v=v, h=h, g=g, beta=beta, h0=initial_state, ht=ht,
        cu_seqlens=cu_seqlens, chunk_offsets=chunk_offsets,
        T=T, H=H, HKV=HKV, K=K, V=V, BT=BT, GQA_GROUP_SIZE=GQA_GROUP_SIZE,
    )

    # Compute output using chunk_fwd_o_gqa
    o = chunk_fwd_o_gqa(
        q=q, k=k, v=v, h=h, g=g, beta=beta,
        scale=scale, cu_seqlens=cu_seqlens, chunk_size=BT,
    )

    return o, ht


class ChunkSimpleGLAGQAFunction(torch.autograd.Function):
    @staticmethod
    @input_guard
    @torch.amp.custom_fwd(device_type='cuda')
    def forward(ctx, q, k, v, g, beta, scale, initial_state, output_final_state, cu_seqlens):
        T = q.shape[1]
        chunk_size = min(64, max(16, triton.next_power_of_2(T)))
        if g is not None:
            g = chunk_local_cumsum(g, chunk_size=chunk_size, cu_seqlens=cu_seqlens)
        o, ht = chunk_simple_gla_gqa_fwd(
            q=q, k=k, v=v, g=g, beta=beta, scale=scale,
            initial_state=initial_state, output_final_state=output_final_state,
            cu_seqlens=cu_seqlens, chunk_size=chunk_size
        )
        return o.to(q.dtype), ht


@torch.compiler.disable
def chunk_simple_gla_gqa(
    q: torch.Tensor,  # [B, T, H, K]
    k: torch.Tensor,  # [B, T, HKV, K]
    v: torch.Tensor,  # [B, T, HKV, V]
    g: Optional[torch.Tensor] = None,  # [B, T, H]
    beta: Optional[torch.Tensor] = None,  # [B, T, H]
    scale: Optional[float] = None,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    cu_seqlens: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Chunked Simple GLA with native GQA support."""
    if scale is None:
        scale = k.shape[-1] ** -0.5
    return ChunkSimpleGLAGQAFunction.apply(
        q, k, v, g, beta, scale, initial_state, output_final_state, cu_seqlens
    )
