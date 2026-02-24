# Adapted from vLLM's fused_recurrent.py with native GQA support.
# Eliminates repeat_interleave by computing KV head index in kernel.

import torch
from vllm.triton_utils import tl, triton

from vllm.model_executor.layers.fla.ops.op import exp


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=w, num_stages=s)
        for w in [2, 4, 8]
        for s in [1, 2, 3, 4]
    ],
    key=['K', 'V', 'BK', 'BV']
)
@triton.heuristics({
    "USE_INITIAL_STATE": lambda args: args["h0"] is not None,
    "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    "IS_CONTINUOUS_BATCHING": lambda args: args["ssm_state_indices"] is not None,
    "IS_SPEC_DECODING": lambda args: args["num_accepted_tokens"] is not None,
})
@triton.jit(do_not_specialize=["N", "T"])
def fused_recurrent_gated_delta_rule_gqa_fwd_kernel(
    q, k, v, g, beta, o, h0, ht,
    cu_seqlens, ssm_state_indices, num_accepted_tokens,
    scale,
    N: tl.int64,
    T: tl.int64,
    B: tl.constexpr,
    H: tl.constexpr,      # num query heads
    HKV: tl.constexpr,    # num kv heads
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    GQA_GROUP_SIZE: tl.constexpr,
    stride_init_state_token: tl.constexpr,
    stride_final_state_token: tl.constexpr,
    stride_indices_seq: tl.constexpr,
    stride_indices_tok: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    INPLACE_FINAL_STATE: tl.constexpr,
    IS_BETA_HEADWISE: tl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    IS_CONTINUOUS_BATCHING: tl.constexpr,
    IS_SPEC_DECODING: tl.constexpr,
):
    i_k, i_v, i_nh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_n, i_h = i_nh // H, i_nh % H
    # GQA: map query head to kv head
    i_kv_h = i_h // GQA_GROUP_SIZE

    if IS_VARLEN:
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int64),
            tl.load(cu_seqlens + i_n + 1).to(tl.int64),
        )
        all = T
        T = eos - bos
    else:
        bos, eos = i_n * T, i_n * T + T
        all = B * T

    if T == 0:
        return

    o_k = i_k * BK + tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)

    # Q uses full head count H
    p_q = q + (bos * H + i_h) * K + o_k
    # K, V use kv head count HKV
    p_k = k + (bos * HKV + i_kv_h) * K + o_k
    p_v = v + (bos * HKV + i_kv_h) * V + o_v

    # g and beta indexed by query head (H)
    if IS_BETA_HEADWISE:
        p_beta = beta + (bos * H + i_h) * V + o_v
    else:
        p_beta = beta + bos * H + i_h

    p_g = g + bos * H + i_h

    # output indexed by query head (H)
    p_o = o + ((i_k * all + bos) * H + i_h) * V + o_v

    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_k[:, None] & mask_v[None, :]

    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_INITIAL_STATE:
        if IS_CONTINUOUS_BATCHING:
            if IS_SPEC_DECODING:
                i_t = tl.load(num_accepted_tokens + i_n).to(tl.int64) - 1
            else:
                i_t = 0
            p_h0 = (
                h0
                + tl.load(ssm_state_indices + i_n * stride_indices_seq + i_t).to(tl.int64)
                * stride_init_state_token
            )
        else:
            p_h0 = h0 + bos * H * K * V
        p_h0 = p_h0 + i_h * K * V + o_k[:, None] * V + o_v[None, :]
        b_h += tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)

    for _ in range(0, T):
        b_q = tl.load(p_q, mask=mask_k, other=0).to(tl.float32)
        b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)

        if USE_QK_L2NORM_IN_KERNEL:
            b_q = b_q / tl.sqrt(tl.sum(b_q * b_q) + 1e-6)
            b_k = b_k / tl.sqrt(tl.sum(b_k * b_k) + 1e-6)
        b_q = b_q * scale

        b_g = tl.load(p_g).to(tl.float32)
        b_h *= exp(b_g)

        b_v -= tl.sum(b_h * b_k[:, None], 0)
        if IS_BETA_HEADWISE:
            b_beta = tl.load(p_beta, mask=mask_v, other=0).to(tl.float32)
        else:
            b_beta = tl.load(p_beta).to(tl.float32)
        b_v *= b_beta

        b_h += b_k[:, None] * b_v[None, :]
        b_o = tl.sum(b_h * b_q[:, None], 0)
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)

        if INPLACE_FINAL_STATE:
            p_ht = (
                ht
                + tl.load(ssm_state_indices + i_n * stride_indices_seq).to(tl.int64)
                * stride_final_state_token
            )
        else:
            p_ht = ht + bos * stride_final_state_token
        p_ht = p_ht + i_h * K * V + o_k[:, None] * V + o_v[None, :]
        tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_h)

        # Advance pointers
        p_q += H * K
        p_k += HKV * K
        p_o += H * V
        p_v += HKV * V
        p_g += H
        p_beta += H * (V if IS_BETA_HEADWISE else 1)
        if INPLACE_FINAL_STATE:
            bos += 1  # for state index lookup


def fused_recurrent_gated_delta_rule_gqa_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    inplace_final_state: bool = True,
    cu_seqlens: torch.LongTensor | None = None,
    ssm_state_indices: torch.Tensor | None = None,
    num_accepted_tokens: torch.Tensor | None = None,
    use_qk_l2norm_in_kernel: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    GQA-native fused recurrent forward.

    Args:
        q: [B, T, H, K] - query with full head count
        k: [B, T, HKV, K] - key with kv head count (NOT expanded)
        v: [B, T, HKV, V] - value with kv head count (NOT expanded)
        g: [B, T, H] - gating with full head count
        beta: [B, T, H] or [B, T, H, V] - beta with full head count
    """
    B, T, H, K = q.shape
    HKV = k.shape[2]
    V = v.shape[-1]
    GQA_GROUP_SIZE = H // HKV

    N = B if cu_seqlens is None else len(cu_seqlens) - 1
    BK, BV = triton.next_power_of_2(K), min(triton.next_power_of_2(V), 32)
    NK, NV = triton.cdiv(K, BK), triton.cdiv(V, BV)
    assert NK == 1, "NK > 1 is not supported yet"

    o = q.new_empty(NK, B, T, H, V)
    if inplace_final_state:
        final_state = initial_state
    else:
        final_state = q.new_empty(B, T, H, K, V, dtype=initial_state.dtype)

    stride_init_state_token = initial_state.stride(0)
    stride_final_state_token = final_state.stride(0)

    if ssm_state_indices is None:
        stride_indices_seq, stride_indices_tok = 1, 1
    elif ssm_state_indices.ndim == 1:
        stride_indices_seq, stride_indices_tok = ssm_state_indices.stride(0), 1
    else:
        stride_indices_seq, stride_indices_tok = ssm_state_indices.stride()

    grid = (NK, NV, N * H)
    fused_recurrent_gated_delta_rule_gqa_fwd_kernel[grid](
        q=q, k=k, v=v, g=g, beta=beta, o=o, h0=initial_state, ht=final_state,
        cu_seqlens=cu_seqlens,
        ssm_state_indices=ssm_state_indices,
        num_accepted_tokens=num_accepted_tokens,
        scale=scale,
        N=N, T=T, B=B, H=H, HKV=HKV, K=K, V=V, BK=BK, BV=BV,
        GQA_GROUP_SIZE=GQA_GROUP_SIZE,
        stride_init_state_token=stride_init_state_token,
        stride_final_state_token=stride_final_state_token,
        stride_indices_seq=stride_indices_seq,
        stride_indices_tok=stride_indices_tok,
        IS_BETA_HEADWISE=beta.ndim == q.ndim,
        USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
        INPLACE_FINAL_STATE=inplace_final_state,
    )
    return o.squeeze(0), final_state


def fused_recurrent_gated_delta_rule_gqa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor = None,
    scale: float = None,
    initial_state: torch.Tensor = None,
    inplace_final_state: bool = True,
    cu_seqlens: torch.LongTensor | None = None,
    ssm_state_indices: torch.Tensor | None = None,
    num_accepted_tokens: torch.Tensor | None = None,
    use_qk_l2norm_in_kernel: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    GQA-native fused recurrent gated delta rule.

    Unlike the standard version, k and v should NOT be expanded via repeat_interleave.
    The kernel handles GQA broadcasting internally.

    Args:
        q: [B, T, H, K] - queries with full head count
        k: [B, T, HKV, K] - keys with kv head count
        v: [B, T, HKV, V] - values with kv head count
        g: [B, T, H] - gating
        beta: [B, T, H] - betas
    """
    if cu_seqlens is not None and q.shape[0] != 1:
        raise ValueError(
            f"Batch size expected to be 1 with cu_seqlens, got {q.shape[0]}"
        )
    if scale is None:
        scale = k.shape[-1] ** -0.5
    if beta is None:
        beta = torch.ones_like(g)

    o, final_state = fused_recurrent_gated_delta_rule_gqa_fwd(
        q=q.contiguous(),
        k=k.contiguous(),
        v=v.contiguous(),
        g=g.contiguous(),
        beta=beta.contiguous(),
        scale=scale,
        initial_state=initial_state,
        inplace_final_state=inplace_final_state,
        cu_seqlens=cu_seqlens,
        ssm_state_indices=ssm_state_indices,
        num_accepted_tokens=num_accepted_tokens,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
    )
    return o, final_state
