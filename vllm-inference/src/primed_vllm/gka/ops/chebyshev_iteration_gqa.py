# chebyshev_decode with GQA support - K uses HKV heads, Q/output/state use H heads.

import torch
from vllm.triton_utils import tl, triton, tldevice


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=w, num_stages=num_stages)
        for w in [2, 4, 8]
        for num_stages in [1, 2, 3, 4]
    ],
    key=['D', 'V', 'num_iter']
)
@triton.heuristics({
    "USE_INITIAL_STATE": lambda args: args["prev_h_kk"] is not None,
    "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    "IS_CONTINUOUS_BATCHING": lambda args: args["ssm_state_indices"] is not None,
    "USE_GK": lambda args: args["gk"] is not None,
})
@triton.jit(do_not_specialize=["N", "T"])
def chebyshev_iteration_gqa_fwd_kernel(
        prev_h_kk, prev_h_kv,
        k, q, v, beta, alpha, gk,
        out, h_kk_out, h_kv_out,
        cu_seqlens, ssm_state_indices, scale,
        N: tl.int64, T: tl.int64,
        B: tl.constexpr, H: tl.constexpr, HKV: tl.constexpr,
        D: tl.constexpr, V: tl.constexpr,
        GQA_GROUP_SIZE: tl.constexpr,
        stride_init_state_seq_kk: tl.constexpr, stride_final_state_seq_kk: tl.constexpr,
        stride_init_state_seq_kv: tl.constexpr, stride_final_state_seq_kv: tl.constexpr,
        stride_indices_seq: tl.constexpr,
        num_iter: tl.constexpr, ridge_ratio: tl.constexpr, USE_BETA: tl.constexpr,
        USE_INITIAL_STATE: tl.constexpr, STORE_FINAL_STATE: tl.constexpr,
        INPLACE_FINAL_STATE: tl.constexpr, IS_VARLEN: tl.constexpr,
        IS_CONTINUOUS_BATCHING: tl.constexpr, USE_GK: tl.constexpr,
):
    i_nh = tl.program_id(0)
    i_n, i_h = i_nh // H, i_nh % H
    i_kv_h = i_h // GQA_GROUP_SIZE  # Map query head to KV head

    if IS_VARLEN:
        bos = tl.load(cu_seqlens + i_n).to(tl.int64)
        eos = tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        seq_len = eos - bos
    else:
        bos = i_n * T
        seq_len = T

    if seq_len == 0:
        return

    if USE_INITIAL_STATE:
        if IS_CONTINUOUS_BATCHING:
            state_idx = tl.load(ssm_state_indices + i_n * stride_indices_seq).to(tl.int64)
            p_h0 = prev_h_kk + state_idx * stride_init_state_seq_kk + i_h * D * D
        else:
            p_h0 = prev_h_kk + i_n * stride_init_state_seq_kk + i_h * D * D

        p_h0_block = tl.make_block_ptr(p_h0, (D, D), (D, 1), (0, 0), (D, D), (1, 0))
        h_kk_prev = tl.load(p_h0_block, boundary_check=(0, 1)).to(tl.float32)
    else:
        h_kk_prev = tl.zeros([D, D], dtype=tl.float32)

    # Q uses H heads, K/V use HKV heads
    p_q = q + (bos * H + i_h) * D + tl.arange(0, D)
    p_k = k + (bos * HKV + i_kv_h) * D + tl.arange(0, D)
    p_alpha = alpha + bos * H + i_h
    if USE_GK:
        p_gk = gk + bos * H + i_h

    if USE_BETA:
        p_beta = beta + bos * H + i_h


    # Note: We specialize the below for seq_len 1 for speedup (by avoiding register spills).
    #   The seq_len 0 is already handled above, so cuda graph padding still works fine.
    for _i in tl.static_range(0, 1):
        b_k_unnorm = tl.load(p_k).to(tl.float32)
        b_q_unnorm = tl.load(p_q).to(tl.float32)

        b_k = b_k_unnorm / tl.sqrt(tl.sum(b_k_unnorm * b_k_unnorm))
        b_q = b_q_unnorm / tl.sqrt(tl.sum(b_q_unnorm * b_q_unnorm))

        if USE_GK:
            exp_gk = tldevice.exp(tl.load(p_gk).to(tl.float32))
        else:
            exp_gk = 1.0

        if USE_BETA:
            beta_val = tl.load(p_beta).to(tl.float32) + 1e-6
            b_k = b_k * beta_val

        h_kk = h_kk_prev * exp_gk + b_k[:, None] * b_k[None, :]

        b_alpha = tl.load(p_alpha).to(tl.float32)

        if STORE_FINAL_STATE:
            if INPLACE_FINAL_STATE and IS_CONTINUOUS_BATCHING:
                # state_idx = tl.load(ssm_state_indices + i_n * stride_indices_seq).to(tl.int64)
                p_ht = h_kk_out + state_idx * stride_final_state_seq_kk + i_h * D * D
            elif INPLACE_FINAL_STATE:
                p_ht = h_kk_out + i_n * stride_final_state_seq_kk + i_h * D * D
            else:
                p_ht = h_kk_out + i_nh * D * D

            p_ht_block = tl.make_block_ptr(p_ht, (D, D), (D, 1), (0, 0), (D, D), (1, 0))
            tl.store(
                p_ht_block,
                h_kk.to(p_ht_block.dtype.element_ty, fp_downcast_rounding="rtne"),
                boundary_check=(0, 1),
            )

        fro = tl.sqrt(tl.sum(h_kk * h_kk))
        ridge_strength = ridge_ratio * fro
        denom = 2.0 * ridge_strength + fro
        inv_denom = 1.0 / denom
        stepsize = 2.0 * inv_denom
        rho_sq_4 = (fro * inv_denom * 0.5) * (fro * inv_denom * 0.5)

        x_prev = tl.zeros([D], dtype=tl.float32)
        x = stepsize * b_q.to(tl.float32)
        omega = 2.0

        for _ in range(num_iter):
            omega = 1.0 / (1.0 - rho_sq_4 * omega)

            h_kk_x = tl.sum(x.to(h_kk.dtype)[:, None] * h_kk, axis=0)
            temp1 = h_kk_x + ridge_strength * x - b_q
            temp2 = (omega - 1.0) * x_prev
            grad_and_x_prev = stepsize * omega * temp1 + temp2
            x_prev = x
            x = omega * x - grad_and_x_prev

        x = b_q + b_alpha * (x - b_q)

        p_v = v + (bos * HKV + i_kv_h) * V + tl.arange(0, V)
        p_out = out + (bos * H + i_h) * V + tl.arange(0, V)
        b_v = tl.load(p_v).to(tl.float32)
        if USE_BETA:
            b_v = b_v * beta_val

        if USE_INITIAL_STATE:
            if IS_CONTINUOUS_BATCHING:
                p_h0_kv = prev_h_kv + state_idx * stride_init_state_seq_kv + i_h * D * V
            else:
                p_h0_kv = prev_h_kv + i_n * stride_init_state_seq_kv + i_h * D * V
            p_h0_block_kv = tl.make_block_ptr(p_h0_kv, (D, V), (V, 1), (0, 0), (D, V), (1, 0))
            h_kv_prev = tl.load(p_h0_block_kv, boundary_check=(0, 1)).to(tl.float32)
        else:
            h_kv_prev = tl.zeros([D, V], dtype=tl.float32)

        h_kv = h_kv_prev * exp_gk + b_k[:, None] * b_v[None, :]

        b_o = tl.sum(h_kv * (x * scale)[:, None], 0)
        tl.store(p_out, b_o.to(p_out.dtype.element_ty, fp_downcast_rounding="rtne"))

        h_kv_prev = h_kv

    if STORE_FINAL_STATE:
        if INPLACE_FINAL_STATE and IS_CONTINUOUS_BATCHING:
            p_ht_kv = h_kv_out + state_idx * stride_final_state_seq_kv + i_h * D * V
        elif INPLACE_FINAL_STATE:
            p_ht_kv = h_kv_out + i_n * stride_final_state_seq_kv + i_h * D * V
        else:
            p_ht_kv = h_kv_out + i_nh * D * V

        p_ht_block_kv = tl.make_block_ptr(p_ht_kv, (D, V), (V, 1), (0, 0), (D, V), (1, 0))
        tl.store(p_ht_block_kv, h_kv_prev.to(p_ht_block_kv.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))


def chebyshev_iteration_gqa(
        k: torch.Tensor,  # [B, T, HKV, D]
        q: torch.Tensor,  # [B, T, H, D]
        v: torch.Tensor,  # [B, T, HKV, V]
        alpha: torch.Tensor,  # [B, T, H]
        gk: torch.Tensor | None = None,  # [B, T, H]
        beta: torch.Tensor | None = None,  # [B, T, H]
        num_iter: int = 15,
        scale: float | None = None,
        ridge_ratio: float = 0.1,
        initial_state_kk: torch.Tensor | None = None,  # [N, H, D, D]
        initial_state_kv: torch.Tensor | None = None,  # [N, H, D, V]
        output_final_state: bool = True,
        inplace_final_state: bool = True,
        cu_seqlens: torch.LongTensor | None = None,
        ssm_state_indices: torch.Tensor | None = None,
):
    """GQA version: K/V have HKV heads, Q/output/state have H heads."""
    if cu_seqlens is not None and q.shape[0] != 1:
        raise ValueError(f"Batch size must be 1 when using cu_seqlens, got {q.shape[0]}.")

    B, T, H, D = q.shape
    HKV = k.shape[2]
    V = v.shape[-1]
    N = B if cu_seqlens is None else len(cu_seqlens) - 1
    GQA_GROUP_SIZE = H // HKV

    if scale is None:
        scale = D ** -0.5
    if initial_state_kk is None:
        initial_state_kk = q.new_zeros(N, H, D, D, dtype=torch.float32)
    if initial_state_kv is None:
        initial_state_kv = q.new_zeros(N, H, D, V, dtype=torch.float32)

    has_beta = beta is not None
    if has_beta:
        beta = beta.contiguous()

    o = q.new_empty(B, T, H, V)
    if output_final_state:
        h_kk_out = initial_state_kk if inplace_final_state else torch.empty_like(initial_state_kk)
        h_kv_out = initial_state_kv if inplace_final_state else torch.empty_like(initial_state_kv)
    else:
        h_kk_out = h_kv_out = None

    stride_init_kk = initial_state_kk.stride(0) if initial_state_kk is not None else 0
    stride_final_kk = h_kk_out.stride(0) if h_kk_out is not None else 0
    stride_init_kv = initial_state_kv.stride(0) if initial_state_kv is not None else 0
    stride_final_kv = h_kv_out.stride(0) if h_kv_out is not None else 0
    stride_indices_seq = ssm_state_indices.stride(0) if ssm_state_indices is not None else 1

    chebyshev_iteration_gqa_fwd_kernel[(N * H,)](
        prev_h_kk=initial_state_kk, prev_h_kv=initial_state_kv,
        k=k.contiguous(), q=q.contiguous(), v=v.contiguous(), beta=beta,
        alpha=alpha.contiguous(), gk=gk.contiguous() if gk is not None else None,
        out=o, h_kk_out=h_kk_out, h_kv_out=h_kv_out,
        cu_seqlens=cu_seqlens, ssm_state_indices=ssm_state_indices, scale=scale,
        N=N, T=T, B=B, H=H, HKV=HKV, D=D, V=V, GQA_GROUP_SIZE=GQA_GROUP_SIZE,
        stride_init_state_seq_kk=stride_init_kk, stride_final_state_seq_kk=stride_final_kk,
        stride_init_state_seq_kv=stride_init_kv, stride_final_state_seq_kv=stride_final_kv,
        stride_indices_seq=stride_indices_seq,
        num_iter=num_iter, ridge_ratio=ridge_ratio,
        USE_BETA=has_beta,
        STORE_FINAL_STATE=output_final_state, INPLACE_FINAL_STATE=inplace_final_state,
    )
    return o, h_kk_out, h_kv_out
