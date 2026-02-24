# Adapted from vLLM's chunk.py with native GQA support.
# Avoids repeat_interleave on K/V by handling GQA in kernels.

import torch
from vllm.model_executor.layers.fla.ops.cumsum import chunk_local_cumsum
from vllm.model_executor.layers.fla.ops.chunk_scaled_dot_kkt import chunk_scaled_dot_kkt_fwd
from vllm.model_executor.layers.fla.ops.solve_tril import solve_tril
from vllm.model_executor.layers.fla.ops.l2norm import l2norm_fwd

from .wy_fast_gqa import recompute_w_u_gqa_fwd
from .chunk_delta_h_gqa import chunk_gated_delta_rule_gqa_fwd_h
from .chunk_o_gqa import chunk_gqa_fwd_o


def chunk_gated_delta_rule_gqa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float = None,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    use_qk_l2norm_in_kernel: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    GQA-native chunk_gated_delta_rule for prefill.

    Unlike the standard version, k and v should NOT be expanded via repeat_interleave.
    The kernels handle GQA broadcasting internally.

    Args:
        q: [B, T, H, K] - queries with full head count
        k: [B, T, HKV, K] - keys with KV head count
        v: [B, T, HKV, V] - values with KV head count
        g: [B, T, H] - gating with full head count
        beta: [B, T, H] - betas with full head count
        initial_state: [N, H, K, V] - initial state with full head count

    Returns:
        o: [B, T, H, V] - output with full head count
        final_state: [N, H, K, V] - final state with full head count
    """
    assert q.dtype == k.dtype == v.dtype
    assert q.dtype != torch.float32, "Does not support float32. Use bfloat16."
    assert len(beta.shape) == 3, "beta must be [B, T, H]"

    if cu_seqlens is not None and q.shape[0] != 1:
        raise ValueError(f"Batch size expected to be 1 with cu_seqlens, got {q.shape[0]}")
    if cu_seqlens is not None and initial_state is not None:
        if initial_state.shape[0] != len(cu_seqlens) - 1:
            raise ValueError(f"Initial states count mismatch: {initial_state.shape[0]} vs {len(cu_seqlens) - 1}")

    if scale is None:
        scale = k.shape[-1] ** -0.5

    if use_qk_l2norm_in_kernel:
        q = l2norm_fwd(q)
        k = l2norm_fwd(k)

    # g cumsum
    g = chunk_local_cumsum(g, chunk_size=64, cu_seqlens=cu_seqlens)

    # A = beta * K * K^T (already supports GQA for K via Hg)
    A = chunk_scaled_dot_kkt_fwd(k=k, beta=beta, g=g, cu_seqlens=cu_seqlens, output_dtype=torch.float32)
    A = solve_tril(A=A, cu_seqlens=cu_seqlens, output_dtype=k.dtype)

    # w, u = recompute (GQA version)
    w, u = recompute_w_u_gqa_fwd(k=k, v=v, beta=beta, g_cumsum=g, A=A, cu_seqlens=cu_seqlens)

    # h, v_new, final_state (GQA version)
    h, v_new, final_state = chunk_gated_delta_rule_gqa_fwd_h(
        k=k, w=w, u=u, g=g,
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
    )

    # o (GQA version)
    o = chunk_gqa_fwd_o(q=q, k=k, v=v_new, h=h, g=g, scale=scale, cu_seqlens=cu_seqlens)

    return o.to(q.dtype), final_state
