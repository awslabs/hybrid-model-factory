# Adapted from vLLM's qwen3_next.py with fused triton gating kernels.
# Added non-contiguous input support via explicit stride parameters.

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _fused_gdn_gating_kernel(
    g_ptr,
    beta_ptr,
    A_log_ptr,
    a_ptr,
    b_ptr,
    dt_bias_ptr,
    num_tokens,
    NUM_HEADS: tl.constexpr,
    a_stride_0,
    a_stride_1,
    b_stride_0,
    b_stride_1,
    softplus_beta: tl.constexpr,
    softplus_threshold: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused triton kernel for GDN gating computation with stride support.

    This kernel uses explicit stride parameters to handle non-contiguous inputs,
    avoiding the need for .contiguous() calls and their associated memory copies.

    Computes:
        g = -exp(A_log) * softplus(a + dt_bias)
        beta = sigmoid(b)

    Input shapes:
        A_log: [num_heads]
        a: [num_tokens, num_heads] (can be non-contiguous)
        b: [num_tokens, num_heads] (can be non-contiguous)
        dt_bias: [num_heads]

    Output shapes (always contiguous):
        g: [num_tokens, num_heads]
        beta: [num_tokens, num_heads]
    """
    token_idx = tl.program_id(0)
    head_offsets = tl.arange(0, BLOCK_SIZE)

    for head_start in range(0, NUM_HEADS, BLOCK_SIZE):
        head_idx = head_start + head_offsets
        mask = head_idx < NUM_HEADS

        # Output offset (contiguous layout)
        out_offset = token_idx * NUM_HEADS + head_idx

        # Strided input access for non-contiguous tensors
        a_offset = token_idx * a_stride_0 + head_idx * a_stride_1
        b_offset = token_idx * b_stride_0 + head_idx * b_stride_1

        A_log = tl.load(A_log_ptr + head_idx, mask=mask).to(tl.float32)
        a = tl.load(a_ptr + a_offset, mask=mask).to(tl.float32)
        b = tl.load(b_ptr + b_offset, mask=mask).to(tl.float32)
        dt_bias = tl.load(dt_bias_ptr + head_idx, mask=mask).to(tl.float32)

        x = a + dt_bias
        softplus_x = tl.where(
            softplus_beta * x <= softplus_threshold,
            (1.0 / softplus_beta) * tl.log(1.0 + tl.exp(softplus_beta * x)),
            x,
        )
        g = -tl.exp(A_log) * softplus_x
        beta = tl.sigmoid(b)

        tl.store(g_ptr + out_offset, g, mask=mask)
        tl.store(beta_ptr + out_offset, beta, mask=mask)


def fused_gdn_gating(
    A_log: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    dt_bias: torch.Tensor,
    beta: float = 1.0,
    threshold: float = 20.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Fused computation of g and beta for GDN using triton kernel (strided version).

    This version supports non-contiguous inputs by passing explicit strides to the
    triton kernel. This avoids the implicit .contiguous() copies that occur when
    inputs come from torch.split() or similar operations.

    HF reference implementation (pure PyTorch):
        g = -self.A_log.float().exp() * F.softplus(gk_proj_out.float() + self.dt_bias)
        beta = b_proj_out.sigmoid()

    Args:
        A_log: A_log tensor of shape [num_heads] (will be exp'd)
        a: gk_proj output of shape [num_tokens, num_heads] (can be non-contiguous)
        b: b_proj output of shape [num_tokens, num_heads] (can be non-contiguous)
        dt_bias: dt_bias tensor of shape [num_heads]
        beta: Softplus beta parameter (default 1.0)
        threshold: Softplus threshold for numerical stability (default 20.0)

    Returns:
        g: gating tensor of shape [1, num_tokens, num_heads]
        beta_out: beta tensor of shape [1, num_tokens, num_heads]
    """
    num_tokens, num_heads = a.shape

    g = torch.empty(num_tokens, num_heads, dtype=a.dtype, device=a.device)
    beta_out = torch.empty(num_tokens, num_heads, dtype=a.dtype, device=a.device)

    BLOCK_SIZE = triton.next_power_of_2(min(num_heads, 128))

    _fused_gdn_gating_kernel[(num_tokens,)](
        g, beta_out, A_log, a, b, dt_bias,
        num_tokens, num_heads,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        beta, threshold, BLOCK_SIZE,
    )

    return g.unsqueeze(0), beta_out.unsqueeze(0)
