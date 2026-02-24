"""
BMOJO-F Attention Utilities.

This module provides utility functions and classes for implementing BMOJO-F attention,
which combines in-context and fading memory attention mechanisms using Flash Attention.

The key components include:
    - roll_batch_first_half: Utility for sequence parallel processing.
    - bmojo_self_attention: Main attention function combining in-context and fading paths.
    - add_logsumexps: Numerically stable log-sum-exp computation.
    - bmojo_flash_attn_func: Flash attention wrapper for BMOJO.
    - BMOJOFlashAttnFunc: Custom autograd function for forward/backward passes.

Shape Notation:
    Throughout this module, tensor shapes use the following notation:
        - B: Batch size
        - W: Window size (typically defined in config.bmojo_config['window_size']).
            Note: The window_size tuple parameter equals (W-1, 0), so window_size[0] + 1 = W.
        - L: Sequence length (in-context tokens)
        - Lf: Fading sequence length, where Lf = L - W
        - H: Number of query heads
        - Hkv: Number of key/value heads (may differ from H in grouped-query attention)
        - D: Head dimension (hidden_size / num_attention_heads)

"""

import math
import warnings
from importlib.metadata import version as get_pkg_version
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from transformers.configuration_utils import PretrainedConfig
from packaging.version import Version as PkgVersion
from torch import Tensor

from flash_attn.flash_attn_interface import (
    _flash_attn_backward,
    _flash_attn_forward,
    flash_attn_func,
)

# Version check for Flash Attention
_flash_attn_version = PkgVersion(get_pkg_version("flash-attn"))
if _flash_attn_version != PkgVersion("2.8.3"):
    warnings.warn(
        f"BMOJO attention has been validated with flash-attn version 2.8.3. "
        f"Detected version {_flash_attn_version}, which may cause compatibility issues.",
        UserWarning,
    )


def roll_batch_first_half(tensor: Tensor, k: int) -> Tensor:
    """
    Roll the first half of a tensor along dimension 1 while keeping the second half unchanged.

    This utility is used for sequence parallel processing to handle boundary tokens
    that need special treatment in the first sequence parallel rank.

    Args:
        tensor: Input tensor with shape [B, L, ...] where B is batch size and L is sequence length.
        k: Number of positions to roll. Positive values roll right, negative values roll left.

    Returns:
        Tensor with the same shape as input, where the first half (along dim 0) is rolled
        by k positions along dim 1, and the second half remains unchanged.
    """
    half_bs = tensor.shape[0] // 2
    return torch.cat(
        [
            torch.roll(tensor[:half_bs], shifts=k, dims=1),  # roll first half
            tensor[half_bs:],  # keep second half as is
        ],
        dim=0,
    )


def bmojo_self_attention(
    query_in_ctx: Tensor,
    key_in_ctx: Tensor,
    value_in_ctx: Tensor,
    query_fading: Optional[Tensor],
    key_fading: Optional[Tensor],
    value_fading: Optional[Tensor],
    config: PretrainedConfig,
    window_size: Tuple[int, int],
    attn_mask_type: str = "causal",
    softmax_scale: Optional[float] = None,
    sequence_parallel_group: Optional[torch.distributed.ProcessGroup] = None,
    original_seqlen: int = 0,
    sp_rank: int = 1,
) -> Tensor:
    """
    Compute BMOJO self-attention with two paths:

    1. In-context path: Standard sliding window attention over recent tokens
    2. Fading path: Sliding window attention over the tokens preceding the in-context
        tokens that have been processed by an SSM.

    This function handles sequence parallel processing, applies Flash Attention,
    and properly combines outputs from both paths.

    Args:
        query_in_ctx: Query tensor for in-context tokens with shape [L, B, H, D].
        key_in_ctx: Key tensor for in-context tokens with shape [L, B, Hkv, D].
        value_in_ctx: Value tensor for in-context tokens with shape [L, B, Hkv, D].
        query_fading: Query tensor for fading tokens with shape [Lf, B, H, D] or None.
        key_fading: Key tensor for fading tokens with shape [Lf, B, Hkv, D] or None.
        value_fading: Value tensor for fading tokens with shape [Lf, B, Hkv, D] or None.
        config: Model configuration containing hidden_size, num_attention_heads, and attention_dropout.
        window_size: Tuple of (left, right) window sizes for sliding window attention.
        attn_mask_type: Type of attention mask, typically "causal" for autoregressive models.
        softmax_scale: Scaling factor for attention scores. If None, defaults to 1/sqrt(head_dim).
        sequence_parallel_group: Process group for sequence parallelism. None if not using SP.
        original_seqlen: Original sequence length before SP reordering (used for SP rank 0).
        sp_rank: Rank in the sequence parallel group (used to determine special handling).

    Returns:
        Attention output tensor with shape [B, L, H*D] where H*D equals hidden_size.

    Note:
        - When fading tensors are None, only in-context SWA is computed.
        - For sequence parallel rank 0, special token rolling is applied to handle boundary conditions.
        - Output tokens are trimmed appropriately for sequence parallel processing.
    """
    kv_channels = config.hidden_size // config.num_attention_heads

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(
            kv_channels if isinstance(kv_channels, int) else kv_channels[0]
        )

    # Our qkv layout is (seq, batch, heads, dim), but flash attention expects it
    # in the form (batch, seq, head, dim).
    query_in_ctx, key_in_ctx, value_in_ctx = [
        x.transpose(0, 1) for x in (query_in_ctx, key_in_ctx, value_in_ctx)
    ]

    # This is needed because the first (window_size[0] + 1) * 2 - 1 tokens are zero for the first SP rank.
    # These need to be ignored, and a simple way of doing this is to roll them towards the end and
    # then drop the tokens afterwards.
    if sequence_parallel_group and sp_rank == 0:
        query_in_ctx = roll_batch_first_half(query_in_ctx, original_seqlen)
        key_in_ctx = roll_batch_first_half(key_in_ctx, original_seqlen)
        value_in_ctx = roll_batch_first_half(value_in_ctx, original_seqlen)

    if key_fading is None:
        out = flash_attn_func(
            query_in_ctx,
            key_in_ctx,
            value_in_ctx,
            dropout_p=config.attention_dropout,
            softmax_scale=softmax_scale,
            causal="causal" in attn_mask_type,
            window_size=window_size,
        )
    else:
        query_fading, key_fading, value_fading = [
            x.transpose(0, 1) for x in (query_fading, key_fading, value_fading)
        ]

        # Apply same boundary token handling as in-context tokens above, but adjusted
        # for the shorter fading sequence length
        if sequence_parallel_group and sp_rank == 0:
            query_fading = roll_batch_first_half(
                query_fading, original_seqlen - (window_size[0] + 1)
            )
            key_fading = roll_batch_first_half(
                key_fading, original_seqlen - (window_size[0] + 1)
            )
            value_fading = roll_batch_first_half(
                value_fading, original_seqlen - (window_size[0] + 1)
            )

        out = bmojo_flash_attn_func(
            query_in_ctx,
            key_in_ctx,
            value_in_ctx,
            query_fading,
            key_fading,
            value_fading,
            dropout_p=config.attention_dropout,
            softmax_scale=softmax_scale,
            causal="causal" in attn_mask_type,
            window_size=window_size,
        )
    bs, seqlen, n_h, h_d = out.shape

    if sequence_parallel_group:
        if sp_rank > 0:
            out = out[:, (window_size[0] + 1) * 2 - 1 :].contiguous()
        else:
            drop_tokens = (window_size[0] + 1) * 2 - 1
            half_bs = out.shape[0] // 2
            out = torch.cat(
                [
                    out[:half_bs, :-drop_tokens],  # first half: drop last tokens
                    out[half_bs:, drop_tokens:],  # second half: drop first tokens
                ],
                dim=0,
            ).contiguous()

        if bs == 2:
            out = out.view(bs // 2, -1, n_h, h_d)
        else:
            out = torch.cat([out[: bs // 2], out[bs // 2 :]], dim=1).contiguous()

        bs, seqlen, _, _ = out.shape

    return out.view(bs, seqlen, -1).contiguous()


def add_logsumexps(lse_a: Tensor, lse_b: Tensor) -> Tensor:
    """
    Compute log(exp(lse_a) + exp(lse_b)) in a numerically stable way.

    This function uses the log-sum-exp trick to avoid numerical overflow/underflow
    when combining log-space values. It's used to properly normalize attention outputs
    from the in-context and fading paths.

    Args:
        lse_a: First log-sum-exp tensor with shape [B, H, L].
        lse_b: Second log-sum-exp tensor with shape [B, H, L].

    Returns:
        Combined log-sum-exp tensor with shape [B, H, L].

    Mathematical formula:
        log(exp(a) + exp(b)) = max(a,b) + log(exp(a-max) + exp(b-max))
    """
    # Use the log-sum-exp trick
    max_val = torch.max(lse_a, lse_b)
    return max_val + torch.log(torch.exp(lse_a - max_val) + torch.exp(lse_b - max_val))


def bmojo_flash_attn_func(
    q_in_ctx: Tensor,
    k_in_ctx: Tensor,
    v_in_ctx: Tensor,
    q_fading: Tensor,
    k_fading: Tensor,
    v_fading: Tensor,
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size: Tuple[int, int] = (-1, -1),
    alibi_slopes: Optional[Tensor] = None,
    deterministic: bool = False,
    return_attn_probs: bool = False,
) -> Tensor:
    """
    Lower-level function that directly interfaces with Flash Attention's
    autograd implementation. See BMOJOFlashAttnFunc for implementation details.

    Args:
        q_in_ctx: Query tensor for in-context path with shape [B, L, H, D].
        k_in_ctx: Key tensor for in-context path with shape [B, L, Hkv, D].
        v_in_ctx: Value tensor for in-context path with shape [B, L, Hkv, D].
        q_fading: Query tensor for fading path with shape [B, Lf, H, D].
        k_fading: Key tensor for fading path with shape [B, Lf, Hkv, D].
        v_fading: Value tensor for fading path with shape [B, Lf, Hkv, D].
        dropout_p: Dropout probability. Should be 0.0 during evaluation.
        softmax_scale: Scaling factor for QK^T before softmax. Defaults to 1/sqrt(D).
        causal: Whether to apply causal attention mask for autoregressive modeling.
        window_size: Tuple of (left, right) for sliding window attention.
        alibi_slopes: ALiBi slopes with shape [H] or [B, H]. Adds bias of
            (-alibi_slope * |i + seqlen_k - seqlen_q - j|) to attention scores.
        deterministic: Whether to use deterministic backward pass (slower, more memory).
        return_attn_probs: Whether to return attention probabilities (for testing only).

    Returns:
        Attention output tensor with shape [B, L, H, D].
    """
    return BMOJOFlashAttnFunc.apply(
        q_in_ctx,
        k_in_ctx,
        v_in_ctx,
        q_fading,
        k_fading,
        v_fading,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        alibi_slopes,
        deterministic,
        return_attn_probs,
    )


class BMOJOFlashAttnFunc(torch.autograd.Function):
    """
    Custom autograd function for BMOJO Flash Attention.

    Implements forward and backward passes for BMOJO Attention, which combines
    in-context and fading memory paths using Flash Attention. The forward pass
    computes attention separately for each path, then combines outputs using
    numerically stable log-sum-exp normalization. The backward pass computes
    gradients for both paths independently.
    """

    @staticmethod
    def forward(
        ctx,
        q_in_ctx: Tensor,
        k_in_ctx: Tensor,
        v_in_ctx: Tensor,
        q_fading: Tensor,
        k_fading: Tensor,
        v_fading: Tensor,
        dropout_p: float,
        softmax_scale: Optional[float],
        causal: bool,
        window_size: Tuple[int, int],
        alibi_slopes: Optional[Tensor],
        deterministic: bool,
        return_softmax: bool,
    ) -> Tensor:
        """
        Forward pass for BMOJO attention.

        Computes attention outputs for both in-context and fading paths using Flash
        Attention, then combines them with proper softmax normalization across paths.

        The combination formula is:
            out = (out_in_ctx * exp(lse_in_ctx) + out_fading * exp(lse_fading)) / Z
        where Z = exp(lse_in_ctx) + exp(lse_fading) ensures proper normalization.

        Args:
            ctx: Context object for saving tensors needed in backward pass.
            q_in_ctx: In-context query tensor [B, L, H, D].
            k_in_ctx: In-context key tensor [B, L, Hkv, D].
            v_in_ctx: In-context value tensor [B, L, Hkv, D].
            q_fading: Fading query tensor [B, Lf, H, D].
            k_fading: Fading key tensor [B, Lf, Hkv, D].
            v_fading: Fading value tensor [B, Lf, Hkv, D].
            dropout_p: Dropout probability.
            softmax_scale: Attention score scaling factor.
            causal: Whether to use causal masking.
            window_size: Sliding window size tuple (left, right).
            alibi_slopes: Optional ALiBi slopes.
            deterministic: Whether to use deterministic backward.
            return_softmax: Whether to return softmax outputs (for dropout > 0).

        Returns:
            Combined attention output tensor [B, L, H, D] in bfloat16.

        Note:
            - Requires head_dim to be divisible by 8 (Flash Attention constraint)
            - Saves tensors to ctx for backward pass
        """
        if softmax_scale is None:
            softmax_scale = q_in_ctx.shape[-1] ** (-0.5)

        assert q_in_ctx.shape[-1] % 8 == 0, (
            f"head_dim must be divisible by 8 for Flash Attention, got {q_in_ctx.shape[-1]}"
        )

        # Forward for self attention over in-context memory
        (
            out_in_ctx,
            softmax_lse_in_ctx,
            S_dmask_in_ctx,
            rng_state_in_ctx,
        ) = _flash_attn_forward(
            q_in_ctx,
            k_in_ctx,
            v_in_ctx,
            dropout_p,
            softmax_scale,
            causal=causal,
            window_size_left=window_size[0],
            window_size_right=window_size[1],
            softcap=0.0,
            alibi_slopes=alibi_slopes,
            return_softmax=return_softmax and dropout_p > 0,
        )

        # Forward for self attention over fading memory
        (
            out_fading,
            softmax_lse_fading,
            S_dmask_fading,
            rng_state_fading,
        ) = _flash_attn_forward(
            q_fading,
            k_fading,
            v_fading,
            dropout_p,
            softmax_scale,
            causal=causal,
            window_size_left=window_size[0],
            window_size_right=window_size[1],
            softcap=0.0,
            alibi_slopes=alibi_slopes,
            return_softmax=return_softmax and dropout_p > 0,
        )

        # Pad fading output if needed (initial tokens have no fading memory)
        if out_in_ctx.shape[1] > out_fading.shape[1]:
            out_fading = F.pad(
                out_fading, (0, 0, 0, 0, window_size[0] + 1, 0), "constant", 0
            )
            softmax_lse_fading = F.pad(
                softmax_lse_fading, (window_size[0] + 1, 0), "constant", -float("inf")
            )

        # Combine log-sum-exp values from both paths
        softmax_lse = add_logsumexps(softmax_lse_in_ctx, softmax_lse_fading)

        # Reshape for broadcasting: [B, H, L] -> [B, L, H, 1]
        softmax_lse_fading_ = softmax_lse_fading.permute(0, 2, 1).unsqueeze(-1)
        softmax_lse_in_ctx_ = softmax_lse_in_ctx.permute(0, 2, 1).unsqueeze(-1)

        # Compute normalized combination using log-sum-exp trick
        max_val = torch.max(softmax_lse_fading_, softmax_lse_in_ctx_)
        softmax_lse_fading_ = softmax_lse_fading_ - max_val
        softmax_lse_in_ctx_ = softmax_lse_in_ctx_ - max_val

        denominator = torch.exp(softmax_lse_in_ctx_) + torch.exp(softmax_lse_fading_)

        unnormalized_attention_output = out_in_ctx * torch.exp(
            softmax_lse_in_ctx_
        ) + out_fading * torch.exp(softmax_lse_fading_)
        normalized_attention_output = unnormalized_attention_output / denominator

        out_padded = normalized_attention_output

        # Save tensors for backward pass
        ctx.save_for_backward(
            q_in_ctx,
            k_in_ctx,
            v_in_ctx,
            q_fading,
            k_fading,
            v_fading,
            out_padded.to(torch.bfloat16).contiguous(),
            softmax_lse.contiguous(),
            rng_state_in_ctx,
            rng_state_fading,
        )
        ctx.dropout_p = dropout_p
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.window_size = window_size
        ctx.alibi_slopes = alibi_slopes
        ctx.deterministic = deterministic

        return normalized_attention_output.to(torch.bfloat16).contiguous()

    @staticmethod
    def backward(ctx, dout: Tensor, *args) -> Tuple[Optional[Tensor], ...]:
        """
        Backward pass for BMOJO attention.

        Computes gradients for all input tensors by running Flash Attention backward
        passes for both in-context and fading paths independently, using the combined
        output and log-sum-exp values from the forward pass.

        Args:
            ctx: Context object containing saved tensors from forward pass.
            dout: Gradient of loss w.r.t. output [B, L, H, D].
            *args: Additional arguments (unused).

        Returns:
            Tuple of gradients for all forward inputs:
                (dq_in_ctx, dk_in_ctx, dv_in_ctx, dq_fading, dk_fading, dv_fading,
                None, None, None, None, None, None, None)
            where None values correspond to non-tensor arguments (dropout_p,
            softmax_scale, causal, window_size, alibi_slopes, deterministic,
            return_softmax).
        """

        (
            q_in_ctx,
            k_in_ctx,
            v_in_ctx,
            q_fading,
            k_fading,
            v_fading,
            out,
            softmax_lse,
            rng_state_in_ctx,
            rng_state_fading,
        ) = ctx.saved_tensors
        dq_in_ctx, dk_in_ctx, dv_in_ctx = (
            torch.empty_like(q_in_ctx),
            torch.empty_like(k_in_ctx),
            torch.empty_like(v_in_ctx),
        )
        dq_fading, dk_fading, dv_fading = (
            torch.empty_like(q_fading),
            torch.empty_like(k_fading),
            torch.empty_like(v_fading),
        )

        # Backward for self attention over in-context tokens
        _flash_attn_backward(
            dout,
            q_in_ctx,
            k_in_ctx,
            v_in_ctx,
            out,
            softmax_lse,
            dq_in_ctx,
            dk_in_ctx,
            dv_in_ctx,
            ctx.dropout_p,
            ctx.softmax_scale,
            ctx.causal,
            ctx.window_size[0],
            ctx.window_size[1],
            0.0,
            ctx.alibi_slopes,
            ctx.deterministic,
            rng_state=rng_state_in_ctx,
        )
        dq_in_ctx = dq_in_ctx[
            ..., : dout.shape[-1]
        ]  # Handle potential head dimension padding
        dk_in_ctx = dk_in_ctx[..., : dout.shape[-1]]
        dv_in_ctx = dv_in_ctx[..., : dout.shape[-1]]

        # Backward for cross attention over fading tokens
        if q_in_ctx.shape[1] > q_fading.shape[1]:
            # Fading sequence is shorter, use subset of gradients
            dout_ = dout[:, ctx.window_size[0] + 1 :].contiguous()
            softmax_lse_ = softmax_lse[:, :, ctx.window_size[0] + 1 :].contiguous()
            out_ = out[:, ctx.window_size[0] + 1 :].contiguous()
        else:
            dout_ = dout
            softmax_lse_ = softmax_lse
            out_ = out

        _flash_attn_backward(
            dout_,
            q_fading,
            k_fading,
            v_fading,
            out_,
            softmax_lse_,
            dq_fading,
            dk_fading,
            dv_fading,
            ctx.dropout_p,
            ctx.softmax_scale,
            ctx.causal,
            ctx.window_size[0],
            ctx.window_size[1],
            0.0,
            ctx.alibi_slopes,
            ctx.deterministic,
            rng_state=rng_state_fading,
        )
        dq_fading = dq_fading[
            ..., : dout_.shape[-1]
        ]  # Handle potential head dimension padding
        dk_fading = dk_fading[..., : dout_.shape[-1]]
        dv_fading = dv_fading[..., : dout_.shape[-1]]

        return (
            dq_in_ctx,
            dk_in_ctx,
            dv_in_ctx,
            dq_fading,
            dk_fading,
            dv_fading,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def test_bmojof(
    q_in_ctx: Tensor,
    k_in_ctx: Tensor,
    v_in_ctx: Tensor,
    q_fading: Tensor,
    k_fading: Tensor,
    v_fading: Tensor,
    window_size: int,
) -> None:
    """
    Test function for BMOJO Flash Attention.

    This function creates tensors with gradients enabled, runs the BMOJO attention
    forward and backward passes, and prints output shapes for verification.

    Args:
        q_in_ctx: In-context query tensor.
        k_in_ctx: In-context key tensor.
        v_in_ctx: In-context value tensor.
        q_fading: Fading query tensor.
        k_fading: Fading key tensor.
        v_fading: Fading value tensor.
        window_size: Window size for sliding window attention.

    Returns:
        None. Prints output and gradient shapes to stdout.
    """
    q_in_ctx_, k_in_ctx_, v_in_ctx_, q_fading_, k_fading_, v_fading_ = (
        torch.tensor(q_in_ctx, requires_grad=True),
        torch.tensor(k_in_ctx, requires_grad=True),
        torch.tensor(v_in_ctx, requires_grad=True),
        torch.tensor(q_fading, requires_grad=True),
        torch.tensor(k_fading, requires_grad=True),
        torch.tensor(v_fading, requires_grad=True),
    )

    out_flash_bmojo = bmojo_flash_attn_func(
        q_in_ctx_,
        k_in_ctx_,
        v_in_ctx_,
        q_fading_,
        k_fading_,
        v_fading_,
        causal=True,
        window_size=(window_size - 1, 0),
    )

    out_flash_bmojo.sum().backward()

    print(out_flash_bmojo.shape)
    print(q_in_ctx_.grad.shape)


if __name__ == "__main__":
    torch.manual_seed(42)
    batch_size, seq_len, num_heads, head_dim = 1, 8192, 32, 128
    window_size = 2048

    q_in_ctx = torch.randn(
        batch_size,
        seq_len,
        num_heads,
        head_dim,
        requires_grad=True,
        dtype=torch.bfloat16,
        device="cuda",
    )
    k_in_ctx = torch.randn(
        batch_size,
        seq_len,
        num_heads,
        head_dim,
        requires_grad=True,
        dtype=torch.bfloat16,
        device="cuda",
    )
    v_in_ctx = torch.randn(
        batch_size,
        seq_len,
        num_heads,
        head_dim,
        requires_grad=True,
        dtype=torch.bfloat16,
        device="cuda",
    )

    q_fading = torch.randn(
        batch_size,
        seq_len - window_size,
        num_heads,
        head_dim,
        requires_grad=True,
        dtype=torch.bfloat16,
        device="cuda",
    )
    k_fading = torch.randn(
        batch_size,
        seq_len - window_size,
        num_heads,
        head_dim,
        requires_grad=True,
        dtype=torch.bfloat16,
        device="cuda",
    )
    v_fading = torch.randn(
        batch_size,
        seq_len - window_size,
        num_heads,
        head_dim,
        requires_grad=True,
        dtype=torch.bfloat16,
        device="cuda",
    )

    test_bmojof(q_in_ctx, k_in_ctx, v_in_ctx, q_fading, k_fading, v_fading, window_size)
