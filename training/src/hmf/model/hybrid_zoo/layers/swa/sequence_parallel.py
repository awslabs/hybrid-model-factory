"""
Sequence parallel helpers for Sliding Window Attention (SWA).

This module provides utilities for running SWA with zig-zag context parallelism.
Standard ring attention does not support sliding window, so SWA uses a P2P-based
approach: each GPU gathers only the boundary tokens (last `window_size` tokens)
from its neighboring chunk, runs flash attention with the sliding window locally,
then trims the overlapping tokens from the output.

The zig-zag pattern splits each GPU's local sequence into two discontiguous chunks
stacked along the batch dimension. For rank 0, the first chunk has no predecessor,
so zero-padding is introduced and must be rolled out of the way before attention
(via `roll_first_half`) and the corresponding output tokens trimmed afterward.

Dependencies:
    - reorder_for_ssm_p2p: P2P communication primitive from Mamba2 SP utilities.
      Exchanges the last `d_conv - 1` tokens between adjacent chunks.
"""

from contextlib import contextmanager

import torch
import torch.distributed as dist
from typing import Tuple

from ..sp_p2p_utils import reorder_for_ssm_p2p


@contextmanager
def swa_sp_attn_override(config):
    """
    Temporarily set config._attn_implementation to flash_attention_2 for SWA SP.

    SWA handles its own sequence parallelism via P2P gather, so it must use
    flash_attention_2 rather than the globally-patched sequence_parallel_attention
    (ring attention), which doesn't support sliding window.

    We must mutate config._attn_implementation directly because transformers'
    flash_attention_forward (integrations/flash_attention.py) re-reads
    module.config._attn_implementation and passes it to _flash_attention_forward,
    which uses it to load the flash kernel via lazy_import_flash_attention.
    Passing the correct string to get_interface alone is not sufficient.
    """
    original = config._attn_implementation
    config._attn_implementation = "flash_attention_2"
    try:
        yield
    finally:
        config._attn_implementation = original


def roll_first_half(tensor: torch.Tensor, seq_len: int) -> torch.Tensor:
    """
    Roll zero-padding tokens to the end for rank 0's first chunk.

    When using SP, `reorder_for_ssm_p2p` prepends `sliding_window` tokens from the
    previous chunk. For rank 0's first chunk there is no previous chunk, so these are
    zero-padded. This function rolls those zeros to the end of the sequence so they
    don't participate in the causal attention computation. The corresponding output
    positions are trimmed in `swa_sp_post_attn`.

    Args:
        tensor: Input tensor of shape [2*B, L, ...] where the first half ([:B])
            contains the first chunk with zero-padding at the start.
        seq_len: The original (non-padded) sequence length per chunk. The roll
            shifts by this amount.

    Returns:
        Tensor of same shape with the first half's elements rolled.
    """
    half_bs = tensor.shape[0] // 2
    return torch.cat(
        [torch.roll(tensor[:half_bs], shifts=seq_len, dims=1), tensor[half_bs:]], dim=0
    )


def swa_sp_gather(
    hidden_states: torch.Tensor,
    position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    sp_group: dist.ProcessGroup,
    sliding_window: int,
) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], int, int]:
    """
    Pre-attention: gather boundary tokens from neighboring chunks via P2P.

    Each GPU holds a local sequence of shape [B, L_local, D]. This function:
    1. Calls `reorder_for_ssm_p2p` to exchange the last `sliding_window` tokens
       with neighboring chunks and reshape to [2*B, L_local//2 + sliding_window, D].
    2. Applies the same reordering to position embeddings (cos, sin).

    Args:
        hidden_states: Local hidden states [B, L_local, D].
        position_embeddings: Tuple of (cos, sin), each [B, L_local, head_dim].
        sp_group: Sequence parallel process group.
        sliding_window: Sliding window size.

    Returns:
        Tuple of:
            - hidden_states: Reordered with boundary context [2*B, L_local//2 + sliding_window, D].
            - position_embeddings: Reordered (cos, sin) with same shape transformation.
            - original_bsz: Original batch size B (needed for post-attention recombination).
            - original_seq_len: Original local sequence length L_local.
    """
    sp_size = dist.get_world_size(group=sp_group)
    sp_rank = dist.get_rank(group=sp_group)
    stream = torch.cuda.Stream()
    d_conv = sliding_window + 1

    original_bsz = hidden_states.shape[0]
    original_seq_len = hidden_states.shape[1]

    hidden_states = reorder_for_ssm_p2p(
        hidden_states, sp_group, stream, sp_size, sp_rank, d_conv
    )

    cos = reorder_for_ssm_p2p(
        position_embeddings[0], sp_group, stream, sp_size, sp_rank, d_conv
    )
    sin = reorder_for_ssm_p2p(
        position_embeddings[1], sp_group, stream, sp_size, sp_rank, d_conv
    )

    return hidden_states, (cos, sin), original_bsz, original_seq_len


def swa_sp_post_attn(
    attn_output: torch.Tensor,
    sp_group: dist.ProcessGroup,
    sliding_window: int,
    original_bsz: int,
) -> torch.Tensor:
    """
    Post-attention: trim overlapping boundary tokens and recombine batch dimension.

    After attention, the output has shape [2*B, L_chunk + sliding_window, ...] where
    the extra `sliding_window` tokens are boundary context that must be removed.

    For rank > 0: simply drop the first `sliding_window` tokens (they were prepended
    context from the previous chunk).

    For rank 0: the first chunk had its zeros rolled to the end (by `roll_first_half`),
    so we drop the last `sliding_window` tokens from the first half-batch and the first
    `sliding_window` tokens from the second half-batch.

    Finally, the doubled batch dimension is collapsed back to the original batch size.

    Args:
        attn_output: Attention output with shape [2*B, L_chunk + sliding_window, ...].
            Can be 3D (after o_proj: [..., hidden_size]) or 4D (before o_proj:
            [..., num_heads, head_dim]).
        sp_group: Sequence parallel process group.
        sliding_window: Sliding window size.
        original_bsz: Original batch size before SP reordering.

    Returns:
        Trimmed and recombined output [B, L_local, ...].
    """
    sp_rank = dist.get_rank(group=sp_group)
    bsz = attn_output.shape[0]  # This is 2 * original_bsz

    if sp_rank > 0:
        attn_output = attn_output[:, sliding_window:].contiguous()
    else:
        half_bs = bsz // 2
        attn_output = torch.cat(
            [
                attn_output[:half_bs, :-sliding_window],
                attn_output[half_bs:, sliding_window:],
            ],
            dim=0,
        ).contiguous()

    # Recombine the doubled batch back to original shape
    if bsz == 2:
        # original_bsz was 1; merge along the sequence dimension
        remaining_dims = attn_output.shape[2:]
        attn_output = attn_output.view(1, -1, *remaining_dims)
    else:
        half_bs = bsz // 2
        attn_output = torch.cat(
            [attn_output[:half_bs], attn_output[half_bs:]], dim=1
        ).contiguous()

    return attn_output


def swa_sp_reorder_cache_position(
    cache_position: torch.Tensor,
    sp_group: dist.ProcessGroup,
    sliding_window: int,
) -> torch.Tensor:
    """
    Reorders cache_position to match the gathered sequence layout from swa_sp_gather.

    swa_sp_gather transforms tensors from [B, L, D] to [2*B, L//2 + sliding_window, D]
    via zig-zag split + boundary token prepend (reorder_for_ssm_p2p). cache_position is
    a 1-D tensor [L] that must undergo the same transformation so that per-position
    operations (e.g., _get_llama_4_attn_scale) see matching sequence dimensions.

    We reshape cache_position to [1, L, 1], run it through reorder_for_ssm_p2p (which
    handles the zig-zag split and P2P exchange of boundary tokens), then take the first
    chunk and squeeze back to 1D [L//2 + sliding_window].

    Args:
        cache_position: 1-D position tensor [L].
        sp_group: Sequence parallel process group.
        sliding_window: Sliding window size.

    Returns:
        Reordered 1-D cache_position [L//2 + sliding_window] matching the gathered
        sequence length.
    """
    sp_size = dist.get_world_size(group=sp_group)
    sp_rank = dist.get_rank(group=sp_group)
    stream = torch.cuda.Stream()
    d_conv = sliding_window + 1

    # Reshape to 3-D so reorder_for_ssm_p2p can handle it: [1, L, 1]
    cp_3d = cache_position.unsqueeze(0).unsqueeze(-1).float()
    cp_reordered = reorder_for_ssm_p2p(cp_3d, sp_group, stream, sp_size, sp_rank, d_conv)
    # cp_reordered shape: [2, L//2 + sliding_window, 1]
    # Both chunks have the same positional layout, take the first and squeeze back to 1-D
    return cp_reordered[0, :, 0].long()
