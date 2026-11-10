"""
Splits attention heads that are larger than a sequence mixer's kernels support into heads of the largest size that
they support, with as many more heads. A hybrid layer with split heads keeps the projection sizes of the attention
that it replaces, so priming can start it from that attention's weights, after reordering them so that the pieces of
each query head pair with the pieces of its own key head (see split_attention_heads in priming/hybrid_layer_init.py).
"""

from typing import Tuple


def split_heads(head_dim: int, num_q_heads: int, num_kv_heads: int, max_head_dim: int) -> Tuple[int, int, int]:
    """
    Returns the head size and the numbers of query and key/value heads of a mixer whose kernels support heads of at
    most max_head_dim. Heads larger than max_head_dim are split into head_dim // max_head_dim heads of max_head_dim,
    e.g., Gemma 4's full attention layers, with 8 query and 1 key/value heads of 512, give GKA 32 query and 4
    key/value heads of 128.

    Raises:
        ValueError: If head_dim is larger than max_head_dim but not a multiple of it.
    """
    if head_dim <= max_head_dim:
        return head_dim, num_q_heads, num_kv_heads
    if head_dim % max_head_dim:
        raise ValueError(
            f"Attention heads of {head_dim} are larger than the kernels support ({max_head_dim}), and can't be split "
            f"into heads of {max_head_dim}, since {head_dim} is not a multiple of {max_head_dim}."
        )
    split = head_dim // max_head_dim
    return max_head_dim, num_q_heads * split, num_kv_heads * split
