"""Turns the mask a model passes to a linear sequence mixer into the padding mask the mixer expects."""

from typing import Optional

import torch


def padding_mask_for_new_tokens(attention_mask: Optional[torch.Tensor], q_len: int) -> Optional[torch.Tensor]:
    """
    Returns a [batch, q_len] mask of the new tokens (1 for real tokens, 0 for padding), or None if none of them
    is padding.

    Models pass their linear sequence mixers (GDN, GKA) the mask they build for attention, which is a 2D padding
    mask under flash attention and a 4D mask of shape [batch, 1, q_len, kv_len] under SDPA and eager. In a 4D
    mask, the last query attends to every real key, whether the batch is left- or right-padded, so its row marks
    the padding. SDPA masks are boolean (True to attend) and eager masks are additive (0 to attend).

    Args:
        attention_mask: A 2D padding mask, a 4D attention mask, or None.
        q_len: The number of new tokens, which are the last q_len keys of the mask.
    """
    if attention_mask is None:
        return None
    if attention_mask.dim() == 4:
        row = attention_mask[:, 0, -1, -q_len:]
        mask = row if row.dtype == torch.bool else row == 0
    elif attention_mask.dim() == 2:
        mask = attention_mask[:, -q_len:].bool()
    else:
        raise ValueError(f"Expected a 2D padding mask or a 4D attention mask, got shape {tuple(attention_mask.shape)}")
    return None if mask.all() else mask.int()
