"""
Sequence parallelism (SP) for hybrid Gemma 4's attention layers, in the zigzag layout of HMF's SP: the sequence is
split into 2 * sp_size chunks of equal length, and rank r holds chunks r and 2 * sp_size - 1 - r, one after the
other. Ring attention, which SP registers for other models' attention, runs on flash attention, which supports
heads of at most 256 and no sliding window, while Gemma 4's full attention layers have heads of 512, its sliding
layers have a window, and its last layers reuse the keys and values of earlier layers instead of computing their
own. So Gemma 4's attention exchanges keys and values instead:

    - Full attention (sp_full_attention) gathers every rank's keys and values, and runs SDPA for each local chunk's
      queries against the keys up to the chunk's end. On CUDA it forces SDPA's memory-efficient kernel, the only
      one that runs heads of 512 at long sequences (the math kernel needs tens of GB per layer at 128K tokens).
    - Sliding window attention (sp_windowed_attention) gathers the last window - 1 keys and values of every chunk,
      and runs each local chunk's queries against the tail of the chunk before it and its own keys, with flash
      attention 2 and its window where it can, or with SDPA and the window as a mask otherwise.

Both run on whichever keys and values the layer has, its own or those that it shares, and the all-gathers pass the
gradients of the gathered keys and values back to the ranks that computed them.
"""

from contextlib import nullcontext

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.attention.bias import causal_lower_right

try:
    from flash_attn import flash_attn_func
except ImportError:
    flash_attn_func = None

# The largest head size of flash attention 2
FLASH_ATTENTION_MAX_HEAD_DIM = 256


def zigzag_chunk_indices(sp_rank: int, sp_size: int) -> tuple[int, int]:
    """Returns the indices, in the whole sequence, of the two chunks that a rank holds."""
    return sp_rank, 2 * sp_size - 1 - sp_rank


def _repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeats each key and value head n_rep times, as grouped-query attention pairs them with the query heads."""
    return hidden_states if n_rep == 1 else hidden_states.repeat_interleave(n_rep, dim=1)


class _AllGather(torch.autograd.Function):
    """
    Gathers a tensor from every rank of a group, and in the backward pass sums the gradients that each rank's
    gathered copies get and returns each rank its own. The sum uses a reduce-scatter on NCCL, and an all-reduce on
    other backends (such as gloo, which has no reduce-scatter).
    """

    @staticmethod
    def forward(ctx, tensor: torch.Tensor, group: dist.ProcessGroup) -> tuple[torch.Tensor, ...]:
        ctx.group = group
        gathered = [torch.empty_like(tensor) for _ in range(dist.get_world_size(group=group))]
        dist.all_gather(gathered, tensor.contiguous(), group=group)
        return tuple(gathered)

    @staticmethod
    def backward(ctx, *grads: torch.Tensor) -> tuple[torch.Tensor, None]:
        rank = dist.get_rank(group=ctx.group)
        if dist.get_backend(ctx.group) == dist.Backend.NCCL:
            grad = torch.empty_like(grads[rank])
            dist.reduce_scatter_tensor(grad, torch.cat([g.contiguous() for g in grads]), group=ctx.group)
            return grad, None
        summed = torch.stack(grads)
        dist.all_reduce(summed, group=ctx.group)
        return summed[rank], None


def _sdpa_kernels(device: torch.device):
    """
    Returns a new context manager for the SDPA kernels of full attention (one can be entered only once): the
    memory-efficient kernel on CUDA, the only one that runs heads of 512 at long sequences, and any on other devices.
    """
    return sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION) if device.type == "cuda" else nullcontext()


def _chunks_in_sequence_order(local: torch.Tensor, group: dist.ProcessGroup) -> list[torch.Tensor]:
    """
    Gathers a tensor of shape (batch, heads, 2, ...) from every rank, whose third dimension holds a slice for each of
    the rank's two chunks, and returns one slice per chunk of the whole sequence, in the sequence's order.
    """
    sp_size = dist.get_world_size(group=group)
    gathered = _AllGather.apply(local, group)
    chunks = []
    for chunk in range(2 * sp_size):
        rank, half = (chunk, 0) if chunk < sp_size else (2 * sp_size - 1 - chunk, 1)
        chunks.append(gathered[rank][:, :, half])
    return chunks


def sp_full_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
    group: dist.ProcessGroup,
    dropout: float = 0.0,
) -> torch.Tensor:
    """
    Returns the causal attention of a rank's queries against the keys and values of the whole sequence.

    Args:
        query: The rank's queries, of shape (batch, heads, 2 * chunk_len, head_dim), its two chunks one after the
            other.
        key: The rank's keys, of shape (batch, kv_heads, 2 * chunk_len, head_dim).
        value: The rank's values, of the same shape as key.
        scale: The scale of the attention scores.
        group: The sequence parallel group.
        dropout: The dropout probability of the attention weights.

    Returns:
        The attention output of the rank's queries, of shape (batch, heads, 2 * chunk_len, head_dim).
    """
    sp_rank, sp_size = dist.get_rank(group=group), dist.get_world_size(group=group)
    batch, heads, local_len, head_dim = query.shape
    chunk_len = local_len // 2
    n_rep = heads // key.shape[1]

    def whole(local):
        chunks = _chunks_in_sequence_order(local.reshape(*local.shape[:2], 2, chunk_len, head_dim), group)
        return torch.cat(chunks, dim=2)

    keys, values = _repeat_kv(whole(key), n_rep), _repeat_kv(whole(value), n_rep)
    outputs = []
    for half, chunk in enumerate(zigzag_chunk_indices(sp_rank, sp_size)):
        end = (chunk + 1) * chunk_len
        with _sdpa_kernels(query.device):
            outputs.append(
                F.scaled_dot_product_attention(
                    query[:, :, half * chunk_len : (half + 1) * chunk_len],
                    keys[:, :, :end],
                    values[:, :, :end],
                    attn_mask=causal_lower_right(chunk_len, end),
                    dropout_p=dropout,
                    scale=scale,
                )
            )
    return torch.cat(outputs, dim=2)


def _windowed_attention(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, window: int, scale: float, dropout: float
) -> torch.Tensor:
    """
    Returns the attention of a chunk's queries against keys that end with the chunk's own, in which each query
    attends to the keys up to its own position and fewer than window positions back, with the shapes of
    sp_windowed_attention. It uses flash attention 2 where it can, and SDPA with the window as a mask otherwise.
    """
    q_len, k_len = query.shape[2], key.shape[2]
    if (
        flash_attn_func is not None
        and query.is_cuda
        and query.dtype in (torch.float16, torch.bfloat16)
        and query.shape[-1] <= FLASH_ATTENTION_MAX_HEAD_DIM
    ):
        # Flash attention aligns the causal mask to the last keys when there are more keys than queries
        output = flash_attn_func(
            query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2), dropout_p=dropout,
            softmax_scale=scale, causal=True, window_size=(window - 1, 0),
        )
        return output.transpose(1, 2)
    query_positions = torch.arange(k_len - q_len, k_len, device=query.device)[:, None]
    key_positions = torch.arange(k_len, device=query.device)[None, :]
    mask = (key_positions <= query_positions) & (query_positions - key_positions < window)
    n_rep = query.shape[1] // key.shape[1]
    return F.scaled_dot_product_attention(
        query, _repeat_kv(key, n_rep), _repeat_kv(value, n_rep), attn_mask=mask, dropout_p=dropout, scale=scale
    )


def sp_windowed_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    window: int,
    scale: float,
    group: dist.ProcessGroup,
    dropout: float = 0.0,
) -> torch.Tensor:
    """
    Returns the causal sliding window attention of a rank's queries, in which each query attends to the keys at its
    own position and fewer than window positions back, which may lie in the chunk before (on another rank).

    Args:
        query: The rank's queries, of shape (batch, heads, 2 * chunk_len, head_dim), its two chunks one after the
            other.
        key: The rank's keys, of shape (batch, kv_heads, 2 * chunk_len, head_dim).
        value: The rank's values, of the same shape as key.
        window: The window, as in transformers' sliding_window: a query attends to window keys, its own included.
        scale: The scale of the attention scores.
        group: The sequence parallel group.
        dropout: The dropout probability of the attention weights.

    Returns:
        The attention output of the rank's queries, of shape (batch, heads, 2 * chunk_len, head_dim).

    Raises:
        ValueError: If a chunk is shorter than window - 1, since a query would then reach further back than the
            chunk before.
    """
    sp_rank, sp_size = dist.get_rank(group=group), dist.get_world_size(group=group)
    local_len = query.shape[2]
    chunk_len = local_len // 2
    tail = window - 1
    if chunk_len < tail:
        raise ValueError(
            f"Sliding window attention under sequence parallelism needs chunks of at least window - 1 = {tail} "
            f"tokens, got {chunk_len} ({local_len} tokens per rank). Use longer sequences or fewer ranks."
        )

    def chunked(local):
        return local.reshape(*local.shape[:2], 2, chunk_len, local.shape[-1])

    key_chunks, value_chunks = chunked(key), chunked(value)
    if tail > 0:
        key_tails = _chunks_in_sequence_order(key_chunks[:, :, :, chunk_len - tail :], group)
        value_tails = _chunks_in_sequence_order(value_chunks[:, :, :, chunk_len - tail :], group)
    outputs = []
    for half, chunk in enumerate(zigzag_chunk_indices(sp_rank, sp_size)):
        chunk_keys, chunk_values = key_chunks[:, :, half], value_chunks[:, :, half]
        if tail > 0 and chunk > 0:
            chunk_keys = torch.cat([key_tails[chunk - 1], chunk_keys], dim=2)
            chunk_values = torch.cat([value_tails[chunk - 1], chunk_values], dim=2)
        outputs.append(
            _windowed_attention(
                query[:, :, half * chunk_len : (half + 1) * chunk_len], chunk_keys, chunk_values, window, scale,
                dropout,
            )
        )
    return torch.cat(outputs, dim=2)
