"""
Sequence parallelism utilities for zig-zag context parallel pattern.

This module provides utilities for gathering and scattering tensors across GPUs
in a zig-zag pattern for sequence parallelism. The zig-zag pattern distributes
sequence chunks across GPUs in a way that balances load and minimizes communication
overhead.

Zig-Zag Pattern Example (SP=4, NC=8):
    GPU 0: chunks [0, 7]
    GPU 1: chunks [1, 6]
    GPU 2: chunks [2, 5]
    GPU 3: chunks [3, 4]

    Pattern: 0→1→2→3→3→2→1→0

Throughout this module, tensor shapes use the following notation:
    B: Batch size
    L: Full sequence length
    L_local: Local sequence length per GPU
    Dm: Model/hidden dimension
    SP: Context/sequence parallel size (cp_size in code)
    NC: Total number of chunks across all GPUs (num_chunks in code, typically 2 × SP)
    C: Chunk size (L // NC)

Note: Each GPU holds (NC // SP) chunks, so L_local = C × (NC // SP).
    When NC = 2 × SP (the typical case for the zig-zag pattern), L_local = L // SP.
"""

import torch
import torch.distributed as dist
from torch.autograd import Function
from typing import List


def get_zigzag_indices(cp_size: int, num_chunks: int) -> List[List[int]]:
    """
    Create zig-zag distribution pattern for sequence chunks across GPUs.

    The zig-zag pattern alternates the direction of chunk assignment across "waves"
    where each wave assigns one chunk to each GPU. Even waves go forward (0→SP-1),
    odd waves go backward (SP-1→0).

    Args:
        cp_size: Number of GPUs in context parallel group (SP).
        num_chunks: Total number of chunks to distribute (NC).

    Returns:
        List of lists where indices[gpu_id] contains the chunk IDs assigned to that GPU.

    Example:
        >>> get_zigzag_indices(cp_size=4, num_chunks=8)  # SP=4, NC=8
        [[0, 7], [1, 6], [2, 5], [3, 4]]
    """
    indices = [[] for _ in range(cp_size)]

    for chunk_id in range(num_chunks):
        # Determine which "wave" we're in
        wave = chunk_id // cp_size
        position_in_wave = chunk_id % cp_size

        # Even waves go forward (0,1,2,3), odd waves go backward (3,2,1,0)
        if wave % 2 == 0:
            gpu_id = position_in_wave
        else:
            gpu_id = cp_size - 1 - position_in_wave

        indices[gpu_id].append(chunk_id)

    return indices


class ZigZagGatherScatter(Function):
    """
    Autograd function for gathering zig-zag distributed chunks into contiguous sequence.

    Forward: Gather all chunks from all GPUs → Rearrange to contiguous order → Return full sequence
    Backward: Split gradient → Rearrange to zig-zag order → Return local gradient
    """

    @staticmethod
    def forward(ctx, input_tensor, cp_group, cp_rank, cp_size, num_chunks):
        """
        Gather zig-zag distributed chunks from all GPUs and rearrange to contiguous sequence.

        Args:
            ctx: Autograd context for saving tensors.
            input_tensor: Local chunks concatenated [B, L_local, Dm].
            cp_group: Context parallel process group.
            cp_rank: Current rank in context parallel group.
            cp_size: Number of ranks in context parallel group (SP).
            num_chunks: Total number of chunks across all GPUs (NC).

        Returns:
            Full sequence with chunks in contiguous order [B, L, Dm].
        """
        ctx.cp_group = cp_group
        ctx.cp_rank = cp_rank
        ctx.cp_size = cp_size
        ctx.num_chunks = num_chunks

        input_tensor = input_tensor.contiguous()

        batch_size, local_seq_len, hidden_dim = input_tensor.shape
        ctx.local_seq_len = local_seq_len

        # Each GPU has multiple chunks, calculate chunk size
        chunks_per_gpu = num_chunks // cp_size
        chunk_size = local_seq_len // chunks_per_gpu
        ctx.chunk_size = chunk_size
        ctx.chunks_per_gpu = chunks_per_gpu

        # Gather all chunks from all GPUs
        gathered_tensors = [torch.empty_like(input_tensor) for _ in range(cp_size)]
        dist.all_gather(gathered_tensors, input_tensor, group=cp_group)

        # Get zig-zag mapping
        zigzag_map = get_zigzag_indices(cp_size, num_chunks)

        # Rearrange from zig-zag to contiguous order
        contiguous_chunks = [None] * num_chunks

        for gpu_rank in range(cp_size):
            # Split this GPU's tensor into its chunks
            gpu_tensor = gathered_tensors[gpu_rank]
            gpu_chunks = list(torch.chunk(gpu_tensor, chunks_per_gpu, dim=1))

            # Place each chunk in its contiguous position
            for local_idx, chunk_id in enumerate(zigzag_map[gpu_rank]):
                contiguous_chunks[chunk_id] = gpu_chunks[local_idx].contiguous()

        # Concatenate to form full sequence
        full_sequence = torch.cat(contiguous_chunks, dim=1)

        return full_sequence

    @staticmethod
    def backward(ctx, grad_output):
        """
        Scatter gradient from contiguous sequence back to zig-zag distributed chunks.

        Args:
            ctx: Autograd context with saved values.
            grad_output: Gradient of full sequence [B, L, Dm].

        Returns:
            Tuple of (grad_input [B, L_local, Dm], None, None, None, None).
        """
        cp_group = ctx.cp_group
        cp_rank = ctx.cp_rank
        cp_size = ctx.cp_size
        num_chunks = ctx.num_chunks
        chunk_size = ctx.chunk_size
        chunks_per_gpu = ctx.chunks_per_gpu

        grad_output = grad_output.contiguous()

        # Split gradient into contiguous chunks
        grad_chunks = list(torch.chunk(grad_output, num_chunks, dim=1))
        grad_chunks = [chunk.contiguous() for chunk in grad_chunks]

        # Get zig-zag mapping
        zigzag_map = get_zigzag_indices(cp_size, num_chunks)

        # Rearrange from contiguous to zig-zag order for this rank
        local_grad_chunks = []
        for chunk_id in zigzag_map[cp_rank]:
            local_grad_chunks.append(grad_chunks[chunk_id])

        # Concatenate local chunks
        local_grad = torch.cat(local_grad_chunks, dim=1).contiguous()

        # Scale by cp_size to account for all_gather in forward pass
        return local_grad * cp_size, None, None, None, None


class ZigZagScatter(Function):
    """
    Autograd function for scattering contiguous sequence into zig-zag distributed chunks.

    Forward: Split full sequence → Rearrange to zig-zag order → Return local chunks
    Backward: Gather local grads → Rearrange to contiguous order → Return full gradient
    """

    @staticmethod
    def forward(ctx, full_tensor, cp_group, cp_rank, cp_size, num_chunks):
        """
        Scatter contiguous sequence into zig-zag distributed chunks for this rank.

        Args:
            ctx: Autograd context for saving tensors.
            full_tensor: Full sequence [B, L, Dm].
            cp_group: Context parallel process group.
            cp_rank: Current rank in context parallel group.
            cp_size: Number of ranks in context parallel group (SP).
            num_chunks: Total number of chunks across all GPUs (NC).

        Returns:
            Local chunks for this rank concatenated [B, L_local, Dm].
        """
        ctx.cp_group = cp_group
        ctx.cp_rank = cp_rank
        ctx.cp_size = cp_size
        ctx.num_chunks = num_chunks

        full_tensor = full_tensor.contiguous()

        batch_size, full_seq_len, hidden_dim = full_tensor.shape
        chunk_size = full_seq_len // num_chunks
        chunks_per_gpu = num_chunks // cp_size
        local_seq_len = chunk_size * chunks_per_gpu

        ctx.chunk_size = chunk_size
        ctx.chunks_per_gpu = chunks_per_gpu
        ctx.local_seq_len = local_seq_len

        # Split into contiguous chunks
        chunks = list(torch.chunk(full_tensor, num_chunks, dim=1))
        chunks = [chunk.contiguous() for chunk in chunks]

        # Get zig-zag mapping
        zigzag_map = get_zigzag_indices(cp_size, num_chunks)

        # Get the chunks for this rank
        local_chunks = []
        for chunk_id in zigzag_map[cp_rank]:
            local_chunks.append(chunks[chunk_id])

        # Concatenate local chunks
        local_tensor = torch.cat(local_chunks, dim=1).contiguous()

        return local_tensor

    @staticmethod
    def backward(ctx, grad_local):
        """
        Gather zig-zag distributed gradients and rearrange to contiguous sequence.

        Args:
            ctx: Autograd context with saved values.
            grad_local: Gradient of local chunks [B, L_local, Dm].

        Returns:
            Tuple of (grad_full [B, L, Dm], None, None, None, None).
        """
        cp_group = ctx.cp_group
        cp_rank = ctx.cp_rank
        cp_size = ctx.cp_size
        num_chunks = ctx.num_chunks
        chunk_size = ctx.chunk_size
        chunks_per_gpu = ctx.chunks_per_gpu

        grad_local = grad_local.contiguous()

        # Gather all gradients from all GPUs
        gathered_grads = [torch.empty_like(grad_local) for _ in range(cp_size)]
        dist.all_gather(gathered_grads, grad_local, group=cp_group)

        # Get zig-zag mapping
        zigzag_map = get_zigzag_indices(cp_size, num_chunks)

        # Rearrange from zig-zag to contiguous order
        contiguous_grads = [None] * num_chunks

        for gpu_rank in range(cp_size):
            # Split this GPU's gradient into its chunks
            gpu_grad = gathered_grads[gpu_rank]
            gpu_grad_chunks = list(torch.chunk(gpu_grad, chunks_per_gpu, dim=1))

            # Place each chunk in its contiguous position
            for local_idx, chunk_id in enumerate(zigzag_map[gpu_rank]):
                contiguous_grads[chunk_id] = gpu_grad_chunks[local_idx].contiguous()

        # Concatenate to form full gradient
        grad_full = torch.cat(contiguous_grads, dim=1)

        # Scale by 1/cp_size to account for all_gather
        return grad_full / cp_size, None, None, None, None
