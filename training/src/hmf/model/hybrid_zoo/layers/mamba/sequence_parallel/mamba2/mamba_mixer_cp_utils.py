"""
Mamba2 state passing utilities for zig-zag sequence parallelism.

This module implements P2P state passing for Mamba2 SSM layers in a zig-zag
context parallelism pattern. States are passed sequentially between ranks
to maintain temporal dependencies across sequence chunks.

Tensor shape notation:
    B: Batch size
    H: Number of heads
    D: Head dimension
    N: State dimension
    C: Number of chunks (always 1 in state passing context)
    K: Chunk size (sequence length per chunk)

Note: deltaA_cumprod has shape [B, H, 1, K] where the third dimension (C)
is always 1, representing a single chunk per rank.
"""

import torch
from torch.distributed import get_global_rank


def state_passing_p2p(curr_last_state, deltaA_cumprod, cp_rank, cp_size, cp_group, bs):
    """
    Pass Mamba states between sequence parallel ranks using P2P communication.

    Args:
        curr_last_state: Current chunk's final state [B, H, D, N].
        deltaA_cumprod: Cumulative product of delta*A [B, H, 1, K].
        cp_rank: Current rank in context parallel group.
        cp_size: Number of ranks in context parallel group.
        cp_group: Context parallel process group.
        bs: Batch size.

    Returns:
        Previous chunk's final state for this rank [B, H, D, N].
    """
    return State_Passing_P2P.apply(
        curr_last_state, deltaA_cumprod, cp_rank, cp_size, cp_group, bs
    )


def mamba_state_passing_fwd(deltaA_cumprod, prev_last_state, curr_last_state):
    """
    Forward pass for Mamba state passing computation.

    Computes the contribution of the previous chunk's state to the current chunk
    by applying the cumulative decay factor. The decay factor is extracted from the
    last position of the last chunk in deltaA_cumprod.

    Args:
        deltaA_cumprod: Cumulative product of delta*A [B, H, 1, K].
        prev_last_state: Previous chunk's final state [B, H, D, N].
        curr_last_state: Current chunk's final state [B, H, D, N].

    Returns:
        Updated current state incorporating previous chunk's contribution [B, H, D, N].
    """
    contrib_prev_chunk_to_state = (
        deltaA_cumprod[:, :, -1, -1][:, :, None, None] * prev_last_state
    )
    curr_last_state = curr_last_state + contrib_prev_chunk_to_state

    return curr_last_state


def mamba_state_passing_bwd(
    next_grads, deltaA_cumprod, prev_last_state, curr_last_state
):
    """
    Backward pass for Mamba state passing computation.

    Computes gradients with respect to deltaA_cumprod, prev_last_state, and curr_last_state.

    Args:
        next_grads: Gradient from next rank [B, H, D, N].
        deltaA_cumprod: Cumulative product of delta*A [B, H, 1, K].
        prev_last_state: Previous chunk's final state [B, H, D, N].
        curr_last_state: Current chunk's final state [B, H, D, N].

    Returns:
        Tuple of (grad_deltaA_cumprod [B, H, 1, K], grad_prev_last_state [B, H, D, N], grad_curr_last_state [B, H, D, N]).
    """
    b, h, _, _ = next_grads.shape
    _, _, c, k = deltaA_cumprod.shape

    grad_deltaA_cumprod = prev_last_state * next_grads
    grad_prev_last_state = deltaA_cumprod[:, :, -1, -1][:, :, None, None] * next_grads

    ddeltaA_cumprod = torch.zeros((b, h, c, k), device=next_grads.device)
    ddeltaA_cumprod[:, :, -1, -1] = grad_deltaA_cumprod.sum(dim=(2, 3))

    dprev_last_state = grad_prev_last_state

    dcurr_last_state = next_grads

    return ddeltaA_cumprod, dprev_last_state, dcurr_last_state


class State_Passing_P2P(torch.autograd.Function):
    """
    Autograd function for passing Mamba states between ranks using P2P communication.

    In zig-zag pattern, each rank processes two chunks. This function passes states
    sequentially through ranks for the first half of chunks (forward in time), then
    handles the second half of chunks (backward in time).
    """

    @staticmethod
    def forward(ctx, curr_last_state, deltaA_cumprod, cp_rank, cp_size, cp_group, bs):
        """
        Forward pass: receive state from previous rank, update, and send to next rank.

        Args:
            ctx: Autograd context for saving tensors.
            curr_last_state: Current chunk's final state [B, H, D, N].
            deltaA_cumprod: Cumulative product of delta*A [B, H, 1, K].
            cp_rank: Current rank in context parallel group.
            cp_size: Number of ranks in context parallel group.
            cp_group: Context parallel process group.
            bs: Batch size.

        Returns:
            Concatenated previous states for both chunks [B, H, D, N].
        """
        # Get prev state from last cp_rank, update and send to next cp_rank
        prev_last_state_chunk = []

        # Go from 0 to cp_size for the first half of cp chunks
        for rank in range(cp_size):
            if rank == cp_rank:
                # Get state from previous rank
                prev_last_state = torch.zeros_like(curr_last_state[0 : bs // 2])
                if cp_rank > 0:
                    torch.distributed.recv(
                        prev_last_state,
                        src=get_global_rank(cp_group, cp_rank - 1),
                        group=None,
                    )
                    updated_current_state = mamba_state_passing_fwd(
                        deltaA_cumprod[0 : bs // 2],
                        prev_last_state,
                        curr_last_state[0 : bs // 2],
                    )

                elif cp_rank == 0:
                    updated_current_state = curr_last_state[0 : bs // 2]

                prev_last_state_chunk.append(prev_last_state)

            torch.distributed.barrier(group=cp_group)

            # Send state to next rank
            if cp_rank < cp_size - 1 and rank == cp_rank:
                torch.distributed.send(
                    updated_current_state,
                    dst=get_global_rank(cp_group, cp_rank + 1),
                    group=None,
                )

        # On last cp_rank the two chunks are consecutive in time, compute locally
        if cp_rank == cp_size - 1:
            prev_last_state = updated_current_state
            updated_current_state = mamba_state_passing_fwd(
                deltaA_cumprod[bs // 2 : bs],
                prev_last_state,
                curr_last_state[bs // 2 : bs],
            )
            prev_last_state_chunk.append(prev_last_state)

        torch.distributed.barrier(group=cp_group)

        # Go from cp_size to 0 for the second half of cp chunks
        for rank in range(cp_size - 1, -1, -1):
            if rank == cp_rank:
                # Get state from next rank
                prev_last_state_ = torch.zeros_like(curr_last_state[bs // 2 : bs])

                if cp_rank < cp_size - 1:
                    torch.distributed.recv(
                        prev_last_state_,
                        src=get_global_rank(cp_group, cp_rank + 1),
                        group=None,
                    )
                    updated_current_state = mamba_state_passing_fwd(
                        deltaA_cumprod[bs // 2 : bs],
                        prev_last_state_,
                        curr_last_state[bs // 2 : bs],
                    )
                    prev_last_state_chunk.append(prev_last_state_)

            torch.distributed.barrier(group=cp_group)

            # Send state to previous rank
            if cp_rank > 0 and cp_rank == rank:
                torch.distributed.send(
                    updated_current_state,
                    dst=get_global_rank(cp_group, cp_rank - 1),
                    group=None,
                )

        ctx.cp_rank = cp_rank
        ctx.cp_group = cp_group
        ctx.cp_size = cp_size
        ctx.bs = bs

        cat_prev_last_state_chunk = torch.cat(prev_last_state_chunk, dim=0)
        ctx.save_for_backward(
            curr_last_state, deltaA_cumprod, cat_prev_last_state_chunk
        )
        return cat_prev_last_state_chunk

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass: reverse the forward communication pattern.

        Args:
            ctx: Autograd context with saved tensors.
            grad_output: Gradient from next operation [B, H, D, N].

        Returns:
            Tuple of gradients (grad_curr_last_state, grad_deltaA_cumprod, None, None, None, None).
        """
        cp_rank = ctx.cp_rank
        cp_size = ctx.cp_size
        cp_group = ctx.cp_group
        bs = ctx.bs

        curr_last_state, deltaA_cumprod, prev_last_state_chunk = ctx.saved_tensors

        chunks_grads_A = []
        chunks_grads_current_state = []

        # Do reverse of forward pass
        for rank in range(cp_size):
            if rank == cp_rank:
                # Get gradient from previous rank
                next_grad_output = torch.empty_like(grad_output[bs // 2 : bs])
                if cp_rank > 0:
                    torch.distributed.recv(
                        next_grad_output,
                        src=get_global_rank(cp_group, cp_rank - 1),
                        group=None,
                    )
                    ddeltaA_cumprod, dprev_last_state, dcurr_last_state = mamba_state_passing_bwd(
                        next_grad_output,
                        deltaA_cumprod[bs // 2 : bs],
                        prev_last_state_chunk[bs // 2 : bs],
                        curr_last_state[bs // 2 : bs],
                    )

                elif cp_rank == 0:
                    ddeltaA_cumprod = torch.zeros_like(deltaA_cumprod[bs // 2 : bs])
                    dcurr_last_state = torch.zeros_like(curr_last_state[bs // 2 : bs])
                    dprev_last_state = dcurr_last_state

                chunks_grads_A.append(ddeltaA_cumprod)
                chunks_grads_current_state.append(dcurr_last_state)

            torch.distributed.barrier(group=cp_group)

            # Send gradient to next rank
            if cp_rank < cp_size - 1 and rank == cp_rank:
                torch.distributed.send(
                    grad_output[bs // 2 : bs] + dprev_last_state,
                    dst=get_global_rank(cp_group, cp_rank + 1),
                    group=None,
                )

        # On last cp_rank, compute locally
        if cp_rank == cp_size - 1:
            next_grad_output = grad_output[bs // 2 : bs] + dprev_last_state
            ddeltaA_cumprod, dprev_last_state, dcurr_last_state = mamba_state_passing_bwd(
                next_grad_output,
                deltaA_cumprod[0 : bs // 2],
                prev_last_state_chunk[0 : bs // 2],
                curr_last_state[0 : bs // 2],
            )
            chunks_grads_A.append(ddeltaA_cumprod)
            chunks_grads_current_state.append(dcurr_last_state)

        torch.distributed.barrier(group=cp_group)

        # Go from cp_size to 0 for the second half of cp chunks
        for rank in range(cp_size - 1, -1, -1):
            if rank == cp_rank:
                # Get gradient from next rank
                next_grad_output_ = torch.empty_like(grad_output[0 : bs // 2])

                if cp_rank < cp_size - 1:
                    torch.distributed.recv(
                        next_grad_output_,
                        src=get_global_rank(cp_group, cp_rank + 1),
                        group=None,
                    )
                    ddeltaA_cumprod, dprev_last_state, dcurr_last_state = mamba_state_passing_bwd(
                        next_grad_output_,
                        deltaA_cumprod[0 : bs // 2],
                        prev_last_state_chunk[0 : bs // 2],
                        curr_last_state[0 : bs // 2],
                    )
                    chunks_grads_A.append(ddeltaA_cumprod)
                    chunks_grads_current_state.append(dcurr_last_state)

            torch.distributed.barrier(group=cp_group)

            # Send gradient to previous rank
            if cp_rank > 0 and cp_rank == rank:
                torch.distributed.send(
                    grad_output[0 : bs // 2] + dprev_last_state,
                    dst=get_global_rank(cp_group, cp_rank - 1),
                    group=None,
                )

        dcurr_last_state = torch.cat(chunks_grads_current_state[::-1], dim=0)
        ddA_cumsum_grad_output = torch.cat(chunks_grads_A[::-1], dim=0)

        return dcurr_last_state, ddA_cumsum_grad_output, None, None, None, None


