"""
Shared sequence parallelism P2P utilities for zig-zag context parallelism.

This module provides the reorder-and-exchange primitive used by multiple layer types
(Mamba, SWA, GKA, BMojo) to exchange convolution/window context between sequence
parallel ranks in a zig-zag pattern. Each GPU processes two discontiguous sequence
chunks for load balancing.

Tensor shape notation:
    B: Batch size
    L: Sequence length
    Dxz: Projection dimension (model dimension before head split)
"""

import torch
import torch.nn.functional as F
from torch.distributed import get_global_rank

class Reorder_For_SSM_P2P(torch.autograd.Function):
    """
    Autograd function for reordering sequence chunks and exchanging convolution context.

    In zig-zag pattern, each rank holds two discontiguous chunks. This function exchanges
    the last d_conv-1 timesteps between adjacent chunks to provide convolution context.
    """

    @staticmethod
    def forward(ctx, xz, cp_group, cp_stream, cp_size, cp_rank, d_conv):
        """
        Forward pass: exchange convolution context between adjacent sequence chunks.

        Args:
            ctx: Autograd context for saving tensors.
            xz: Input tensor [B, L, Dxz].
            cp_group: Context parallel process group.
            cp_stream: CUDA stream for context parallel operations.
            cp_size: Number of ranks in context parallel group.
            cp_rank: Current rank in context parallel group.
            d_conv: Convolution kernel size.

        Returns:
            Tensor with convolution context prepended [2*B, L//2 + d_conv - 1, Dxz].
        """
        xz = xz.contiguous()

        # Reshape for zig-zag pattern: [B, L, Dxz] -> [2*B, L//2, Dxz]
        b, s, d = xz.shape

        assert (
            s // 2 >= d_conv - 1
        ), f"SP chunk-size {s // 2} smaller than dconv size {d_conv} not supported."
        if b == 1:
            xz = xz.view(2 * b, s // 2, d)
        else:
            xz = torch.cat([xz[:, : s // 2], xz[:, s // 2 :]], dim=0).contiguous()

        ops = []

        for i in range(cp_size):
            if cp_rank == i:
                # In zig-zag pattern each rank has 2 discontiguous chunks stacked along dim=0
                if cp_rank % 2 == 0:
                    # Even ranks: send first, then receive
                    if cp_rank < cp_size - 1:
                        # Chunk 0: send last d_conv-1 timesteps to next rank
                        send_tensor_0 = xz[0:b, -d_conv + 1 :, :].contiguous()
                        ops.append(
                            torch.distributed.P2POp(
                                torch.distributed.isend,
                                tensor=send_tensor_0,
                                peer=get_global_rank(cp_group, cp_rank + 1),
                                group=cp_group,
                            )
                        )

                    if cp_rank > 0:
                        recv_tensor_0 = torch.empty_like(
                            xz[0:b, -d_conv + 1 :, :]
                        ).contiguous()
                        ops.append(
                            torch.distributed.P2POp(
                                torch.distributed.irecv,
                                tensor=recv_tensor_0,
                                peer=get_global_rank(cp_group, cp_rank - 1),
                                group=cp_group,
                            )
                        )

                    if cp_rank > 0:
                        # Chunk 1: send last d_conv-1 timesteps to previous rank
                        send_tensor_1 = xz[b : 2 * b, -d_conv + 1 :, :].contiguous()
                        ops.append(
                            torch.distributed.P2POp(
                                torch.distributed.isend,
                                tensor=send_tensor_1,
                                peer=get_global_rank(cp_group, cp_rank - 1),
                                group=cp_group,
                            )
                        )

                    if cp_rank < cp_size - 1:
                        recv_tensor_1 = torch.empty_like(
                            xz[b : 2 * b, -d_conv + 1 :, :]
                        ).contiguous()
                        ops.append(
                            torch.distributed.P2POp(
                                torch.distributed.irecv,
                                tensor=recv_tensor_1,
                                peer=get_global_rank(cp_group, cp_rank + 1),
                                group=cp_group,
                            )
                        )

                if cp_rank % 2 == 1:
                    # Odd ranks: receive first, then send
                    if cp_rank > 0:
                        recv_tensor_0 = torch.empty_like(
                            xz[0:b, -d_conv + 1 :, :]
                        ).contiguous()
                        ops.append(
                            torch.distributed.P2POp(
                                torch.distributed.irecv,
                                recv_tensor_0,
                                get_global_rank(cp_group, cp_rank - 1),
                                cp_group,
                            )
                        )

                    if cp_rank < cp_size - 1:
                        # Chunk 0: send last d_conv-1 timesteps to next rank
                        send_tensor_0 = xz[0:b, -d_conv + 1 :, :].contiguous()
                        ops.append(
                            torch.distributed.P2POp(
                                torch.distributed.isend,
                                tensor=send_tensor_0,
                                peer=get_global_rank(cp_group, cp_rank + 1),
                                group=cp_group,
                            )
                        )

                    if cp_rank < cp_size - 1:
                        recv_tensor_1 = torch.empty_like(
                            xz[b : 2 * b, -d_conv + 1 :, :]
                        ).contiguous()
                        ops.append(
                            torch.distributed.P2POp(
                                torch.distributed.irecv,
                                tensor=recv_tensor_1,
                                peer=get_global_rank(cp_group, cp_rank + 1),
                                group=cp_group,
                            )
                        )

                    if cp_rank > 0:
                        # Chunk 1: send last d_conv-1 timesteps to previous rank
                        send_tensor_1 = xz[b : 2 * b, -d_conv + 1 :, :].contiguous()
                        ops.append(
                            torch.distributed.P2POp(
                                torch.distributed.isend,
                                tensor=send_tensor_1,
                                peer=get_global_rank(cp_group, cp_rank - 1),
                                group=cp_group,
                            )
                        )

        reqs = torch.distributed.batch_isend_irecv(ops)

        for req in reqs:
            req.wait()

        torch.distributed.barrier(group=cp_group)

        # Pad tensors to prepend the last d_conv-1 states from previous chunk
        xz_ = F.pad(xz, (0, 0, d_conv - 1, 0, 0, 0))

        if cp_rank > 0:
            xz_[0:b, : d_conv - 1] = xz_[0:b, : d_conv - 1] + recv_tensor_0

        if cp_rank < cp_size - 1:
            xz_[b : 2 * b, : d_conv - 1] = xz_[b : 2 * b, : d_conv - 1] + recv_tensor_1

        if cp_rank == cp_size - 1:
            # Last rank: chunks are consecutive, use local data
            xz_[b : 2 * b, : d_conv - 1] = (
                xz_[b : 2 * b, : d_conv - 1] + xz[0:b, -d_conv + 1 :, :]
            )

        ctx.cp_rank = cp_rank
        ctx.cp_group = cp_group
        ctx.d_conv = d_conv
        ctx.cp_size = cp_size
        ctx.cp_stream = cp_stream
        return xz_

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass: reverse the reordering and exchange gradients.

        Args:
            ctx: Autograd context with saved values.
            grad_output: Gradient from next operation [2*B, L//2 + d_conv - 1, Dxz].

        Returns:
            Tuple of (grad_input [B, L, Dxz], None, None, None, None, None).
        """
        recv_tensor_0 = None
        recv_tensor_1 = None
        ops = []
        b, l, d = grad_output.shape

        for i in range(ctx.cp_size - 1, -1, -1):
            if ctx.cp_rank == i:
                # In zig-zag pattern each rank has 2 discontiguous chunks stacked along dim=0
                if ctx.cp_rank % 2 == 0:
                    # Even ranks: send first, then receive
                    if ctx.cp_rank > 0:
                        # Chunk 0: send first d_conv-1 gradient timesteps to previous rank
                        send_tensor_0 = grad_output[
                            0 : b // 2, : ctx.d_conv - 1, :
                        ].contiguous()
                        ops.append(
                            torch.distributed.P2POp(
                                torch.distributed.isend,
                                tensor=send_tensor_0,
                                peer=get_global_rank(ctx.cp_group, ctx.cp_rank - 1),
                                group=ctx.cp_group,
                            )
                        )

                    if ctx.cp_rank < ctx.cp_size - 1:
                        recv_tensor_0 = torch.empty_like(
                            grad_output[0 : b // 2, : ctx.d_conv - 1, :]
                        ).contiguous()
                        ops.append(
                            torch.distributed.P2POp(
                                torch.distributed.irecv,
                                tensor=recv_tensor_0,
                                peer=get_global_rank(ctx.cp_group, ctx.cp_rank + 1),
                                group=ctx.cp_group,
                            )
                        )

                    if ctx.cp_rank < ctx.cp_size - 1:
                        # Chunk 1: send first d_conv-1 gradient timesteps to next rank
                        send_tensor_1 = grad_output[
                            b // 2 : b, : ctx.d_conv - 1, :
                        ].contiguous()
                        ops.append(
                            torch.distributed.P2POp(
                                torch.distributed.isend,
                                tensor=send_tensor_1,
                                peer=get_global_rank(ctx.cp_group, ctx.cp_rank + 1),
                                group=ctx.cp_group,
                            )
                        )

                    if ctx.cp_rank > 0:
                        recv_tensor_1 = torch.empty_like(
                            grad_output[b // 2 : b, : ctx.d_conv - 1, :]
                        ).contiguous()
                        ops.append(
                            torch.distributed.P2POp(
                                torch.distributed.irecv,
                                tensor=recv_tensor_1,
                                peer=get_global_rank(ctx.cp_group, ctx.cp_rank - 1),
                                group=ctx.cp_group,
                            )
                        )

                if ctx.cp_rank % 2 == 1:
                    # Odd ranks: receive first, then send
                    if ctx.cp_rank < ctx.cp_size - 1:
                        recv_tensor_0 = torch.empty_like(
                            grad_output[0 : b // 2, : ctx.d_conv - 1, :]
                        ).contiguous()
                        ops.append(
                            torch.distributed.P2POp(
                                torch.distributed.irecv,
                                tensor=recv_tensor_0,
                                peer=get_global_rank(ctx.cp_group, ctx.cp_rank + 1),
                                group=ctx.cp_group,
                            )
                        )

                    if ctx.cp_rank > 0:
                        # Chunk 0: send first d_conv-1 gradient timesteps to previous rank
                        send_tensor_0 = grad_output[
                            0 : b // 2, : ctx.d_conv - 1, :
                        ].contiguous()
                        ops.append(
                            torch.distributed.P2POp(
                                torch.distributed.isend,
                                tensor=send_tensor_0,
                                peer=get_global_rank(ctx.cp_group, ctx.cp_rank - 1),
                                group=ctx.cp_group,
                            )
                        )

                    if ctx.cp_rank > 0:
                        recv_tensor_1 = torch.empty_like(
                            grad_output[b // 2 : b, : ctx.d_conv - 1, :]
                        ).contiguous()
                        ops.append(
                            torch.distributed.P2POp(
                                torch.distributed.irecv,
                                tensor=recv_tensor_1,
                                peer=get_global_rank(ctx.cp_group, ctx.cp_rank - 1),
                                group=ctx.cp_group,
                            )
                        )

                    if ctx.cp_rank < ctx.cp_size - 1:
                        # Chunk 1: send first d_conv-1 gradient timesteps to next rank
                        send_tensor_1 = grad_output[
                            b // 2 : b, : ctx.d_conv - 1, :
                        ].contiguous()
                        ops.append(
                            torch.distributed.P2POp(
                                torch.distributed.isend,
                                tensor=send_tensor_1,
                                peer=get_global_rank(ctx.cp_group, ctx.cp_rank + 1),
                                group=ctx.cp_group,
                            )
                        )

        reqs = torch.distributed.batch_isend_irecv(ops)

        for req in reqs:
            req.wait()

        torch.distributed.barrier(group=ctx.cp_group)

        # Accumulate gradients from adjacent chunks
        if ctx.cp_rank < ctx.cp_size - 1:
            grad_output[0 : b // 2, -ctx.d_conv + 1 :] = (
                grad_output[0 : b // 2, -ctx.d_conv + 1 :] + recv_tensor_0
            )

        if ctx.cp_rank > 0:
            grad_output[b // 2 : b, -ctx.d_conv + 1 :] = (
                grad_output[b // 2 : b, -ctx.d_conv + 1 :] + recv_tensor_1
            )

        if ctx.cp_rank == ctx.cp_size - 1:
            # Last rank: accumulate gradients from consecutive chunks locally
            grad_output[0 : b // 2, -ctx.d_conv + 1 :] = (
                grad_output[0 : b // 2, -ctx.d_conv + 1 :]
                + grad_output[b // 2 : b, : ctx.d_conv - 1]
            )

        # Remove the prepended d_conv-1 timesteps
        grad_input = grad_output[:, ctx.d_conv - 1 :, :].contiguous()

        # Reshape back to original format: [2*B, L//2, Dxz] -> [B, L, Dxz]
        if b == 2:
            grad_input = grad_input.view(b // 2, (l - ctx.d_conv + 1) * 2, d)
        else:
            grad_input = torch.cat(
                [grad_input[: b // 2], grad_input[b // 2 :]], dim=1
            ).contiguous()

        return grad_input, None, None, None, None, None

def reorder_for_ssm_p2p(xz, cp_group, cp_stream, cp_size, cp_rank, d_conv):
    """
    Reorder and exchange convolution context between sequence parallel ranks.

    Args:
        xz: Input tensor [B, L, Dxz].
        cp_group: Context parallel process group.
        cp_stream: CUDA stream for context parallel operations.
        cp_size: Number of ranks in context parallel group.
        cp_rank: Current rank in context parallel group.
        d_conv: Convolution kernel size.

    Returns:
        Reordered tensor with convolution context from adjacent chunks [2*B, L//2 + d_conv - 1, Dxz].
    """
    return Reorder_For_SSM_P2P.apply(xz, cp_group, cp_stream, cp_size, cp_rank, d_conv)