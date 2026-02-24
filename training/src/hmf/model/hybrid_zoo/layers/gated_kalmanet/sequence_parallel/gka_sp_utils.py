"""
Sequence parallel utilities for Gated KalmaNet layer.
Handles state passing for kk, gkk, and h_kv states across chunks in zigzag pattern.
"""

import torch
import torch.distributed as dist
from torch.distributed import get_global_rank
from einops import rearrange

from fla.ops.simple_gla import chunk_simple_gla


def gka_state_passing_bwd_kk(grad_output, curr_gkk, prev_kk, curr_kk):
    """
    Backward pass for kk state passing.

    Forward: updated_kk = curr_kk + exp(curr_gkk) * prev_kk

    Args:
        grad_output: Gradient w.r.t. updated_kk [B, H, D, D]
        curr_gkk: gkk cumsum at LAST token (used in forward) [B, H]
        prev_kk: Previous kk state [B, H, D, D]
        curr_kk: Current kk state [B, H, D, D]

    Returns:
        grad_curr_gkk: Gradient w.r.t. curr_gkk [B, H]
        grad_prev_kk: Gradient w.r.t. prev_kk [B, H, D, D]
        grad_curr_kk_state: Gradient w.r.t. curr_kk [B, H, D, D]
    """
    if curr_gkk is not None and prev_kk is not None:
        decay = torch.exp(curr_gkk)  # [B, H]

        # Gradient w.r.t. curr_kk (direct path only)
        grad_curr_kk_state = grad_output

        # Gradient w.r.t. prev_kk
        grad_prev_kk = decay[:, :, None, None] * grad_output  # [B, H, D, D]

        # Gradient w.r.t. curr_gkk (via exp decay)
        # d(updated)/d(curr_gkk) = exp(curr_gkk) * prev_kk
        grad_decay = (prev_kk * grad_output).sum(dim=(2, 3))  # [B, H]
        grad_curr_gkk = decay * grad_decay  # [B, H]
    else:
        grad_curr_kk_state = grad_output
        grad_prev_kk = grad_output if prev_kk is not None else None
        grad_curr_gkk = torch.zeros_like(curr_gkk) if curr_gkk is not None else None

    return grad_curr_gkk, grad_prev_kk, grad_curr_kk_state




def gka_state_passing_bwd_h_kv(grad_output, g_cumsum_last, prev_h_kv, curr_h_kv):
    """
    Backward pass for h_kv state passing.

    Forward: updated_h_kv = curr_h_kv + exp(g_cumsum_last) * prev_h_kv
    """
    if g_cumsum_last is not None and prev_h_kv is not None:
        decay = torch.exp(g_cumsum_last)  # [B, H]

        grad_curr_h_kv = grad_output
        grad_prev_h_kv = decay[:, :, None, None] * grad_output  # [B, H, D_k, D_v]

        # Gradient w.r.t. g_cumsum_last
        grad_decay = (prev_h_kv * grad_output).sum(dim=(2, 3))  # [B, H]
        grad_g_cumsum_last = decay * grad_decay  # [B, H]
    else:
        grad_curr_h_kv = grad_output
        grad_prev_h_kv = grad_output if prev_h_kv is not None else None
        grad_g_cumsum_last = torch.zeros_like(g_cumsum_last) if g_cumsum_last is not None else None

    return grad_g_cumsum_last, grad_prev_h_kv, grad_curr_h_kv


def gka_state_passing_fwd(prev_kk, prev_h_kv,
                          curr_kk, curr_gkk, curr_h_kv,
                          g_cumsum_last):
    """
    Update current states with contribution from previous chunk.

    For kk: updated_kk = curr_kk + exp(curr_gkk) * prev_kk
    For h_kv: updated_h_kv = curr_h_kv + exp(g_cumsum_last) * prev_h_kv

    Args:
        prev_kk, curr_kk: [B, H, D, D]
        prev_h_kv, curr_h_kv: [B, H, D_k, D_v]
        curr_gkk: [B, H] - curr_gkk is cumsum at LAST token (LOCAL to current chunk, not passed)
        g_cumsum_last: g cumsum at LAST token [B, H] - total decay across chunk

    Returns:
        updated_kk, updated_h_kv - states to be passed forward
    """
    # Update kk: Apply decay using TOTAL gating across current chunk (curr_gkk = gkk at LAST token)
    # prev_kk is at END of previous chunk, we decay it to END of current chunk
    if curr_gkk is not None and prev_kk is not None:
        decay_kk = torch.exp(curr_gkk[:, :, None, None])  # [B, H, 1, 1] - use curr_gkk (final)!
        updated_kk = curr_kk + decay_kk * prev_kk
    else:
        updated_kk = curr_kk + (prev_kk if prev_kk is not None else 0)

    # Update h_kv: Apply decay using TOTAL g across current chunk (g_cumsum_last)
    if g_cumsum_last is not None and prev_h_kv is not None:
        decay_hkv = torch.exp(g_cumsum_last[:, :, None, None])  # [B, H, 1, 1] - use g_cumsum_last!
        updated_h_kv = curr_h_kv + decay_hkv * prev_h_kv
    else:
        updated_h_kv = curr_h_kv + (prev_h_kv if prev_h_kv is not None else 0)

    return updated_kk, updated_h_kv


class State_Passing_GKA_P2P(torch.autograd.Function):
    @staticmethod
    def forward(ctx, kk_final, gkk_final, h_kv_final, g_cumsum_last,
                cp_group, cp_rank, cp_size, bs):
        """
        Pass GKA states between chunks using P2P communication in zigzag pattern.

        Args:
            kk_final: Final kk state from LOCAL computation [B, H, D, D]
            gkk_final: Final gkk cumsum from LOCAL computation [B, H] (LOCAL only, not passed)
            h_kv_final: Final h_kv state from LOCAL computation [B, H, D_k, D_v]
            g_cumsum_last: g cumsum at LAST token of current chunk [B, H]
            cp_group: Communication group
            cp_rank: Current rank
            cp_size: Group size
            bs: Batch size (includes both chunks per GPU)

        Returns:
            prev_kk: kk state received from previous chunk [B, H, D, D]
            prev_h_kv: h_kv state received from previous chunk [B, H, D_k, D_v]
        """
        prev_kk_chunks = []
        prev_h_kv_chunks = []

        # Forward pass for first half of chunks (0 → 1 → 2 → 3)
        for rank in range(cp_size):
            if rank == cp_rank:
                # Receive state from previous rank
                prev_kk = torch.zeros_like(kk_final[0:bs//2])
                prev_h_kv = torch.zeros_like(h_kv_final[0:bs//2])

                if cp_rank > 0:
                    dist.recv(prev_kk, src=get_global_rank(cp_group, cp_rank - 1), group=None)
                    dist.recv(prev_h_kv, src=get_global_rank(cp_group, cp_rank - 1), group=None)

                    # Update current state with prev contribution
                    updated_kk, updated_h_kv = gka_state_passing_fwd(
                        prev_kk, prev_h_kv,
                        kk_final[0:bs//2], gkk_final[0:bs//2], h_kv_final[0:bs//2],
                        g_cumsum_last[0:bs//2] if g_cumsum_last is not None else None
                    )
                elif cp_rank == 0:
                    # First rank has no previous state
                    updated_kk = kk_final[0:bs//2]
                    updated_h_kv = h_kv_final[0:bs//2]

                # Save received prev state (for return/corrections)
                prev_kk_chunks.append(prev_kk)
                prev_h_kv_chunks.append(prev_h_kv)

            dist.barrier(group=cp_group)

            # Send UPDATED state to next rank
            if cp_rank < cp_size - 1 and rank == cp_rank:
                dist.send(updated_kk, dst=get_global_rank(cp_group, cp_rank + 1), group=None)
                dist.send(updated_h_kv, dst=get_global_rank(cp_group, cp_rank + 1), group=None)

        # On last rank, the two chunks are consecutive - use local update
        if cp_rank == cp_size - 1:
            prev_kk = updated_kk
            prev_h_kv = updated_h_kv

            updated_kk, updated_h_kv = gka_state_passing_fwd(
                prev_kk, prev_h_kv,
                kk_final[bs//2:bs], gkk_final[bs//2:bs], h_kv_final[bs//2:bs],
                g_cumsum_last[bs//2:bs] if g_cumsum_last is not None else None
            )

            prev_kk_chunks.append(prev_kk)
            prev_h_kv_chunks.append(prev_h_kv)

        dist.barrier(group=cp_group)

        # Backward pass for second half of chunks (3 → 2 → 1 → 0)
        for rank in range(cp_size - 1, -1, -1):
            if rank == cp_rank:
                # Receive state from next rank
                prev_kk_ = torch.zeros_like(kk_final[bs//2:bs])
                prev_h_kv_ = torch.zeros_like(h_kv_final[bs//2:bs])

                if cp_rank < cp_size - 1:
                    dist.recv(prev_kk_, src=get_global_rank(cp_group, cp_rank + 1), group=None)
                    dist.recv(prev_h_kv_, src=get_global_rank(cp_group, cp_rank + 1), group=None)

                    # Update current state with prev contribution
                    updated_kk, updated_h_kv = gka_state_passing_fwd(
                        prev_kk_, prev_h_kv_,
                        kk_final[bs//2:bs], gkk_final[bs//2:bs], h_kv_final[bs//2:bs],
                        g_cumsum_last[bs//2:bs] if g_cumsum_last is not None else None
                    )

                    prev_kk_chunks.append(prev_kk_)
                    prev_h_kv_chunks.append(prev_h_kv_)

            dist.barrier(group=cp_group)

            # Send UPDATED state to previous rank
            if cp_rank > 0 and cp_rank == rank:
                dist.send(updated_kk, dst=get_global_rank(cp_group, cp_rank - 1), group=None)
                dist.send(updated_h_kv, dst=get_global_rank(cp_group, cp_rank - 1), group=None)

        # Concatenate prev states (what we received, for corrections)
        cat_prev_kk = torch.cat(prev_kk_chunks, dim=0)
        cat_prev_h_kv = torch.cat(prev_h_kv_chunks, dim=0)

        # Save for backward
        ctx.cp_rank = cp_rank
        ctx.cp_group = cp_group
        ctx.cp_size = cp_size
        ctx.bs = bs
        ctx.save_for_backward(kk_final, gkk_final, h_kv_final, g_cumsum_last,
                             cat_prev_kk, cat_prev_h_kv)

        return cat_prev_kk, cat_prev_h_kv

    @staticmethod
    def backward(ctx, grad_prev_kk, grad_prev_h_kv):
        """
        Backward pass for state passing.

        Gradients flow in reverse order of forward pass.
        """
        cp_rank = ctx.cp_rank
        cp_size = ctx.cp_size
        cp_group = ctx.cp_group
        bs = ctx.bs

        kk_final, gkk_final, h_kv_final, g_cumsum_last, \
            prev_kk_cat, prev_h_kv_cat = ctx.saved_tensors

        grad_kk_chunks = []
        grad_h_kv_chunks = []
        grad_g_cumsum_last_chunks = []

        # Reverse of forward pass: start with first half chunks (0 → 1 → 2 → 3)
        for rank in range(cp_size):
            if rank == cp_rank:
                # Receive gradients from next rank
                next_grad_kk = torch.empty_like(grad_prev_kk[bs//2:bs])
                next_grad_h_kv = torch.empty_like(grad_prev_h_kv[bs//2:bs])

                if cp_rank > 0:
                    dist.recv(next_grad_kk, src=get_global_rank(cp_group, cp_rank - 1), group=None)
                    dist.recv(next_grad_h_kv, src=get_global_rank(cp_group, cp_rank - 1), group=None)

                    # Compute gradients
                    grad_curr_gkk, grad_prev_kk_val, grad_curr_kk = gka_state_passing_bwd_kk(
                        next_grad_kk,
                        gkk_final[bs//2:bs],  # Use gkk_final for local decay computation
                        prev_kk_cat[bs//2:bs],
                        kk_final[bs//2:bs]
                    )
                    grad_g_cumsum_last_val, grad_prev_h_kv_val, grad_curr_h_kv = gka_state_passing_bwd_h_kv(
                        next_grad_h_kv,
                        g_cumsum_last[bs//2:bs] if g_cumsum_last is not None else None,
                        prev_h_kv_cat[bs//2:bs],
                        h_kv_final[bs//2:bs]
                    )
                elif cp_rank == 0:
                    grad_curr_kk = torch.zeros_like(kk_final[bs//2:bs])
                    grad_curr_gkk = torch.zeros_like(gkk_final[bs//2:bs])
                    grad_curr_h_kv = torch.zeros_like(h_kv_final[bs//2:bs])
                    grad_prev_kk_val = torch.zeros_like(grad_prev_kk[bs//2:bs])
                    grad_prev_h_kv_val = torch.zeros_like(grad_prev_h_kv[bs//2:bs])
                    grad_g_cumsum_last_val = torch.zeros_like(g_cumsum_last[bs//2:bs]) if g_cumsum_last is not None else None

                grad_kk_chunks.append(grad_curr_kk)
                grad_h_kv_chunks.append(grad_curr_h_kv)
                grad_g_cumsum_last_chunks.append(grad_g_cumsum_last_val)

            dist.barrier(group=cp_group)

            # Send accumulated gradients to next rank
            # CRITICAL: grad_prev_kk has TWO contributions:
            # 1. grad_prev_kk_val: from state passing computation
            # 2. grad_prev_kk[bs//2:bs]: from local Chebyshev h0 usage
            if cp_rank < cp_size - 1 and rank == cp_rank:
                send_grad_kk = grad_prev_kk_val + grad_prev_kk[bs//2:bs]
                send_grad_h_kv = grad_prev_h_kv_val + grad_prev_h_kv[bs//2:bs]
                dist.send(send_grad_kk, dst=get_global_rank(cp_group, cp_rank + 1), group=None)
                dist.send(send_grad_h_kv, dst=get_global_rank(cp_group, cp_rank + 1), group=None)

        # Handle last rank local update
        if cp_rank == cp_size - 1:
            next_grad_kk = grad_prev_kk_val + grad_prev_kk[bs//2:bs]
            next_grad_h_kv = grad_prev_h_kv_val + grad_prev_h_kv[bs//2:bs]

            grad_curr_gkk, grad_prev_kk_val, grad_curr_kk = gka_state_passing_bwd_kk(
                next_grad_kk,
                gkk_final[0:bs//2],  # Use gkk_final for local decay computation
                prev_kk_cat[0:bs//2],
                kk_final[0:bs//2]
            )
            grad_g_cumsum_last_val, grad_prev_h_kv_val, grad_curr_h_kv = gka_state_passing_bwd_h_kv(
                next_grad_h_kv,
                g_cumsum_last[0:bs//2] if g_cumsum_last is not None else None,
                prev_h_kv_cat[0:bs//2],
                h_kv_final[0:bs//2]
            )

            grad_kk_chunks.append(grad_curr_kk)
            grad_h_kv_chunks.append(grad_curr_h_kv)
            grad_g_cumsum_last_chunks.append(grad_g_cumsum_last_val)

        dist.barrier(group=cp_group)

        # Reverse pass for second half chunks (3 → 2 → 1 → 0)
        for rank in range(cp_size - 1, -1, -1):
            if rank == cp_rank:
                next_grad_kk_ = torch.empty_like(grad_prev_kk[0:bs//2])
                next_grad_h_kv_ = torch.empty_like(grad_prev_h_kv[0:bs//2])

                if cp_rank < cp_size - 1:
                    dist.recv(next_grad_kk_, src=get_global_rank(cp_group, cp_rank + 1), group=None)
                    dist.recv(next_grad_h_kv_, src=get_global_rank(cp_group, cp_rank + 1), group=None)

                    grad_curr_gkk, grad_prev_kk_val, grad_curr_kk = gka_state_passing_bwd_kk(
                        next_grad_kk_,
                        gkk_final[0:bs//2],  # Use gkk_final for local decay computation
                        prev_kk_cat[0:bs//2],
                        kk_final[0:bs//2]
                    )
                    grad_g_cumsum_last_val, grad_prev_h_kv_val, grad_curr_h_kv = gka_state_passing_bwd_h_kv(
                        next_grad_h_kv_,
                        g_cumsum_last[0:bs//2] if g_cumsum_last is not None else None,
                        prev_h_kv_cat[0:bs//2],
                        h_kv_final[0:bs//2]
                    )

                    grad_kk_chunks.append(grad_curr_kk)
                    grad_h_kv_chunks.append(grad_curr_h_kv)
                    grad_g_cumsum_last_chunks.append(grad_g_cumsum_last_val)

            dist.barrier(group=cp_group)

            if cp_rank > 0 and cp_rank == rank:
                send_grad_kk = grad_prev_kk[0:bs//2] + grad_prev_kk_val
                send_grad_h_kv = grad_prev_h_kv[0:bs//2] + grad_prev_h_kv_val
                dist.send(send_grad_kk, dst=get_global_rank(cp_group, cp_rank - 1), group=None)
                dist.send(send_grad_h_kv, dst=get_global_rank(cp_group, cp_rank - 1), group=None)

        # Concatenate gradients (reverse order to match forward)
        grad_kk_final = torch.cat(grad_kk_chunks[::-1], dim=0)
        grad_h_kv_final = torch.cat(grad_h_kv_chunks[::-1], dim=0)
        grad_g_cumsum_last = torch.cat(grad_g_cumsum_last_chunks[::-1], dim=0) if g_cumsum_last is not None else None

        # Return gradients: None for gkk_final (only used internally for decay, not an output)
        return grad_kk_final, None, grad_h_kv_final, grad_g_cumsum_last, None, None, None, None


def state_passing_gka_p2p(kk_final, gkk_final, h_kv_final, g_cumsum_last,
                          cp_group, cp_rank, cp_size, bs):
    """Wrapper function for State_Passing_GKA_P2P.apply

    Args:
        kk_final: Final kk state [B, H, D, D]
        gkk_final: Final gkk cumsum [B, H] (LOCAL only, used for decay computation)
        h_kv_final: Final h_kv state [B, H, D_k, D_v]
        g_cumsum_last: g cumsum at last token [B, H]

    Returns:
        prev_kk: Previous kk state [B, H, D, D]
        prev_h_kv: Previous h_kv state [B, H, D_k, D_v]
    """
    return State_Passing_GKA_P2P.apply(
        kk_final, gkk_final, h_kv_final, g_cumsum_last,
        cp_group, cp_rank, cp_size, bs
    )


def recompute_with_initial_states(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    alpha: torch.Tensor,
    g: torch.Tensor,
    gk: torch.Tensor,
    prev_kk: torch.Tensor,
    prev_h_kv: torch.Tensor,
    gla_scale: float,
    ridge_strength: float,
    solver_type: str,
    num_iter: int,
    bp_lambda: bool,
    chunk_size: int = 64,
):
    """
    Recompute Chebyshev + GLA with initial states from previous chunk.

    Args:
        q, k, v: Input tensors [B, T, H, D]
        alpha: Interpolation parameter [B, T, H]
        g: Forgetting gate for h_kv [B, T, H]
        gk: Forgetting gate for h_kk [B, T, H]
        prev_kk: Initial h_kk state from previous chunk [B, H, D, D]
        prev_h_kv: Initial h_kv state from previous chunk [B, H, D_k, D_v]
        gla_scale: Scaling factor
        ridge_strength: Ridge regularization
        solver_type: Solver type
        num_iter: Number of iterations
        bp_lambda: Backprop through lambda

    Returns:
        o_corrected: Corrected output [B, T, H, D_v]
        h_kk_final: Final h_kk state [B, H, D, D]
        h_kv_final: Final h_kv state [B, H, D_k, D_v]
    """
    q_proj = q
    k_proj = k

    if num_iter > 0:
        # Recompute Chebyshev with initial h_kk (prev_kk)
        if solver_type == 'chebyshev':
            from ..ops.chebyshev.chebyshev_iteration import ChebyshevIteration
            q_corrected, h_kk_final = ChebyshevIteration.apply(
                k_proj, q_proj, gk, num_iter, ridge_strength, bp_lambda,
                prev_kk, chunk_size  # Pass prev_kk as h0
            )
        else:
            raise NotImplementedError(f"Solver type {solver_type} not supported for SP correction")

        # Apply alpha interpolation
        q_corrected = q + alpha[..., None] * (q_corrected - q)
    else:
        q_corrected = q
        h_kk_final = None

    # Recompute GLA with initial h_kv and corrected q
    o_corrected, h_kv_final = chunk_simple_gla(
        q=q_corrected,
        k=k,
        v=v,
        scale=gla_scale,
        g=g,
        initial_state=prev_h_kv,
        output_final_state=True
    )

    return o_corrected, h_kk_final, h_kv_final


def apply_gla_correction(
    o_local: torch.Tensor,
    q: torch.Tensor,
    prev_h_kv: torch.Tensor,
    g: torch.Tensor,
    gla_scale: float
) -> torch.Tensor:
    """
    Apply GLA correction using previous chunk's h_kv state.

    The correction accounts for the contribution of previous timesteps to the output:
    o_corrected = o_local + q @ (decay * prev_h_kv)

    Args:
        o_local: Output computed assuming zero initial state [B, T, H, D]
        q: Query tensor [B, T, H, D]
        prev_h_kv: h_kv state from previous chunk [B, H, D_k, D_v]
        g: Forgetting gate values [B, T, H]
        gla_scale: Scaling factor

    Returns:
        o_corrected: Corrected output [B, T, H, D]
    """
    if prev_h_kv is None or (prev_h_kv == 0).all():
        return o_local

    # Compute cumulative decay from start of chunk
    # g shape: [B, T, H]
    if g is not None:
        # Cumsum of g gives us cumulative log-decay
        g_cumsum = torch.cumsum(g, dim=1)  # [B, T, H]
        decay = torch.exp(g_cumsum)  # [B, T, H]
    else:
        decay = torch.ones_like(o_local[..., 0])  # [B, T, H]

    # Apply decay to previous state and compute contribution
    # prev_h_kv: [B, H, D_k, D_v]
    # decay: [B, T, H]
    # q: [B, T, H, D_k]

    # Broadcast and apply: decay[:,:,:,None,None] * prev_h_kv[:,None,:,:,:]
    decayed_prev_h_kv = decay[:, :, :, None, None] * prev_h_kv[:, None, :, :, :]
    # Shape: [B, T, H, D_k, D_v]

    # Compute correction: q @ decayed_prev_h_kv
    # q: [B, T, H, D_k] -> [B, T, H, D_k, 1]
    # result: [B, T, H, 1, D_v] -> [B, T, H, D_v]
    correction = torch.einsum('bthk,bthkv->bthv', q * gla_scale, decayed_prev_h_kv)

    return o_local + correction
