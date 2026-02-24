"""
Sequence parallelism utilities for distributed attention computation.

This module provides sequence parallelism support using ring attention and Ulysses attention
patterns. It monkey-patches transformers' flash attention implementation to enable distributed
sequence processing across multiple GPUs.

Adapted from:
  - https://github.com/zhuzilin/ring-flash-attention
  - https://github.com/jzhang38/EasyContext

Dependencies:
  - ring_flash_attn: Ring attention implementation
  - yunchang: Ulysses attention implementation

Note:
    This module modifies transformers.modeling_flash_attention_utils._flash_attention_forward
    globally. Ensure compatibility with your transformers version.
"""

import logging
from functools import partial
from typing import Optional, Tuple

import torch
import torch.distributed as dist
import transformers
import transformers.modeling_flash_attention_utils
from ring_flash_attn.zigzag_ring_flash_attn import zigzag_ring_flash_attn_func
from yunchang import UlyssesAttention
from yunchang.kernels import AttnType

from ...extras.packages import is_transformers_version_greater_than

logger = logging.getLogger(__name__)


def new_flash_attn_forward(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    q_len: Optional[int] = None,
    dropout: float = 0.0,
    deterministic: bool = False,
    sliding_window: Optional[int] = None,
    is_causal: bool = True,
    group: Optional[dist.ProcessGroup] = None,
    mode: str = "zigzag-ring",
    **kwargs,
) -> torch.Tensor:
    """
    Distributed flash attention forward pass with sequence parallelism.

    Args:
        query_states: Query tensor [B, L, H, D].
        key_states: Key tensor [B, L, H, D].
        value_states: Value tensor [B, L, H, D].
        attention_mask: Optional attention mask (unused in current implementations).
        q_len: Query sequence length (unused in current implementations). Defaults to None.
        dropout: Dropout probability. Defaults to 0.0.
        deterministic: Whether to use deterministic operations. Defaults to False.
        sliding_window: Sliding window size for local attention (unused in current implementations). Defaults to None.
        is_causal: Whether to apply causal masking. Defaults to True.
        group: Process group for distributed communication. Defaults to None.
        mode: Sequence parallel mode ("zigzag-ring" or "ulysses"). Defaults to "zigzag-ring".

    Returns:
        Attention output tensor [B, L, H, D].

    Raises:
        NotImplementedError: If mode is not supported.
    """
    if mode == "zigzag-ring":
        attn_output = zigzag_ring_flash_attn_func(
            query_states,
            key_states,
            value_states,
            dropout,
            deterministic=deterministic,
            causal=is_causal,
            group=group,
        )
    elif mode == "ulysses":
        dist_attn = UlyssesAttention(
            sequence_process_group=group, attn_type=AttnType.FA
        )
        attn_output = dist_attn(
            query_states,
            key_states,
            value_states,
            deterministic=deterministic,
            dropout_p=dropout,
            causal=is_causal,
        )
    else:
        raise NotImplementedError(
            f"Sequence parallel mode '{mode}' is not implemented. Supported modes: 'zigzag-ring', 'ulysses'."
        )

    return attn_output


def init_sp_group(sp_size: int) -> Tuple[dist.ProcessGroup, int]:
    """
    Initialize sequence parallel process groups.

    Args:
        sp_size: Number of GPUs per sequence parallel group.

    Returns:
        Tuple of (process_group, local_rank) for this process.

    Raises:
        AssertionError: If distributed is not initialized or world_size is not divisible by sp_size.
    """
    assert dist.is_initialized(), "PyTorch distributed must be initialized"

    world_size = dist.get_world_size()  # Total number of GPUs available

    assert (
        world_size % sp_size == 0
    ), f"World size ({world_size}) must be divisible by sequence_parallel_size ({sp_size})"

    # Compute how many sequence parallel groups we have
    sp_group_num = world_size // sp_size

    # Build list of rank groups, where each sublist contains the GPU ranks for one SP group
    # Example: with world_size=8 and sp_size=4, creates [[0,1,2,3], [4,5,6,7]]
    sp_ranks_list = [
        list(range(i * sp_size, (i + 1) * sp_size)) for i in range(sp_group_num)
    ]

    # Create PyTorch distributed process groups from the rank lists
    sp_groups = [dist.new_group(ranks) for ranks in sp_ranks_list]

    # Determine which SP group this process belongs to
    global_rank = dist.get_rank()  # This process's global rank ID
    sp_idx = global_rank // sp_size  # Which SP group index (0, 1, 2, ...)
    local_rank = global_rank % sp_size  # Rank within the SP group (0 to sp_size-1)

    return sp_groups[sp_idx], local_rank


def apply_sequence_parallel(
    model_args, full_determinism: bool = False
) -> Optional[dist.ProcessGroup]:
    """
    Apply sequence parallelism by monkey-patching transformers attention.

    This function modifies the global flash attention implementation in transformers
    to use distributed sequence parallel attention across multiple GPUs.

    Args:
        model_args: Model arguments containing sequence_parallel_size and sequence_parallel_mode.
        full_determinism: Whether to enforce deterministic operations. Defaults to False.

    Returns:
        The process group for this sequence parallel instance, or None if sp_size == 1.

    Raises:
        NotImplementedError: If sequence_parallel_mode is not supported.
    """
    if model_args.sequence_parallel_size == 1:
        return None  # No sequence parallelism needed

    # Initialize sequence parallel groups
    sp_size = model_args.sequence_parallel_size
    group, local_rank = init_sp_group(sp_size)

    try:
        # Create partial function with group and mode
        if model_args.sequence_parallel_mode in ["zigzag-ring", "ulysses"]:
            new_flash_attention_forward = partial(
                new_flash_attn_forward,
                group=group,
                mode=model_args.sequence_parallel_mode,
                deterministic=full_determinism,
            )
        else:
            raise NotImplementedError(
                f"Sequence parallel mode '{model_args.sequence_parallel_mode}' is not implemented. "
                "Supported modes: 'zigzag-ring', 'ulysses'."
            )

        # Monkey patch transformers flash attention
        transformers.modeling_flash_attention_utils._flash_attention_forward = (
            new_flash_attention_forward
        )

        # Register AttentionInterface for newer transformers versions (Qwen3+)
        if is_transformers_version_greater_than("4.51.0"):
            _register_attention_interface(new_flash_attention_forward)

        logger.info(
            f"Sequence parallelism enabled: mode={model_args.sequence_parallel_mode}, "
            f"size={sp_size}, local_rank={local_rank}"
        )

    except Exception as e:
        logger.error(f"Failed to apply sequence parallelism: {e}", exc_info=True)
        raise

    return group


def _register_attention_interface(new_flash_attention_forward) -> None:
    """
    Register custom attention interface for transformers >= 4.51.0.

    Args:
        new_flash_attention_forward: The patched flash attention forward function.
    """
    from transformers import AttentionInterface
    from transformers.modeling_flash_attention_utils import (
        flash_attn_supports_top_left_mask,
    )
    from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS


    _use_top_left_mask = flash_attn_supports_top_left_mask()

    def sequence_parallel_attention(
        module: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        dropout: float = 0.0,
        scaling: Optional[float] = None,
        sliding_window: Optional[int] = None,
        softcap: Optional[float] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, None]:
        """
        Sequence parallel attention interface for transformers >= 4.51.0.

        Args:
            module: The attention module.
            query: Query tensor [B, H, L, D].
            key: Key tensor [B, H, L, D].
            value: Value tensor [B, H, L, D].
            attention_mask: Optional attention mask.
            dropout: Dropout probability. Defaults to 0.0.
            scaling: Attention scaling factor (passed as softmax_scale). Defaults to None.
            sliding_window: Sliding window size. Defaults to None.
            softcap: Softcap value for attention scores. Defaults to None.

        Returns:
            Tuple of (attention_output [B, H, L, D], None).
        """

        # Store sequence length before transpose
        seq_len = query.shape[2]

        # Transpose to FA2 format: [B, H, L, D] -> [B, L, H, D]
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        # Handle dtype casting for PEFT compatibility
        target_dtype = None
        if query.dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            elif hasattr(module.config, "_pre_quantization_dtype"):
                target_dtype = module.config._pre_quantization_dtype
            else:
                # Find first linear layer's dtype
                target_dtype = next(
                    layer
                    for layer in module.modules()
                    if isinstance(layer, torch.nn.Linear)
                ).weight.dtype

        # Remove is_causal from kwargs to avoid duplicate argument
        kwargs.pop("is_causal", None)

        attn_output = new_flash_attention_forward(
            query,
            key,
            value,
            attention_mask,
            q_len=seq_len,
            is_causal=module.is_causal,
            dropout=dropout,
            softmax_scale=scaling,
            sliding_window=sliding_window,
            softcap=softcap,
            use_top_left_mask=_use_top_left_mask,
            target_dtype=target_dtype,
            **kwargs,
        )

        return attn_output, None

    AttentionInterface.register(
        "sequence_parallel_attention", sequence_parallel_attention
    )
    ALL_MASK_ATTENTION_FUNCTIONS.register("sequence_parallel_attention", ALL_MASK_ATTENTION_FUNCTIONS["flash_attention_2"])

