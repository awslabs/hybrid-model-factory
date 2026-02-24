"""
Utility classes and functions for primed attention layers.

This module contains shared utilities for various linear attention variants:
- Mamba (Mamba1, Mamba2)
- GatedDeltaNet (GDN)
- Linear Attention
- And other state-space models
"""

from vllm.distributed import divide


class PrimedStateShapeCalculator:
    """
    State shape calculator for various primed attention mechanisms.

    This class provides methods to calculate cache state shapes for different
    linear attention variants, accounting for tensor parallelism and speculative
    decoding.
    """

    @classmethod
    def linear_attention_state_shape(
        cls,
        num_heads: int,
        tp_size: int,
        head_dim: int,
    ) -> tuple[tuple[int, int, int], ...]:
        """
        Calculate state shape for generic linear attention.

        Args:
            num_heads: Number of attention heads
            tp_size: Tensor parallel world size
            head_dim: Dimension per head

        Returns:
            Tuple containing state shape
        """
        state_shape = (num_heads // tp_size, head_dim, head_dim)
        return (state_shape,)

    @classmethod
    def mamba1_state_shape(
        cls,
        tp_world_size: int,
        intermediate_size: int,
        state_size: int,
        conv_kernel: int,
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        """
        Calculate state shapes for Mamba1 (original Mamba).

        Args:
            tp_world_size: Tensor parallel world size
            intermediate_size: Intermediate dimension size
            state_size: SSM state size
            conv_kernel: Convolution kernel size

        Returns:
            Tuple of (conv_state_shape, temporal_state_shape)
        """
        # V1: conv_state shape is swapped compared to V0
        conv_state_shape = (
            conv_kernel - 1,
            divide(intermediate_size, tp_world_size),
        )

        temporal_state_shape = (
            divide(intermediate_size, tp_world_size),
            state_size,
        )

        return conv_state_shape, temporal_state_shape

    @classmethod
    def mamba2_state_shape(
        cls,
        tp_world_size: int,
        intermediate_size: int,
        n_groups: int,
        num_heads: int,
        head_dim: int,
        state_size: int,
        conv_kernel: int,
        conv_dim: int,
    ) -> tuple[tuple[int, int], tuple[int, int, int]]:
        """
        Calculate state shapes for Mamba2 (SSD).

        Args:
            tp_world_size: Tensor parallel world size
            intermediate_size: Intermediate dimension size
            n_groups: Number of groups for grouped query attention
            num_heads: Number of attention heads
            head_dim: Dimension per head
            state_size: SSM state size
            conv_kernel: Convolution kernel size
            conv_dim: Total convolution dimension

        Returns:
            Tuple of (conv_state_shape, temporal_state_shape)
        """
        # If n_groups is not divisible by world_size, need to extend the shards
        # to ensure all groups needed by a head is sharded along with it
        n_groups = n_groups + cls.extra_groups_for_head_shards(
            n_groups, tp_world_size
        )

        # heads and n_groups are TP-ed
        # conv_dim = intermediate_size + 2 * n_groups * state_size

        # V1: contiguous along 'dim' axis
        conv_state_shape = (conv_kernel - 1, divide(conv_dim, tp_world_size))

        # These are not TP-ed as they depend on A, dt_bias, D
        # - they are typically small
        #   e.g., (num_heads, head_dim, state_size) = (128, 64, 128)
        temporal_state_shape = (
            divide(num_heads, tp_world_size),
            head_dim,
            state_size,
        )
        return conv_state_shape, temporal_state_shape

    @classmethod
    def gated_delta_net_state_shape(
        cls,
        tp_world_size: int,
        num_heads: int,
        num_kv_heads: int,
        hidden_size: int,
        head_dim: int,
        conv_kernel_size: int,
        num_spec: int = 0,
    ) -> tuple[tuple[int, int], tuple[int, int, int]]:
        """
        Calculate state shapes for checkpoint-compatible GDN with GQA.

        This differs from vLLM's native GDN state shape calculation because
        it uses checkpoint-compatible dimensions with GQA:
        - Native vLLM: conv_dim = head_k_dim * num_k_heads * 2 + head_v_dim * num_v_heads
        - Checkpoint-compatible: conv_dim = key_dim + kv_k_dim + kv_v_dim

        Args:
            tp_world_size: Tensor parallel world size
            num_heads: Number of attention heads (full)
            num_kv_heads: Number of key-value heads (for GQA)
            hidden_size: Hidden dimension size
            head_dim: Dimension per head
            conv_kernel_size: Convolution kernel size
            num_spec: Number of speculative tokens

        Returns:
            Tuple of (conv_state_shape, temporal_state_shape)
        """
        # Calculate dimensions using expand factors (checkpoint-compatible)
        num_kv_groups = num_heads // num_kv_heads
        expand_k = (num_heads * head_dim) / hidden_size
        expand_v = (num_heads * head_dim) / hidden_size
        key_dim = int(hidden_size * expand_k)
        value_dim = int(hidden_size * expand_v)

        # GQA dimensions (for conv - matches checkpoint structure)
        kv_k_dim = key_dim // num_kv_groups
        kv_v_dim = value_dim // num_kv_groups

        # Conv dim for checkpoint-compatible GDN with GQA
        # Checkpoint has: q_conv1d (key_dim), k_conv1d (kv_k_dim), v_conv1d (kv_v_dim)
        conv_dim = key_dim + kv_k_dim + kv_v_dim

        # Conv state shape (V1 format: swapped dimensions)
        conv_state_shape = (
            conv_kernel_size - 1 + num_spec,
            divide(conv_dim, tp_world_size),
        )

        # Temporal state shape (recurrent state)
        head_k_dim = key_dim // num_heads
        head_v_dim = value_dim // num_heads
        temporal_state_shape = (
            divide(num_heads, tp_world_size),
            head_k_dim,
            head_v_dim,
        )

        return conv_state_shape, temporal_state_shape

    @classmethod
    def gka_state_shape(
        cls,
        tp_world_size: int,
        num_heads: int,
        num_kv_heads: int,
        hidden_size: int,
        head_dim: int,
        conv_kernel_size: int,
        num_spec: int,
        num_sketches_per_head: int = 1,
    ) -> tuple[tuple[int, int], tuple[int, int, int], tuple[int, int, int]]:
        """
        Returns (conv_state_shape, h_kk_shape, h_kv_shape) for GKA.

        Args:
            tp_world_size: Tensor parallel world size
            num_heads: Number of attention heads
            num_kv_heads: Number of key-value heads
            hidden_size: Hidden dimension size
            head_dim: Dimension per head
            conv_kernel_size: Convolution kernel size
            num_spec: Number of speculative tokens
            num_sketches_per_head: Number of sketches per head

        Returns:
            Tuple of (conv_state_shape, h_kk_shape, h_kv_shape)
        """
        # Conv state (same as GDN)
        conv_state_shape, _ = cls.gated_delta_net_state_shape(
            tp_world_size, num_heads, num_kv_heads, hidden_size, 
            head_dim, conv_kernel_size, num_spec
        )

        # h_kk state: Chebyshev solver state [H*num_sketches, sketch_dim, sketch_dim]
        # With num_sketches_per_head=1: sketch_dim = head_dim
        sketch_dim = head_dim // num_sketches_per_head
        num_sketch_heads = (num_heads // tp_world_size) * num_sketches_per_head
        h_kk_shape = (num_sketch_heads, sketch_dim, sketch_dim)

        # h_kv state: Simple GLA state [H, head_dim, head_dim]
        h_kv_shape = (num_heads // tp_world_size, head_dim, head_dim)

        return conv_state_shape, h_kk_shape, h_kv_shape

    @classmethod
    def extra_groups_for_head_shards(cls, ngroups: int, tp_size: int) -> int:
        """
        Compute the increase in group numbers to account for replication
        in order to accompany the head shards.

        This is needed when ngroups is not divisible by tp_size to ensure
        all groups needed by a head are sharded along with it.

        Args:
            ngroups: Number of groups
            tp_size: Tensor parallel world size

        Returns:
            Number of extra groups needed
        """
        # In the case ngroups % tp_size == 0, this will be zero
        if ngroups % tp_size == 0:
            return 0

        # For n_groups == 1, this is exactly tp_size - n_groups
        return tp_size - ngroups
