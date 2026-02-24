"""
BMOJO-F Attention layer implementation.

This module implements the B'MOJO-F layer. It takes as input the hidden states from the previous layer
(referred to as 'in-context tokens') as well as these hidden states after having been processed
by an SSM, such as Mamba2 or GKA (referred to as 'fading tokens').

B'MOJO computes attention via two parallel paths:

    1. **In-context path**: Computes queries, keys, and values all from the in-context tokens.
    When processing query i, we apply sliding window attention (SWA) with window size w-1
    (where w is specified in config.bmojo_config['window_size']), so query i attends to
    keys and values at positions [i-w+1, i].

    2. **Fading path**: Computes queries from the in-context tokens, but keys and values from
    the fading tokens. We apply SWA with window size w-1 over the fading tokens, excluding
    the tokens in the in-context window. Hence, query i attends to keys and values at
    positions [i-2w+1, i-w].

The outputs from both paths are combined using proper softmax normalization, effectively
providing a 2w token attention window.

Example:
    Consider tokens [0, 1, 2, 3, 4, 5, 6, 7, 8] with config.bmojo_config['window_size'] = 3.
    For token 6:
        - In-context tokens: [4, 5, 6]
        - Fading tokens: [1, 2, 3]

Key Variables:
    - num_heads: Number of query attention heads.
    - num_key_value_heads: Number of key/value heads (for GQA).
    - num_key_value_groups: Repetition factor for GQA, computed as num_heads // num_key_value_heads.
    - head_dim: Dimension of each attention head.
    - window_size: Window size for sliding window attention. The effective window size is 2*window_size.
    - tie_attn_weights: Whether to share Q/K/V projection weights between in-context and fading tokens.

Shape Notation:
    Throughout this module, tensor shapes use the following notation:
        - B: Batch size
        - L: Sequence length
        - SP: Sequence parallel size (1 if not using sequence parallel)
        - Dh: Hidden dimension (self.hidden_size)
        - H: Number of query heads (self.num_heads)
        - Hkv: Number of KV heads (self.num_key_value_heads)
        - D: Head dimension (self.head_dim)
        - W: Window size (self.window_size[0] + 1)
        - Din: Inner dimension, H * D
        - Dkv: Key and value dimension, Hkv * D

        - B_sp: Batch size after SP reordering. B_sp = B*2 if self.sequence_parallel_group else B_sp = B
        - L_sp: Sequence length after SP reordering. L_sp = L//2 + 2*W - 1 if self.sequence_parallel_group else L_sp = L
        - Lp: Sequence length after SP reordering for fading tokens. Lp = L//2 + W - 1 if self.sequence_parallel_group else Lp = L
"""

from typing import TYPE_CHECKING, Callable, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from transformers.configuration_utils import PretrainedConfig

if TYPE_CHECKING:
    from hmf.model.hybrid_zoo.models.cache import HybridCache, BMojoCache

from ..sp_p2p_utils import reorder_for_ssm_p2p
from .bmojo_attention_utils import bmojo_self_attention


class BMOJOAttention(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        apply_rotary_pos_emb_fn: Optional[
            Callable[[Tensor, Tensor, Tensor, Tensor], Tuple[Tensor, Tensor]]
        ] = None,
        layer_idx: Optional[int] = None,
        qk_norm: Optional[nn.Module] = None,
    ) -> None:
        """
        Initialize the BMOJO-F attention layer.

        Args:
            config: Model configuration. See module docstring for required config parameters.
            apply_rotary_pos_emb_fn: Function to apply rotary positional embeddings to Q and K.
            layer_idx: Layer index for KV caching during inference.
            qk_norm: Normalization module class (e.g., RMSNorm) to apply to queries and keys.
        """

        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = getattr(config, "head_dim", self.hidden_size // self.num_heads)
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.sequence_parallel_group = None

        # NOTE: Rather than specifying the desired window size - 1 in the model config, we subtract 1 here.
        # Hence, if a window size of 4096 is provided in the model config, we really use 4095.
        self.window_size = (self.config.bmojo_config["window_size"] - 1, 0)

        self.apply_rotary_pos_emb_fn = apply_rotary_pos_emb_fn

        qkv_bias = getattr(config, "attention_bias", True)
        o_bias = getattr(config, "attention_bias", False)

        # Projection layers for in-context tokens (and fading tokens when self.tie_attn_weights is True)
        self.q_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=qkv_bias
        )
        self.k_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=qkv_bias
        )
        self.v_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=qkv_bias
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=o_bias
        )

        # Optional QK normalization
        self.q_norm = None
        self.k_norm = None
        self.q_norm_ = None  # For fading tokens when not tying weights
        self.k_norm_ = None  # For fading tokens when not tying weights
        self.softmax_scale = None

        if qk_norm is not None:
            self.q_norm = qk_norm(self.head_dim, config.rms_norm_eps)
            self.k_norm = qk_norm(self.head_dim, config.rms_norm_eps)
            self.softmax_scale = self.head_dim ** -0.5

        # Optional separate projection weights for fading tokens (when self.tie_attn_weights=False).
        self.tie_attn_weights = config.bmojo_config["tie_attn_weights"]
        if not self.tie_attn_weights:
            self.q_proj_ = nn.Linear(
                self.hidden_size, self.num_heads * self.head_dim, bias=qkv_bias
            )
            self.k_proj_ = nn.Linear(
                self.hidden_size,
                self.num_key_value_heads * self.head_dim,
                bias=qkv_bias,
            )
            self.v_proj_ = nn.Linear(
                self.hidden_size,
                self.num_key_value_heads * self.head_dim,
                bias=qkv_bias,
            )

            if qk_norm is not None:
                self.q_norm_ = qk_norm(self.head_dim, config.rms_norm_eps)
                self.k_norm_ = qk_norm(self.head_dim, config.rms_norm_eps)

    def get_qkv_proj(
        self,
        hidden_states: Tensor,
        key_value_states: Tensor,
        is_in_context: bool,
        position_ids: Optional[Tensor] = None,
        position_embeddings: Optional[Tuple[Tensor, Tensor]] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Compute query, key, and value projections with optional rotary embeddings.

        This method handles projection for either in-context tokens (is_in_context=True) or fading
        tokens (is_in_context=False). When tie_attn_weights=False and is_in_context=False, separate projection
        weights are used for fading tokens.

        Args:
            hidden_states: Input tensor for query projection with shape [B, L, Dh].
            key_value_states: Input tensor for key/value projection with shape [B, L, Dh].
            is_in_context: Whether processing in-context tokens (True) or fading tokens (False).
            position_ids: Optional position indices with shape [B, L] (not currently used).
            position_embeddings: Optional tuple of (cos, sin) tensors for rotary embeddings,
                each with shape [B, L, D] or [1, L, D].

        Returns:
            Tuple of (query_states, key_states, value_states) where:
                - query_states: Query tensor with shape [L, B, H, D].
                - key_states: Key tensor with shape [L, B, Hkv, D].
                - value_states: Value tensor with shape [L, B, Hkv, D].

            Note: Output tensors are permuted to [L, B, H, D] format for compatibility
            with the bmojo_self_attention kernel.
        """
        bsz, q_len, _ = hidden_states.size()
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        # Determine which projection weights to use
        use_untied = (not self.tie_attn_weights) and (not is_in_context)

        q_proj = self.q_proj_ if use_untied else self.q_proj
        k_proj = self.k_proj_ if use_untied else self.k_proj
        v_proj = self.v_proj_ if use_untied else self.v_proj

        q_norm = self.q_norm_ if use_untied else self.q_norm
        k_norm = self.k_norm_ if use_untied else self.k_norm

        # For in-context tokens, use hidden_states for both Q and KV
        # For fading tokens, use key_value_states for KV
        kv_hidden_states = hidden_states if is_in_context else key_value_states

        # Apply projections
        query_states = q_proj(hidden_states)  # [B, L, Din]

        key_states = k_proj(kv_hidden_states)  # [B, L, Dkv]

        value_states = v_proj(kv_hidden_states)  # [B, L, Dkv]

        # Apply optional QK normalization
        if q_norm is not None:
            query_states = q_norm(query_states.view(hidden_shape)).view(
                bsz, q_len, -1
            )  # [B, L, Din]
            key_states = k_norm(key_states.view(hidden_shape)).view(
                bsz, q_len, -1
            )  # [B, L, Dkv]

        # Handle sequence parallelism: reorder chunks and add overlap for convolution
        if self.sequence_parallel_group is not None:
            sp_size = torch.distributed.get_world_size(
                group=self.sequence_parallel_group
            )
            sp_rank = torch.distributed.get_rank(group=self.sequence_parallel_group)
            assert (self.window_size[0] + 1) * 2 <= query_states.shape[1] // 2, (
                f"Sliding window size of BMOJO-F layers = {(self.window_size[0] + 1) * 2} "
                f"larger than CP chunk size = {query_states.shape[1] // 2} is not supported."
            )

            # Reorder sequence chunks for sequence parallelism and add overlap from the previous
            # chunk to support the sliding window attention operation.
            query_states = reorder_for_ssm_p2p(
                query_states,
                self.sequence_parallel_group,
                torch.cuda.Stream(),
                sp_size,
                sp_rank,
                (self.window_size[0] + 1) * 2,
            )  # [B_sp, L_sp, Din]

            key_states = reorder_for_ssm_p2p(
                key_states,
                self.sequence_parallel_group,
                torch.cuda.Stream(),
                sp_size,
                sp_rank,
                (self.window_size[0] + 1) * 2,
            )  # [B_sp, L_sp, Din]

            value_states = reorder_for_ssm_p2p(
                value_states,
                self.sequence_parallel_group,
                torch.cuda.Stream(),
                sp_size,
                sp_rank,
                (self.window_size[0] + 1) * 2,
            )  # [B_sp, L_sp, Din]

            bsz, q_len, _ = query_states.shape

        query_states = query_states.view(
            bsz, q_len, self.num_heads, self.head_dim
        ).transpose(
            1, 2
        )  # [B_sp, L_sp, H, D] -> [B_sp, H, L_sp, D]

        key_states = key_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(
            1, 2
        )  # [B_sp, L_sp, Hkv, D] -> [B_sp, Hkv, L_sp, D]

        value_states = value_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(
            1, 2
        )  # [B_sp, L_sp, Hkv, D] -> [B_sp, Hkv, L_sp, D]

        # Apply rotary positional embeddings
        if self.sequence_parallel_group:
            # Reorder position embeddings for sequence parallel
            position_embeddings_cos = reorder_for_ssm_p2p(
                position_embeddings[0],
                self.sequence_parallel_group,
                torch.cuda.Stream(),
                sp_size,
                sp_rank,
                (self.window_size[0] + 1) * 2,
            )
            position_embeddings_sin = reorder_for_ssm_p2p(
                position_embeddings[1],
                self.sequence_parallel_group,
                torch.cuda.Stream(),
                sp_size,
                sp_rank,
                (self.window_size[0] + 1) * 2,
            )
            cos, sin = position_embeddings_cos, position_embeddings_sin
        else:
            cos, sin = position_embeddings

        query_states, key_states = self.apply_rotary_pos_emb_fn(
            query_states, key_states, cos, sin
        )
        # query_states: [B_sp, H, L_sp, D]
        # key_states: [B_sp, Hkv, L_sp, D]

        # Reshape for bmojo attention, which expects [L_sp, B_sp, H, D]
        query_states = query_states.permute(
            2, 0, 1, 3
        ).contiguous()  # [B_sp, H, L_sp, D] -> [L_sp, B_sp, H, D]

        key_states = key_states.permute(
            2, 0, 1, 3
        ).contiguous()  # [B_sp, Hkv, L_sp, D] -> [L_sp, B_sp, Hkv, D]

        value_states = value_states.permute(
            2, 0, 1, 3
        ).contiguous()  # [B_sp, Hkv, L_sp, D] -> [L_sp, B_sp, Hkv, D]

        return query_states, key_states, value_states

    def _adjust_key_value_for_inference(
        self,
        query_in_ctx: Tensor,
        key_in_ctx: Tensor,
        value_in_ctx: Tensor,
        query_fading: Tensor,
        key_fading: Tensor,
        value_fading: Tensor,
        past_key_values,
    ) -> Tuple[
        Tensor,
        Optional[Tensor],
        Tensor,
        Optional[Tensor],
        Optional[Tensor],
        Optional[Tensor],
    ]:
        """
        Adjust key-value tensors for incremental inference with caching.

        This method handles different scenarios:
        - No cache (past_key_values is None): Training or inference without cache object
        - With cache, prefill (seqlen_offset == 0): First pass, populates cache
        - With cache, decoding (seqlen_offset > 0): Incremental generation, one token at a time

        During cached inference, this method:
        - Allocates cache buffers on first call
        - Updates cache with new KV pairs and retrieves cached tokens
        - Adjusts fading memory to exclude the last W tokens (handled by in-context path)

        The fading tensors (query_fading, key_fading, value_fading) become None when the total sequence
        length is <= W (where W = window_size[0] + 1). During this phase, only in-context
        attention is used until sufficient history accumulates.

        Args:
            query_in_ctx: In-context query tensor with shape [L, B, H, D].
            key_in_ctx: In-context key tensor with shape [L, B, Hkv, D].
            value_in_ctx: In-context value tensor with shape [L, B, Hkv, D].
            query_fading: Fading memory query tensor with shape [L, B, H, D].
            key_fading: Fading memory key tensor with shape [L, B, Hkv, D].
            value_fading: Fading memory value tensor with shape [L, B, Hkv, D].
            past_key_values: Cache object for storing KV pairs. None during training or uncached inference.

        Returns:
            Tuple of (query_in_ctx, key_in_ctx, value_in_ctx, query_fading, key_fading, value_fading) where:
                - query_in_ctx: Unchanged [L, B, H, D].
                - key_in_ctx, value_in_ctx: Unchanged without cache, or expanded to all cached tokens with cache.
                - query_fading, key_fading, value_fading: None when sequence length <= W, otherwise adjusted to
                    exclude last W tokens (which are handled by in-context attention).

            During cached prefill only, query_fading is further trimmed to skip the first W positions.
        """

        window_size = self.window_size[0]

        # Handle case without cache (e.g., during training or prefill)
        if past_key_values is None:
            if len(query_fading) <= window_size + 1:
                # Sequence too short for fading memory
                query_fading, key_fading, value_fading = None, None, None
            else:
                # Adjust fading memory to exclude overlap with in-context window
                query_fading = query_fading[window_size + 1 :]  # [L-W, B, H, D]
                key_fading = key_fading[: -window_size - 1]  # [L-W, B, Hkv, D]
                value_fading = value_fading[: -window_size - 1]  # [L-W, B, Hkv, D]

            return (
                query_in_ctx,
                key_in_ctx,
                value_in_ctx,
                query_fading,
                key_fading,
                value_fading,
            )

        # Allocate cache buffers on first use
        if self.layer_idx not in past_key_values.key_value_memory_dict:
            inf_max_seq_length = past_key_values.max_sequence_length
            inf_max_batch_size = past_key_values.max_batch_size

            inference_key_memory = {
                "in-context": self._allocate_memory(
                    inf_max_seq_length,
                    inf_max_batch_size,
                    self.head_dim,
                    key_in_ctx.dtype,
                ),
                "fading-memory": self._allocate_memory(
                    inf_max_seq_length,
                    inf_max_batch_size,
                    self.head_dim,
                    key_fading.dtype,
                ),
            }

            inference_value_memory = {
                "in-context": self._allocate_memory(
                    inf_max_seq_length,
                    inf_max_batch_size,
                    self.head_dim,
                    value_in_ctx.dtype,
                ),
                "fading-memory": self._allocate_memory(
                    inf_max_seq_length,
                    inf_max_batch_size,
                    self.head_dim,
                    value_fading.dtype,
                ),
            }

            past_key_values.key_value_memory_dict[self.layer_idx] = (
                inference_key_memory,
                inference_value_memory,
            )
        else:
            # Retrieve pre-allocated buffers
            inference_key_memory, inference_value_memory = past_key_values.key_value_memory_dict[
                self.layer_idx
            ]

        # Determine batch and sequence slices for cache update
        batch_start = past_key_values.batch_size_offset
        batch_end = batch_start + key_in_ctx.size(1)
        assert batch_end <= inference_key_memory["in-context"].size(1)

        sequence_start = past_key_values.seqlen_offset
        sequence_end = sequence_start + key_in_ctx.size(0)
        assert sequence_end <= inference_key_memory["in-context"].size(0)

        # Update in-context cache
        inference_key_memory["in-context"][
            sequence_start:sequence_end, batch_start:batch_end, ...
        ] = key_in_ctx
        inference_value_memory["in-context"][
            sequence_start:sequence_end, batch_start:batch_end, ...
        ] = value_in_ctx
        key_in_ctx = inference_key_memory["in-context"][
            :sequence_end, batch_start:batch_end, ...
        ]
        value_in_ctx = inference_value_memory["in-context"][
            :sequence_end, batch_start:batch_end, ...
        ]

        # Update fading memory cache
        inference_key_memory["fading-memory"][
            sequence_start:sequence_end, batch_start:batch_end, ...
        ] = key_fading
        inference_value_memory["fading-memory"][
            sequence_start:sequence_end, batch_start:batch_end, ...
        ] = value_fading
        key_fading = inference_key_memory["fading-memory"][
            :sequence_end, batch_start:batch_end, ...
        ]
        value_fading = inference_value_memory["fading-memory"][
            :sequence_end, batch_start:batch_end, ...
        ]

        # Adjust fading memory based on window size
        if len(key_fading) <= window_size + 1:
            # Not enough tokens for fading memory
            return (query_in_ctx, key_in_ctx, value_in_ctx, None, None, None)

        # Exclude last window_size+1 tokens from fading memory (handled by in-context)
        key_fading = key_fading[: -window_size - 1]
        value_fading = value_fading[: -window_size - 1]

        # During prefill, adjust query to skip tokens without fading memory
        if past_key_values.seqlen_offset == 0:
            query_fading = query_fading[window_size + 1 :]  # [L-W, B, H, D]

        return (
            query_in_ctx,
            key_in_ctx,
            value_in_ctx,
            query_fading,
            key_fading,
            value_fading,
        )

    def _allocate_memory(
        self,
        inference_max_sequence_length: int,
        batch_size: int,
        dim: int,
        dtype: torch.dtype,
    ) -> Tensor:
        """
        Allocate memory buffer to store KV cache during inference.

        Args:
            inference_max_sequence_length: Maximum sequence length for inference.
            batch_size: Batch size for inference.
            dim: Head dimension (self.head_dim).
            dtype: Data type for the cache tensor.

        Returns:
            Allocated tensor with shape [inference_max_sequence_length, batch_size,
                num_key_value_heads, dim] on the current CUDA device.
        """
        return torch.empty(
            inference_max_sequence_length,
            batch_size,
            self.num_key_value_heads,
            dim,
            dtype=dtype,
            device=torch.cuda.current_device(),
        )

    def forward(
        self,
        hidden_states: Tensor,
        key_value_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        past_key_values: Optional["BMojoCache"] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[Tensor] = None,
        position_embeddings: Optional[Tuple[Tensor, Tensor]] = None,
    ) -> Tuple[Tensor, None, Optional[Tensor]]:
        """
        BMOJO-F attention forward pass with dual in-context and fading memory paths.

        Args:
            hidden_states: Input tensor containing in-context tokens with shape [B, L//SP, Dh].
            key_value_states: Input tensor containing fading memory tokens with shape [B, L//SP, Dh].
                This is typically the output of processing hidden_states through an SSM layer.
            attention_mask: Optional attention mask (not currently used).
            position_ids: Optional position indices (not currently used).
            past_key_values: BMojoCache object containing KV cache for in-context and fading tokens.
            output_attentions: Whether to return attention weights (not currently used).
            use_cache: Whether to use KV caching (not currently used).
            cache_position: Optional cache position indices (not currently used).
            position_embeddings: Tuple of (cos, sin) tensors for rotary positional embeddings,
                each with shape [B, L, D] or [1, L, D]. Required for proper position encoding.

        Returns:
            Tuple of (attn_output, None, past_key_values) where:
                - attn_output: Output tensor with shape [B, L, Dh]
                - None: Placeholder for attention weights (not returned)
                - past_key_values: Updated cache object (same as input if provided, else None)
        """
        original_seqlen = hidden_states.shape[1]

        # Compute projections for fading memory tokens
        query_fading, key_fading, value_fading = self.get_qkv_proj(
            hidden_states=hidden_states,
            key_value_states=key_value_states,
            is_in_context=False,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
        )
        # query_fading: [L_sp, B_sp, H, D]
        # key_fading and value_fading: [L_sp, B_sp, Hkv, D]

        # Compute projections for in-context tokens
        query_in_ctx, key_in_ctx, value_in_ctx = self.get_qkv_proj(
            hidden_states=hidden_states,
            key_value_states=key_value_states,
            is_in_context=True,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
        )
        # query_in_ctx: [L_sp, B_sp, H, D]
        # key_in_ctx and value_in_ctx: [L_sp, B_sp, Hkv, D]

        # Adjust for inference with caching
        query_in_ctx, key_in_ctx, value_in_ctx, query_fading, key_fading, value_fading = self._adjust_key_value_for_inference(
            query_in_ctx,
            key_in_ctx,
            value_in_ctx,
            query_fading,
            key_fading,
            value_fading,
            past_key_values=past_key_values,
        )
        # query_in_ctx: [L_sp, B_sp, H, D]
        # key_in_ctx and value_in_ctx: [L_sp, B_sp, Hkv, D]
        # query_fading: [Lp, B_sp, H, D] or None
        # key_fading and value_fading: [Lp, B_sp, Hkv, D] or None

        # Apply BMOJO self-attention kernel
        attn_output = bmojo_self_attention(
            query_in_ctx,
            key_in_ctx,
            value_in_ctx,
            query_fading,
            key_fading,
            value_fading,
            config=self.config,
            window_size=self.window_size,
            softmax_scale=self.softmax_scale,
            sequence_parallel_group=self.sequence_parallel_group,
            original_seqlen=original_seqlen // 2
            if self.sequence_parallel_group
            else original_seqlen,
            sp_rank=torch.distributed.get_rank(group=self.sequence_parallel_group)
            if self.sequence_parallel_group
            else 1,
        )  # [B, L, Din]

        # Project output back to hidden dimension
        attn_output = self.o_proj(attn_output)  # [B, L, Dh]

        return attn_output, None, past_key_values
