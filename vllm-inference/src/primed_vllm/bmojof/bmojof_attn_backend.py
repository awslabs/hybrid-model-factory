"""BMOJO-F C-Stream Attention Backend (CUDA Graph Compatible)

Registers two custom ops as torch.compile graph break points (splitting_ops):
  - vllm::bmojof_attention_s_forward  (s-stream)
  - vllm::bmojof_attention_c_forward  (c-stream)
Added to CompilationConfig._attention_ops via vllm_patches.py.

CONSTRAINT
----------
flash_attn_varlen_func requires q_len <= seqused_k per sequence.

For c-stream hollow window [i-2w+1 : i-w], seqused_k = seq_len - w.
During prefill q_len can exceed seqused_k (e.g. seq_len=300, w=256
gives q_len=300 vs seqused_k=44), violating the constraint.

SOLUTION: VIRTUAL SEQUENCE SPLITTING
-------------------------------------
Split each prefill at position w into two virtual sequences:
  1. Invalid part (positions 0..w-1): seqused_k = q_len  (constraint satisfied,
     output masked to 0 afterward)
  2. Valid part (positions w..end):   seqused_k = seq_len - w  (naturally valid)

Decode (q_len=1): if seq_len > w, seqused_k = seq_len - w >= 1; otherwise
seqused_k = 1 and output is masked to 0.

CHUNKED PREFILL
---------------
Validity is based on absolute position, not chunk-relative position.
first_token_abs_pos = num_computed_tokens; tokens with abs_pos <= w are invalid.
A chunk straddling position w is split at the boundary.

CUDA GRAPH COMPATIBILITY
-------------------------
All metadata buffers are pre-allocated at max size (max_virtual_seqs,
max_num_tokens) and padded rather than sliced, so kernel grid sizes and
argument shapes are fixed across replays:
  - cu_seqlens padded with last value (zero-length virtual sequences)
  - seqused_k padded with 1 (minimum valid)
  - max_seqlen_q/k use static upper bounds (Python scalars baked at capture)
"""

from dataclasses import dataclass
import copy

import torch


from vllm.v1.attention.backends.registry import (
    AttentionBackendEnum,
    register_backend,
)
from vllm.v1.attention.backend import AttentionCGSupport
from vllm.v1.attention.backends.flash_attn import (
    FlashAttentionBackend,
    FlashAttentionImpl,
    FlashAttentionMetadata,
    FlashAttentionMetadataBuilder,
)
from vllm.v1.attention.backends.fa_utils import (
    flash_attn_varlen_func,
    reshape_and_cache_flash,
    get_flash_attn_version
)
from vllm.forward_context import get_forward_context
from vllm.platforms import current_platform
from vllm.utils.torch_utils import direct_register_custom_op


class BMOJOFAttentionSBackend(FlashAttentionBackend):
    """Custom attention backend for BMOJO-F s-stream."""

    @staticmethod
    def get_name() -> str:
        return "BMOJOF_ATTN_S"

    @staticmethod
    def get_impl_cls():
        return BMOJOFAttentionSImpl


class BMOJOFAttentionSImpl(FlashAttentionImpl):
    """Attention implementation for BMOJO-F s-stream with LSE support."""

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
        return_lse: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for s-stream attention with optional LSE return."""

        if not return_lse:
            num_tokens = query.shape[0]
            query_3d = query.view(num_tokens, self.num_heads, self.head_size)
            key_3d = key.view(num_tokens, self.num_kv_heads, self.head_size)
            value_3d = value.view(num_tokens, self.num_kv_heads, self.head_size)

            output_3d = torch.empty(
                (num_tokens, self.num_heads, self.head_size),
                dtype=query.dtype,
                device=query.device,
            )

            super().forward(
                layer, query_3d, key_3d, value_3d, kv_cache, attn_metadata,
                output_3d, output_scale, output_block_scale
            )

            output_2d = output_3d.view(num_tokens, self.num_heads * self.head_size)
            if output is not None:
                output.copy_(output_2d)
                return output
            return output_2d

        if output is None:
            output = torch.empty_like(query)

        if attn_metadata is None:
            lse = torch.zeros(
                (self.num_heads, query.shape[0]),
                dtype=torch.float32,
                device=query.device,
            )
            return output.fill_(0), lse

        num_actual_tokens = attn_metadata.num_actual_tokens
        key_cache, value_cache = kv_cache.unbind(0)

        if key is not None and value is not None:
            key_3d = key.view(key.shape[0], self.num_kv_heads, self.head_size)
            value_3d = value.view(value.shape[0], self.num_kv_heads, self.head_size)

            reshape_and_cache_flash(
                key_3d, value_3d, key_cache, value_cache,
                attn_metadata.slot_mapping,
                self.kv_cache_dtype,
                layer._k_scale, layer._v_scale,
            )

        if self.kv_cache_dtype.startswith("fp8"):
            dtype = FlashAttentionBackend.get_fp8_dtype_for_flashattn(self.kv_cache_dtype)
            key_cache = key_cache.view(dtype)
            value_cache = value_cache.view(dtype)

        # Use full bucket-sized tensors for CG compatibility
        num_query_tokens = query.shape[0]  # Bucket size
        descale_shape = (attn_metadata.query_start_loc.shape[0] - 1, self.num_kv_heads)

        q_descale = layer._q_scale.expand(descale_shape)
        k_descale = layer._k_scale.expand(descale_shape)
        v_descale = layer._v_scale.expand(descale_shape)

        query_3d = query.view(num_query_tokens, self.num_heads, self.head_size)
        output_3d = output.view(num_query_tokens, self.num_heads, self.head_size)

        sliding_window_size = (
            list(self.sliding_window) if self.sliding_window is not None else None
        )
        attn_output, lse_raw = flash_attn_varlen_func(
            q=query_3d,
            k=key_cache,
            v=value_cache,
            out=output_3d,
            cu_seqlens_q=attn_metadata.query_start_loc,
            max_seqlen_q=attn_metadata.max_query_len,
            seqused_k=attn_metadata.seq_lens,
            max_seqlen_k=attn_metadata.max_seq_len,
            softmax_scale=self.scale,
            causal=attn_metadata.causal,
            alibi_slopes=self.alibi_slopes,
            window_size=sliding_window_size,
            block_table=attn_metadata.block_table,
            softcap=self.logits_soft_cap,
            return_softmax_lse=True,
            q_descale=q_descale,
            k_descale=k_descale,
            v_descale=v_descale,
        )

        # lse_raw is [num_heads, num_query_tokens] - already bucket-sized
        # Return full bucket-sized tensors for CG compatibility
        # bmojof_layer.py will slice to num_actual_tokens after merge
        return output, lse_raw


@register_backend(AttentionBackendEnum.CUSTOM)
class BMOJOFAttentionBackend(FlashAttentionBackend):
    """Custom attention backend for BMOJO-F c-stream - CUDA Graph Compatible v4."""

    @staticmethod
    def get_name() -> str:
        return "BMOJOF_ATTN"

    @staticmethod
    def get_impl_cls():
        return BMOJOFAttentionImpl

    @staticmethod
    def get_builder_cls():
        return BMOJOFAttentionMetadataBuilder


@dataclass
class BMOJOFAttentionMetadata(FlashAttentionMetadata):
    """Attention metadata for BMOJO-F c-stream with virtual sequence splitting.

    Virtual sequences are used to satisfy flash_attn constraint: q_len <= seqused_k.
    Prefill sequences are split at position w into invalid (first w) and valid (rest) parts.

    CUDA Graph Compatibility:
    - All buffers are FULL size (max_virtual_seqs, max_num_tokens)
    - cu_seqlens padded with last value (creates zero-length sequences)
    - seqused_k padded with 1 (minimum valid value)
    """
    c_stream_window_size: int = 256

    # Virtual batch metadata - FULL FIXED SIZE for CG compatibility
    cu_seqlens_q_virtual: torch.Tensor | None = None   # [max_virtual_seqs + 1] - FULL
    seqused_k_virtual: torch.Tensor | None = None      # [max_virtual_seqs] - FULL
    block_table_virtual: torch.Tensor | None = None    # [max_virtual_seqs, max_blocks] - FULL

    # Actual counts (for reference, but kernels use full buffers)
    num_virtual_seqs: int = 0
    max_virtual_seqs: int = 0  # The fixed size used for all buffers
    max_seqlen_q_virtual: int = 0
    max_seqlen_k_virtual: int = 0

    # Valid token mask for output masking [max_num_tokens] - FULL
    valid_token_mask: torch.Tensor | None = None
    max_num_tokens: int = 0  # The fixed size for token buffers

    # Mapping from virtual to original sequences (for update_block_table)
    virtual_to_orig: torch.Tensor | None = None  # [max_virtual_seqs]

class BMOJOFAttentionMetadataBuilder(FlashAttentionMetadataBuilder):
    """Metadata builder for BMOJO-F c-stream with virtual sequence splitting.

    Constructs virtual batch metadata in build() to satisfy flash_attn constraint.
    Pre-allocates buffers for CG compatibility.

    CUDA Graph Fix: Returns FULL buffers, not sliced views.
    """

    # CG support: we use fixed-size buffers, so should be compatible
    _cudagraph_support = (
        AttentionCGSupport.ALWAYS
        if get_flash_attn_version() == 3
        else AttentionCGSupport.UNIFORM_BATCH
    )
    supports_update_block_table: bool = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.c_stream_window_size = None

        # Get max sizes for buffer allocation
        self.max_num_seqs = self.vllm_config.scheduler_config.max_num_seqs
        self.max_num_tokens = self.vllm_config.scheduler_config.max_num_batched_tokens

        # Max virtual seqs: worst case is all prefills, each split into 2
        self.max_virtual_seqs = 2 * self.max_num_seqs

        self.use_full_cuda_graph = (
            self.compilation_config.cudagraph_mode.has_full_cudagraphs()
        )

        # Pre-allocate persistent buffers for CG compatibility
        # Use self.block_size (set by parent from kv_cache_spec) for correct calculation
        max_model_len = self.vllm_config.model_config.max_model_len
        max_blocks_per_seq = (max_model_len + self.block_size - 1) // self.block_size
        self.max_blocks_per_seq = max_blocks_per_seq

        # Always allocate these buffers (needed for both CG and non-CG paths)
        # Initialize cu_seqlens with 0 (will be padded properly in build)
        self._cu_seqlens_q_virtual = torch.zeros(
            self.max_virtual_seqs + 1, dtype=torch.int32, device=self.device
        )
        # Initialize seqused_k with 1 (minimum valid value for padding)
        self._seqused_k_virtual = torch.ones(
            self.max_virtual_seqs, dtype=torch.int32, device=self.device
        )
        self._block_table_virtual = torch.zeros(
            self.max_virtual_seqs, max_blocks_per_seq, 
            dtype=torch.int32, device=self.device
        )
        # Initialize valid_token_mask to False (padding tokens are invalid)
        self._valid_token_mask = torch.zeros(
            self.max_num_tokens, dtype=torch.bool, device=self.device
        )

        # Pre-allocate GPU work buffers for build() - avoid allocations per call
        self._gpu_q_lens = torch.zeros(self.max_num_seqs, dtype=torch.int32, device=self.device)
        self._gpu_seq_starts = torch.zeros(self.max_num_seqs, dtype=torch.int32, device=self.device)
        self._gpu_seq_ends = torch.zeros(self.max_num_seqs, dtype=torch.int32, device=self.device)
        self._gpu_is_decode = torch.zeros(self.max_num_seqs, dtype=torch.bool, device=self.device)
        self._gpu_is_valid_seq = torch.zeros(self.max_num_seqs, dtype=torch.bool, device=self.device)
        self._gpu_needs_split = torch.zeros(self.max_num_seqs, dtype=torch.bool, device=self.device)
        self._gpu_all_valid = torch.zeros(self.max_num_seqs, dtype=torch.bool, device=self.device)
        self._gpu_all_invalid = torch.zeros(self.max_num_seqs, dtype=torch.bool, device=self.device)
        self._gpu_num_virtual_per_seq = torch.zeros(self.max_num_seqs, dtype=torch.int32, device=self.device)
        self._gpu_virtual_to_orig = torch.zeros(self.max_virtual_seqs, dtype=torch.int64, device=self.device)
        self._gpu_is_second_part = torch.zeros(self.max_virtual_seqs, dtype=torch.bool, device=self.device)
        self._gpu_split_points = torch.zeros(self.max_num_seqs, dtype=torch.int32, device=self.device)
        self._gpu_invalid_tokens_in_chunk = torch.zeros(self.max_num_seqs, dtype=torch.int32, device=self.device)

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata,
        fast_build: bool = False,
    ) -> BMOJOFAttentionMetadata:
        """Main build method - uses GPU-native fast build."""
        return self.build_fast(common_prefix_len, common_attn_metadata, fast_build)

    def build_fast(
        self,
        common_prefix_len: int,
        common_attn_metadata,
        fast_build: bool = False,
    ) -> BMOJOFAttentionMetadata:
        """GPU-native build - no CPU sync points, all tensor ops on GPU."""
        base_metadata = super().build(common_prefix_len, common_attn_metadata, fast_build)

        # Get window size from config (one-time)
        if self.c_stream_window_size is None:
            if hasattr(self.model_config, 'hf_config'):
                hf_config = self.model_config.hf_config
                if hasattr(hf_config, 'bmojo_config'):
                    self.c_stream_window_size = hf_config.bmojo_config.get('window_size', 256)
                else:
                    self.c_stream_window_size = 256
            else:
                self.c_stream_window_size = 256

        w = self.c_stream_window_size
        num_actual_tokens = base_metadata.num_actual_tokens

        # All tensors stay on GPU - NO .cpu() calls!
        query_start_loc = base_metadata.query_start_loc  # [num_seqs + 1]
        seq_lens = base_metadata.seq_lens                # [num_seqs]
        num_seqs = seq_lens.shape[0]

        # Compute derived values on GPU - NO CPU SYNC!
        # q_lens[i] = query_start_loc[i+1] - query_start_loc[i]
        q_lens = query_start_loc[1:num_seqs+1] - query_start_loc[:num_seqs]
        seq_starts = query_start_loc[:num_seqs]
        seq_ends = query_start_loc[1:num_seqs+1]

        # num_computed_tokens = seq_lens - q_lens (computed on GPU, no CPU sync!)
        # This is the number of tokens already processed for each sequence
        num_computed = seq_lens - q_lens

        # Classification flags (all vectorized on GPU)
        first_token_abs_pos = num_computed
        is_decode = (q_lens == 1)
        is_valid_seq = (seq_lens > w)

        # For prefill sequences, determine if they need splitting
        # needs_split: chunk straddles window boundary w
        # all_valid: chunk starts at or after w
        # all_invalid: chunk ends before w OR seq_len <= w
        chunk_end_pos = first_token_abs_pos + q_lens
        needs_split = ~is_decode & is_valid_seq & (first_token_abs_pos < w) & (chunk_end_pos > w)
        all_valid = ~is_decode & is_valid_seq & (first_token_abs_pos >= w)
        all_invalid = ~is_decode & (~is_valid_seq | (chunk_end_pos <= w))

        # Number of virtual sequences per real sequence
        # decode: 1, prefill without split: 1, prefill with split: 2
        num_virtual_per_seq = torch.where(needs_split, 2, 1).to(torch.int32)
        num_virtual_seqs = num_virtual_per_seq.sum().item()  # Single scalar sync - unavoidable for sizing

        # Build virtual sequence mapping using repeat_interleave
        # This expands [0,1,2] with counts [1,2,1] -> [0,1,1,2]
        seq_indices = torch.arange(num_seqs, device=self.device, dtype=torch.int64)
        virtual_to_orig = torch.repeat_interleave(seq_indices, num_virtual_per_seq.to(torch.int64))

        # For split sequences, mark which virtual seq is the "second part" (valid tokens)
        # 
        # Since repeat_interleave guarantees split parts are adjacent (e.g., [0, 1, 1, 2]),
        # we simply check if current maps to same orig seq as previous.
        # First part: prev != curr -> False
        # Second part: prev == curr -> True
        prev_orig = torch.cat([torch.tensor([-1], device=self.device, dtype=torch.int64), virtual_to_orig[:-1]])
        is_second_part = (virtual_to_orig == prev_orig)

        # Compute split points for sequences that need splitting
        # split_point = seq_start + (w - first_token_abs_pos) = seq_start + invalid_tokens_in_chunk
        invalid_tokens_in_chunk = torch.clamp(w - first_token_abs_pos, min=0)
        split_points = seq_starts + invalid_tokens_in_chunk

        # Build cu_seqlens_q_virtual
        # cu_seqlens format: [start0, start1, ..., startN, end_of_last]
        # where start_i is the offset in the flattened query tensor where virtual seq i begins

        orig_seq_starts = seq_starts[virtual_to_orig]
        orig_seq_ends = seq_ends[virtual_to_orig]
        orig_split_points = split_points[virtual_to_orig]
        orig_needs_split = needs_split[virtual_to_orig]

        # Virtual seq start:
        # - If split and second part: split_point
        # - Otherwise: orig_seq_start
        virtual_seq_starts = torch.where(
            is_second_part,
            orig_split_points,
            orig_seq_starts
        )

        # Virtual seq end:
        # - If split and first part: split_point
        # - Otherwise: orig_seq_end
        virtual_seq_ends = torch.where(
            orig_needs_split & ~is_second_part,
            orig_split_points,
            orig_seq_ends
        )

        # cu_seqlens_q = [start0, start1, ..., startN, endN]
        # This is the standard varlen format
        self._cu_seqlens_q_virtual[:num_virtual_seqs] = virtual_seq_starts.to(torch.int32)
        if num_virtual_seqs > 0:
            self._cu_seqlens_q_virtual[num_virtual_seqs] = virtual_seq_ends[-1].to(torch.int32)
        else:
            self._cu_seqlens_q_virtual[0] = 0

        # Pad with last value (creates zero-length sequences for padding)
        if num_virtual_seqs < self.max_virtual_seqs:
            last_val = self._cu_seqlens_q_virtual[num_virtual_seqs]
            self._cu_seqlens_q_virtual[num_virtual_seqs + 1:] = last_val

        # Build seqused_k_virtual
        # seqused_k determines how many K tokens to attend to
        orig_seq_lens = seq_lens[virtual_to_orig]
        orig_q_lens = q_lens[virtual_to_orig]
        orig_is_valid = is_valid_seq[virtual_to_orig]
        orig_is_decode = is_decode[virtual_to_orig]
        orig_all_valid = all_valid[virtual_to_orig]
        orig_all_invalid = all_invalid[virtual_to_orig]
        orig_invalid_tokens = invalid_tokens_in_chunk[virtual_to_orig]

        # seqused_k logic:
        # - decode + valid: seq_len - w
        # - decode + invalid: 1 (minimum)
        # - prefill + all_valid: seq_len - w
        # - prefill + all_invalid: q_len (to satisfy constraint)
        # - prefill + split + first_part (invalid): invalid_tokens_in_chunk
        # - prefill + split + second_part (valid): seq_len - w

        seqused_k = torch.ones(num_virtual_seqs, dtype=torch.int32, device=self.device)

        # Decode cases
        decode_valid_mask = orig_is_decode & orig_is_valid
        decode_invalid_mask = orig_is_decode & ~orig_is_valid

        # Prefill cases (not decode)
        prefill_all_valid_mask = ~orig_is_decode & orig_all_valid
        prefill_all_invalid_mask = ~orig_is_decode & orig_all_invalid
        prefill_split_first_mask = ~orig_is_decode & orig_needs_split & ~is_second_part
        prefill_split_second_mask = ~orig_is_decode & orig_needs_split & is_second_part

        # Apply seqused_k values
        seqused_k = torch.where(decode_valid_mask, (orig_seq_lens - w).to(torch.int32), seqused_k)
        seqused_k = torch.where(decode_invalid_mask, torch.ones_like(seqused_k), seqused_k)
        seqused_k = torch.where(prefill_all_valid_mask, (orig_seq_lens - w).to(torch.int32), seqused_k)
        seqused_k = torch.where(prefill_all_invalid_mask, orig_q_lens.to(torch.int32), seqused_k)
        seqused_k = torch.where(prefill_split_first_mask, orig_invalid_tokens.to(torch.int32), seqused_k)
        seqused_k = torch.where(prefill_split_second_mask, (orig_seq_lens - w).to(torch.int32), seqused_k)

        self._seqused_k_virtual[:num_virtual_seqs] = seqused_k
        if num_virtual_seqs < self.max_virtual_seqs:
            self._seqused_k_virtual[num_virtual_seqs:] = 1

        # Build block_table_virtual by gathering from original
        # block_table_virtual[i] = block_table[virtual_to_orig[i]]
        bt_cols = min(base_metadata.block_table.shape[1], self.max_blocks_per_seq)
        self._block_table_virtual[:num_virtual_seqs, :bt_cols] = base_metadata.block_table[virtual_to_orig, :bt_cols]
        if num_virtual_seqs < self.max_virtual_seqs:
            self._block_table_virtual[num_virtual_seqs:, :] = 0

        # Store virtual_to_orig mapping for update_block_table
        self._gpu_virtual_to_orig[:num_virtual_seqs].copy_(virtual_to_orig, non_blocking=True)

        # Build valid_token_mask
        # Valid tokens are those that can attend to the hollow window
        # - decode + valid: True
        # - prefill + all_valid: True for all tokens
        # - prefill + split: True only for tokens after split_point
        self._valid_token_mask.zero_()

        # For each sequence, set valid tokens
        # This is trickier to vectorize - use scatter or a simple approach
        # Create token indices and check validity
        token_indices = torch.arange(num_actual_tokens, device=self.device)

        # For each token, find which sequence it belongs to
        # Use searchsorted on query_start_loc
        token_seq_idx = torch.searchsorted(query_start_loc[1:num_seqs+1], token_indices, right=True)
        token_seq_idx = torch.clamp(token_seq_idx, 0, num_seqs - 1)

        # Get per-token properties from its sequence
        token_is_decode = is_decode[token_seq_idx]
        token_is_valid_seq = is_valid_seq[token_seq_idx]
        token_all_valid = all_valid[token_seq_idx]
        token_needs_split = needs_split[token_seq_idx]
        token_split_point = split_points[token_seq_idx]

        # Token is valid if:
        # - decode + valid_seq
        # - prefill + all_valid
        # - prefill + split + token_idx >= split_point
        token_valid = (
            (token_is_decode & token_is_valid_seq) |
            token_all_valid |
            (token_needs_split & (token_indices >= token_split_point))
        )

        self._valid_token_mask[:num_actual_tokens] = token_valid

        # Scalar args (max_seqlen_q/k) are baked into CUDA graphs at capture time,
        # so use static upper bounds when CG is enabled.
        if num_virtual_seqs > 0:
            virtual_q_lens = self._cu_seqlens_q_virtual[1:num_virtual_seqs+1] - self._cu_seqlens_q_virtual[:num_virtual_seqs]
            actual_max_q = virtual_q_lens.max().item()
            actual_max_k = self._seqused_k_virtual[:num_virtual_seqs].max().item()
        else:
            actual_max_q = 0
            actual_max_k = 0

        if self.use_full_cuda_graph:
            # Use static constants for CG compatibility
            # Q is limited by chunk size/batch limit, so this is safe:
            max_seqlen_q_virtual = self.max_num_tokens 
            # K can grow to full context length, so use max_model_len:
            max_seqlen_k_virtual = self.vllm_config.model_config.max_model_len
        else:
            # Use actual values for non-CG path (slightly more efficient)
            max_seqlen_q_virtual = actual_max_q
            max_seqlen_k_virtual = actual_max_k

        # Return full (unsliced) buffers for CG compatibility
        return BMOJOFAttentionMetadata(
            num_actual_tokens=base_metadata.num_actual_tokens,
            max_query_len=base_metadata.max_query_len,
            query_start_loc=base_metadata.query_start_loc,
            max_seq_len=base_metadata.max_seq_len,
            seq_lens=base_metadata.seq_lens,
            block_table=base_metadata.block_table,
            slot_mapping=base_metadata.slot_mapping,
            use_cascade=base_metadata.use_cascade,
            common_prefix_len=base_metadata.common_prefix_len,
            cu_prefix_query_lens=base_metadata.cu_prefix_query_lens,
            prefix_kv_lens=base_metadata.prefix_kv_lens,
            suffix_kv_lens=base_metadata.suffix_kv_lens,
            max_dcp_context_kv_len=base_metadata.max_dcp_context_kv_len,
            dcp_context_kv_lens=base_metadata.dcp_context_kv_lens,
            scheduler_metadata=base_metadata.scheduler_metadata,
            prefix_scheduler_metadata=base_metadata.prefix_scheduler_metadata,
            max_num_splits=base_metadata.max_num_splits,
            causal=base_metadata.causal,
            # C-stream virtual batch fields - FULL BUFFERS
            c_stream_window_size=self.c_stream_window_size,
            cu_seqlens_q_virtual=self._cu_seqlens_q_virtual,  # FULL [max_virtual_seqs + 1]
            seqused_k_virtual=self._seqused_k_virtual,        # FULL [max_virtual_seqs]
            block_table_virtual=self._block_table_virtual,    # FULL [max_virtual_seqs, max_blocks]
            num_virtual_seqs=num_virtual_seqs,                # Actual count (for reference)
            max_virtual_seqs=self.max_virtual_seqs,           # Fixed size used
            max_seqlen_q_virtual=max_seqlen_q_virtual,
            max_seqlen_k_virtual=max_seqlen_k_virtual,
            valid_token_mask=self._valid_token_mask,          # FULL [max_num_tokens]
            max_num_tokens=self.max_num_tokens,               # Fixed size used
            virtual_to_orig=self._gpu_virtual_to_orig,        # Mapping for update_block_table
        )

    def update_block_table(
        self,
        metadata: BMOJOFAttentionMetadata,
        blk_table: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> BMOJOFAttentionMetadata:
        """Update block table for existing metadata.

        This is used when metadata structure is unchanged but block table needs
        updating (e.g., for hybrid KV-cache groups). Much faster than rebuild.

        Args:
            metadata: Existing metadata with virtual structure
            blk_table: New block table [num_seqs, max_blocks]
            slot_mapping: New slot mapping [num_tokens]

        Returns:
            New metadata with updated block tables
        """
        # First, update the base metadata using parent's update
        new_metadata = copy.copy(metadata)
        new_metadata.block_table = blk_table
        new_metadata.slot_mapping = slot_mapping

        # Now update the virtual block table using the saved mapping
        # block_table_virtual[v] = blk_table[virtual_to_orig[v]]
        if metadata.virtual_to_orig is not None and metadata.num_virtual_seqs > 0:
            virtual_to_orig = metadata.virtual_to_orig[:metadata.num_virtual_seqs]

            # Gather blocks for each virtual sequence from the original block table
            # Handle both CG (persistent buffer) and non-CG paths
            if metadata.block_table_virtual is self._block_table_virtual:
                # CUDA graph path - update persistent buffer in-place
                self._block_table_virtual[:metadata.num_virtual_seqs].copy_(
                    blk_table[virtual_to_orig], non_blocking=True
                )
                # No change to reference, metadata.block_table_virtual already points to it
            else:
                # Non-CG path - create new tensor
                block_table_virtual = blk_table[virtual_to_orig]
                new_metadata.block_table_virtual = block_table_virtual

        return new_metadata


class BMOJOFAttentionImpl(FlashAttentionImpl):
    """Attention implementation for BMOJO-F c-stream with virtual sequence splitting.

    Uses virtual batch metadata to satisfy flash_attn constraint (q_len <= seqused_k).
    The query tensor is unchanged - only the metadata (cu_seqlens, seqused_k, block_table)
    is transformed to create virtual sequences.

    After flash_attn, invalid token outputs are masked to 0 using torch.where.

    CUDA Graph Compatibility:
    - Uses FULL fixed-size buffers (no dynamic slicing)
    - descale expanded to max_virtual_seqs
    - Masking done on full buffer
    """

    def __init__(self, *args, c_stream_window_size: int = 256, prefix: str = "", **kwargs):
        super().__init__(*args, **kwargs)
        self.c_stream_window_size = c_stream_window_size
        self.prefix = prefix

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: BMOJOFAttentionMetadata,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
        return_lse: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Forward pass using virtual sequence splitting - CUDA Graph Compatible.

        Key CG fixes:
        1. Use FULL buffers from metadata (no slicing)
        2. descale expanded to max_virtual_seqs (fixed size)
        3. Masking on full buffer using valid_token_mask
        4. Query/output used at full size (vllm pads to bucket size)
        """

        if output is None:
            output = torch.empty_like(query)

        if attn_metadata is None:
            # No metadata means no actual tokens - return empty-ish tensors
            # Note: This case shouldn't happen in normal operation
            if return_lse:
                lse = torch.zeros((self.num_heads, query.shape[0]), dtype=torch.float32, device=query.device)
                return output.fill_(0), lse
            return output.fill_(0)

        num_actual_tokens = attn_metadata.num_actual_tokens
        num_virtual_seqs = attn_metadata.num_virtual_seqs
        max_virtual_seqs = attn_metadata.max_virtual_seqs
        max_num_tokens = attn_metadata.max_num_tokens
        w = self.c_stream_window_size

        # Early exit if no virtual sequences
        if num_virtual_seqs == 0:
            output.zero_()  # Zero full buffer
            # Return full bucket-sized tensors (CG compatible)
            # bmojof_layer.py will handle slicing after merge
            if return_lse:
                lse = torch.full(
                    (self.num_heads, query.shape[0]),
                    float('-inf'),
                    dtype=torch.float32,
                    device=query.device,
                )
                return output, lse
            return output

        # Step 1: Cache K/V
        key_cache, value_cache = kv_cache.unbind(0)

        if key is not None and value is not None:
            key_3d = key.view(key.shape[0], self.num_kv_heads, self.head_size)
            value_3d = value.view(value.shape[0], self.num_kv_heads, self.head_size)
            reshape_and_cache_flash(
                key_3d, value_3d, key_cache, value_cache,
                attn_metadata.slot_mapping,
                self.kv_cache_dtype,
                layer._k_scale, layer._v_scale,
            )

        if self.kv_cache_dtype.startswith("fp8"):
            dtype = FlashAttentionBackend.get_fp8_dtype_for_flashattn(self.kv_cache_dtype)
            key_cache = key_cache.view(dtype)
            value_cache = value_cache.view(dtype)

        # Use full query tensor (vllm pads to bucket size for CG compatibility)
        num_query_tokens = query.shape[0]
        query_3d = query.view(num_query_tokens, self.num_heads, self.head_size)
        output_3d = output.view(num_query_tokens, self.num_heads, self.head_size)

        # Use actual count, not max — vLLM re-captures graphs per batch size
        n = num_virtual_seqs
        cu_seqlens_q = attn_metadata.cu_seqlens_q_virtual[:n + 1]
        seqused_k = attn_metadata.seqused_k_virtual[:n]
        block_table = attn_metadata.block_table_virtual[:n, :]

        # Hollow window for c-stream: attend to [i - 2w + 1 : i - w]
        window_size = (w - 1, 0)

        # Use actual num_virtual_seqs for descale shape
        descale_shape = (n, self.num_kv_heads)

        q_descale = layer._q_scale.expand(descale_shape)
        k_descale = layer._k_scale.expand(descale_shape)
        v_descale = layer._v_scale.expand(descale_shape)

        # Step 3: Call flash_attn with actual virtual batch size
        result = flash_attn_varlen_func(
            q=query_3d,
            k=key_cache,
            v=value_cache,
            out=output_3d,
            cu_seqlens_q=cu_seqlens_q,                         # [num_virtual_seqs + 1]
            max_seqlen_q=attn_metadata.max_seqlen_q_virtual,
            max_seqlen_k=attn_metadata.max_seqlen_k_virtual,
            seqused_k=seqused_k,                               # [num_virtual_seqs]
            softmax_scale=self.scale,
            causal=True,
            window_size=window_size,
            block_table=block_table,                           # [num_virtual_seqs, ...]
            alibi_slopes=self.alibi_slopes,
            softcap=self.logits_soft_cap,
            return_softmax_lse=return_lse,
            q_descale=q_descale,
            k_descale=k_descale,
            v_descale=v_descale,
        )

        # Mask invalid/padding tokens to 0.
        # Slice [:num_query_tokens] is CG-safe: vllm captures per bucket size,
        # so query.shape[0] is fixed within each captured graph.
        mask_for_output = attn_metadata.valid_token_mask[:num_query_tokens]
        mask_expanded = mask_for_output.view(-1, 1, 1)  # [T, 1, 1] for broadcasting

        output_3d.mul_(mask_expanded.to(output_3d.dtype))

        # Return full bucket-sized tensors; bmojof_layer.py slices after merge
        if return_lse:
            _, lse_raw = result
            # lse_raw shape: [num_heads, num_query_tokens]
            # Mask invalid tokens to -inf
            mask_lse = mask_for_output.unsqueeze(0)  # [1, T]
            lse = torch.where(mask_lse, lse_raw, float('-inf'))
            return output, lse

        return output


# ============================================================================
# Custom op registration for torch.compile + cudagraph compatibility.
# Wrapping in splitting_ops makes the ops opaque to torch.compile, preventing
# it from tracing into flash_attn_varlen_func and breaking cudagraph capture.
# Pattern follows vLLM's unified_attention_with_output in attention/layer.py.
# ============================================================================


def _get_bmojof_attention_context(layer_name: str):
    """Get attention context for BMOJO-F layers.

    Similar to get_attention_context() in vllm/attention/layer.py but for
    BMOJO-F attention layers (BMOJOFAttentionS and BMOJOFAttentionC).
    """
    forward_context = get_forward_context()
    attn_metadata = forward_context.attn_metadata

    # Handle dict-based metadata (for hybrid models)
    if isinstance(attn_metadata, dict):
        attn_metadata = attn_metadata[layer_name]

    attn_layer = forward_context.no_compile_layers[layer_name]
    kv_cache = attn_layer.kv_cache[forward_context.virtual_engine]

    return attn_metadata, attn_layer, kv_cache


def bmojof_attention_s_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    lse_output: torch.Tensor,
    layer_name: str,
) -> None:
    """Custom op for BMOJO-F s-stream attention forward.

    This wraps BMOJOFAttentionSImpl.forward() in a custom op for torch.compile
    compatibility. The op is registered as a splitting_op so torch.compile
    treats it as a graph break point.

    Args:
        query: Query tensor [num_tokens, num_heads * head_size]
        key: Key tensor [num_tokens, num_kv_heads * head_size]
        value: Value tensor [num_tokens, num_kv_heads * head_size]
        output: Output tensor (mutated in place)
        lse_output: LSE output tensor (mutated in place)
        layer_name: Layer name for context lookup
    """
    attn_metadata, attn_layer, kv_cache = _get_bmojof_attention_context(layer_name)

    # Call the implementation's forward with return_lse=True
    out, lse = attn_layer.impl.forward(
        layer=attn_layer,
        query=query,
        key=key,
        value=value,
        kv_cache=kv_cache,
        attn_metadata=attn_metadata,
        output=output,
        return_lse=True,
    )

    # Copy results to output tensors (in-place mutation)
    output.copy_(out)
    lse_output.copy_(lse)


def bmojof_attention_s_forward_fake(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    lse_output: torch.Tensor,
    layer_name: str,
) -> None:
    """Fake implementation for torch.compile tracing."""
    return


direct_register_custom_op(
    op_name="bmojof_attention_s_forward",
    op_func=bmojof_attention_s_forward,
    mutates_args=["output", "lse_output"],
    fake_impl=bmojof_attention_s_forward_fake,
)


def bmojof_attention_c_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    lse_output: torch.Tensor,
    layer_name: str,
) -> None:
    """Custom op for BMOJO-F c-stream attention forward.

    This wraps BMOJOFAttentionImpl.forward() (c-stream) in a custom op for
    torch.compile compatibility. The op is registered as a splitting_op so
    torch.compile treats it as a graph break point.

    Args:
        query: Query tensor [num_tokens, num_heads * head_size]
        key: Key tensor [num_tokens, num_kv_heads * head_size]
        value: Value tensor [num_tokens, num_kv_heads * head_size]
        output: Output tensor (mutated in place)
        lse_output: LSE output tensor (mutated in place)
        layer_name: Layer name for context lookup
    """
    attn_metadata, attn_layer, kv_cache = _get_bmojof_attention_context(layer_name)

    # Call the implementation's forward with return_lse=True
    out, lse = attn_layer.impl.forward(
        layer=attn_layer,
        query=query,
        key=key,
        value=value,
        kv_cache=kv_cache,
        attn_metadata=attn_metadata,
        output=output,
        return_lse=True,
    )

    # Copy results to output tensors (in-place mutation)
    output.copy_(out)
    lse_output.copy_(lse)


def bmojof_attention_c_forward_fake(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    lse_output: torch.Tensor,
    layer_name: str,
) -> None:
    """Fake implementation for torch.compile tracing."""
    return


direct_register_custom_op(
    op_name="bmojof_attention_c_forward",
    op_func=bmojof_attention_c_forward,
    mutates_args=["output", "lse_output"],
    fake_impl=bmojof_attention_c_forward_fake,
)
