"""
Optimized GDN attention backend that pre-splits non-spec tokens into
separate decode and prefill subsets in the metadata builder, so the layer
can route each subset to its optimal kernel without per-layer overhead.

Stock vLLM GDN uses an if/elif that sends ALL non-spec tokens through the
chunked kernel whenever any prefill is present. This wastes FLOPs
per decode token (chunk_size=64, but only 1 valid token per decode chunk).

This backend pre-computes the split so the layer can run:
  - fused_recurrent for decodes (O(1) per token)
  - chunk_gated_delta_rule for prefills (O(chunk_size) per token)
simultaneously in mixed batches.

Design follows vLLM's Mamba2's pattern:
  - Builder pre-computes metadata that is expensive or requires CPU work:
    query_start_loc splits, conv metadata (nums_dict/batch_ptr/token_chunk_offset_ptr).
  - Layer does cheap GPU tensor splits on state indices and hidden states.
  - nums_dict/batch_ptr/token_chunk_offset_ptr are overwritten with prefill-only
    versions in mixed batches, since causal_conv1d_fn is the only consumer and
    it only runs on prefills. The layer passes metadata=attn_metadata directly.
"""

from dataclasses import dataclass

import torch

from vllm.v1.attention.backends.gdn_attn import (
    GDNAttentionBackend,
    GDNAttentionMetadata,
    GDNAttentionMetadataBuilder,
)
from vllm.v1.attention.backends.registry import (
    MambaAttentionBackendEnum,
    register_backend,
)
from vllm.v1.attention.backends.utils import compute_causal_conv1d_metadata


@dataclass
class PrimedGDNAttentionMetadata(GDNAttentionMetadata):
    """Extends GDNAttentionMetadata with pre-split decode/prefill fields.

    Only populated in mixed batches (both num_decodes > 0 and num_prefills > 0).
    In pure decode or pure prefill batches these are None and the layer uses
    the base fields directly, identical to stock behavior.

    The inherited nums_dict/batch_ptr/token_chunk_offset_ptr are overwritten
    with prefill-only conv metadata in mixed batches so the layer can pass
    metadata=attn_metadata to causal_conv1d_fn unchanged (same as Mamba2).
    """

    # Prefill-only query_start_loc, re-offset from 0: shape [num_prefills + 1]
    non_spec_query_start_loc_p: torch.Tensor | None = None
    # Decode-only query_start_loc: shape [num_decodes + 1]
    non_spec_query_start_loc_d: torch.Tensor | None = None


class PrimedGDNAttentionMetadataBuilder(GDNAttentionMetadataBuilder):

    def build(self, common_prefix_len, common_attn_metadata, *args, **kwargs) -> PrimedGDNAttentionMetadata:
        base = super().build(common_prefix_len, common_attn_metadata, *args, **kwargs)

        num_decodes = base.num_decodes
        num_prefills = base.num_prefills

        # Fast path: pure decode or pure prefill — no split needed.
        # Stock conv metadata is already correct (prefill-only or unused).
        if num_prefills == 0 or num_decodes == 0:
            return PrimedGDNAttentionMetadata(
                **{f.name: getattr(base, f.name)
                   for f in base.__dataclass_fields__.values()}
            )

        # === Mixed batch: pre-compute query_start_loc splits and
        #     prefill-only conv metadata ===
        meta = PrimedGDNAttentionMetadata(
            **{f.name: getattr(base, f.name)
               for f in base.__dataclass_fields__.values()}
        )

        if meta.non_spec_query_start_loc is not None:
            # Decode qsl: first num_decodes+1 entries (already offset from 0)
            meta.non_spec_query_start_loc_d = (
                meta.non_spec_query_start_loc[:num_decodes + 1]
            )
            # Prefill qsl: remaining entries, re-offset from 0
            raw_p = meta.non_spec_query_start_loc[num_decodes:]
            meta.non_spec_query_start_loc_p = raw_p - raw_p[0]

        # Overwrite inherited conv metadata with prefill-only versions.
        # causal_conv1d_fn is the only consumer and only runs on prefills.
        # Derive CPU qsl without GPU->CPU sync (same as Mamba2 base builder).
        num_decode_tokens = meta.num_decode_tokens

        if meta.spec_sequence_masks is not None:
            # Spec+mixed (rare): CPU qsl includes spec seqs, can't slice
            qsl_p_cpu = meta.non_spec_query_start_loc_p.cpu()
        else:
            # No spec (common): CPU qsl == full batch, decodes first
            qsl_p_cpu = (
                common_attn_metadata.query_start_loc_cpu[-num_prefills - 1:]
                - num_decode_tokens
            )

        device = meta.non_spec_query_start_loc.device
        (
            meta.nums_dict,
            meta.batch_ptr,
            meta.token_chunk_offset_ptr,
        ) = compute_causal_conv1d_metadata(qsl_p_cpu, device=device)

        return meta


@register_backend(MambaAttentionBackendEnum.GDN_ATTN, is_mamba=True)
class PrimedGDNAttentionBackend(GDNAttentionBackend):

    @staticmethod
    def get_name() -> str:
        return "PRIMED_GDN_ATTN"

    @staticmethod
    def get_builder_cls() -> type["PrimedGDNAttentionMetadataBuilder"]:
        return PrimedGDNAttentionMetadataBuilder
