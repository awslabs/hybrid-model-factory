"""VLM reassembly: wire a distilled hybrid text backbone back into the
original Vision-Language wrapper so it loads as a hybrid VL model.

Used after Stage 1 distillation. The Stage 1 output is a text-only hybrid
backbone (HybridQwen2 with GKA layers). To fine-tune or evaluate the full
VL model, we need to put the hybrid layers back inside the VL wrapper
(vision encoder + processor + connector).

The result is saved as ``model_type=hybrid_qwen2_5_vl`` so that
``AutoModelForImageTextToText.from_pretrained`` rebuilds the *hybrid* VL
class (`HybridQwen2_5_VLForConditionalGeneration`) instead of the stock
Qwen2.5-VL class — which would silently drop GKA weights at load time.

Saving as ``hybrid_qwen2_5_vl`` requires that ``model_register`` has been
imported so the AutoConfig / AutoModelForImageTextToText registrations
have run.
"""
from __future__ import annotations

import os

import torch
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoProcessor

# Registers HybridQwen2_5_VLConfig + HybridQwen2_5_VLForConditionalGeneration
# with the Auto* classes; required for save→load roundtrip of hybrid VLMs.
from ..model.hybrid_zoo.models import model_register  # noqa: F401
from ..model.hybrid_zoo.models.hybrid_qwen2_5_vl.configuration_hybrid_qwen2_5_vl import (
    HybridQwen2_5_VLConfig,
)
from ..model.hybrid_zoo.models.hybrid_qwen2_5_vl.modeling_hybrid_qwen2_5_vl import (
    HybridQwen2_5_VLForConditionalGeneration,
)


def reassemble_vlm(
    vl_model_path: str,
    text_backbone_path: str,
    output_path: str,
    save_max_shard_size: str = "5GB",
    dtype: torch.dtype = torch.bfloat16,
) -> None:
    """Combine an original VL model (vision encoder + processor) with a
    distilled hybrid text backbone and save as a hybrid VL checkpoint.

    The output checkpoint:
        - has ``model_type=hybrid_qwen2_5_vl`` (or upstream-equivalent)
        - has the GKA layers' weights inside the ``model.language_model.layers``
          slot, exactly where the wrapper expects them
        - reloads via ``AutoModelForImageTextToText.from_pretrained`` (with
          ``model_register`` imported) into the hybrid VL class with no
          missing/unexpected key warnings.

    Args:
        vl_model_path: Path or HF id of the original VL model. Provides the
            vision encoder, processor, and structural config.
        text_backbone_path: Path to the distilled hybrid text backbone
            (output of ``hmf prime-init`` + Stage 1 distillation +
            ``hmf prime-unfuse``).
        output_path: Where to save the reassembled VLM.
        save_max_shard_size: Passed to ``save_pretrained``.
        dtype: Load dtype (default bfloat16).
    """
    print(f"Loading VL wrapper (vision + connector): {vl_model_path}")
    vl_src = AutoModelForImageTextToText.from_pretrained(
        vl_model_path, dtype=dtype, device_map="cpu", trust_remote_code=True
    )

    print(f"Loading hybrid text backbone: {text_backbone_path}")
    hybrid_text = AutoModelForCausalLM.from_pretrained(
        text_backbone_path, dtype=dtype, device_map="cpu", trust_remote_code=True
    )

    # Build a HybridQwen2_5_VLConfig: vision_config from upstream Qwen2.5-VL,
    # text_config from the distilled hybrid backbone (carries
    # hybrid_override_pattern, gka_config, layer_types).
    src_cfg = vl_src.config
    hybrid_cfg = HybridQwen2_5_VLConfig(
        vision_config=src_cfg.vision_config.to_dict(),
        text_config=hybrid_text.config.to_dict(),
    )
    # Preserve top-level wrapper fields the upstream config carries.
    for attr in (
        "image_token_id", "video_token_id", "vision_start_token_id",
        "vision_end_token_id", "vision_token_id", "bos_token_id",
        "eos_token_id", "pad_token_id", "tie_word_embeddings",
    ):
        if hasattr(src_cfg, attr):
            setattr(hybrid_cfg, attr, getattr(src_cfg, attr))

    print("Building hybrid VL wrapper from merged config")
    hybrid_vl = HybridQwen2_5_VLForConditionalGeneration(hybrid_cfg)

    # Copy weights from sources, preserving the per-parameter dtypes.
    # Using assign=True puts the source tensors *as-is* into the new module,
    # so float32 GKA scalars (A_log) and bfloat16 projections both keep their
    # original dtype rather than being silently cast.
    hybrid_vl.model.visual.load_state_dict(vl_src.model.visual.state_dict(), assign=True)
    hybrid_vl.lm_head.load_state_dict(vl_src.lm_head.state_dict(), assign=True)

    hybrid_inner = hybrid_text.model if hasattr(hybrid_text, "model") else hybrid_text
    hybrid_vl.model.language_model.embed_tokens.load_state_dict(
        hybrid_inner.embed_tokens.state_dict(), assign=True
    )
    hybrid_vl.model.language_model.norm.load_state_dict(
        hybrid_inner.norm.state_dict(), assign=True
    )
    hybrid_vl.model.language_model.layers.load_state_dict(
        hybrid_inner.layers.state_dict(), assign=True
    )

    # rope-related buffers (e.g. cos/sin caches) follow the language_model
    # construction, no need to copy.

    n_params = sum(p.numel() for p in hybrid_vl.parameters())
    print(f"Reassembled hybrid VLM: {n_params:,} params (model_type={hybrid_cfg.model_type})")

    print(f"Saving to: {output_path}")
    os.makedirs(output_path, exist_ok=True)
    hybrid_vl.save_pretrained(output_path, max_shard_size=save_max_shard_size)

    processor = AutoProcessor.from_pretrained(vl_model_path, trust_remote_code=True)
    processor.save_pretrained(output_path)
    print("Done")


__all__ = ["reassemble_vlm"]
