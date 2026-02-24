import inspect
import logging

from functools import partial
from types import MethodType
from typing import Callable
from typing import Optional

import transformers

from packaging import version
from transformers import PreTrainedModel

from liger_kernel.transformers.cross_entropy import LigerCrossEntropyLoss
from liger_kernel.transformers.functional import liger_cross_entropy
# from liger_kernel.transformers.geglu import LigerGEGLUMLP
# from liger_kernel.transformers.layer_norm import LigerLayerNorm
from liger_kernel.transformers.model.qwen2 import lce_forward as qwen2_lce_forward
from liger_kernel.transformers.model.qwen2 import lce_forward_deprecated as qwen2_lce_forward_deprecated
# from liger_kernel.transformers.model.smollm3 import lce_forward as smollm3_lce_forward
from liger_kernel.transformers.qwen2vl_mrope import liger_multimodal_rotary_pos_emb
from liger_kernel.transformers.rms_norm import LigerRMSNorm
from liger_kernel.transformers.rope import liger_rotary_pos_emb
# from liger_kernel.transformers.rope import liger_rotary_pos_emb_with_cast
# from liger_kernel.transformers.rope import liger_rotary_pos_emb_with_cast_and_leading_batch
# from liger_kernel.transformers.swiglu import LigerBlockSparseTop2MLP
# from liger_kernel.transformers.swiglu import LigerPhi3SwiGLUMLP
from liger_kernel.transformers.swiglu import LigerSwiGLUMLP
from liger_kernel.transformers.monkey_patch import _patch_rms_norm_module, _patch_layer_norm_module, _patch_swiglu_module


def apply_liger_kernel_to_hybridqwen3_moe(
    rope: bool = True,
    cross_entropy: bool = False,
    fused_linear_cross_entropy: bool = True,
    rms_norm: bool = True,
    swiglu: bool = True,
    model: PreTrainedModel = None,
) -> None:
    """
    Apply Liger kernels to replace original implementation in HuggingFace Qwen3 models.
    """
    assert not (cross_entropy and fused_linear_cross_entropy), (
        "cross_entropy and fused_linear_cross_entropy cannot both be True."
    )

    from . import modeling_hybrid_qwen3_moe
    from .modeling_hybrid_qwen3_moe import HybridQwen3MoeModel

    from liger_kernel.transformers.model.qwen3_moe import lce_forward as qwen3_lce_forward
    from liger_kernel.transformers.swiglu import LigerQwen3MoeSwiGLUMLP

    if rope:
        modeling_hybrid_qwen3_moe.apply_rotary_pos_emb = liger_rotary_pos_emb

    if rms_norm:
        modeling_hybrid_qwen3_moe.Qwen3MoeRMSNorm = LigerRMSNorm

    if cross_entropy:
        from transformers.loss.loss_utils import nn

        nn.functional.cross_entropy = liger_cross_entropy

    if fused_linear_cross_entropy:
        modeling_hybrid_qwen3_moe.HybridQwen3MoeForCausalLM.forward = qwen3_lce_forward

    if swiglu:
        modeling_hybrid_qwen3_moe.Qwen3MoeMLP = LigerQwen3MoeSwiGLUMLP

    if model is not None:
        # The model instance already exists, so we need to additionally patch the
        # instance variables that reference already-instantiated modules

        # get the base model from the model instance
        base_model: HybridQwen3MoeModel = getattr(model, model.base_model_prefix, model)

        if rms_norm:
            _patch_rms_norm_module(base_model.norm)
        for decoder_layer in base_model.layers:
            if swiglu:
                _patch_swiglu_module(decoder_layer.mlp, LigerQwen3MoeSwiGLUMLP)
            if rms_norm:
                _patch_rms_norm_module(decoder_layer.input_layernorm)
                _patch_rms_norm_module(decoder_layer.post_attention_layernorm)
