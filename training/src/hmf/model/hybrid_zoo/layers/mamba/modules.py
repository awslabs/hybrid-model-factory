"""
Mamba2 layer wrapper that initializes from config and handles cache extraction.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
from transformers.configuration_utils import PretrainedConfig

from .mamba2 import Mamba2
from .mamba2_config import construct_mamba2_config
from hmf.model.hybrid_zoo.models.cache import HybridCache, Mamba2Cache


class Mamba2Layer(Mamba2):
    def __init__(
        self,
        config: PretrainedConfig,
        layer_idx: int,
        apply_rotary_pos_emb_fn,
        qk_norm: Optional[nn.Module] = None,
    ) -> None:
        """
        Mamba2 layer wrapper for Hybrid models.

        Args:
            config: Model config with optional mamba2_config dict
            layer_idx: Layer index in the model
            apply_rotary_pos_emb_fn: Function for rotary position embeddings
            qk_norm: Optional normalization for queries (C in Mamba2) and keys (B in Mamba2), such
                as Qwen3RMSNorm.
        """

        mamba2_config_dict = getattr(config, "mamba2_config", None)
        mamba2_config = construct_mamba2_config(
            config, mamba2_exp_config=mamba2_config_dict
        )

        dtype = config.torch_dtype
        factory_kwargs = {"dtype": dtype}

        super().__init__(
            d_model=mamba2_config.d_model,
            d_xb=mamba2_config.d_xb,
            d_inner=mamba2_config.d_inner,
            layer_idx=layer_idx,
            use_pos_emb=mamba2_config.use_pos_emb,
            apply_rotary_pos_emb_fn=apply_rotary_pos_emb_fn,
            use_qk_norm=mamba2_config.use_qk_norm,
            qk_norm=qk_norm,
            **mamba2_config.ssm_cfg,
            **factory_kwargs,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[HybridCache] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, None, Optional[Mamba2Cache]]:
        """
        Forward pass through the Mamba2 layer.

        Args:
            hidden_states: Input tensor of shape [batch, seq_len, hidden_size]
            attention_mask: Ignored, kept for consistency
            past_key_values: HybridCache object, extracts Mamba2-specific cache if present.
            use_cache: Ignored, kept for consistency
            output_attentions: Ignored, kept for consistency
            position_embeddings: Optional tuple of (cos, sin) for rotary embeddings
            **kwargs: Additional arguments, ignored

        Returns:
            Tuple of (hidden_states, None, cache) where:
                - hidden_states: Output tensor of shape [batch, seq_len, hidden_size]
                - None: Placeholder for consistency with attention layers
                - cache: Mamba2Cache or None
        """

        if hasattr(past_key_values, "mamba2"):
            mamba_past_key_values = past_key_values.mamba2
        else:
            mamba_past_key_values = None

        return super().forward(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            past_key_values=mamba_past_key_values,
        )
