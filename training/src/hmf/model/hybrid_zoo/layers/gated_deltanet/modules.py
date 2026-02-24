"""
Gated DeltaNet (GDN) layer wrapper that initializes from config and handles cache extraction.
"""

from typing import Optional, Tuple, TYPE_CHECKING

import torch
import torch.nn as nn
from torch import Tensor
from transformers.configuration_utils import PretrainedConfig

from .gdn import GatedDeltaNet
from .gdn_config import construct_gdn_config

if TYPE_CHECKING:
    from hmf.model.hybrid_zoo.models.cache import (
        HybridCache,
        GatedDeltaNetCache,
    )


class GatedDeltaNetLayer(GatedDeltaNet):
    def __init__(
        self, config: PretrainedConfig, layer_idx: int, mode: str = "chunk"
    ) -> None:
        """
        Gated DeltaNet layer wrapper for Hybrid models.

        Args:
            config: Model config with optional gdn_config dict
            layer_idx: Layer index in the model
            mode: Operating mode for GatedDeltaNet, defaults to "chunk"
        """

        gdn_config_dict = getattr(config, "gdn_config", None)
        gdn_config = construct_gdn_config(config, gdn_exp_config=gdn_config_dict)

        super().__init__(
            mode=mode,
            hidden_size=gdn_config.hidden_size,
            head_dim=gdn_config.head_dim,
            num_q_heads=gdn_config.num_q_heads,
            num_k_heads=gdn_config.num_k_heads,
            num_v_heads=gdn_config.num_v_heads,
            use_gate=gdn_config.use_gate,
            use_short_conv=gdn_config.use_short_conv,
            allow_neg_eigval=gdn_config.allow_neg_eigval,
            conv_size=gdn_config.conv_size,
            conv_bias=gdn_config.conv_bias,
            norm_eps=gdn_config.norm_eps,
            layer_idx=layer_idx,
            kv_proj_rank=gdn_config.kv_proj_rank,
            kv_learnable_residual=gdn_config.kv_learnable_residual,
        )

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        past_key_values: Optional["HybridCache"] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[Tensor, None, Optional["GatedDeltaNetCache"]]:
        """
        Forward pass through the Gated DeltaNet layer.

        Args:
            hidden_states: Input tensor of shape [batch, seq_len, hidden_size]
            attention_mask: Optional attention mask tensor
            past_key_values: HybridCache object, extracts GDN-specific cache if present
            use_cache: Whether to use caching
            output_attentions: Whether to output attention weights
            **kwargs: Additional arguments passed to parent forward

        Returns:
            Tuple of (hidden_states, None, cache) where:
                - hidden_states: Output tensor of shape [batch, seq_len, hidden_size]
                - None: Placeholder for consistency with attention layers
                - cache: GatedDeltaNetCache or None

        """

        if hasattr(past_key_values, "gated_deltanet"):
            gated_deltanet_past_key_values = past_key_values.gated_deltanet
        else:
            gated_deltanet_past_key_values = None

        return super().forward(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            past_key_values=gated_deltanet_past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            **kwargs,
        )