"""
GKA layer wrapper that initializes from config and handles cache extraction.
"""

from typing import Optional, Tuple, TYPE_CHECKING

from torch import Tensor
from transformers.configuration_utils import PretrainedConfig

from .gka import GatedKalmaNet
from .gka_config import construct_gka_config

if TYPE_CHECKING:
    from hmf.model.hybrid_zoo.models.cache import HybridCache, GatedKalmaNetCache


class GatedKalmaNetLayer(GatedKalmaNet):
    def __init__(self, config: PretrainedConfig, layer_idx: int) -> None:
        """
        GKA layer wrapper for Hybrid models.

        Args:
            config: Model config with optional gka_config dict
            layer_idx: Layer index in the model
        """

        # Construct GKA config
        gka_config_dict = getattr(config, "gka_config", None)
        gka_config = construct_gka_config(config, gka_exp_config=gka_config_dict)

        super().__init__(
            hidden_size=gka_config.hidden_size,
            head_dim=gka_config.head_dim,
            num_q_heads=gka_config.num_q_heads,
            num_k_heads=gka_config.num_k_heads,
            num_v_heads=gka_config.num_v_heads,
            use_alpha_connection=gka_config.use_alpha_connection,
            use_v_conv=gka_config.use_v_conv,
            use_forgetting_gate=gka_config.use_forgetting_gate,
            gla_rescale=gka_config.gla_rescale,
            solver_type=gka_config.solver_type,
            bp_lambda=gka_config.bp_lambda,
            num_iter=gka_config.num_iter,
            ridge_strength=gka_config.ridge_strength,
            use_gate=gka_config.use_gate,
            conv_size=gka_config.conv_size,
            norm_eps=gka_config.norm_eps,
            use_beta_gate=gka_config.use_beta_gate,
            use_forgetting_gate_kk=gka_config.use_forgetting_gate_kk,
            layer_idx=layer_idx,
            chunk_size=gka_config.chunk_size,
            kv_proj_rank=gka_config.kv_proj_rank,
            kv_learnable_residual=gka_config.kv_learnable_residual,
        )

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        past_key_values: Optional["HybridCache"] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[Tensor, Optional[Tensor], Optional["GatedKalmaNetCache"]]:
        """
        Forward pass through GKA layer. Extracts layer-specific cache from
        past_key_values if provided.

        Args:
            hidden_states: Input tensor of shape [batch, seq_len, hidden_size]
            attention_mask: Optional attention mask tensor
            past_key_values: HybridCache object, extracts GKA-specific cache if present
            use_cache: Whether to use caching
            output_attentions: Whether to output attention weights
            **kwargs: Additional arguments passed to parent forward

        Returns:
            Tuple of (hidden_states, attention_weights, cache) where:
                - hidden_states: Output tensor of shape [batch, seq_len, hidden_size]
                - attention_weights: Optional attention weights tensor or None
                - cache: GatedKalmaNetCache or None
        """

        if hasattr(past_key_values, "gated_kalmanet"):
            gka_past_key_values = past_key_values.gated_kalmanet
        else:
            gka_past_key_values = None

        return super().forward(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            past_key_values=gka_past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            **kwargs,
        )
