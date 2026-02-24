"""
B'MOJO layer wrapper that initializes from config and handles cache extraction.
"""

from typing import Optional, Tuple, TYPE_CHECKING, Union

import torch
import torch.nn as nn
from torch import Tensor
from transformers.configuration_utils import PretrainedConfig

from .....extras import logging
from .bmojo_attention import BMOJOAttention
from .bmojo_config import construct_bmojo_config
from ..mamba.modules import Mamba2Layer
from ..gated_deltanet.modules import GatedDeltaNetLayer
from ..gated_kalmanet.modules import GatedKalmaNetLayer

if TYPE_CHECKING:
    from hmf.model.hybrid_zoo.models.cache import (
        HybridCache,
        BMojoCache,
        Mamba2Cache,
        GatedDeltaNetCache,
        GatedKalmaNetCache,
    )

logger = logging.get_logger(__name__)


class BMojoLayer(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        apply_rotary_pos_emb_fn,
        layer_idx: int,
        qk_norm: Optional[nn.Module] = None,
    ) -> None:
        """
        B'MOJO layer wrapper for Hybrid models.

        B'MOJO applies sliding window attention (SWA) to 'in-context' tokens and 'fading' tokens.
        When processing a token, the in-context tokens are those tokens within a local window.
        Fading tokens are tokens within a window preceding the in-context tokens that have been
        processed by an SSM.

        Args:
            config: Model config with optional bmojo_config dict.
            apply_rotary_pos_emb_fn: Function for applying rotary position embeddings.
            layer_idx: Layer index in the model.
            qk_norm: Optional normalization for queries and keys, such as Qwen3RMSNorm.
        """
        super().__init__()

        # Construct B'MOJO config
        bmojo_config_dict = getattr(config, "bmojo_config", None)
        bmojo_config = construct_bmojo_config(
            config, bmojo_exp_config=bmojo_config_dict
        )

        self.sequence_parallel_group = None
        self.hidden_size = bmojo_config.hidden_size
        self.ssm_mixer_type = bmojo_config.ssm_mixer

        # Initialize the appropriate SSM mixer based on config
        if self.ssm_mixer_type.lower() == "mamba2":
            self.mamba = Mamba2Layer(
                config,
                layer_idx,
                apply_rotary_pos_emb_fn=apply_rotary_pos_emb_fn,
                qk_norm=qk_norm,
            )
            logger.info_rank0("B'MOJO fading tokens sequence mixer set to Mamba2.")
        elif self.ssm_mixer_type.lower() in ["gated_deltanet", "gdn"]:
            self.gdn = GatedDeltaNetLayer(config, layer_idx)
            logger.info_rank0(
                "B'MOJO fading tokens sequence mixer set to Gated DeltaNet (GDN)."
            )
        elif self.ssm_mixer_type.lower() in ["gated_kalmanet", "gka"]:
            self.gka = GatedKalmaNetLayer(config, layer_idx)
            logger.info_rank0(
                "B'MOJO fading tokens sequence mixer set to Gated KalmaNet (GKA)."
            )
        else:
            raise ValueError(
                f"{self.ssm_mixer_type} is not a valid fading tokens sequence mixer type for B'MOJO. "
                "Must be one of `mamba2`, `gdn`, or `gka`."
            )

        self.bmojo_attn = BMOJOAttention(
            config,
            apply_rotary_pos_emb_fn=apply_rotary_pos_emb_fn,
            layer_idx=layer_idx,
            qk_norm=qk_norm,
        )

    @property
    def ssm_mixer(self):
        """
        Returns the active SSM mixer module based on configuration.

        Returns:
            The configured SSM mixer (Mamba2, GDN, or GKA).
        """
        if self.ssm_mixer_type.lower() == "mamba2":
            return self.mamba
        elif self.ssm_mixer_type.lower() in ["gated_deltanet", "gdn"]:
            return self.gdn
        elif self.ssm_mixer_type.lower() in ["gated_kalmanet", "gka"]:
            return self.gka

    def _get_fading_cache(
        self, past_key_values: Optional["HybridCache"] = None
    ) -> Union[
        "Mamba2Cache", "GatedDeltaNetCache", "GatedKalmaNetCache"
    ]:
        """
        Extracts the appropriate cache for the configured SSM mixer.

        Args:
            past_key_values: HybridCache object containing caches for different layer types

        Returns:
            The cache specific to the configured SSM mixer, or None if not present
        """
        if self.ssm_mixer_type.lower() == "mamba2" and hasattr(
            past_key_values, "mamba2"
        ):
            return past_key_values.mamba2
        elif self.ssm_mixer_type.lower() in ["gated_deltanet", "gdn"] and hasattr(
            past_key_values, "gated_deltanet"
        ):
            return past_key_values.gated_deltanet
        elif self.ssm_mixer_type.lower() in ["gated_kalmanet", "gka"] and hasattr(
            past_key_values, "gka"
        ):
            return past_key_values.gka
        else:
            return None

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        past_key_values: Optional["HybridCache"] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[Tensor] = None,
        position_embeddings: Optional[Tuple[Tensor, Tensor]] = None,
        **kwargs,
    ) -> Tuple[Tensor, None, Optional["BMojoCache"]]:
        """
        Forward pass through the B'MOJO layer.

        The layer processes hidden states through the SSM mixer to generate fading tokens,
        then performs B'MOJO attention which blends two separate self-attention operations:
        one over in-context tokens (from hidden_states) and another over fading tokens
        (from the SSM mixer output). The outputs are combined using their softmax scores.

        Args:
            hidden_states: Input tensor of shape [batch, seq_len, hidden_size].
            attention_mask: Optional attention mask tensor (not currently used).
            position_ids: Optional position indices tensor (not currently used).
            past_key_values: HybridCache object, extracts B'MOJO-specific cache if present.
            use_cache: Whether to use caching. Not used--caching determined based on whether
                past_key_values is provided.
            cache_position: Optional cache position tensor (not currently used).
            position_embeddings: Optional tuple of (cos, sin) for rotary embeddings.
            **kwargs: Additional arguments, ignored.

        Returns:
            Tuple of (hidden_states, None, cache) where:
                - hidden_states: Output tensor of shape [batch, seq_len, hidden_size].
                - None: Placeholder for consistency with attention layers.
                - cache: BMojoCache or None.
        """
        self_attn_input = hidden_states

        # Extract the appropriate cache for the SSM mixer
        fading_past_key_values = self._get_fading_cache(past_key_values=past_key_values)
        fading_kwargs = {
            "position_embeddings": position_embeddings,
            "past_key_values": fading_past_key_values,
        }

        # Extract B'MOJO-specific cache
        if hasattr(past_key_values, "bmojo_f"):
            bmojo_f_past_key_values = past_key_values.bmojo_f
        else:
            bmojo_f_past_key_values = None

        # Generate fading tokens through SSM mixer
        fading_tokens_ = self.ssm_mixer(hidden_states, **fading_kwargs)

        # Sequence mixers often return (hidden_states, None, cache).
        # Extract only the final hidden states here.
        if isinstance(fading_tokens_, tuple):
            fading_tokens_ = fading_tokens_[0]

        # Apply B'MOJO attention with original hidden_states and fading tokens
        outputs, _, _ = self.bmojo_attn(
            self_attn_input,
            key_value_states=fading_tokens_,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=bmojo_f_past_key_values,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )

        return outputs, None, past_key_values
