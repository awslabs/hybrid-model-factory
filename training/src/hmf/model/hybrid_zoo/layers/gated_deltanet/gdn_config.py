from dataclasses import asdict, dataclass
from typing import Optional, Union, TYPE_CHECKING

from transformers.configuration_utils import PretrainedConfig

if TYPE_CHECKING:
    from ..hybrid_dataclasses import HybridGatedDeltaNetConfig


@dataclass
class GDNConfig:
    """
    Configuration class for Gated DeltaNet layer.

    This config contains both derived parameters (computed from the base model config)
    and user-configurable parameters (can be customized per layer).

    Attributes:
        hidden_size: Model dimension (hidden_size). Derived from base config.
        head_dim: Dimension of each attention head. Derived from base config.
        num_q_heads: Number of query heads. Derived from base config.
        num_k_heads: Number of key heads. Derived from base config.
        num_v_heads: Number of value heads. Derived from base config.
        use_gate: Whether to use gating mechanism. User-configurable.
        use_short_conv: Whether to use short convolution. User-configurable.
        allow_neg_eigval: Whether to allow negative eigenvalues. User-configurable.
        conv_size: Size of the convolution kernel. User-configurable.
        conv_bias: Whether to use bias in convolution. User-configurable.
        norm_eps: Epsilon for normalization. User-configurable.
        kv_proj_rank: Rank for low-rank KV projection. None disables projection. User-configurable.
        kv_learnable_residual: Whether to use a learnable residual for the KV head expansion instead
            of a fixed repeat. User-configurable.
    """

    # Core parameters
    hidden_size: int
    head_dim: int
    num_q_heads: int
    num_k_heads: int
    num_v_heads: int

    # User-configurable parameters
    use_gate: bool = True
    use_short_conv: bool = True
    allow_neg_eigval: bool = False
    conv_size: int = 4
    conv_bias: bool = False
    norm_eps: float = 1e-5
    kv_proj_rank: Optional[int] = None
    kv_learnable_residual: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, config_dict):
        if isinstance(config_dict, dict):
            return cls(**config_dict)
        elif isinstance(config_dict, cls):
            return config_dict
        return None


def _is_qwen_gdn_config(config: PretrainedConfig) -> bool:
    """Check if config is from a Qwen model with GDN layers (e.g., Qwen3-Next and Qwen3.5)"""
    return hasattr(config, "linear_num_key_heads") and hasattr(
        config, "linear_num_value_heads"
    )


def _construct_from_qwen_gdn(
    config: PretrainedConfig, gdn_exp_config: Optional[dict] = None
) -> GDNConfig:
    """Construct GDNConfig from a GDN QWen config (e.g., Qwen3-Next and Qwen3.5)."""
    gdn_exp_config = gdn_exp_config or {}

    return GDNConfig(
        hidden_size=config.hidden_size,
        head_dim=config.linear_key_head_dim,
        num_q_heads=config.linear_num_key_heads,
        num_k_heads=config.linear_num_key_heads,
        num_v_heads=config.linear_num_value_heads,
        # Qwen3-Next specific defaults
        use_gate=gdn_exp_config.get("use_gate", True),
        use_short_conv=gdn_exp_config.get("use_short_conv", True),
        allow_neg_eigval=gdn_exp_config.get("allow_neg_eigval", False),
        conv_size=gdn_exp_config.get("conv_size", config.linear_conv_kernel_dim),
        conv_bias=gdn_exp_config.get("conv_bias", False),
        norm_eps=gdn_exp_config.get("norm_eps", config.rms_norm_eps),
        kv_proj_rank=gdn_exp_config.get("kv_proj_rank", None),
        kv_learnable_residual=gdn_exp_config.get("kv_learnable_residual", False),
    )


def _construct_from_transformer(
    config: PretrainedConfig, gdn_exp_config: Optional[dict] = None
) -> GDNConfig:
    """Construct GDNConfig from standard transformer config (attention -> GDN conversion)."""
    gdn_exp_config = gdn_exp_config or {}

    num_heads = config.num_attention_heads
    num_kv_heads = getattr(config, "num_key_value_heads", num_heads)
    head_dim = getattr(config, "head_dim", config.hidden_size // num_heads)

    # Standard GQA: Q has more heads, K/V have fewer
    return GDNConfig(
        hidden_size=config.hidden_size,
        head_dim=head_dim,
        num_q_heads=num_heads,
        num_k_heads=num_kv_heads,
        num_v_heads=num_kv_heads,
        use_gate=gdn_exp_config.get("use_gate", True),
        use_short_conv=gdn_exp_config.get("use_short_conv", True),
        allow_neg_eigval=gdn_exp_config.get("allow_neg_eigval", False),
        conv_size=gdn_exp_config.get("conv_size", 4),
        conv_bias=gdn_exp_config.get("conv_bias", False),
        norm_eps=gdn_exp_config.get("norm_eps", 1e-5),
        kv_proj_rank=gdn_exp_config.get("kv_proj_rank", None),
        kv_learnable_residual=gdn_exp_config.get("kv_learnable_residual", False),
    )


def construct_gdn_config(
    config: PretrainedConfig,
    gdn_exp_config: Optional[Union[dict, "HybridGatedDeltaNetConfig"]] = None,
) -> GDNConfig:
    """
    Constructs GDNConfig from a model config.
    
    Automatically detects whether the config is from:
        1. A GDN-based Qwen model (such as Qwen3-Next and Qwen3.5): In this case
            uses linear_* params directly
        2. Standard transformer: In this case, we derive GKA params from the
            Transformer's config.
    
    Args:
        config: The model's PretrainedConfig
        gdn_exp_config: Optional dict with user-configurable overrides
        
    Returns:
        A GDNConfig instance
    """
    if gdn_exp_config is not None and hasattr(gdn_exp_config, "to_dict"):
        gdn_exp_config = gdn_exp_config.to_dict()

    if _is_qwen_gdn_config(config):
        return _construct_from_qwen_gdn(config, gdn_exp_config)
    else:
        return _construct_from_transformer(config, gdn_exp_config)
