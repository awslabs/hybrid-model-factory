from dataclasses import asdict, dataclass
from typing import Optional, Union, TYPE_CHECKING

from transformers.configuration_utils import PretrainedConfig

if TYPE_CHECKING:
    from ..hybrid_dataclasses import HybridGKAConfig


@dataclass
class GKAConfig:
    """
    Configuration class for Gated KalmaNet (GKA) layer.

    This config contains both derived parameters (computed from the base model config)
    and user-configurable parameters (can be customized per layer).

    Attributes:
        hidden_size: Model dimension (hidden_size). Derived from base config.
        head_dim: Dimension of each attention head. Derived from base config.
        num_q_heads: Number of query heads. Derived from base config.
        num_k_heads: Number of key heads. Derived from base config.
        num_v_heads: Number of value heads. Derived from base config.
        use_alpha_connection: Whether to use alpha connection for residual paths. User-configurable.
        use_v_conv: Whether to apply convolution to value vectors. User-configurable.
        use_forgetting_gate: Whether to use forgetting gate mechanism. User-configurable.
        gla_rescale: Whether to apply GLA-style rescaling. User-configurable.
        solver_type: Type of solver for regression problem. Currently only support 'chebyshev'. User-configurable.
        bp_lambda: Whether to backpropagate through lambda parameters. User-configurable.
        num_iter: Number of iterations for iterative solvers. User-configurable.
        ridge_strength: Regularization strength for ridge regression. User-configurable.
        use_gate: Whether to use gating mechanism. User-configurable.
        conv_size: Size of the convolution kernel. User-configurable.
        norm_eps: Epsilon for normalization. User-configurable.
        use_forgetting_gate_kk: Whether to use forgetting gate for key-key interactions. User-configurable.
        use_beta_gate: Whether to use beta gating mechanism. User-configurable.
        chunk_size: Triton block size. User-configurable.
        kv_proj_rank: Rank for low-rank KV projection. None disables projection. User-configurable.
        kv_learnable_residual: Whether to use a learnable residual for the KV head expansion
            instead of a fixed repeat. User-configurable.
    """

    # Derived from base model config
    hidden_size: int
    head_dim: int
    num_q_heads: int
    num_k_heads: int
    num_v_heads: int

    # User-configurable parameters
    use_alpha_connection: bool = True
    use_v_conv: bool = True
    use_forgetting_gate: bool = True
    gla_rescale: bool = True
    solver_type: str = "chebyshev"
    bp_lambda: bool = True
    num_iter: int = 30
    ridge_strength: float = 0.02
    use_gate: bool = True
    conv_size: int = 4
    norm_eps: float = 1e-6
    use_forgetting_gate_kk: bool = True
    use_beta_gate: bool = True
    chunk_size: int = 64
    kv_proj_rank: Optional[int] = None
    kv_learnable_residual: bool = False

    def to_dict(self) -> dict:
        """Convert the config to a dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, config_dict):
        """
        Create GKAConfig from a dictionary.

        Args:
            config_dict: Dictionary containing config parameters or GKAConfig instance

        Returns:
            GKAConfig instance or None if input is invalid
        """
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
    config: PretrainedConfig, gka_exp_config: Optional[dict] = None
) -> GKAConfig:
    """Construct GKAConfig from a GDN QWen config (e.g., Qwen3-Next and Qwen3.5)."""
    gka_exp_config = gka_exp_config or {}

    return GKAConfig(
        hidden_size=config.hidden_size,
        head_dim=config.linear_key_head_dim,
        num_q_heads=config.linear_num_key_heads,
        num_k_heads=config.linear_num_key_heads,
        num_v_heads=config.linear_num_value_heads,
        # User-configurable parameters (from config or defaults)
        use_alpha_connection=gka_exp_config.get("use_alpha_connection", True),
        use_v_conv=gka_exp_config.get("use_v_conv", True),
        use_forgetting_gate=gka_exp_config.get("use_forgetting_gate", True),
        gla_rescale=gka_exp_config.get("gla_rescale", True),
        solver_type=gka_exp_config.get("solver_type", "chebyshev"),
        bp_lambda=gka_exp_config.get("bp_lambda", True),
        num_iter=gka_exp_config.get("num_iter", 30),
        ridge_strength=gka_exp_config.get("ridge_strength", 0.02),
        use_gate=gka_exp_config.get("use_gate", True),
        conv_size=gka_exp_config.get("conv_size", config.linear_conv_kernel_dim),
        norm_eps=gka_exp_config.get("norm_eps", config.rms_norm_eps),
        use_forgetting_gate_kk=gka_exp_config.get("use_forgetting_gate_kk", True),
        use_beta_gate=gka_exp_config.get("use_beta_gate", True),
        chunk_size=gka_exp_config.get("chunk_size", 64),
        kv_proj_rank=gka_exp_config.get("kv_proj_rank", None),
        kv_learnable_residual=gka_exp_config.get("kv_learnable_residual", False),
    )


def _construct_from_transformer(
    config: PretrainedConfig, gka_exp_config: Optional[dict] = None
) -> GKAConfig:
    """Construct GKAConfig from standard transformer config (Attention -> GKA conversion)."""
    gka_exp_config = gka_exp_config or {}

    # Derive parameters from base model
    hidden_size = config.hidden_size
    num_heads = config.num_attention_heads
    num_kv_heads = config.num_key_value_heads
    head_dim = getattr(config, "head_dim", hidden_size // num_heads)

    # Standard GQA: Q has more heads, K/V have fewer
    return GKAConfig(
        hidden_size=config.hidden_size,
        head_dim=head_dim,
        num_q_heads=num_heads,
        num_k_heads=num_kv_heads,
        num_v_heads=num_kv_heads,
        # User-configurable parameters (from config or defaults)
        use_alpha_connection=gka_exp_config.get("use_alpha_connection", True),
        use_v_conv=gka_exp_config.get("use_v_conv", True),
        use_forgetting_gate=gka_exp_config.get("use_forgetting_gate", True),
        gla_rescale=gka_exp_config.get("gla_rescale", True),
        solver_type=gka_exp_config.get("solver_type", "chebyshev"),
        bp_lambda=gka_exp_config.get("bp_lambda", True),
        num_iter=gka_exp_config.get("num_iter", 30),
        ridge_strength=gka_exp_config.get("ridge_strength", 0.02),
        use_gate=gka_exp_config.get("use_gate", True),
        conv_size=gka_exp_config.get("conv_size", 4),
        norm_eps=gka_exp_config.get("norm_eps", 1e-6),
        use_forgetting_gate_kk=gka_exp_config.get("use_forgetting_gate_kk", True),
        use_beta_gate=gka_exp_config.get("use_beta_gate", True),
        chunk_size=gka_exp_config.get("chunk_size", 64),
        kv_proj_rank=gka_exp_config.get("kv_proj_rank", None),
        kv_learnable_residual=gka_exp_config.get("kv_learnable_residual", False),
    )


def construct_gka_config(
    config: PretrainedConfig,
    gka_exp_config: Optional[Union[dict, "HybridGKAConfig"]] = None,
) -> GKAConfig:
    """
    Constructs GKAConfig from a model config.
    
    Automatically detects whether the config is from:
        1. A GDN-based Qwen model (such as Qwen3-Next and Qwen3.5): In this case
            uses linear_* params directly
        2. Standard transformer: In this case, we derive GKA params from the
            Transformer's config.
    
    Args:
        config: The model's PretrainedConfig
        gka_exp_config: Optional dict with user-configurable overrides
        
    Returns:
        A GKAConfig instance
    """
    if gka_exp_config is not None and hasattr(gka_exp_config, "to_dict"):
        gka_exp_config = gka_exp_config.to_dict()

    if _is_qwen_gdn_config(config):
        return _construct_from_qwen_gdn(config, gka_exp_config)
    else:
        return _construct_from_transformer(config, gka_exp_config)
