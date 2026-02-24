from dataclasses import asdict, dataclass
from typing import Optional, Union, TYPE_CHECKING
from transformers.configuration_utils import PretrainedConfig

if TYPE_CHECKING:
    from ..hybrid_dataclasses import HybridBMOJOConfig


@dataclass
class BMojoConfig:
    """
    Configuration class for B'MOJO layers.

    B'MOJO combines an SSM (State Space Model) sequence mixer with sliding window
    attention. The SSM mixer can be one of several types (Mamba2, GDN, GKA).

    Attributes:
        hidden_size: Model dimension (hidden_size from base config)
        num_heads: Number of attention heads
        num_key_value_heads: Number of key-value heads (for GQA)
        head_dim: Dimension of each attention head
        rms_norm_eps: Epsilon for RMS normalization
        window_size: Size of the sliding attention window (default: 2048)
        tie_attn_weights: Whether to tie input/output projection weights in attention (default: True)
        ssm_mixer: Type of SSM mixer to use, e.g., "mamba2", "gdn", "gka" (default: "mamba2")
    """

    # Derived from base model config
    hidden_size: int  # Model dimension
    num_heads: int  # Number of attention heads
    num_key_value_heads: int  # Number of key-value heads (for GQA)
    head_dim: int  # Dimension per attention head
    rms_norm_eps: float  # RMS norm epsilon (used for QK norm)

    # User-configurable parameters
    window_size: int = 2048  # Sliding window size for local attention
    tie_attn_weights: bool = True  # Tie attention projection weights
    ssm_mixer: str = "mamba2"  # SSM mixer type: "mamba2", "gdn", "gka"

    def to_dict(self) -> dict:
        """Convert the config to a dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, config_dict):
        """
        Create BMojoConfig from a dictionary.

        Args:
            config_dict: Dictionary containing config parameters or BMojoConfig instance

        Returns:
            BMojoConfig instance or None if input is invalid
        """
        if isinstance(config_dict, dict):
            return cls(**config_dict)
        elif isinstance(config_dict, cls):
            return config_dict
        return None


def construct_bmojo_config(
    config: PretrainedConfig,
    bmojo_exp_config: Optional[Union[dict, "HybridBMOJOConfig"]] = None,
) -> BMojoConfig:
    """
    Constructs B'MOJO config from base model config.

    Derives necessary parameters from the base model config. User-configurable
    parameters are taken from bmojo_exp_config if provided, otherwise uses
    defaults.

    Derived parameter mappings:
        - hidden_size: Set to config.hidden_size
        - num_heads: Set to config.num_attention_heads
        - num_key_value_heads: Set to config.num_key_value_heads
        - head_dim: Set to config.head_dim or computed as hidden_size // num_heads
        - rms_norm_eps: Set to config.rms_norm_eps

    Args:
        config: The base model's PretrainedConfig containing standard transformer parameters
        bmojo_exp_config: Dict with user-configurable params (or HybridBMOJOConfig)

    Returns:
        A BMojoConfig instance with all derived and user-specified parameters
    """
    # Convert dataclass to dict if needed
    if bmojo_exp_config is not None and hasattr(bmojo_exp_config, "to_dict"):
        bmojo_exp_config = bmojo_exp_config.to_dict()

    # Derive parameters from base model config
    hidden_size = config.hidden_size
    num_heads = config.num_attention_heads
    num_key_value_heads = config.num_key_value_heads

    # Handle head_dim: use explicit value if available, otherwise compute from hidden_size
    head_dim = getattr(config, "head_dim", hidden_size // num_heads)

    rms_norm_eps = config.rms_norm_eps

    # Get user-configurable params with defaults
    if bmojo_exp_config is None:
        bmojo_exp_config = {}

    return BMojoConfig(
        # Derived parameters from base model
        hidden_size=hidden_size,
        num_heads=num_heads,
        num_key_value_heads=num_key_value_heads,
        head_dim=head_dim,
        rms_norm_eps=rms_norm_eps,
        # User-configurable parameters (from config or defaults)
        window_size=bmojo_exp_config.get("window_size", 2048),
        tie_attn_weights=bmojo_exp_config.get("tie_attn_weights", True),
        ssm_mixer=bmojo_exp_config.get("ssm_mixer", "mamba2"),
    )
