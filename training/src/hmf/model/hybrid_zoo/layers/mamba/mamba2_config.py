from dataclasses import asdict, dataclass, field
from typing import Optional, Union, TYPE_CHECKING

from transformers.configuration_utils import PretrainedConfig

if TYPE_CHECKING:
    from ..hybrid_dataclasses import HybridMamba2Config


@dataclass
class MambaConfig:
    """
    Configuration class for Mamba2 layers.

    Attributes:
        d_model: Model dimension (hidden_size)
        d_inner: Inner dimension (num_attention_heads * head_dim)
        d_xb: Key/value dimension (num_key_value_heads * head_dim)
        intermediate_size: Size of the intermediate layer in FFN
        hidden_act: Activation function name
        rms_norm_eps: Epsilon for RMS normalization
        n_layer: Number of layers
        ssm_cfg: SSM-specific configuration dict containing expand, ngroups, d_state
        use_qk_norm: Whether to apply normalization to queries (C in Mamba2) and keys (B in Mamba2)
        use_pos_emb: Whether to apply positional embeddings to B and C in Mamba2.
    """

    # Derived from base model config
    d_model: int  # hidden_size
    d_inner: int  # num_attention_heads * head_dim
    d_xb: int  # num_key_value_heads * head_dim
    intermediate_size: int
    hidden_act: str
    rms_norm_eps: float
    n_layer: int

    ssm_cfg: dict = field(
        default_factory=dict
    )  # Derived. Contains: expand, ngroups, d_state

    # User-configurable parameters
    use_qk_norm: bool = True
    use_pos_emb: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, config_dict):
        """Create MambaConfig from a dictionary."""
        if isinstance(config_dict, dict):
            return cls(**config_dict)
        elif isinstance(config_dict, cls):
            return config_dict
        return None


def construct_mamba2_config(
    config: PretrainedConfig,
    mamba2_exp_config: Optional[Union[dict, "HybridMamba2Config"]] = None,
) -> MambaConfig:
    """
    Constructs Mamba2 config from base model config.

    Recomputes derived parameters from the base config. User-configurable
    parameters are taken from mamba2_exp_config if provided, otherwise uses
    defaults.

    Derived parameter mappings:
        - d_model: Set to config.hidden_size
        - d_inner: Set to num_attention_heads * head_dim
        - d_xb: Set to num_key_value_heads * head_dim
        - d_state (in ssm_cfg): Set to head_dim
        - ngroups (in ssm_cfg): Set to num_attention_heads
        - expand (in ssm_cfg): Set from mamba2_exp_config or defaults to 1

    Args:
        config: The base model's PretrainedConfig
        mamba2_exp_config: Dict with user-configurable params (or HybridMamba2Config)

    Returns:
        A config for Mamba2 with derived + user params
    """
    # Get user-configurable params with defaults
    if mamba2_exp_config is None:
        mamba2_exp_config = {}
    elif hasattr(mamba2_exp_config, "to_dict"):
        mamba2_exp_config = mamba2_exp_config.to_dict()

    # Derive parameters from base model.
    # NOTE: For some models, like Qwen3-4B, hidden_size != num_attention_heads*head_dim. As such,
    # we set d_inner to num_attention_heads*head_dim whenever head_dim is present in the config,
    # otherwise we derive the head dimension from hidden_size.
    if not hasattr(config, "head_dim"):
        head_dim = config.hidden_size // config.num_attention_heads
        d_xb = config.num_key_value_heads * head_dim
        d_inner = config.hidden_size
        d_state = head_dim
    else:
        head_dim = config.head_dim
        d_xb = config.num_key_value_heads * head_dim
        d_inner = config.num_attention_heads * head_dim
        d_state = head_dim

    return MambaConfig(
        # Derived parameters
        d_model=config.hidden_size,
        d_inner=d_inner,
        d_xb=d_xb,
        intermediate_size=config.intermediate_size,
        hidden_act=config.hidden_act,
        n_layer=config.num_hidden_layers,
        rms_norm_eps=config.rms_norm_eps,
        ssm_cfg={
            "expand": 1,
            "ngroups": config.num_attention_heads,
            "d_state": d_state,
        },
        # User-configurable parameters (from config or defaults)
        use_qk_norm=mamba2_exp_config.get("use_qk_norm", True),
        use_pos_emb=mamba2_exp_config.get("use_pos_emb", False),
    )
