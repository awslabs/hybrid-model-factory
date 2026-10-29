"""Decision model configuration.

A decision model is a pretrained language model (the backbone) with its LM head removed and a
pointer head added. Given a prompt that lists options, the pointer head returns one logit per
option, the scaled dot product of a query from the hidden state at the answer position and a key
from the hidden state at the last token of each option's line. The backbone can be any model
loadable with `AutoModel`, either a stock Transformer or one of our Hybrid models.

Decision models are created from a causal LM with `hmf decision-init` (see
priming/decision_init.py).
"""

from typing import Optional, Union

from transformers.configuration_utils import PreTrainedConfig
from transformers.models.auto import CONFIG_MAPPING, AutoConfig


class DecisionConfig(PreTrainedConfig):
    """
    Configuration class for a DecisionModel. It holds the backbone's configuration and the
    pointer head's hyperparameters.

    Args:
        text_config: The configuration of the backbone, as a config or a dict. A dict must contain
            model_type, which selects the config class. Defaults to None.
        decision_head_dim: Width of the pointer head's query and key projections. Defaults to 256.
        decision_head_norm: Whether to apply a LayerNorm, shared by the query and key inputs, before
            the projections. Defaults to True.
        decision_head_dropout: Dropout on the query and key inputs, after the LayerNorm. 0 disables
            it. Defaults to 0.05.
        decision_head_init: How the head was initialized by hmf decision-init. Recorded for
            provenance; loading a checkpoint always restores the saved weights. Defaults to "random".
        source_model_name_or_path: The causal LM the backbone was taken from. Recorded for
            provenance, e.g. to reattach the vision tower of a multimodal source model.
            Defaults to None.
    """

    model_type = "hmf_decision"
    sub_configs = {"text_config": AutoConfig}

    def __init__(
        self,
        text_config: Optional[Union[PreTrainedConfig, dict]] = None,
        decision_head_dim: int = 256,
        decision_head_norm: bool = True,
        decision_head_dropout: float = 0.05,
        decision_head_init: str = "random",
        source_model_name_or_path: Optional[str] = None,
        **kwargs,
    ):
        if isinstance(text_config, dict):
            if "model_type" not in text_config:
                raise ValueError("`text_config` must contain `model_type`.")
            text_config = CONFIG_MAPPING[text_config["model_type"]](**text_config)
        self.text_config = text_config

        if decision_head_dim <= 0:
            raise ValueError(f"`decision_head_dim` must be positive, got {decision_head_dim}.")
        if not 0.0 <= decision_head_dropout < 1.0:
            raise ValueError(f"`decision_head_dropout` must be in [0, 1), got {decision_head_dropout}.")

        self.decision_head_dim = decision_head_dim
        self.decision_head_norm = decision_head_norm
        self.decision_head_dropout = decision_head_dropout
        self.decision_head_init = decision_head_init
        self.source_model_name_or_path = source_model_name_or_path

        # The decision model has no LM head, so there is nothing to tie
        kwargs["tie_word_embeddings"] = False
        super().__init__(**kwargs)
