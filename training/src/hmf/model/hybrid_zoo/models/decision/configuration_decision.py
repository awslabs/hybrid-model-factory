"""Decision model configuration.

A decision model is a pretrained language model (the backbone) with its LM head removed and a
decision head added. Given a prompt that lists options, the head returns one logit per option. The
pointer head, the default, gives the scaled dot product of a query from the hidden state at the
answer position and a key from the hidden state at the last token of each option's line. The label
heads instead keep the LM head's rows of the options' labels (see decision_head_type). The backbone
can be any model loadable with `AutoModel`, either a stock Transformer or one of our Hybrid models.

Decision models are created from a causal LM with `hmf decision-init` (see
priming/decision_init.py).
"""

from typing import Optional, Union

from transformers.configuration_utils import PreTrainedConfig
from transformers.models.auto import CONFIG_MAPPING, AutoConfig


# How the backbone can attend over the prompt (decision_attention)
DECISION_ATTENTION = ("causal", "bidirectional")

# The decision heads (decision_head_type). The label heads read the LM head's rows of the options' labels.
DECISION_HEAD_TYPES = ("pointer", "label", "label_trained", "label_pointer")
LABEL_HEAD_TYPES = ("label", "label_trained", "label_pointer")


class DecisionConfig(PreTrainedConfig):
    """
    Configuration class for a DecisionModel. It holds the backbone's configuration and the
    decision head's hyperparameters.

    Args:
        text_config: The configuration of the backbone, as a config or a dict. A dict must contain
            model_type, which selects the config class. Defaults to None.
        decision_head_type: The decision head (see modeling_decision.py):
            - "pointer": the pointer head, which scores each option from its own hidden state.
            - "label": the LM head's rows of the options' labels (A, B, ...), frozen, applied to the
              hidden state at the answer marker, so the untrained model answers as its source causal LM
              would with the next token after the answer marker. Only the backbone (e.g., its LoRA
              adapter) trains.
            - "label_trained": the same rows, trained.
            - "label_pointer": the frozen rows plus a pointer head whose scores start at 0, so the
              untrained model is the label head's, and the pointer learns to correct it.
            Defaults to "pointer".
        decision_num_labels: The number of label rows of a label head, which is the number of the
            tokenizer's letter codes (see data/decision_prompt.py) and the most options it can read.
            Defaults to None, for the pointer head.
        decision_label_softcap: The cap of the label logits, cap * tanh(logit / cap), for a source model
            that caps its LM head's logits (Gemma's final_logit_softcapping), or None. Defaults to None.
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
        decision_attention: How the backbone attends over the prompt, which is "causal", as it was
            pretrained, or "bidirectional", where every token attends to every token of the prompt
            (padding stays masked). On a hybrid backbone such as Qwen3.5, only the full attention
            layers become bidirectional, and the recurrent layers stay causal. Bidirectional
            attention needs a stock Hugging Face backbone and no sequence parallelism. Defaults to
            "causal".
    """

    model_type = "hmf_decision"
    sub_configs = {"text_config": AutoConfig}

    def __init__(
        self,
        text_config: Optional[Union[PreTrainedConfig, dict]] = None,
        decision_head_type: str = "pointer",
        decision_num_labels: Optional[int] = None,
        decision_label_softcap: Optional[float] = None,
        decision_head_dim: int = 256,
        decision_head_norm: bool = True,
        decision_head_dropout: float = 0.05,
        decision_head_init: str = "random",
        source_model_name_or_path: Optional[str] = None,
        decision_attention: str = "causal",
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
        if decision_attention not in DECISION_ATTENTION:
            raise ValueError(f"`decision_attention` must be one of {DECISION_ATTENTION}, got {decision_attention!r}.")
        if decision_head_type not in DECISION_HEAD_TYPES:
            raise ValueError(f"`decision_head_type` must be one of {DECISION_HEAD_TYPES}, got {decision_head_type!r}.")
        if decision_head_type in LABEL_HEAD_TYPES and (decision_num_labels is None or decision_num_labels < 2):
            raise ValueError(f"A {decision_head_type} head needs `decision_num_labels` of at least 2.")

        self.decision_head_type = decision_head_type
        self.decision_num_labels = decision_num_labels if decision_head_type in LABEL_HEAD_TYPES else None
        self.decision_label_softcap = decision_label_softcap
        # The labels that a label head reads and that prompts must show (decision_option_labels), or None for the
        # pointer head
        self.decision_option_labels = "letters" if decision_head_type in LABEL_HEAD_TYPES else None

        self.decision_head_dim = decision_head_dim
        self.decision_head_norm = decision_head_norm
        self.decision_head_dropout = decision_head_dropout
        self.decision_head_init = decision_head_init
        self.source_model_name_or_path = source_model_name_or_path
        self.decision_attention = decision_attention

        # The decision model has no LM head, so there is nothing to tie
        kwargs["tie_word_embeddings"] = False
        super().__init__(**kwargs)
