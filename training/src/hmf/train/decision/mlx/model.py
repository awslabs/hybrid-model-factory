"""
A decision model in MLX, which mirrors the PyTorch DecisionModel of model/hybrid_zoo/models/decision/, so that the
same checkpoint gives the same answers. The backbone is mlx-lm's text model (see backbone.py), and the decision head
is the one that the checkpoint's decision_head_type names, with the parameter names of the PyTorch heads, and every
head computes in float32.

    - The pointer head ("pointer") scores each option from its own hidden state, as a single attention score between
      a query from the answer position and a key from each option's position.
    - The label heads ("label" and "label_trained") score the option shown at position k with row k, a copy of the
      source causal LM's LM-head row of label k's token (A, B, ...), applied to the hidden state at the answer
      position. The rows are frozen for "label" and trained for "label_trained".
    - The label pointer head ("label_pointer") adds the scores of a frozen label head and of a pointer head.

PyTorch keeps frozen rows as a buffer, which never trains. MLX has no buffers, so a label head whose rows are frozen
freezes them itself, and lora.set_trainable freezes them again after unfreezing the head.

The model takes no attention mask. Decision prompts are right-padded, and every position the head reads (the answer
position and the options' positions) comes before the padding, so causal attention, Gated DeltaNet's causal
recurrence, and its causal convolution give these positions the same hidden states with or without a mask.
"""

from typing import Dict, Optional, Tuple

import mlx.core as mx
from mlx import nn

from .backbone import Backbone, load_backbone


__all__ = [
    "MASK_VALUE",
    "DecisionModelMLX",
    "LabelHead",
    "LabelPointerHead",
    "PointerHead",
    "freeze_fixed_rows",
    "load_decision_model",
    "masked_log_softmax",
]

# These are the values of decision_head_type whose heads hold label rows
LABEL_HEAD_TYPES = ("label", "label_trained", "label_pointer")

# The logit of a padded option, as in the PyTorch DecisionModel
MASK_VALUE = -1e4
# PyTorch's default LayerNorm epsilon, which the PyTorch pointer head uses
LAYER_NORM_EPS = 1e-5


def masked_log_softmax(logits: mx.array, option_mask: mx.array) -> mx.array:
    """
    Log-softmax over each row's real options, in float32. Padded options are set to -inf after the softmax, so
    they carry no probability mass.

    Args:
        logits: Option logits of shape [batch, num_options].
        option_mask: Boolean mask of shape [batch, num_options], True for real options.

    Returns:
        log_probs: Log-probabilities of shape [batch, num_options].
    """
    logits = mx.where(option_mask, logits, MASK_VALUE).astype(mx.float32)
    log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    return mx.where(option_mask, log_probs, -mx.inf)


class PointerHead(nn.Module):
    """
    The pointer head of the PyTorch DecisionModel in MLX, with the same parameter names (norm, q, and k), so that it
    loads the head's weights from a decision checkpoint. It always computes in float32, whatever the dtype of its
    inputs.

    Args:
        hidden_size: The hidden size of the backbone.
        dim: Width of the query and key projections.
        use_norm: Whether to apply a LayerNorm, shared by the query and key inputs, before the projections.
        dropout: Dropout on the query and key inputs, after the LayerNorm, in training mode.
    """

    def __init__(self, hidden_size: int, dim: int, use_norm: bool, dropout: float):
        super().__init__()
        if use_norm:
            self.norm = nn.LayerNorm(hidden_size, eps=LAYER_NORM_EPS)
        self.use_norm = use_norm
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        self.q = nn.Linear(hidden_size, dim)
        self.k = nn.Linear(hidden_size, dim)
        self.scale = dim**-0.5

    def _inputs(self, x: mx.array) -> mx.array:
        x = x.astype(mx.float32)
        if self.use_norm:
            x = self.norm(x)
        return self.dropout(x) if self.dropout is not None else x

    def __call__(self, answer_states: mx.array, option_states: mx.array, option_mask: mx.array) -> mx.array:
        """
        Args:
            answer_states: Hidden states at the answer position, of shape [batch, hidden_size].
            option_states: Hidden states at each option's position, of shape [batch, num_options, hidden_size].
            option_mask: Boolean mask of shape [batch, num_options], True for real options.

        Returns:
            logits: Option logits of shape [batch, num_options], in float32. Padded options hold MASK_VALUE.
        """
        q = self.q(self._inputs(answer_states))  # [batch, dim]
        k = self.k(self._inputs(option_states))  # [batch, num_options, dim]
        logits = (k @ q[:, :, None]).squeeze(-1) * self.scale
        return mx.where(option_mask, logits, MASK_VALUE)


class LabelHead(nn.Module):
    """
    The label head of the PyTorch DecisionModel in MLX, with the same parameter name (rows), which scores the option
    shown at position k with row k applied to the hidden state at the answer position. For a source LM that caps its
    logits, the scores are capped as cap * tanh(logit / cap). The head always computes in float32 and has no dropout,
    so that an untrained head answers as the source LM does.

    Args:
        hidden_size: The hidden size of the backbone.
        num_labels: The number of rows, which is the most options a question can have.
        trainable: Whether the rows are trained. Frozen rows are frozen here and by freeze_fixed_rows.
        softcap: The cap of the logits, or None.
    """

    def __init__(self, hidden_size: int, num_labels: int, trainable: bool = False, softcap: Optional[float] = None):
        super().__init__()
        self.rows = mx.zeros((num_labels, hidden_size), dtype=mx.float32)
        self.trainable_rows = trainable
        self.softcap = softcap
        if not trainable:
            self.freeze(keys=["rows"], recurse=False)

    def __call__(self, answer_states: mx.array, option_states: mx.array, option_mask: mx.array) -> mx.array:
        """
        Args:
            answer_states: Hidden states at the answer position, of shape [batch, hidden_size].
            option_states: Unused, for the pointer head's signature.
            option_mask: Boolean mask of shape [batch, num_options], True for real options.

        Returns:
            logits: Option logits of shape [batch, num_options], in float32. Padded options hold MASK_VALUE.
        """
        num_options = option_mask.shape[1]
        if num_options > self.rows.shape[0]:
            raise ValueError(f"The label head reads at most {self.rows.shape[0]} options, got {num_options}.")
        logits = answer_states.astype(mx.float32) @ self.rows[:num_options].astype(mx.float32).T
        if self.softcap:
            logits = self.softcap * mx.tanh(logits / self.softcap)
        return mx.where(option_mask, logits, MASK_VALUE)


class LabelPointerHead(nn.Module):
    """
    The label pointer head of the PyTorch DecisionModel in MLX, whose logits are those of a frozen label head plus
    those of a pointer head. Its parameter names are PyTorch's, label.rows for the label head, and pointer.norm,
    pointer.q, and pointer.k for the pointer head.

    Args:
        hidden_size: The hidden size of the backbone.
        num_labels: The number of label rows.
        softcap: The cap of the label logits, or None. The pointer's scores are not capped.
        dim, use_norm, dropout: The pointer head's (see PointerHead).
    """

    def __init__(
        self,
        hidden_size: int,
        num_labels: int,
        softcap: Optional[float] = None,
        dim: int = 256,
        use_norm: bool = True,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.label = LabelHead(hidden_size, num_labels, trainable=False, softcap=softcap)
        self.pointer = PointerHead(hidden_size, dim=dim, use_norm=use_norm, dropout=dropout)

    def __call__(self, answer_states: mx.array, option_states: mx.array, option_mask: mx.array) -> mx.array:
        logits = self.label(answer_states, option_states, option_mask) + self.pointer(
            answer_states, option_states, option_mask
        )
        return mx.where(option_mask, logits, MASK_VALUE)


def freeze_fixed_rows(model: nn.Module) -> None:
    """
    Freezes the rows of every label head in a model that does not train its rows, i.e., the heads of "label" and
    "label_pointer" models.
    """
    for _, module in model.named_modules():
        if isinstance(module, LabelHead) and not module.trainable_rows:
            module.freeze(keys=["rows"], recurse=False)


class DecisionModelMLX(nn.Module):
    """
    A decision model in MLX, which is a backbone (mlx-lm's decoder) and a decision head.

    Args:
        decoder: mlx-lm's decoder, which maps token ids to the final normalized hidden states.
        head: The decision head (a PointerHead, LabelHead, or LabelPointerHead).
    """

    def __init__(self, decoder: nn.Module, head: nn.Module):
        super().__init__()
        self.model = decoder
        self.decision_head = head

    def __call__(self, input_ids: mx.array, opt_idx: mx.array, answer_idx: mx.array) -> Tuple[mx.array, mx.array]:
        """
        Args:
            input_ids: Right-padded token ids of shape [batch, seq_len].
            opt_idx: The position of each option's token, of shape [batch, num_options], with -1 for padded options.
            answer_idx: The answer position of each row, of shape [batch].

        Returns:
            logits: Option logits of shape [batch, num_options], in float32, with MASK_VALUE for padded options.
            log_probs: Log-probabilities of shape [batch, num_options], in float32, with -inf for padded options.
        """
        hidden_states = self.model(input_ids)  # [batch, seq_len, hidden_size]
        rows = mx.arange(input_ids.shape[0])
        answer_states = hidden_states[rows, answer_idx]  # [batch, hidden_size]
        option_mask = opt_idx >= 0
        option_states = hidden_states[rows[:, None], mx.maximum(opt_idx, 0)]  # [batch, num_options, hidden_size]
        logits = self.decision_head(answer_states, option_states, option_mask)
        return logits, masked_log_softmax(logits, option_mask)


def _head_from_config(config: Dict, hidden_size: int, head_weights: Dict[str, mx.array]) -> nn.Module:
    """Builds the decision head that the configuration's decision_head_type names, and loads its weights."""
    head_type = config.get("decision_head_type") or "pointer"
    pointer_kwargs = {
        "dim": config.get("decision_head_dim", 256),
        "use_norm": config.get("decision_head_norm", True),
        "dropout": config.get("decision_head_dropout", 0.05),
    }
    if head_type in LABEL_HEAD_TYPES:
        num_labels, softcap = config["decision_num_labels"], config.get("decision_label_softcap")
        if head_type == "label_pointer":
            head = LabelPointerHead(hidden_size, num_labels, softcap=softcap, **pointer_kwargs)
        else:
            head = LabelHead(hidden_size, num_labels, trainable=head_type == "label_trained", softcap=softcap)
    elif head_type == "pointer":
        head = PointerHead(hidden_size, **pointer_kwargs)
    else:
        raise ValueError(f"The MLX backend has no decision head {head_type!r}.")
    head.set_dtype(mx.float32)
    head.load_weights(list(head_weights.items()), strict=True)
    return head


def load_decision_model(model_path: str, dtype: str = "bfloat16") -> DecisionModelMLX:
    """
    Loads a decision model (an hmf_decision checkpoint) in MLX. The backbone is in dtype and frozen, and the head is
    in float32 and trainable, apart from the frozen rows of a "label" or "label_pointer" head. The model starts in
    evaluation mode, so dropout is off until model.train() is called.

    Args:
        model_path: The folder of a decision model.
        dtype: The dtype of the backbone, one of backbone.DTYPES. Defaults to "bfloat16".

    Returns:
        The decision model.
    """
    backbone: Backbone = load_backbone(model_path, dtype)
    head = _head_from_config(backbone.config, backbone.text_config["hidden_size"], backbone.head_weights)
    model = DecisionModelMLX(backbone.decoder, head)
    model.eval()
    return model
