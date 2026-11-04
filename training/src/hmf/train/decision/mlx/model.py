"""
A decision model in MLX, which mirrors the PyTorch DecisionModel of model/hybrid_zoo/models/decision/, so that the
same checkpoint gives the same answers. The backbone is mlx-lm's text model (see backbone.py). The pointer head, which
always computes in float32, scores each option from its own hidden state, as a single attention score between a query
from the answer position and a key from each option's position.

The model takes no attention mask. Decision prompts are right-padded, and every position the head reads (the answer
position and the options' positions) comes before the padding, so causal attention, Gated DeltaNet's causal
recurrence, and its causal convolution give these positions the same hidden states with or without a mask.
"""

from typing import Dict, Tuple

import mlx.core as mx
from mlx import nn

from .backbone import Backbone, load_backbone


__all__ = ["MASK_VALUE", "DecisionModelMLX", "PointerHead", "load_decision_model", "masked_log_softmax"]

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


class DecisionModelMLX(nn.Module):
    """
    A decision model in MLX, which is a backbone (mlx-lm's decoder) and a pointer head.

    Args:
        decoder: mlx-lm's decoder, which maps token ids to the final normalized hidden states.
        head: The pointer head.
    """

    def __init__(self, decoder: nn.Module, head: PointerHead):
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


def _head_from_config(config: Dict, hidden_size: int, head_weights: Dict[str, mx.array]) -> PointerHead:
    head = PointerHead(
        hidden_size,
        dim=config.get("decision_head_dim", 256),
        use_norm=config.get("decision_head_norm", True),
        dropout=config.get("decision_head_dropout", 0.05),
    )
    head.set_dtype(mx.float32)
    head.load_weights(list(head_weights.items()), strict=True)
    return head


def load_decision_model(model_path: str, dtype: str = "bfloat16") -> DecisionModelMLX:
    """
    Loads a decision model (an hmf_decision checkpoint) in MLX. The backbone is in dtype and frozen, and the head is
    in float32 and trainable. The model starts in evaluation mode, so dropout is off until model.train() is called.

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
