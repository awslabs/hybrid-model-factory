"""A decision model, which is a pretrained backbone with its LM head removed and a pointer head added.

The prompt lists a set of options and ends with an answer marker. The backbone reads it once, and
the pointer head scores each option from two hidden states:

    q = W_q · LN(h_answer)                hidden state at the answer marker
    k_j = W_k · LN(h_j)                   hidden state at the last token of option j's line
    logit_j = <q, k_j> / sqrt(dim)

There are no per-option parameters, so the head handles any number of options, and the meaning of
each option comes from the prompt rather than from the weights. The positions of the answer marker
and of each option are computed when the prompt is tokenized and passed in as answer_idx and
opt_idx.

Prompts are right-padded, so under causal attention, and in recurrent layers that read the sequence
left to right, no real position ever reads a pad, and the padding mask changes nothing the head reads.
HMF's hybrid backbones are therefore run without it (see uses_padding_mask), as their sequence
mixers do not all accept one (e.g. GKA refuses a padding mask in eval mode).
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, PreTrainedModel
from transformers.utils import ModelOutput

from .configuration_decision import DecisionConfig


# A large negative value rather than -inf for masked logits, which keeps the softmax finite even if a
# malformed row has every option masked.
MASK_VALUE = -1e4

# The package of HMF's hybrid models (hmf.model.hybrid_zoo)
HYBRID_ZOO_PACKAGE = __name__.split(".models.decision.")[0]


def uses_padding_mask(backbone: nn.Module) -> bool:
    """
    Whether a backbone is given the padding mask of right-padded prompts. Stock Hugging Face models
    are, as they expect it. HMF's hybrid models are not, because their sequence mixers do not all accept one
    (e.g. GKA refuses a padding mask in eval mode), and for right-padded prompts leaving it out is
    exact, since no real position reads a pad.

    Args:
        backbone: The backbone, or a causal LM.
    """
    return not type(backbone).__module__.startswith(HYBRID_ZOO_PACKAGE + ".")


@dataclass
class DecisionModelOutput(ModelOutput):
    """
    Output of DecisionModel.

    Args:
        logits: Option logits of shape [batch, num_options]. Padded options hold MASK_VALUE.
        log_probs: Log-probabilities over each row's options, of shape [batch, num_options].
            Padded options hold -inf.
    """

    logits: Optional[torch.FloatTensor] = None
    log_probs: Optional[torch.FloatTensor] = None


def masked_log_softmax(logits: torch.Tensor, option_mask: torch.Tensor) -> torch.Tensor:
    """
    Log-softmax over each row's real options. Padded options are set to -inf after the softmax,
    so they carry no probability mass.

    Args:
        logits: Option logits of shape [batch, num_options].
        option_mask: Boolean mask of shape [batch, num_options], True for real options.

    Returns:
        log_probs: Log-probabilities of shape [batch, num_options], in float32.
    """
    log_probs = F.log_softmax(logits.masked_fill(~option_mask, MASK_VALUE).float(), dim=-1)
    return log_probs.masked_fill(~option_mask, float("-inf"))


class FP32Linear(nn.Linear):
    """A Linear layer that computes in float32, whatever the dtype of its weights and inputs."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = self.bias.float() if self.bias is not None else None
        return F.linear(x.float(), self.weight.float(), bias)


class FP32LayerNorm(nn.LayerNorm):
    """A LayerNorm that computes in float32, whatever the dtype of its weights and inputs."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.weight.float() if self.weight is not None else None
        bias = self.bias.float() if self.bias is not None else None
        return F.layer_norm(x.float(), self.normalized_shape, weight, bias, self.eps)


class PointerHead(nn.Module):
    """
    Scores each option from its own hidden state, as a single attention score between a query from
    the answer position and a key from each option's position.

    The head always computes in float32, whatever the dtype of its weights or inputs and whether or
    not autocast is active, because it is small and a low-precision readout is a needless source of
    calibration error. Its layers do the casting in their own forward (FP32Linear, FP32LayerNorm),
    so that frameworks that act on a layer's forward, such as DeepSpeed ZeRO-3 (which gathers a
    layer's weights when its forward runs), see every use of the weights.

    Args:
        hidden_size: The hidden size of the backbone.
        dim: Width of the query and key projections. Defaults to 256.
        use_norm: Whether to apply a LayerNorm, shared by the query and key inputs, before the
            projections. Defaults to True.
        dropout: Dropout on the query and key inputs, after the LayerNorm. Defaults to 0.05.
    """

    def __init__(
        self,
        hidden_size: int,
        dim: int = 256,
        use_norm: bool = True,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.norm = FP32LayerNorm(hidden_size) if use_norm else None
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.q = FP32Linear(hidden_size, dim)
        self.k = FP32Linear(hidden_size, dim)
        self.scale = dim**-0.5

        # The head is initialized by hmf decision-init and restored from the checkpoint
        # afterwards; prevent transformers' post_init from re-initializing it.
        for module in self.modules():
            module._is_hf_initialized = True

    def reset_parameters(self) -> None:
        """Re-initialize the head with PyTorch's default initialization."""
        if self.norm is not None:
            self.norm.reset_parameters()
        self.q.reset_parameters()
        self.k.reset_parameters()

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x) if self.norm is not None else x

    def forward(
        self,
        answer_states: torch.Tensor,
        option_states: torch.Tensor,
        option_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            answer_states: Hidden states at the answer position, of shape [batch, hidden_size].
            option_states: Hidden states at each option's position, of shape
                [batch, num_options, hidden_size].
            option_mask: Boolean mask of shape [batch, num_options], True for real options.

        Returns:
            logits: Option logits of shape [batch, num_options], in float32. Padded options hold
                MASK_VALUE.
        """
        with torch.autocast(device_type=answer_states.device.type, enabled=False):
            q = self.q(self.dropout(self._norm(answer_states.float())))
            k = self.k(self.dropout(self._norm(option_states.float())))
            logits = torch.einsum("bkd,bd->bk", k, q) * self.scale
        return logits.masked_fill(~option_mask, MASK_VALUE)


class DecisionModel(PreTrainedModel):
    """
    A backbone loaded with AutoModel (no LM head) followed by a PointerHead. The backbone is any
    model supported by AutoModel, including our Hybrid models.

    Args:
        config: The DecisionConfig, holding the backbone's configuration as text_config.
    """

    config_class = DecisionConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    # Unlike _keep_in_fp32_modules, which only applies when loading in fp16, this also keeps the
    # head in fp32 when loading in bf16
    _keep_in_fp32_modules_strict = ["decision_head"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True

    def __init__(self, config: DecisionConfig):
        super().__init__(config)
        if config.text_config is None:
            raise ValueError("DecisionConfig.text_config must be set to build a DecisionModel.")

        self.model = AutoModel.from_config(config.text_config)
        self.decision_head = PointerHead(
            hidden_size=config.text_config.hidden_size,
            dim=config.decision_head_dim,
            use_norm=config.decision_head_norm,
            dropout=config.decision_head_dropout,
        )

        # Device placement (e.g. device_map="auto") should not split the backbone's layers
        self._no_split_modules = getattr(self.model, "_no_split_modules", None)
        self.pass_attention_mask = uses_padding_mask(self.model)

        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.model.set_input_embeddings(value)

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        opt_idx: torch.LongTensor,
        answer_idx: torch.LongTensor,
        position_ids: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> DecisionModelOutput:
        """
        Args:
            input_ids: Token ids of shape [batch, seq_len], right-padded.
            attention_mask: Attention mask of shape [batch, seq_len]. Not passed to HMF's hybrid
                backbones (see uses_padding_mask).
            opt_idx: Position of the last token of each option's line, of shape
                [batch, num_options]. Rows with fewer options are padded with -1.
            answer_idx: Position of the answer marker, of shape [batch].
            position_ids: Optional position ids of shape [batch, seq_len].
            kwargs: Passed to the backbone.

        Returns:
            A DecisionModelOutput with the option logits and log-probabilities.
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask if self.pass_attention_mask else None,
            position_ids=position_ids,
            use_cache=False,
            return_dict=True,
            **kwargs,
        )
        hidden_states = outputs.last_hidden_state  # [batch, seq_len, hidden_size]

        batch_idx = torch.arange(hidden_states.size(0), device=hidden_states.device)
        answer_states = hidden_states[batch_idx, answer_idx]  # [batch, hidden_size]

        # Padded options (-1) are clamped to position 0; their logits are masked in the head
        option_mask = opt_idx >= 0
        option_pos = opt_idx.clamp_min(0)
        option_states = hidden_states[batch_idx.unsqueeze(-1), option_pos]  # [batch, num_options, hidden_size]

        logits = self.decision_head(answer_states, option_states, option_mask)
        log_probs = masked_log_softmax(logits, option_mask)

        return DecisionModelOutput(logits=logits, log_probs=log_probs)
