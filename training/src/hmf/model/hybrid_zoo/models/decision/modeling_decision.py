"""A decision model, which is a pretrained backbone with its LM head removed and a decision head added.

The prompt lists a set of options and ends with an answer marker. The backbone reads it once, and
the pointer head, the default, scores each option from two hidden states:

    q = W_q · LN(h_answer)                hidden state at the answer marker
    k_j = W_k · LN(h_j)                   hidden state at the last token of option j's line
    logit_j = <q, k_j> / sqrt(dim)

There are no per-option parameters, so the head handles any number of options, and the meaning of
each option comes from the prompt rather than from the weights. The positions of the answer marker
and of each option are computed when the prompt is tokenized and passed in as answer_idx and
opt_idx.

Label heads (decision_head_type) read the options as a causal LM answers a multiple-choice
question instead. The prompt labels the options with letters (A, B, ...), and the head keeps the
source LM's LM-head rows of those letters' tokens, so that the logit of the option shown at position
k is that row applied to the hidden state at the answer marker, which before training is the
source LM's next-token logit of label k. The "label" head keeps the rows frozen, "label_trained" trains them,
and "label_pointer" adds a pointer head whose scores start at 0 (see LabelHead and LabelPointerHead).

Prompts are right-padded, so under causal attention, and in recurrent layers that read the sequence
left to right, no real position ever reads a pad, and the padding mask changes nothing the head reads.
HMF's hybrid backbones are therefore run without it (see uses_padding_mask), as their sequence
mixers do not all accept one (e.g. GKA refuses a padding mask in eval mode).

Given each prompt's length (seq_lens), a stock backbone with flash attention runs without a mask at all.
Its attention layers get each prompt's boundaries instead (see varlen_kwargs), computed from the lengths
on the GPU, and its recurrent layers (Qwen3.5's Gated DeltaNet) read the right-padded prompts as they
are. Building a padding mask, and transformers' checks of the mask, read values back from the GPU and
make the CPU wait for it, which a CUDA graph cannot record. This path needs no such reads.

With sequence parallelism, every rank of a sequence parallel group is given the same batch. Each
rank runs the backbone on its zig-zag chunks of the prompts (see zigzag_local), and then gathers the
backbone's hidden states of the whole prompts. Every rank therefore computes the head and the loss
on the same hidden states, as without sequence parallelism. tests/decision/model/decision_sp_parity.py
checks this against the same models without sequence parallelism.

A mixture-of-experts backbone (e.g., Qwen3.5's) can run with fewer active experts per token than it was trained
with, by lowering num_experts_per_tok in its config. Transformers' routers then renormalize the weights of the
experts they keep to sum to 1, which scales them up. With moe_topk_renormalize: false in the backbone's config, the
routers instead divide by the sum of the top moe_topk_norm_k weights (8 by default, the number Qwen3.5's and
Qwen3.6's mixtures of experts are trained with), so that each kept expert has exactly the weight it has at
moe_topk_norm_k experts (see keep_topk_weights).
"""

import contextlib
import types
from dataclasses import dataclass
from typing import Any, Generator, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, PreTrainedModel
from transformers.utils import ModelOutput

from hmf.model.model_utils.allgather_sp_helper import ZigZagGatherScatter, get_zigzag_indices

from .configuration_decision import DecisionConfig


# A large negative value rather than -inf for masked logits, which keeps the softmax finite even if a
# malformed row has every option masked.
MASK_VALUE = -1e4

# The arguments that give transformers' flash attention the prompts' boundaries (see varlen_kwargs)
VARLEN_KWARGS = ("cu_seq_lens_q", "cu_seq_lens_k", "max_length_q", "max_length_k")

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


def zigzag_local(tensor: torch.Tensor, sp_rank: int, sp_size: int) -> torch.Tensor:
    """
    The zig-zag chunks of tensor that a sequence parallel rank processes, in the layout that
    ZigZagGatherScatter gathers: the sequence is split into 2 * sp_size chunks, and rank r keeps
    chunks r and 2 * sp_size - 1 - r.

    Args:
        tensor: A tensor of shape [batch, seq_len, ...], with seq_len a multiple of 2 * sp_size.
        sp_rank: The rank within the sequence parallel group.
        sp_size: The size of the sequence parallel group.

    Returns:
        The rank's chunks, concatenated, of shape [batch, seq_len // sp_size, ...].
    """
    chunks = tensor.chunk(2 * sp_size, dim=1)
    return torch.cat([chunks[i] for i in get_zigzag_indices(sp_size, 2 * sp_size)[sp_rank]], dim=1)


def varlen_kwargs(seq_lens: torch.Tensor, seq_len: int) -> dict:
    """
    The arguments that tell flash attention where each right-padded prompt starts and ends, so that it
    needs no mask. Flash attention reads the batch as one sequence of batch * seq_len tokens split at
    these boundaries, and each row's padding is a sequence of its own (empty when a row has none), so
    no real token attends to padding and no padding token to a real one. Everything is computed on the
    GPU, without reading anything back.

    Args:
        seq_lens: Each prompt's length without padding, of shape [batch], on the model's device.
        seq_len: The padded length of the batch.

    Returns:
        cu_seq_lens_q, cu_seq_lens_k, max_length_q and max_length_k, as transformers' flash attention
        takes them.
    """
    batch = seq_lens.size(0)
    starts = torch.arange(batch, device=seq_lens.device, dtype=torch.int32) * seq_len
    bounds = torch.stack([starts, starts + seq_lens.to(torch.int32)], dim=1).flatten()
    end = torch.full((1,), batch * seq_len, device=seq_lens.device, dtype=torch.int32)
    cu_seq_lens = torch.cat([bounds, end])
    return dict(zip(VARLEN_KWARGS, (cu_seq_lens, cu_seq_lens, seq_len, seq_len)))


def _router_forward_keeping_topk_weights(self: nn.Module, hidden_states: torch.Tensor) -> tuple:
    """
    The forward pass of a transformers TopK router (e.g., Qwen3.5's) that weights its top_k experts by their
    probabilities divided by the sum of the top norm_k probabilities, rather than by the sum of the top_k ones (see
    keep_topk_weights). It returns what the router returns, which is every expert's logit (which transformers records
    for the router's auxiliary loss), the top_k weights, and their indices. The weights are computed in float32 and
    returned in the dtype of the router's input, as transformers' router returns them, since the experts multiply
    their outputs by them, and float32 weights would make that product, on every token's every expert, run in
    float32.
    """
    hidden_states = hidden_states.reshape(-1, self.hidden_dim)
    logits = F.linear(hidden_states, self.weight)
    probs = F.softmax(logits, dtype=torch.float, dim=-1)
    top_probs, top_indices = torch.topk(probs, max(self.top_k, self.norm_k), dim=-1)
    weights = top_probs[:, : self.top_k] / top_probs[:, : self.norm_k].sum(dim=-1, keepdim=True)
    return logits, weights.to(logits.dtype), top_indices[:, : self.top_k]


def keep_topk_weights(backbone: nn.Module, norm_k: int) -> int:
    """
    Makes every TopK router of a mixture-of-experts backbone give its active experts the weights they would have
    among the top norm_k experts, instead of renormalizing them to sum to 1. With fewer active experts than norm_k,
    the experts kept then contribute exactly as they do at norm_k, and the ones left out contribute nothing. With
    norm_k active experts, the routers behave exactly as transformers' do.

    Args:
        backbone: The backbone, whose routers are modules named *TopKRouter, with top_k, hidden_dim, and weight.
        norm_k: The number of top experts whose weights sum to 1, which is the number the model was trained with.

    Returns:
        The number of routers changed.
    """
    routers = [m for m in backbone.modules() if type(m).__name__.endswith("TopKRouter")]
    if not routers:
        raise ValueError(f"moe_topk_renormalize is false, but {type(backbone).__name__} has no TopK routers.")
    for router in routers:
        missing = [name for name in ("top_k", "hidden_dim", "weight") if not hasattr(router, name)]
        if missing:
            raise ValueError(
                f"{type(router).__name__} has no {', '.join(missing)}, so it cannot keep its top-k weights."
            )
        router.norm_k = norm_k
        router.forward = types.MethodType(_router_forward_keeping_topk_weights, router)
    return len(routers)


def _drop_varlen_kwargs(module: nn.Module, args: tuple, kwargs: dict) -> tuple:
    """
    A forward pre-hook for a recurrent layer (Qwen3.5's Gated DeltaNet) that removes the prompts' boundaries
    (see varlen_kwargs). Given boundaries, its kernels lay out their chunks on the CPU, which makes the CPU
    wait for the GPU. Without them, the layer reads each right-padded prompt left to right, so no real token
    reads a pad.
    """
    return args, {k: v for k, v in kwargs.items() if k not in VARLEN_KWARGS}


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


class LabelHead(nn.Module):
    """
    Scores the option shown at position k with row k, the LM head's row of label k's token (A, B, ..., see
    data/decision_prompt.py), applied to the hidden state at the answer marker:

        logit_k = <w_k, h_answer>             capped as cap * tanh(logit_k / cap) for a source LM that caps its logits

    hmf decision-init copies the rows from the source causal LM, so the untrained head gives the source LM's own
    next-token distribution over the labels, renormalized over the question's options. Frozen rows are a buffer,
    saved with the model, so that only the backbone trains, and trainable rows are a parameter. Like the pointer
    head, the label head computes in float32.

    Args:
        hidden_size: The hidden size of the backbone.
        num_labels: The number of rows, which is the most options a question can have.
        trainable: Whether the rows are trained. Defaults to False.
        softcap: The cap of the logits, or None. Defaults to None.
    """

    def __init__(self, hidden_size: int, num_labels: int, trainable: bool = False, softcap: Optional[float] = None):
        super().__init__()
        rows = torch.zeros(num_labels, hidden_size)
        if trainable:
            self.rows = nn.Parameter(rows)
        else:
            self.register_buffer("rows", rows)
        self.softcap = softcap
        # No dropout: the rows are pretrained, and an untrained head must equal the source LM
        self.dropout = nn.Identity()
        for module in self.modules():
            module._is_hf_initialized = True

    def reset_parameters(self) -> None:
        """The rows come from the source LM (hmf decision-init), so there is nothing to re-initialize here."""

    def copy_rows(self, rows: torch.Tensor) -> None:
        """Sets the rows, e.g., to the source LM's rows of the label tokens, of shape [num_labels, hidden_size]."""
        if rows.shape != self.rows.shape:
            raise ValueError(f"Expected label rows of shape {tuple(self.rows.shape)}, got {tuple(rows.shape)}.")
        with torch.no_grad():
            self.rows.copy_(rows.float())

    def forward(
        self,
        answer_states: torch.Tensor,
        option_states: torch.Tensor,
        option_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            answer_states: Hidden states at the answer position, of shape [batch, hidden_size].
            option_states: Unused, for the pointer head's signature.
            option_mask: Boolean mask of shape [batch, num_options], True for real options.

        Returns:
            logits: Option logits of shape [batch, num_options], in float32. Padded options hold MASK_VALUE.
        """
        num_options = option_mask.size(1)
        if num_options > self.rows.size(0):
            raise ValueError(f"The label head reads at most {self.rows.size(0)} options, got {num_options}.")
        with torch.autocast(device_type=answer_states.device.type, enabled=False):
            logits = answer_states.float() @ self.rows[:num_options].float().T
            if self.softcap:
                logits = self.softcap * torch.tanh(logits / self.softcap)
        return logits.masked_fill(~option_mask, MASK_VALUE)


class LabelPointerHead(nn.Module):
    """
    A frozen label head plus a pointer head whose scores start at 0 (its query projection is zero), so the
    untrained head is the label head, and the pointer learns from the options' own hidden states what the labels'
    rows get wrong:

        logit_k = <w_k, h_answer> + <q(h_answer), k(h_k)> / sqrt(dim)

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
        self.dropout = self.pointer.dropout

    def reset_parameters(self) -> None:
        """Initializes the pointer as PyTorch does, then zeroes its query projection, so that it scores 0."""
        self.pointer.reset_parameters()
        nn.init.zeros_(self.pointer.q.weight)
        nn.init.zeros_(self.pointer.q.bias)

    def copy_rows(self, rows: torch.Tensor) -> None:
        self.label.copy_rows(rows)

    def forward(
        self,
        answer_states: torch.Tensor,
        option_states: torch.Tensor,
        option_mask: torch.Tensor,
    ) -> torch.Tensor:
        logits = self.label(answer_states, option_states, option_mask) + self.pointer(
            answer_states, option_states, option_mask
        )
        return logits.masked_fill(~option_mask, MASK_VALUE)


def build_head(config: DecisionConfig) -> nn.Module:
    """The decision head of a config (decision_head_type)."""
    hidden_size = config.text_config.hidden_size
    if config.decision_head_type in ("label", "label_trained"):
        return LabelHead(
            hidden_size,
            config.decision_num_labels,
            trainable=config.decision_head_type == "label_trained",
            softcap=config.decision_label_softcap,
        )
    pointer_kwargs = {
        "dim": config.decision_head_dim,
        "use_norm": config.decision_head_norm,
        "dropout": config.decision_head_dropout,
    }
    if config.decision_head_type == "label_pointer":
        return LabelPointerHead(
            hidden_size, config.decision_num_labels, softcap=config.decision_label_softcap, **pointer_kwargs
        )
    return PointerHead(hidden_size=hidden_size, **pointer_kwargs)


class DecisionModel(PreTrainedModel):
    """
    A backbone loaded with AutoModel (no LM head) followed by a decision head (a PointerHead, LabelHead or
    LabelPointerHead, see decision_head_type). The backbone is any model supported by AutoModel, including our
    Hybrid models.

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
        self.decision_head = build_head(config)

        # Device placement (e.g. device_map="auto") should not split the backbone's layers
        self._no_split_modules = getattr(self.model, "_no_split_modules", None)
        self.pass_attention_mask = uses_padding_mask(self.model)
        if config.decision_attention == "bidirectional" and not self.pass_attention_mask:
            raise ValueError(
                f"decision_attention='bidirectional' needs a stock Hugging Face backbone, whose attention takes a "
                f"4D mask, and {type(self.model).__name__} is one of HMF's hybrid backbones."
            )
        # Set by the training workflow when sequence parallelism is enabled
        self.sequence_parallel_group = None
        if getattr(config.text_config, "moe_topk_renormalize", True) is False:
            keep_topk_weights(self.model, getattr(config.text_config, "moe_topk_norm_k", 8))

        self.post_init()

    def _bidirectional_backbone(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor],
        position_ids: Optional[torch.LongTensor],
        **kwargs,
    ) -> Any:
        """
        Runs the backbone with bidirectional attention over the prompt (decision_attention). Its
        attention layers get a 4D mask in which every real token attends to every real token and no
        token attends to padding. On a hybrid backbone with recurrent layers (layer type
        linear_attention, as Qwen3.5's Gated DeltaNet layers), only the full attention layers get
        it, and the recurrent layers, which are causal by construction, get the padding mask of the
        causal path.

        Args:
            input_ids: Token ids of shape [batch, seq_len], right-padded.
            attention_mask: Attention mask of shape [batch, seq_len], or None if nothing is padded.
            position_ids: Optional position ids of shape [batch, seq_len].
            kwargs: Passed to the backbone.

        Returns:
            The backbone's output.
        """
        from transformers.masking_utils import create_bidirectional_mask

        inputs_embeds = self.model.get_input_embeddings()(input_ids)
        if attention_mask is None:
            attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=input_ids.device)
        mask = create_bidirectional_mask(
            config=self.model.config,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            # A skipped (None) mask would let SDPA fall back to its causal mask, so the mask is always built
            allow_is_bidirectional_skip=False,
        )
        layer_types = getattr(self.model.config, "layer_types", None) or []
        if "linear_attention" in layer_types:
            from transformers.masking_utils import create_recurrent_attention_mask

            unknown = set(layer_types) - {"full_attention", "linear_attention"}
            if unknown:
                raise ValueError(f"decision_attention='bidirectional' does not support layer types {sorted(unknown)}.")
            mask = {
                "full_attention": mask,
                "linear_attention": create_recurrent_attention_mask(
                    config=self.model.config,
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    past_key_values=None,
                    position_ids=position_ids,
                ),
            }
        if getattr(self.model.config, "_attn_implementation", None) == "flash_attention_2":
            # Flash attention ignores the mask's shape and needs to be told not to be causal
            kwargs["is_causal"] = False
        return self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=mask,
            position_ids=position_ids,
            use_cache=False,
            return_dict=True,
            **kwargs,
        )

    def get_input_embeddings(self) -> nn.Module:
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.model.set_input_embeddings(value)

    @contextlib.contextmanager
    def _same_head_dropout_in_group(self, device: torch.device) -> Generator[None, None, None]:
        """
        Makes every rank of a sequence parallel group use the same dropout masks in the head, by
        seeding them with a number drawn on the group's first rank. Every rank must compute exactly
        the same loss, because the backward of ZigZagGatherScatter keeps only each rank's own part of
        the gradient and relies on the other ranks to supply theirs.

        Args:
            device: The device of the hidden states.
        """
        group = self.sequence_parallel_group
        if group is None or not self.training or not isinstance(self.decision_head.dropout, nn.Dropout):
            yield
            return
        seed = torch.randint(2**62, (1,), device=device)
        dist.broadcast(seed, src=dist.get_global_rank(group, 0), group=group)
        with torch.random.fork_rng(devices=[device] if device.type == "cuda" else []):
            torch.manual_seed(int(seed.item()))
            yield

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        opt_idx: torch.LongTensor,
        answer_idx: torch.LongTensor,
        position_ids: Optional[torch.LongTensor] = None,
        seq_lens: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> DecisionModelOutput:
        """
        Args:
            input_ids: Token ids of shape [batch, seq_len], right-padded. With sequence
                parallelism, every rank of the group is given the same whole prompts, and seq_len
                must be a multiple of 2 * sequence_parallel_size.
            attention_mask: Attention mask of shape [batch, seq_len]. Not passed to HMF's hybrid
                backbones (see uses_padding_mask), nor to any backbone with sequence parallelism,
                nor read when seq_lens is given.
            opt_idx: Position of the last token of each option's line, of shape
                [batch, num_options]. Rows with fewer options are padded with -1.
            answer_idx: Position of the answer marker, of shape [batch].
            position_ids: Optional position ids of shape [batch, seq_len].
            seq_lens: Optional length of each prompt without padding, of shape [batch], on the model's
                device. Given, the backbone runs without a mask and makes the CPU wait for nothing (see
                the module's docstring). It needs a stock backbone with flash attention, and no
                sequence parallelism.
            kwargs: Passed to the backbone.

        Returns:
            A DecisionModelOutput with the option logits and log-probabilities.
        """
        if seq_lens is not None:
            self._check_varlen()
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=None,
                position_ids=position_ids,
                use_cache=False,
                return_dict=True,
                **varlen_kwargs(seq_lens, input_ids.size(1)),
                **({"is_causal": False} if self.config.decision_attention == "bidirectional" else {}),
                **kwargs,
            )
            return self._head_outputs(outputs.last_hidden_state, opt_idx, answer_idx)

        if not self.pass_attention_mask:
            attention_mask = None

        group = self.sequence_parallel_group
        if group is not None:
            sp_size, sp_rank = dist.get_world_size(group), dist.get_rank(group)
            seq_len = input_ids.size(1)
            if seq_len % (2 * sp_size) != 0:
                raise ValueError(
                    f"With sequence parallelism, the padded prompt length ({seq_len}) must be a multiple "
                    f"of 2 * sequence_parallel_size ({2 * sp_size}); see decision_pad_to_multiple_of."
                )
            if position_ids is None:
                position_ids = torch.arange(seq_len, device=input_ids.device).expand(input_ids.size(0), -1)
            # Each rank runs the backbone on its zig-zag chunks, at their positions in the whole prompts. The
            # padding mask is left out, which is exact for right-padded prompts (see uses_padding_mask).
            input_ids = zigzag_local(input_ids, sp_rank, sp_size)
            position_ids = zigzag_local(position_ids, sp_rank, sp_size)
            attention_mask = None

        if self.config.decision_attention == "bidirectional":
            if group is not None:
                raise ValueError("decision_attention='bidirectional' does not support sequence parallelism.")
            outputs = self._bidirectional_backbone(input_ids, attention_mask, position_ids, **kwargs)
        else:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False,
                return_dict=True,
                **kwargs,
            )
        hidden_states = outputs.last_hidden_state  # [batch, seq_len, hidden_size]
        if group is not None:
            # Every rank gathers the hidden states of the whole prompts, which answer_idx and opt_idx index
            hidden_states = ZigZagGatherScatter.apply(hidden_states, group, sp_rank, sp_size, 2 * sp_size)
        return self._head_outputs(hidden_states, opt_idx, answer_idx)

    def _check_varlen(self) -> None:
        """
        Raises if the model cannot run from prompt lengths (seq_lens) instead of a mask, and the first time
        it can, keeps the prompts' boundaries away from its recurrent layers (see _drop_varlen_kwargs).
        """
        attn_implementation = getattr(self.model.config, "_attn_implementation", None) or ""
        if not self.pass_attention_mask or not attn_implementation.startswith("flash_attention"):
            raise ValueError(
                f"seq_lens needs a stock Hugging Face backbone with flash attention, and {type(self.model).__name__} "
                f"uses {attn_implementation or 'no attention implementation'}."
            )
        if self.sequence_parallel_group is not None:
            raise ValueError("seq_lens does not support sequence parallelism.")
        if not getattr(self, "_varlen_hooks_added", False):
            for layer in getattr(self.model, "layers", []):
                if getattr(layer, "linear_attn", None) is not None:
                    layer.linear_attn.register_forward_pre_hook(_drop_varlen_kwargs, with_kwargs=True)
            self._varlen_hooks_added = True

    def _head_outputs(
        self, hidden_states: torch.Tensor, opt_idx: torch.LongTensor, answer_idx: torch.LongTensor
    ) -> DecisionModelOutput:
        """
        The pointer head's outputs from the backbone's hidden states.

        Args:
            hidden_states: The backbone's last hidden states, of shape [batch, seq_len, hidden_size].
            opt_idx: Position of the last token of each option's line, of shape [batch, num_options],
                padded with -1.
            answer_idx: Position of the answer marker, of shape [batch].

        Returns:
            A DecisionModelOutput with the option logits and log-probabilities.
        """
        batch_idx = torch.arange(hidden_states.size(0), device=hidden_states.device)
        answer_states = hidden_states[batch_idx, answer_idx]  # [batch, hidden_size]

        # Padded options (-1) are clamped to position 0; their logits are masked in the head
        option_mask = opt_idx >= 0
        option_pos = opt_idx.clamp_min(0)
        option_states = hidden_states[batch_idx.unsqueeze(-1), option_pos]  # [batch, num_options, hidden_size]

        with self._same_head_dropout_in_group(hidden_states.device):
            logits = self.decision_head(answer_states, option_states, option_mask)
        log_probs = masked_log_softmax(logits, option_mask)

        return DecisionModelOutput(logits=logits, log_probs=log_probs)
