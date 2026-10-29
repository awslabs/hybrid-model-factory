"""
Trainer for decision models (stage `decision`; see model/hybrid_zoo/models/decision/).

The loss (decision_loss) is the cross-entropy between each row's target distribution over its
options (label_dist, one-hot or ordinally smoothed for score rows) and the model's option
probabilities, averaged over rows with their weights. With a label set (decision_label_folder),
two weighted KL terms can be added, toward the anchor's distribution
(decision_anchor_kl_weight) and toward the teacher's (decision_teacher_kl_weight).

Training rows are tokenized decision_num_augmentations times; an augmentation of a row is one
augmented copy of it, with its options in one order and one phrasing of its question.
DecisionAugmentationSampler trains epoch e on augmentation e (modulo the number of augmentations),
so every epoch shows every row once, with fresh option orders.
"""

import random
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import pyarrow.compute as pc
import torch
from torch.utils.data import Sampler
from transformers import Trainer
from typing_extensions import override

from ...extras import logging
from ..trainer_utils import SaveShardMixin


if TYPE_CHECKING:
    from datasets import Dataset

    from ...hparams import FinetuningArguments


logger = logging.get_logger(__name__)


# Inputs of DecisionModel.forward; the other collated columns are targets
MODEL_INPUTS = ("input_ids", "attention_mask", "opt_idx", "answer_idx")


def param_shape(param: torch.nn.Parameter) -> Tuple[int, ...]:
    """A parameter's shape, also under DeepSpeed ZeRO-3, which leaves each rank an empty placeholder."""
    return tuple(getattr(param, "ds_shape", param.shape))


def param_numel(param: torch.nn.Parameter) -> int:
    """A parameter's number of elements, also under DeepSpeed ZeRO-3."""
    return int(getattr(param, "ds_numel", param.numel()))


def checkpointing_layers(model: torch.nn.Module) -> List[torch.nn.Module]:
    """The modules whose gradient checkpointing is on (those with gradient_checkpointing set to True)."""
    return [module for module in model.modules() if getattr(module, "gradient_checkpointing", None) is True]


def row_lengths(dataset: "Dataset") -> List[int]:
    """The number of tokens of each row of a tokenized dataset."""
    if dataset._indices is None:  # read the lengths from the Arrow table, without loading the tokens
        return pc.list_value_length(dataset.data.table.column("input_ids")).to_pylist()
    return [len(input_ids) for input_ids in dataset["input_ids"]]


def forward_kl(target_probs: torch.Tensor, log_probs: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
    """
    The mean forward KL divergence KL(target || model) over the selected rows:

        KL = sum_k p_k (log p_k - log q_k)

    summed over each row's options, where p is the target and q the model. Options where the target
    is exactly 0 carry no weight in this sum and are skipped, so they cannot give NaN; this covers
    padded options (target 0, model -inf) and targets rounded to 0.

    Args:
        target_probs: The target distributions, of shape [batch, num_options], 0 past each row's
            options.
        log_probs: The model's log-probabilities, of shape [batch, num_options], -inf past each
            row's options.
        rows: Which rows to average over, of shape [batch].

    Returns:
        The mean KL over the selected rows (a scalar).
    """
    target = target_probs[rows].to(log_probs.dtype)  # [rows, num_options]
    keep = target > 0
    log_target = target.clamp_min(1e-12).log().masked_fill(~keep, 0.0)
    log_model = log_probs[rows].masked_fill(~keep, 0.0)
    return (target * (log_target - log_model)).sum(dim=-1).mean()


def decision_loss(
    log_probs: torch.Tensor,
    inputs: Dict[str, Any],
    anchor_kl_weight: float = 0.0,
    teacher_kl_weight: float = 0.0,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    The loss of a batch of decision rows:

        loss = ce_loss + anchor_kl_weight * anchor_kl + teacher_kl_weight * teacher_kl

    - ce_loss: the cross-entropy against label_dist, averaged over the batch's rows with their
      weights (rows of weight 0 do not count).
    - anchor_kl: KL(anchor || model) (see forward_kl), averaged over the batch's rows that have an
      anchor (has_ref). Row weights do not apply.
    - teacher_kl: KL(teacher || model), averaged over the batch's rows that have a teacher
      (has_teacher). Row weights do not apply.

    A KL term is averaged over the rows of the batch that have its target, not over the whole
    batch, so a batch with a single teacher row gives that row the full teacher weight. A batch
    without any rows that have the target adds nothing for that term. In this way, the weight of a
    term does not depend on how many of the batch's rows have its target. The averages are taken on
    each device, in the same way as Strands Decider takes them on each micro-batch.

    Args:
        log_probs: The model's log-probabilities over each row's options, of shape
            [batch, num_options], -inf past each row's options.
        inputs: The collated batch: label_dist and weight, and with a label set ref_probs, has_ref,
            teacher_probs and has_teacher (see DecisionDataCollator).
        anchor_kl_weight: The weight of the anchor term. Defaults to 0 (no anchor term).
        teacher_kl_weight: The weight of the teacher term. Defaults to 0 (no teacher term).

    Returns:
        The loss, and its parts (ce_loss, and anchor_kl and teacher_kl when they are computed),
        unweighted and detached.
    """
    # Padded options are -inf in log_probs and 0 in label_dist; 0 * -inf would be NaN
    finite_log_probs = log_probs.masked_fill(torch.isinf(log_probs), 0.0)
    per_row = -(inputs["label_dist"].to(log_probs.dtype) * finite_log_probs).sum(dim=-1)
    weights = inputs["weight"].to(per_row.dtype)
    ce_loss = (per_row * weights).sum() / weights.sum().clamp_min(1e-8)

    loss, parts = ce_loss, {"ce_loss": ce_loss.detach()}
    for name, weight, probs_key, has_key in (
        ("anchor_kl", anchor_kl_weight, "ref_probs", "has_ref"),
        ("teacher_kl", teacher_kl_weight, "teacher_probs", "has_teacher"),
    ):
        if weight > 0 and has_key in inputs and bool(inputs[has_key].any()):
            kl = forward_kl(inputs[probs_key], log_probs, inputs[has_key])
            loss = loss + weight * kl
            parts[name] = kl.detach()
    return loss, parts


class DecisionAugmentationSampler(Sampler[int]):
    """
    Samples the rows of one augmentation per epoch, augmentation epoch % num_augmentations, shuffled
    with a seed that depends on the epoch. The Trainer calls set_epoch at the start of each epoch, so
    with max_steps the augmentations follow the epochs the Trainer counts, and a resumed run
    reproduces the order of the epoch it resumes.

    Without shuffle, the rows of each augmentation are served in dataset order.

    With lengths, rows of similar length are batched together, to reduce padding. The shuffled rows
    are taken in groups of sort_window batches' worth; each group is sorted by length and cut into
    batches, and all the batches are then shuffled. Sorting within groups, rather than over all
    rows, keeps batches random across training. The batch with the longest row goes first, so one
    that does not fit in memory fails at the first step. Each epoch still sees every row of its
    augmentation once.

    Args:
        augmentations: The augmentation of each row of the dataset.
        batch_size: The per-device batch size. Batches are formed per device, as accelerate
            assigns consecutive batches to the devices in turn.
        seed: Seed of the order.
        lengths: The length of each row, to group rows of similar length. Defaults to None (no
            grouping).
        sort_window: The number of batches whose rows are sorted by length together, when grouping
            by length. Defaults to 50.
        shuffle: Whether to shuffle the rows of each augmentation. Defaults to True.
    """

    def __init__(
        self,
        augmentations: List[int],
        batch_size: int,
        seed: int,
        lengths: Optional[List[int]] = None,
        sort_window: int = 50,
        shuffle: bool = True,
    ):
        self.rows: Dict[int, List[int]] = {}
        for index, augmentation in enumerate(augmentations):
            self.rows.setdefault(augmentation, []).append(index)
        self.num_augmentations = len(self.rows)
        if sorted(self.rows) != list(range(self.num_augmentations)):
            raise ValueError(f"Augmentations must be numbered from 0, got {sorted(self.rows)}.")

        self.batch_size = batch_size
        self.seed = seed
        self.lengths = lengths
        self.sort_window = sort_window
        self.shuffle = shuffle
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _epoch_rows(self) -> List[int]:
        return self.rows[self.epoch % self.num_augmentations]

    def _order(self) -> List[int]:
        rng = random.Random(self.seed * 1_000_003 + self.epoch)
        rows = list(self._epoch_rows())
        if not self.shuffle:
            return rows
        rng.shuffle(rows)
        if self.lengths is None or not rows:
            return rows

        group_size = self.batch_size * self.sort_window
        batches = []
        for start in range(0, len(rows), group_size):
            group = sorted(rows[start : start + group_size], key=lambda i: -self.lengths[i])
            batches += [group[i : i + self.batch_size] for i in range(0, len(group), self.batch_size)]

        # Only the last batch can be short; keep it last so every other batch stays aligned with
        # the devices' batch boundaries
        last = batches.pop() if len(batches[-1]) < self.batch_size else None
        longest = max(range(len(batches)), key=lambda b: self.lengths[batches[b][0]]) if batches else None
        first = [batches.pop(longest)] if longest is not None else []
        rng.shuffle(batches)
        order = first + batches + ([last] if last is not None else [])
        return [i for batch in order for i in batch]

    def __iter__(self):
        return iter(self._order())

    def __len__(self) -> int:
        return len(self._epoch_rows())


class DecisionTrainer(SaveShardMixin, Trainer):
    """
    Trainer for decision models.

    The optimizer has two learning rates, decision_head_lr for the pointer head, which needs a larger
    step when it starts from random weights, and learning_rate for the rest of the trainable parameters
    (e.g. LoRA adapters). weight_decay applies to the head's matrices only.

    With gradient checkpointing on and decision_checkpoint_min_tokens above 0, only batches of more
    padded tokens than decision_checkpoint_min_tokens are checkpointed; shorter ones run without
    recomputation. The result is the same either way; only speed and memory change. In distributed
    training every rank makes the same choice, from the largest of the ranks' batches, because under
    DeepSpeed ZeRO-3, ranks gather each layer's weights together, and a rank that recomputes a layer the others
    do not would wait for them forever.

    The loss is decision_loss. Besides the loss, the training logs show its parts, which are
    ce_loss, and anchor_kl and teacher_kl when their weights are above 0 (unweighted, averaged over the batches
    since the last log, on the main process). Evaluation rows have no label set, so eval_loss is
    the cross-entropy alone.

    Args:
        finetuning_args: The finetuning arguments.
        processor: Unused; decision prompts are text only.
    """

    def __init__(self, finetuning_args: "FinetuningArguments", processor=None, **kwargs) -> None:
        kwargs["processing_class"] = kwargs.pop("tokenizer")
        super().__init__(**kwargs)
        self.finetuning_args = finetuning_args
        # The loss is a mean over the micro-batch, so the Trainer divides it by the number of
        # gradient accumulation steps itself
        self.model_accepts_loss_kwargs = False
        # The sum and count of each part of the training loss since the last log
        self._loss_parts: Dict[str, Tuple[torch.Tensor, int]] = {}
        # The layers whose gradient checkpointing is turned on and off per batch (found at the first batch)
        self._checkpointing_layers: Optional[List[torch.nn.Module]] = None

    @override
    def create_optimizer(self) -> "torch.optim.Optimizer":
        if self.optimizer is not None:
            return self.optimizer

        head_lr = self.finetuning_args.decision_head_lr or self.args.learning_rate
        head_decay, head_no_decay, others = [], [], []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "decision_head" in name:
                (head_decay if len(param_shape(param)) > 1 else head_no_decay).append(param)
            else:
                others.append(param)

        groups = [
            {"params": head_decay, "lr": head_lr, "weight_decay": self.args.weight_decay, "name": "head"},
            {"params": head_no_decay, "lr": head_lr, "weight_decay": 0.0, "name": "head"},
            {"params": others, "lr": self.args.learning_rate, "weight_decay": 0.0, "name": "other"},
        ]
        groups = [group for group in groups if group["params"]]
        logger.info_rank0(
            f"Optimizer: {sum(map(param_numel, head_decay + head_no_decay)):,} head parameters at lr {head_lr} "
            f"({sum(map(param_numel, head_decay)):,} with weight decay {self.args.weight_decay}), "
            f"{sum(map(param_numel, others)):,} other parameters at lr {self.args.learning_rate}."
        )

        optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args, self.model)
        self.optimizer = optimizer_cls(groups, **optimizer_kwargs)
        return self.optimizer

    @override
    def _get_train_sampler(self, *args, **kwargs) -> Optional["torch.utils.data.Sampler"]:
        """
        The augmentation sampler, following train_sampling_strategy. "random" shuffles each
        augmentation, "group_by_length" also batches rows of similar length together, and
        "sequential" (or disable_shuffling) keeps each augmentation in dataset order.
        """
        dataset = self.train_dataset
        strategy = getattr(self.args, "train_sampling_strategy", "random")
        sequential = strategy == "sequential" or self.finetuning_args.disable_shuffling
        return DecisionAugmentationSampler(
            augmentations=dataset["augmentation"],
            batch_size=self.args.per_device_train_batch_size,
            seed=self.args.data_seed if self.args.data_seed is not None else self.args.seed,
            lengths=row_lengths(dataset) if strategy == "group_by_length" and not sequential else None,
            shuffle=not sequential,
        )

    @override
    def log(self, logs: Dict[str, float], *args, **kwargs) -> None:
        """Logs both learning rates, learning_rate for the backbone's trainable parameters (e.g.
        LoRA adapters), and head_learning_rate for the pointer head."""
        if "learning_rate" in logs and self.optimizer is not None:
            rates = {group.get("name"): group["lr"] for group in self.optimizer.param_groups}
            if "other" in rates:
                logs["learning_rate"] = rates["other"]
            if "head" in rates:
                logs["head_learning_rate"] = rates["head"]
        if "loss" in logs:  # a training log, so add the parts of the loss
            for name, (total, count) in self._loss_parts.items():
                if count:
                    logs[name] = float(total) / count
            self._loss_parts = {}
        super().log(logs, *args, **kwargs)

    def _set_checkpointing(self, model, inputs: Dict[str, Any]) -> None:
        """
        Turns gradient checkpointing on for long batches only (see decision_checkpoint_min_tokens),
        the same way on every rank, turning it on if the largest batch of all ranks is long.
        """
        min_tokens = self.finetuning_args.decision_checkpoint_min_tokens
        if not min_tokens or not model.training:
            return
        if self._checkpointing_layers is None:
            self._checkpointing_layers = checkpointing_layers(model)
        num_tokens = inputs["input_ids"].numel()  # padded tokens of the batch
        if torch.distributed.is_available() and torch.distributed.is_initialized() and self.args.world_size > 1:
            largest = torch.tensor(num_tokens, device=inputs["input_ids"].device)
            torch.distributed.all_reduce(largest, op=torch.distributed.ReduceOp.MAX)
            num_tokens = int(largest.item())
        on = num_tokens > min_tokens
        for layer in self._checkpointing_layers:
            layer.gradient_checkpointing = on

    def _loss(self, model, inputs: Dict[str, Any]) -> Tuple[torch.Tensor, Any]:
        """The loss (see decision_loss), and the model outputs."""
        self._set_checkpointing(model, inputs)
        outputs = model(**{name: inputs[name] for name in MODEL_INPUTS})
        loss, parts = decision_loss(
            outputs.log_probs,
            inputs,
            anchor_kl_weight=self.finetuning_args.decision_anchor_kl_weight,
            teacher_kl_weight=self.finetuning_args.decision_teacher_kl_weight,
        )
        if model.training:  # summed on the device, so logging does not wait for the GPU at every step
            for name, value in parts.items():
                total, count = self._loss_parts.get(name, (0.0, 0))
                self._loss_parts[name] = (total + value, count + 1)
        return loss, outputs

    @override
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        loss, outputs = self._loss(model, inputs)
        return (loss, outputs) if return_outputs else loss

    @override
    def prediction_step(
        self,
        model: torch.nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Returns, for each row, [prediction, label, kind_id, log p(label), p(prediction), sum of p^2]:
        a fixed width whatever the number of options, from which the metrics are computed.
        """
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            loss, outputs = self._loss(model, inputs)
        if prediction_loss_only:
            return loss.detach(), None, None

        log_probs = outputs.log_probs
        probs = log_probs.exp()  # padded options are -inf, so 0
        labels = inputs["label"]
        prediction = probs.argmax(dim=-1)
        stats = torch.stack(
            [
                prediction.float(),
                labels.float(),
                inputs["kind_id"].float(),
                log_probs.gather(1, labels.unsqueeze(-1)).squeeze(-1),
                probs.max(dim=-1).values,
                (probs**2).sum(dim=-1),
            ],
            dim=-1,
        )
        return loss.detach(), stats, labels
