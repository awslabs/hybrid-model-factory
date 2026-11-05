"""
Trains decision models with MLX on Apple silicon (decision_backend: mlx), from the same data, arguments, and label
sets as the PyTorch DecisionTrainer, and saves the same checkpoints, so that labeling, evaluation, and serving (which
run in PyTorch) use them unchanged.

The data path is the PyTorch one up to the batches. The datasets come from get_dataset with the label set joined, the
rows are ordered by DecisionAugmentationSampler (one augmentation per epoch), and DecisionDataCollator pads them, so
that the two backends see the same batches. The rest follows what the Hugging Face Trainer does for DecisionTrainer.

    - The optimizer is AdamW with PyTorch's bias correction, in DecisionTrainer's three groups, which are the head's
      matrices at decision_head_lr with weight_decay, the head's other parameters at the same rate without weight
      decay, and the other trainable parameters at learning_rate without weight decay.
    - The learning rate follows lr_scheduler_type (linear, cosine, constant, or constant_with_warmup) with the
      Trainer's warmup, and gradients are clipped to max_grad_norm by their global norm.
    - Each optimizer step averages the gradients of gradient_accumulation_steps batches of
      per_device_train_batch_size rows. The last step of an epoch averages over the batches that are left.
    - Training logs, evaluations, checkpoints, resuming, and the files of the output folder (trainer_state.json,
      train_results.json, eval_results.json, and all_results.json) follow the Trainer's.

Training runs in one process on one device. The backbone runs in bfloat16 with bf16, in float16 with fp16, and in
float32 otherwise, and the trainable parameters are kept in float32.
"""

import json
import math
import os
import re
import shutil
import time
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import mlx.core as mx
import mlx.optimizers as optim
import numpy as np
from mlx import nn
from mlx.utils import tree_flatten, tree_map, tree_unflatten

from ....extras import logging
from ...callbacks import PreserveTokenizerConfigCallback
from ..metric import ComputeDecisionMetrics
from ..trainer import DecisionAugmentationSampler, row_lengths
from .backbone import set_gradient_checkpointing
from .checkpoint import load_adapter, save_adapter, save_full_model
from .lora import attach_lora, lora_targets, merge_lora, set_trainable
from .loss import decision_loss
from .model import DecisionModelMLX, load_decision_model


if TYPE_CHECKING:
    from datasets import Dataset
    from transformers import PreTrainedTokenizer, Seq2SeqTrainingArguments

    from ....hparams import DataArguments, FinetuningArguments, ModelArguments


logger = logging.get_logger(__name__)

__all__ = ["SCHEDULERS", "check_mlx_arguments", "lr_factor", "run_decision_mlx"]

SCHEDULERS = ("linear", "cosine", "constant", "constant_with_warmup")
OPTIMIZERS = ("adamw_torch", "adamw_torch_fused", "adamw_hf")
# MLX keeps freed buffers in a cache for reuse, which by default may grow to the whole memory limit. The batches'
# shapes change from step to step, so without a smaller limit the cache keeps growing (to 22 GB for the 0.6B model,
# whose arrays never exceed 6 GB) and can exhaust the Mac's memory.
CACHE_LIMIT_GB = 2.0
TRAINER_STATE = "trainer_state.json"
TRAINABLE_WEIGHTS = "mlx_trainable.safetensors"
OPTIMIZER_STATE = "mlx_optimizer.safetensors"
CHECKPOINT_PREFIX = "checkpoint"
MODEL_INPUTS = ("input_ids", "opt_idx", "answer_idx")


def _value(option) -> str:
    """The value of an option that may be an enum (e.g. transformers' IntervalStrategy)."""
    return getattr(option, "value", option)


def check_mlx_arguments(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
) -> None:
    """Rejects the options the MLX backend does not support, and warns about the ones it ignores."""
    problems = []
    if training_args.deepspeed:
        problems.append("DeepSpeed")
    if model_args.sequence_parallel_size > 1:
        problems.append("sequence parallelism (sequence_parallel_size > 1)")
    if training_args.world_size > 1:
        problems.append("more than one process (the MLX backend trains on a single device)")
    if model_args.quantization_bit is not None:
        problems.append("quantization (quantization_bit)")
    if finetuning_args.finetuning_type not in ("lora", "full"):
        problems.append(f"finetuning_type {finetuning_args.finetuning_type} (use lora or full)")
    if finetuning_args.use_dora:
        problems.append("DoRA (use_dora)")
    if finetuning_args.pissa_init:
        problems.append("PiSSA (pissa_init)")
    if finetuning_args.finetuning_type == "full" and model_args.adapter_name_or_path:
        problems.append("adapter_name_or_path with full fine-tuning (merge the adapters into the model first)")
    if _value(training_args.optim) not in OPTIMIZERS:
        problems.append(f"optim {_value(training_args.optim)} (use adamw_torch)")
    if _value(training_args.lr_scheduler_type) not in SCHEDULERS:
        scheduler = _value(training_args.lr_scheduler_type)
        problems.append(f"lr_scheduler_type {scheduler} (use one of {', '.join(SCHEDULERS)})")
    if training_args.lr_scheduler_kwargs:
        problems.append("lr_scheduler_kwargs")
    if training_args.load_best_model_at_end:
        problems.append("load_best_model_at_end")
    if _value(training_args.save_strategy) == "best":
        problems.append("save_strategy best")
    if problems:
        raise ValueError(f"decision_backend mlx does not support {'; '.join(problems)}.")

    reporters = [name for name in (training_args.report_to or []) if name != "none"]
    if reporters:
        logger.warning_rank0(
            f"decision_backend mlx logs to the console and trainer_state.json only, not to {', '.join(reporters)}."
        )


def backbone_dtype(training_args: "Seq2SeqTrainingArguments", finetuning_args: "FinetuningArguments") -> str:
    """The dtype of the backbone, which is bfloat16 with bf16 (or pure_bf16), float16 with fp16, else float32."""
    if training_args.bf16 or finetuning_args.pure_bf16:
        return "bfloat16"
    if training_args.fp16:
        return "float16"
    return "float32"


def lr_factor(step: int, max_steps: int, warmup_steps: int, scheduler: str) -> float:
    """
    The factor of the learning rate at an optimizer step (0 for the first), as transformers' get_scheduler computes
    it for the linear, cosine (half a cycle), constant, and constant_with_warmup schedules.

    Args:
        step: The number of optimizer steps taken before this one.
        max_steps: The number of optimizer steps of the training.
        warmup_steps: The number of steps of linear warmup.
        scheduler: The schedule, one of SCHEDULERS.

    Returns:
        The factor by which the base learning rate is multiplied.
    """
    if scheduler == "constant":
        return 1.0
    if step < warmup_steps:
        return step / max(1, warmup_steps)
    if scheduler == "constant_with_warmup":
        return 1.0
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    if scheduler == "linear":
        return max(0.0, (max_steps - step) / max(1, max_steps - warmup_steps))
    if scheduler == "cosine":
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    raise ValueError(f"Unknown lr_scheduler_type {scheduler}. Use one of {', '.join(SCHEDULERS)}.")


def _interval(value: float, max_steps: int) -> int:
    """A number of steps given as the Trainer takes it, a count, or a fraction of the training below 1."""
    return max(1, math.ceil(value * max_steps)) if value < 1 else int(value)


def _peak_memory_gb() -> float:
    """The peak memory of MLX's arrays since the start, which leaves out its cache of freed buffers."""
    return mx.get_peak_memory() / 2**30


def _cache_memory_gb() -> float:
    """The memory held by MLX's cache of freed buffers (see CACHE_LIMIT_GB)."""
    return mx.get_cache_memory() / 2**30


def _batch_to_mlx(batch: Dict[str, Any]) -> Dict[str, mx.array]:
    """A collated batch of PyTorch tensors as MLX arrays."""
    return {name: mx.array(value.numpy()) for name, value in batch.items()}


class _Optimizer:
    """
    DecisionTrainer's three optimizer groups as AdamW optimizers with PyTorch's bias correction, in an MLX
    MultiOptimizer, whose learning rates are set at every step from the schedule.
    """

    def __init__(self, training_args: "Seq2SeqTrainingArguments", finetuning_args: "FinetuningArguments"):
        head_lr = finetuning_args.decision_head_lr or training_args.learning_rate
        betas = [training_args.adam_beta1, training_args.adam_beta2]

        def adamw(weight_decay: float) -> optim.AdamW:
            return optim.AdamW(
                learning_rate=head_lr, betas=betas, eps=training_args.adam_epsilon, weight_decay=weight_decay,
                bias_correction=True,
            )

        self.base_rates = [head_lr, head_lr, training_args.learning_rate]
        self.optimizers = [adamw(training_args.weight_decay), adamw(0.0), adamw(0.0)]
        self.optimizer = optim.MultiOptimizer(
            self.optimizers,
            filters=[
                lambda path, value: path.startswith("decision_head.") and value.ndim > 1,
                lambda path, value: path.startswith("decision_head."),
            ],
        )

    def set_rates(self, factor: float) -> Tuple[float, float]:
        """Sets the learning rates of the groups, and returns the backbone's and the head's."""
        for optimizer, rate in zip(self.optimizers, self.base_rates):
            optimizer.learning_rate = rate * factor
        return self.base_rates[2] * factor, self.base_rates[0] * factor

    @property
    def state(self):
        return self.optimizer.state

    @state.setter
    def state(self, state):
        self.optimizer.state = state


class MLXDecisionTrainer:
    """
    Trains a decision model in MLX (see the module docstring).

    Args:
        model: The decision model, with its trainable parameters set (see lora.set_trainable).
        tokenizer: The tokenizer, saved with every checkpoint.
        collator: The DecisionDataCollator that pads the batches.
        train_dataset: The tokenized training rows, with the label set joined if there is one.
        eval_dataset: The tokenized evaluation rows, or none.
        model_args: The model arguments.
        training_args: The training arguments.
        finetuning_args: The finetuning arguments.
        lora_settings: For LoRA, the adapters' targets, rank, alpha, dropout, and use_rslora, saved with them.
    """

    def __init__(
        self,
        model: DecisionModelMLX,
        tokenizer: "PreTrainedTokenizer",
        collator: Any,
        train_dataset: Optional["Dataset"],
        eval_dataset: Optional["Dataset"],
        model_args: "ModelArguments",
        training_args: "Seq2SeqTrainingArguments",
        finetuning_args: "FinetuningArguments",
        lora_settings: Optional[Dict[str, Any]] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.collator = collator
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.model_args = model_args
        self.args = training_args
        self.finetuning_args = finetuning_args
        self.lora_settings = lora_settings
        self.dtype = backbone_dtype(training_args, finetuning_args)
        self.checkpointing = not model_args.disable_gradient_checkpointing
        self.log_history: List[Dict[str, Any]] = []
        self.global_step = 0
        mx.set_cache_limit(int(CACHE_LIMIT_GB * 2**30))
        self.tokenizer_config = (
            PreserveTokenizerConfigCallback(model_args.model_name_or_path)
            if model_args.preserve_tokenizer_config
            else None
        )

    # The order of the training rows

    def _sampler(self) -> DecisionAugmentationSampler:
        strategy = _value(getattr(self.args, "train_sampling_strategy", "random"))
        sequential = strategy == "sequential" or self.finetuning_args.disable_shuffling
        return DecisionAugmentationSampler(
            augmentations=self.train_dataset["augmentation"],
            batch_size=self.args.per_device_train_batch_size,
            seed=self.args.data_seed if self.args.data_seed is not None else self.args.seed,
            lengths=row_lengths(self.train_dataset) if strategy == "group_by_length" and not sequential else None,
            shuffle=not sequential,
        )

    def _batches(self, sampler: DecisionAugmentationSampler, epoch: int) -> List[List[int]]:
        """The rows of each training batch of an epoch, as the Trainer's data loader cuts the sampler's order."""
        sampler.set_epoch(epoch)
        order = list(sampler)
        size = self.args.per_device_train_batch_size
        return [order[start : start + size] for start in range(0, len(order), size)]

    def _collate(self, dataset: "Dataset", rows: List[int]) -> Dict[str, mx.array]:
        return _batch_to_mlx(self.collator([dataset[int(row)] for row in rows]))

    def steps_per_epoch(self, num_batches: int) -> int:
        """The optimizer steps of an epoch of num_batches batches, as the Trainer counts them."""
        accumulation = self.args.gradient_accumulation_steps
        return max(num_batches // accumulation + int(num_batches % accumulation > 0), 1)

    # One optimizer step

    def _loss(self, model: DecisionModelMLX, batch: Dict[str, mx.array]):
        _, log_probs = model(*(batch[name] for name in MODEL_INPUTS))
        return decision_loss(
            log_probs,
            batch,
            anchor_kl_weight=self.finetuning_args.decision_anchor_kl_weight,
            teacher_kl_weight=self.finetuning_args.decision_teacher_kl_weight,
        )

    def _set_checkpointing(self, batch: Dict[str, mx.array]) -> None:
        """Checkpoints every batch, or with decision_checkpoint_min_tokens only the batches of more padded tokens."""
        min_tokens = self.finetuning_args.decision_checkpoint_min_tokens
        on = self.checkpointing and (not min_tokens or batch["input_ids"].size > min_tokens)
        set_gradient_checkpointing(self.model.model, on)

    def _step(self, batches: List[Dict[str, mx.array]]) -> Tuple[float, Dict[str, float], float]:
        """
        Takes one optimizer step on the gradients of the batches, averaged over them, after clipping their global norm
        to max_grad_norm.

        Returns:
            The mean loss of the batches, the mean of each part of the loss, and the gradients' norm before clipping.
        """
        value_and_grad = nn.value_and_grad(self.model, self._loss)
        grads, loss_sum, part_sums = None, 0.0, {}
        for batch in batches:
            self._set_checkpointing(batch)
            (loss, parts), batch_grads = value_and_grad(self.model, batch)
            batch_grads = tree_map(lambda g: g / len(batches), batch_grads)
            grads = batch_grads if grads is None else tree_map(mx.add, grads, batch_grads)
            mx.eval(grads, loss, parts)
            loss_sum += float(loss)
            for name, value in parts.items():
                part_sums[name] = part_sums.get(name, 0.0) + float(value)
        if self.args.max_grad_norm and self.args.max_grad_norm > 0:
            grads, grad_norm = optim.clip_grad_norm(grads, self.args.max_grad_norm)
        else:
            grad_norm = mx.sqrt(sum((g.astype(mx.float32) ** 2).sum() for _, g in tree_flatten(grads)))
        self.optimizer.optimizer.update(self.model, grads)
        mx.eval(self.model.parameters(), self.optimizer.state)
        parts = {name: total / len(batches) for name, total in part_sums.items()}
        return loss_sum / len(batches), parts, float(grad_norm)

    # Evaluation

    def evaluate(self, metric_key_prefix: str = "eval") -> Dict[str, float]:
        """
        Evaluates the model on the evaluation rows, with the same metrics as DecisionTrainer (ComputeDecisionMetrics)
        and the loss averaged over the rows.
        """
        start = time.time()
        self.model.eval()
        stats, loss_sum, num_rows = [], 0.0, 0
        size = self.args.per_device_eval_batch_size
        for begin in range(0, len(self.eval_dataset), size):
            batch = self._collate(self.eval_dataset, list(range(begin, min(begin + size, len(self.eval_dataset)))))
            _, log_probs = self.model(*(batch[name] for name in MODEL_INPUTS))
            # Evaluation rows have no label set, so the loss is the cross-entropy alone
            loss, _ = decision_loss(log_probs, batch)
            probs = mx.exp(log_probs)  # padded options are -inf, so 0
            labels = batch["label"]
            columns = [
                mx.argmax(probs, axis=-1).astype(mx.float32),
                labels.astype(mx.float32),
                batch["kind_id"].astype(mx.float32),
                mx.take_along_axis(log_probs, labels[:, None], axis=-1)[:, 0],
                probs.max(axis=-1),
                (probs**2).sum(axis=-1),
            ]
            stats.append(np.array(mx.stack(columns, axis=-1)))
            loss_sum += float(loss) * labels.shape[0]
            num_rows += labels.shape[0]
        self.model.train()

        metrics = ComputeDecisionMetrics()(SimpleNamespace(predictions=np.concatenate(stats)))
        metrics = {f"{metric_key_prefix}_{name}": value for name, value in metrics.items()}
        metrics[f"{metric_key_prefix}_loss"] = loss_sum / num_rows
        metrics[f"{metric_key_prefix}_runtime"] = round(time.time() - start, 4)
        metrics[f"{metric_key_prefix}_samples_per_second"] = round(num_rows / max(time.time() - start, 1e-9), 3)
        return metrics

    # Saving and resuming

    def save_model(self, output_dir: str) -> None:
        """Saves the adapter (LoRA) or the model (full fine-tuning), with the tokenizer."""
        if self.lora_settings is not None:
            save_adapter(self.model, output_dir, base_model_name_or_path=self.model_args.model_name_or_path,
                         **self.lora_settings)
        else:
            save_full_model(self.model, output_dir, self.model_args.model_name_or_path, self.model_config,
                            dtype=self.dtype)
        self.tokenizer.save_pretrained(output_dir)
        if self.tokenizer_config is not None:
            self.tokenizer_config._copy_tokenizer_config(output_dir)

    def _state(self, epoch: float) -> Dict[str, Any]:
        return {
            "global_step": self.global_step,
            "epoch": epoch,
            "max_steps": self.max_steps,
            "num_train_epochs": self.num_epochs,
            "logging_steps": self.logging_steps,
            "eval_steps": self.eval_steps,
            "save_steps": self.save_steps,
            "train_batch_size": self.args.per_device_train_batch_size,
            "log_history": self.log_history,
            "backend": "mlx",
        }

    def _save_state(self, output_dir: str, epoch: float) -> None:
        with open(os.path.join(output_dir, TRAINER_STATE), "w", encoding="utf-8") as f:
            json.dump(self._state(epoch), f, indent=2, sort_keys=True)

    def _save_checkpoint(self, epoch: float) -> None:
        """Saves checkpoint-<step>, with the training state unless save_only_model, and rotates old checkpoints."""
        output_dir = os.path.join(self.args.output_dir, f"{CHECKPOINT_PREFIX}-{self.global_step}")
        self.save_model(output_dir)
        if not self.args.save_only_model:
            mx.save_safetensors(
                os.path.join(output_dir, TRAINABLE_WEIGHTS), dict(tree_flatten(self.model.trainable_parameters()))
            )
            mx.save_safetensors(os.path.join(output_dir, OPTIMIZER_STATE), dict(tree_flatten(self.optimizer.state)))
        self._save_state(output_dir, epoch)
        logger.info_rank0(f"Saved checkpoint {output_dir}.")

        limit = self.args.save_total_limit
        if limit:
            checkpoints = sorted(
                (int(match.group(1)), name)
                for name in os.listdir(self.args.output_dir)
                if (match := re.fullmatch(rf"{CHECKPOINT_PREFIX}-(\d+)", name))
            )
            for _, name in checkpoints[: max(0, len(checkpoints) - limit)]:
                shutil.rmtree(os.path.join(self.args.output_dir, name))

    def _resume(self, checkpoint_dir: str) -> None:
        """Restores the trainable weights, the optimizer state, the step, and the training's logs from a checkpoint."""
        weights_path = os.path.join(checkpoint_dir, TRAINABLE_WEIGHTS)
        if not os.path.isfile(weights_path):
            raise ValueError(f"{checkpoint_dir} has no training state to resume from (saved with save_only_model?).")
        self.model.load_weights(list(mx.load(weights_path).items()), strict=False)
        self.optimizer.state = tree_unflatten(list(mx.load(os.path.join(checkpoint_dir, OPTIMIZER_STATE)).items()))
        with open(os.path.join(checkpoint_dir, TRAINER_STATE), encoding="utf-8") as f:
            state = json.load(f)
        self.global_step = state["global_step"]
        self.log_history = state["log_history"]
        mx.eval(self.model.parameters(), self.optimizer.state)
        logger.info_rank0(f"Resumed from {checkpoint_dir}, at step {self.global_step}.")

    def _last_checkpoint(self) -> Optional[str]:
        names = [name for name in os.listdir(self.args.output_dir) if re.fullmatch(rf"{CHECKPOINT_PREFIX}-\d+", name)]
        if not names:
            return None
        return os.path.join(self.args.output_dir, max(names, key=lambda name: int(name.rsplit("-", 1)[1])))

    # Training

    def train(self, resume_from_checkpoint: Optional[Any] = None) -> Dict[str, float]:
        """
        Trains the model, as the Trainer's train does for DecisionTrainer, and returns the training metrics.

        Args:
            resume_from_checkpoint: A checkpoint folder to resume from, or True for the output folder's last one.
                Defaults to none.
        """
        sampler = self._sampler()
        num_batches = math.ceil(len(sampler) / self.args.per_device_train_batch_size)
        steps_per_epoch = self.steps_per_epoch(num_batches)
        if self.args.max_steps > 0:
            self.max_steps = self.args.max_steps
            self.num_epochs = self.max_steps // steps_per_epoch + int(self.max_steps % steps_per_epoch > 0)
        else:
            self.max_steps = math.ceil(self.args.num_train_epochs * steps_per_epoch)
            self.num_epochs = math.ceil(self.args.num_train_epochs)
        warmup_steps = self.args.get_warmup_steps(self.max_steps)
        scheduler = _value(self.args.lr_scheduler_type)
        self.logging_steps = _interval(self.args.logging_steps, self.max_steps)
        eval_strategy, save_strategy = _value(self.args.eval_strategy), _value(self.args.save_strategy)
        self.eval_steps = _interval(self.args.eval_steps or self.args.logging_steps, self.max_steps)
        self.save_steps = _interval(self.args.save_steps, self.max_steps)
        self.optimizer = _Optimizer(self.args, self.finetuning_args)

        if resume_from_checkpoint is True:
            resume_from_checkpoint = self._last_checkpoint()
        if resume_from_checkpoint:
            self._resume(resume_from_checkpoint)
        accumulation = self.args.gradient_accumulation_steps
        logger.info_rank0(
            f"MLX training: {len(sampler):,} rows per epoch, {self.num_epochs} epoch(s), {self.max_steps} steps "
            f"of {accumulation} batch(es) of {self.args.per_device_train_batch_size} rows, backbone in {self.dtype}."
        )

        self.model.train()
        start, tokens, steps_done = time.time(), 0, 0
        loss_total, log_loss, log_parts, log_steps, log_start, log_tokens = 0.0, 0.0, {}, 0, time.time(), 0
        first_epoch, skip_steps = divmod(self.global_step, steps_per_epoch)
        epoch_float = 0.0
        for epoch in range(first_epoch, self.num_epochs):
            if self.global_step >= self.max_steps:
                break
            batches = self._batches(sampler, epoch)
            groups = [batches[i : i + accumulation] for i in range(0, len(batches), accumulation)]
            batches_done = 0
            for step_in_epoch, group in enumerate(groups):
                batches_done += len(group)
                if epoch == first_epoch and step_in_epoch < skip_steps:
                    continue  # done before the run was resumed
                if self.global_step >= self.max_steps:
                    break
                self.optimizer.set_rates(lr_factor(self.global_step, self.max_steps, warmup_steps, scheduler))
                batch_arrays = [self._collate(self.train_dataset, rows) for rows in group]
                loss, parts, grad_norm = self._step(batch_arrays)
                self.global_step += 1
                steps_done += 1
                group_tokens = sum(int(batch["input_ids"].size) for batch in batch_arrays)
                tokens += group_tokens
                log_tokens += group_tokens
                loss_total += loss
                log_loss += loss
                log_steps += 1
                for name, value in parts.items():
                    log_parts[name] = log_parts.get(name, 0.0) + value
                epoch_float = epoch + batches_done / len(batches)  # counted in batches, as the Trainer does
                # The Trainer logs the learning rates after its scheduler has stepped, which are the next step's
                lr, head_lr = (
                    rate * lr_factor(self.global_step, self.max_steps, warmup_steps, scheduler)
                    for rate in (self.optimizer.base_rates[2], self.optimizer.base_rates[0])
                )

                if self.global_step % self.logging_steps == 0 or (
                    self.args.logging_first_step and self.global_step == 1
                ):
                    elapsed = max(time.time() - log_start, 1e-9)
                    entry = {
                        "loss": round(log_loss / log_steps, 4),
                        **{name: total / log_steps for name, total in log_parts.items()},
                        "grad_norm": grad_norm,
                        "learning_rate": lr,
                        "head_learning_rate": head_lr,
                        "epoch": round(epoch_float, 4),
                        "step": self.global_step,
                        "padded_tokens_per_second": round(log_tokens / elapsed, 1),
                        "peak_memory_gb": round(_peak_memory_gb(), 3),
                        "cache_memory_gb": round(_cache_memory_gb(), 3),
                    }
                    self.log_history.append(entry)
                    logger.info_rank0(str(entry))
                    log_loss, log_parts, log_steps, log_start, log_tokens = 0.0, {}, 0, time.time(), 0

                end_of_epoch = step_in_epoch == len(groups) - 1
                if self.eval_dataset is not None and (
                    (eval_strategy == "steps" and self.global_step % self.eval_steps == 0)
                    or (eval_strategy == "epoch" and end_of_epoch)
                ):
                    metrics = self.evaluate()
                    self.log_history.append({**metrics, "epoch": round(epoch_float, 4), "step": self.global_step})
                    logger.info_rank0(str(metrics))
                if (save_strategy == "steps" and self.global_step % self.save_steps == 0) or (
                    save_strategy == "epoch" and end_of_epoch
                ):
                    self._save_checkpoint(epoch_float)

        runtime = max(time.time() - start, 1e-9)
        metrics = {
            "train_runtime": round(runtime, 4),
            "train_samples_per_second": round(
                steps_done * accumulation * self.args.per_device_train_batch_size / runtime, 3
            ),
            "train_steps_per_second": round(steps_done / runtime, 3),
            "train_padded_tokens_per_second": round(tokens / runtime, 1),
            "train_loss": loss_total / max(steps_done, 1),
            "train_peak_memory_gb": round(_peak_memory_gb(), 3),
            "epoch": round(epoch_float, 4),
        }
        self.log_history.append({**metrics, "step": self.global_step})
        self.final_epoch = epoch_float
        return metrics

    def save_final(self, output_dir: str) -> None:
        """Saves the model and the training state in the output folder, as the Trainer's save_model and save_state."""
        self.save_model(output_dir)
        self._save_state(output_dir, getattr(self, "final_epoch", 0.0))


def _save_metrics(output_dir: str, split: str, metrics: Dict[str, float]) -> None:
    """Writes <split>_results.json and merges the metrics into all_results.json, as the Trainer's save_metrics."""
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, f"{split}_results.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4, sort_keys=True)
    all_path = os.path.join(output_dir, "all_results.json")
    merged = {}
    if os.path.isfile(all_path):
        with open(all_path, encoding="utf-8") as f:
            merged = json.load(f)
    merged.update(metrics)
    with open(all_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=4, sort_keys=True)
    logger.info_rank0(f"***** {split} metrics *****\n" + "\n".join(f"  {k} = {v}" for k, v in sorted(metrics.items())))


def build_model(
    model_args: "ModelArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
) -> Tuple[DecisionModelMLX, Optional[Dict[str, Any]]]:
    """
    Loads the decision model in MLX and sets up what trains, as HMF does for PyTorch. With LoRA, the adapters of
    adapter_name_or_path are merged into the model, except the last, which is trained further, unless
    create_new_adapter is set (or the run does not train), in which case all are merged and a new adapter is made.

    Returns:
        The model, and for LoRA the adapters' settings (targets, rank, alpha, dropout, and use_rslora).
    """
    dtype = backbone_dtype(training_args, finetuning_args)
    model = load_decision_model(model_args.model_name_or_path, dtype)
    mx.random.seed(training_args.seed)  # the initialization of new adapters
    adapters = list(model_args.adapter_name_or_path or [])
    train = training_args.do_train
    resume = adapters[-1] if adapters and train and not finetuning_args.create_new_adapter else None
    for adapter in adapters[:-1] if resume else adapters:
        load_adapter(model, adapter)
        merge_lora(model)
        logger.info_rank0(f"Merged the adapter {adapter}.")
    if not train:
        return model, None
    if finetuning_args.finetuning_type == "full":
        set_trainable(model, "full", finetuning_args.hybrid_learnable_params)
        return model, None

    if resume:
        config = load_adapter(model, resume)
        settings = {
            "targets": config["target_modules"], "rank": config["r"], "alpha": config["lora_alpha"],
            "dropout": config.get("lora_dropout", 0.0), "use_rslora": config.get("use_rslora", False),
        }
        logger.info_rank0(f"Training the adapter {resume} further.")
    else:
        settings = {
            "targets": lora_targets(model, finetuning_args.lora_target), "rank": finetuning_args.lora_rank,
            "alpha": finetuning_args.lora_alpha, "dropout": finetuning_args.lora_dropout,
            "use_rslora": finetuning_args.use_rslora,
        }
        attach_lora(model, settings["targets"], settings["rank"], settings["alpha"], settings["dropout"],
                    settings["use_rslora"])
    trainable = set_trainable(model, "lora")
    logger.info_rank0(f"LoRA on {', '.join(settings['targets'])}: {len(trainable)} trainable tensors.")
    return model, settings


def run_decision_mlx(
    dataset_module: Dict[str, Any],
    tokenizer: "PreTrainedTokenizer",
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
) -> None:
    """
    Trains and evaluates a decision model with MLX, as run_decision does with PyTorch, from the datasets that
    run_decision has loaded.
    """
    from ....data import DecisionDataCollator
    from ....extras.ploting import plot_loss
    from ..workflow import _warn_if_augmentations_reused

    check_mlx_arguments(model_args, data_args, training_args, finetuning_args)
    model, lora_settings = build_model(model_args, training_args, finetuning_args)
    trainer = MLXDecisionTrainer(
        model=model,
        tokenizer=tokenizer,
        collator=DecisionDataCollator(
            tokenizer=tokenizer, pad_to_multiple_of=finetuning_args.decision_pad_to_multiple_of
        ),
        train_dataset=dataset_module.get("train_dataset"),
        eval_dataset=dataset_module.get("eval_dataset"),
        model_args=model_args,
        training_args=training_args,
        finetuning_args=finetuning_args,
        lora_settings=lora_settings,
    )
    with open(os.path.join(model_args.model_name_or_path, "config.json"), encoding="utf-8") as f:
        trainer.model_config = json.load(f)

    if training_args.do_train:
        _warn_if_augmentations_reused(SimpleNamespace(train_dataset=trainer.train_dataset), training_args, 1)
        metrics = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        trainer.save_final(training_args.output_dir)
        _save_metrics(training_args.output_dir, "train", metrics)
        if finetuning_args.plot_loss:
            plot_loss(training_args.output_dir, keys=["loss", "eval_loss", "eval_accuracy"])

    if training_args.do_eval:
        if trainer.eval_dataset is None:
            raise ValueError("do_eval needs evaluation rows (val_size or eval_dataset).")
        metrics = trainer.evaluate(metric_key_prefix="eval")
        _save_metrics(training_args.output_dir, "eval", metrics)
