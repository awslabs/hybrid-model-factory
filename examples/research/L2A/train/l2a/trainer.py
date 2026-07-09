# Copyright 2024 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/trainer_seq2seq.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
import math
from types import MethodType
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.distributed as dist
from torch.nn import CrossEntropyLoss
from torch.utils.data import SequentialSampler
from transformers import Seq2SeqTrainer, TrainerCallback
from transformers.trainer import _is_peft_model
from typing_extensions import override

from hmf.extras import logging
from hmf.extras.constants import IGNORE_INDEX, TRAINER_LOG
from hmf.extras.packages import is_transformers_version_greater_than
from hmf.train.callbacks import PissaConvertCallback, SaveProcessorCallback
from hmf.train.trainer_utils import create_custom_optimizer, create_custom_scheduler

if TYPE_CHECKING:
    from torch.utils.data import Dataset
    from transformers import PreTrainedTokenizer, ProcessorMixin
    from transformers.trainer import PredictionOutput

    from hmf.hparams import FinetuningArguments

logger = logging.get_logger(__name__)


class L2ASparsityCallback(TrainerCallback):
    """Callback that appends sparsity metrics to trainer_log.jsonl."""

    def on_log(self, args, state, control, logs=None, model=None, **kwargs):
        if not args.should_save or not state.is_world_process_zero:
            return
        if model is None:
            return

        # Read sparsity directly from model
        unwrapped = model.model if hasattr(model, "model") else model
        if not hasattr(unwrapped, "sparsity_tracker"):
            return

        tracker = unwrapped.sparsity_tracker
        reg_loss = getattr(unwrapped, "reg_loss", 0.0)

        sparsity_entry = {
            "current_steps": state.global_step,
            "total_steps": state.max_steps,
            "loss": state.log_history[-1].get("loss") if state.log_history else None,
            "reg_loss": reg_loss,
            "lr": state.log_history[-1].get("learning_rate")
            if state.log_history
            else None,
            "epoch": state.log_history[-1].get("epoch") if state.log_history else None,
            "sparsity_layerwise": dict(tracker.running_sparsity),
            "sparsity_layerwise_avg": tracker.current_avg_sparsity,
        }
        sparsity_entry = {k: v for k, v in sparsity_entry.items() if v is not None}

        log_path = os.path.join(args.output_dir, TRAINER_LOG)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(sparsity_entry) + "\n")


class CustomTrainer(Seq2SeqTrainer):
    r"""Inherits Seq2SeqTrainer to compute generative metrics such as BLEU and ROUGE."""

    def __init__(
        self,
        finetuning_args: "FinetuningArguments",
        processor: Optional["ProcessorMixin"],
        gen_kwargs: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        if is_transformers_version_greater_than("4.46"):
            kwargs["processing_class"] = kwargs.pop("tokenizer")
        else:
            self.processing_class: PreTrainedTokenizer = kwargs.get("tokenizer")

        super().__init__(**kwargs)
        if processor is not None:
            self.model_accepts_loss_kwargs = False

        self.finetuning_args = finetuning_args

        self._has_dummy_forwarded = False
        if gen_kwargs is not None:
            # https://github.com/huggingface/transformers/blob/v4.45.0/src/transformers/trainer_seq2seq.py#L287
            self._gen_kwargs = gen_kwargs

        if processor is not None:
            self.add_callback(SaveProcessorCallback(processor))

        if finetuning_args.use_badam:
            from badam import BAdamCallback, clip_grad_norm_old_version  # type: ignore

            self.accelerator.clip_grad_norm_ = MethodType(
                clip_grad_norm_old_version, self.accelerator
            )
            self.add_callback(BAdamCallback)

    @override
    def create_optimizer(self) -> "torch.optim.Optimizer":
        if self.optimizer is None:
            self.optimizer = create_custom_optimizer(
                self.model, self.args, self.finetuning_args
            )
        return super().create_optimizer()

    @override
    def create_scheduler(
        self,
        num_training_steps: int,
        optimizer: Optional["torch.optim.Optimizer"] = None,
    ) -> "torch.optim.lr_scheduler.LRScheduler":
        create_custom_scheduler(self.args, num_training_steps, optimizer)
        return super().create_scheduler(num_training_steps, optimizer)

    @override
    def log(self, logs: Dict[str, float], *args, **kwargs) -> None:
        """Inject sparsity and reg_loss metrics into the log dict."""
        if "loss" in logs:
            model = self.model
            if hasattr(model, "model") and hasattr(model.model, "sparsity_tracker"):
                tracker = model.model.sparsity_tracker
                logs["sparsity_avg"] = tracker.current_avg_sparsity
                logs["sparsity_layerwise"] = dict(tracker.running_sparsity)
            if hasattr(model, "model") and hasattr(model.model, "reg_loss"):
                logs["reg_loss"] = model.model.reg_loss
        super().log(logs, *args, **kwargs)

    @override
    def training_step(self, model, inputs, *args, **kwargs):
        if not self._has_dummy_forwarded and model.sequence_parallel_group is not None:
            model.eval()
            with torch.no_grad():
                _ = model(**inputs)
            model.train()
            self._has_dummy_forwarded = True
        return super().training_step(model, inputs, *args, **kwargs)

    @override
    def _get_train_sampler(self, dataset=None):
        if self.model.sequence_parallel_group is not None:
            return SequentialSampler(
                dataset if dataset is not None else self.train_dataset
            )
        else:
            return super()._get_train_sampler(dataset)

    def _get_lambda_reg_scaled(self, current_step: int) -> float:
        """Compute the scaled lambda_reg based on the configured schedule.

        Supports linear ramp-up, cosine ramp-up, or constant scheduling.
        Only called after warmup has completed.
        """
        warmup_steps = self.args.warmup_steps
        total_steps = self.args.max_steps
        progress = min(
            1.0, (current_step - warmup_steps) / (total_steps - warmup_steps + 1e-9)
        )
        if self.args.lambda_reg_scheduler == "linear":
            return progress * float(self.args.lambda_reg)
        elif self.args.lambda_reg_scheduler == "cosine":
            cosine_factor = 0.5 * (1 - math.cos(math.pi * progress))
            return cosine_factor * float(self.args.lambda_reg)
        return float(self.args.lambda_reg)

    @override
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        r"""
        Fixes the loss value for transformers 4.46.0.
        https://github.com/huggingface/transformers/blob/v4.46.0/src/transformers/trainer.py#L3605
        """
        current_step = self.state.global_step
        warmup_steps = self.args.warmup_steps

        if (
            model.sequence_parallel_group is None
        ):  # no sequence parallel, compute as it is
            loss = super().compute_loss(model, inputs, return_outputs, **kwargs)
            # Get reg_loss for current batch, only apply regularization after warmup
            if (
                current_step > warmup_steps
                and hasattr(model.model, "reg_loss_intermediate")
                and model.training
                and model.model.reg_loss_intermediate
            ):
                lambda_reg_scaled = self._get_lambda_reg_scaled(current_step)
                reg_loss = (
                    model.model.reg_loss_intermediate
                ) * lambda_reg_scaled  # last layer has the correct total averaged reg_loss_intermediate we care about
            else:
                reg_loss = torch.tensor(
                    0.0, device=loss.device, dtype=loss.dtype, requires_grad=True
                )

            if is_transformers_version_greater_than("4.46") and not getattr(
                self, "model_accepts_loss_kwargs", False
            ):
                # other model should not scale the loss
                reg_loss = reg_loss / self.args.gradient_accumulation_steps

        else:
            # compute loss without shift labels, as we have already shifted labels in data processing when using sequence parallel
            _, outputs = super().compute_loss(
                model, inputs, return_outputs=True, **kwargs
            )
            # Flatten the tokens
            loss_fct = CrossEntropyLoss(reduction="sum")
            logits, labels = (
                outputs["logits"] if isinstance(outputs, dict) else outputs[1],
                inputs["labels"],
            )
            # Get vocab_size
            unwrapped_model = self.accelerator.unwrap_model(model)
            if _is_peft_model(unwrapped_model):
                vocab_size = unwrapped_model.base_model.model.config.vocab_size
            else:
                vocab_size = unwrapped_model.config.vocab_size
            logits = logits.view(-1, vocab_size)
            labels = labels.view(-1)
            # Enable model parallelism
            labels = labels.to(logits.device)
            loss = loss_fct(logits, labels)

            # weighted reduce within sequence_parallel_group
            sp_group = model.sequence_parallel_group
            dist.all_reduce(loss, op=dist.ReduceOp.SUM, group=sp_group)
            label_num = (labels != loss_fct.ignore_index).sum()
            dist.all_reduce(label_num, op=dist.ReduceOp.SUM, group=sp_group)
            loss /= label_num

            # Apply regularization after warmup
            if (
                current_step > warmup_steps
                and hasattr(model.model, "reg_loss_intermediate")
                and model.training
                and model.model.reg_loss_intermediate
            ):
                lambda_reg_scaled = self._get_lambda_reg_scaled(current_step)
                reg_loss = (
                    model.model.reg_loss_intermediate
                ) * lambda_reg_scaled
            else:
                reg_loss = torch.tensor(
                    0.0, device=loss.device, dtype=loss.dtype, requires_grad=True
                )

            # Sum across the sp_group to account for sequence parallelism
            dist.all_reduce(reg_loss, op=dist.ReduceOp.SUM, group=sp_group)

        if (
            is_transformers_version_greater_than("4.46")
            and model.sequence_parallel_group is not None
            and getattr(self, "model_accepts_loss_kwargs", False)
        ):
            # other model should not scale the loss
            if return_outputs:
                return (loss[0] / self.args.gradient_accumulation_steps, *loss[1:])
            else:
                return loss / self.args.gradient_accumulation_steps

        # Store sparsity/reg_loss for logging (picked up by self.log override)
        if isinstance(reg_loss, torch.Tensor):
            model.model.reg_loss = reg_loss.item()
        else:
            model.model.reg_loss = reg_loss

        if self.args.lambda_reg:
            return loss + reg_loss.squeeze()
        else:
            return loss

    @override
    def prediction_step(
        self,
        model: "torch.nn.Module",
        inputs: dict[str, Union["torch.Tensor", Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[list[str]] = None,
        **gen_kwargs,
    ) -> tuple[Optional[float], Optional["torch.Tensor"], Optional["torch.Tensor"]]:
        r"""Remove the prompt part in the generated tokens.

        Subclass and override to inject custom behavior.
        """
        if self.args.predict_with_generate:  # do not pass labels to model when generate
            labels = inputs.pop("labels", None)
        else:
            labels = inputs.get("labels")

        loss, generated_tokens, _ = super().prediction_step(
            model,
            inputs,
            prediction_loss_only=prediction_loss_only,
            ignore_keys=ignore_keys,
            **gen_kwargs,
        )
        if generated_tokens is not None and self.args.predict_with_generate:
            generated_tokens[
                :, : inputs["input_ids"].size(-1)
            ] = self.processing_class.pad_token_id
            generated_tokens = generated_tokens.contiguous()

        return loss, generated_tokens, labels

    def save_predictions(
        self,
        dataset: "Dataset",
        predict_results: "PredictionOutput",
        skip_special_tokens: bool = True,
    ) -> None:
        r"""Save model predictions to `output_dir`.

        A custom behavior that not contained in Seq2SeqTrainer.
        """
        if not self.is_world_process_zero():
            return

        output_prediction_file = os.path.join(
            self.args.output_dir, "generated_predictions.jsonl"
        )
        logger.info_rank0(f"Saving prediction results to {output_prediction_file}")

        labels = np.where(
            predict_results.label_ids != IGNORE_INDEX,
            predict_results.label_ids,
            self.processing_class.pad_token_id,
        )
        preds = np.where(
            predict_results.predictions != IGNORE_INDEX,
            predict_results.predictions,
            self.processing_class.pad_token_id,
        )

        for i in range(len(preds)):
            pad_len = np.nonzero(preds[i] != self.processing_class.pad_token_id)[0]
            if len(pad_len):  # move pad token to last
                preds[i] = np.concatenate(
                    (preds[i][pad_len[0] :], preds[i][: pad_len[0]]), axis=-1
                )

        decoded_inputs = self.processing_class.batch_decode(
            dataset["input_ids"], skip_special_tokens=False
        )
        decoded_preds = self.processing_class.batch_decode(
            preds, skip_special_tokens=skip_special_tokens
        )
        decoded_labels = self.processing_class.batch_decode(
            labels, skip_special_tokens=skip_special_tokens
        )

        with open(output_prediction_file, "w", encoding="utf-8") as f:
            for text, pred, label in zip(decoded_inputs, decoded_preds, decoded_labels):
                f.write(
                    json.dumps(
                        {"prompt": text, "predict": pred, "label": label},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
