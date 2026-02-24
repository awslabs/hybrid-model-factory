# Copyright 2025 the LlamaFactory team.
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

from types import MethodType
from typing import TYPE_CHECKING, Optional

import torch
import torch.distributed as dist
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, SequentialSampler
from transformers import Trainer
from typing_extensions import override

from ..callbacks import SaveProcessorCallback
from ...extras.packages import is_transformers_version_greater_than
from ..fp8_utils import (
    configure_fp8_environment,
    patch_accelerator_for_fp8,
    verify_fp8_status,
)
from ..trainer_utils import (
    SaveShardMixin,
    SequenceParallelBatchSampler,
    create_custom_optimizer,
    create_custom_scheduler,
)


if TYPE_CHECKING:
    from transformers import ProcessorMixin

    from ...hparams import FinetuningArguments, ModelArguments, TrainingArguments


class CustomTrainer(SaveShardMixin, Trainer):
    r"""Inherit Trainer for custom optimizer."""

    def __init__(
        self,
        finetuning_args: "FinetuningArguments",
        processor: Optional["ProcessorMixin"],
        model_args: Optional["ModelArguments"] = None,
        **kwargs,
    ) -> None:
        kwargs["processing_class"] = kwargs.pop("tokenizer")
        # Configure FP8 environment if enabled
        training_args: TrainingArguments = kwargs.get("args")
        if training_args.fp8:
            configure_fp8_environment(training_args)
            if getattr(training_args, "fp8_backend", "auto") == "te":
                patch_accelerator_for_fp8()

        super().__init__(**kwargs)
        if processor is not None:
            # avoid wrong loss under gradient accumulation
            # https://github.com/huggingface/transformers/pull/36044#issuecomment-2746657112
            self.model_accepts_loss_kwargs = False

        self.finetuning_args = finetuning_args
        self._has_dummy_forwarded = False

        if processor is not None:
            self.add_callback(SaveProcessorCallback(processor))

        if finetuning_args.use_badam:
            from badam import BAdamCallback, clip_grad_norm_old_version  # type: ignore

            self.accelerator.clip_grad_norm_ = MethodType(
                clip_grad_norm_old_version, self.accelerator
            )
            self.add_callback(BAdamCallback)

        if training_args.fp8 and hasattr(
            self, "accelerator"
        ):  # verify FP8 status after trainer initialization
            verify_fp8_status(self.accelerator, training_args)

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
    def get_train_dataloader(self) -> DataLoader:
        """
        Override to use SequenceParallelBatchSampler for SP training with batch_size > 1.
        """
        if (
            self.model.sequence_parallel_group is None
            or self.args.per_device_train_batch_size == 1
        ):
            return super().get_train_dataloader()

        # Use custom BatchSampler for SP with batch_size > 1
        sp_group = self.model.sequence_parallel_group
        sp_size = dist.get_world_size(sp_group)
        batch_size = self.args.per_device_train_batch_size

        batch_sampler = SequenceParallelBatchSampler(
            self.train_dataset, sp_size, batch_size
        )

        # Create DataLoader with batch_sampler (batch_size=1 since batch_sampler returns batches)
        dataloader = DataLoader(
            self.train_dataset,
            batch_sampler=batch_sampler,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

        return self.accelerator.prepare(dataloader)

    @override
    def _get_train_sampler(
        self, *args, **kwargs
    ) -> Optional["torch.utils.data.Sampler"]:
        # For SP with batch_size > 1, we use a custom BatchSampler via get_train_dataloader
        # For SP with batch_size = 1, SequentialSampler works correctly with accelerator distribution
        if (
            self.model.sequence_parallel_group is not None
            or self.finetuning_args.disable_shuffling
        ):
            return SequentialSampler(self.train_dataset)
        return super()._get_train_sampler(*args, **kwargs)

    @override
    def training_step(self, model, inputs, *args, **kwargs):
        # Sequence_parallel modes other than 'zigzag-ring' may not need dummy forward
        if not self._has_dummy_forwarded and model.sequence_parallel_group is not None:
            model.eval()
            with torch.no_grad():
                _ = model(**inputs)
            model.train()
            self._has_dummy_forwarded = True
        return super().training_step(model, inputs, *args, **kwargs)

    @override
    def compute_loss(self, model, inputs, *args, **kwargs):
        if model.sequence_parallel_group is None:
            loss = super().compute_loss(model, inputs, *args, **kwargs)
        else:
            # Compute loss without shifting labels since they get shifted during data preprocessing
            # when using sequence parallel (see src/llamafactory/data/processor/sequence_parallel.py).
            outputs = model(**inputs)
            logits, labels = (
                outputs["logits"] if isinstance(outputs, dict) else outputs[1],
                inputs["labels"],
            )
            vocab_size = logits.shape[-1]

            # Flatten and compute loss
            logits = logits.view(-1, vocab_size)
            labels = labels.view(-1).to(logits.device)
            loss_fct = CrossEntropyLoss(reduction="sum")
            loss = loss_fct(logits, labels)

            # Weighted reduce within sequence_parallel_group
            sp_group = model.sequence_parallel_group
            dist.all_reduce(loss, op=dist.ReduceOp.SUM, group=sp_group)
            label_num = (labels != loss_fct.ignore_index).sum()
            dist.all_reduce(label_num, op=dist.ReduceOp.SUM, group=sp_group)
            loss /= label_num

        if (
            is_transformers_version_greater_than("4.46")
            and model.sequence_parallel_group is not None
            and getattr(self, "model_accepts_loss_kwargs", False)
        ):
            return loss / self.args.gradient_accumulation_steps

        return loss
