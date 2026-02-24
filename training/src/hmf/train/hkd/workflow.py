# Copyright 2025 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/examples/pytorch/language-modeling/run_clm.py
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

import math
import warnings
from typing import TYPE_CHECKING, List, Optional

from ...data import (SFTDataCollatorWith4DAttentionMask, get_dataset,
                     get_template_and_fix_tokenizer)
from ...extras import logging
from ...extras.constants import IGNORE_INDEX
from ...extras.ploting import plot_loss
from ...model import load_model, load_tokenizer
from ..callbacks import PreserveTokenizerConfigCallback
from ..trainer_utils import create_modelcard_and_push, freeze_params_hybrid
from .trainer import CustomTrainer

logger = logging.get_logger(__name__)


if TYPE_CHECKING:
    from transformers import TrainerCallback

    from ...hparams import (DataArguments, FinetuningArguments, ModelArguments,
                            TrainingArguments)

def set_sequence_parallel_group_recursive(module, sequence_parallel_group):
    """Recursively set sequence_parallel_group on all modules"""
    module.sequence_parallel_group = sequence_parallel_group
    for child in module.children():
        set_sequence_parallel_group_recursive(child, sequence_parallel_group)

def run_hkd(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "TrainingArguments",
    finetuning_args: "FinetuningArguments",
    callbacks: Optional[List["TrainerCallback"]] = None,
):
    """
    Runs training for our hybrid layerwise distillation. The key difference between this
    workflow and others is in the use of a different Trainer. In this Trainer, we compute
    the layerwise distillation loss assuming we are using our fused Attention+Mamba2 models.
    """
    # Load tokenizer
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]


    # Load model
    model = load_model(
        tokenizer,
        model_args,
        finetuning_args,
        training_args.do_train,
    )

    # Recursively set the sequence_parallel_group attribute of all modules in model
    set_sequence_parallel_group_recursive(model, model.sequence_parallel_group)

    # Load data
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    dataset_module = get_dataset(
        template, model_args, data_args, training_args, stage="pt", **tokenizer_module
    )

    data_collator = SFTDataCollatorWith4DAttentionMask(
        template=template,
        model=model if not training_args.predict_with_generate else None,
        pad_to_multiple_of=8
        if training_args.do_train
        else None,  # for shift short attention
        label_pad_token_id=IGNORE_INDEX
        if data_args.ignore_pad_token_for_loss
        else tokenizer.pad_token_id,
        block_diag_attn=model_args.block_diag_attn,
        attn_implementation=getattr(model.config, "_attn_implementation", None),
        compute_dtype=model_args.compute_dtype,
        require_position_ids=model_args.sequence_parallel_size > 1,
        **tokenizer_module,
    )

    # Initialize Trainer
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        finetuning_args=finetuning_args,
        data_collator=data_collator,
        callbacks=callbacks,
        **dataset_module,
        **tokenizer_module,
    )

    # Add callback to preserve original tokenizer_config.json
    if model_args.preserve_tokenizer_config:
        trainer.add_callback(PreserveTokenizerConfigCallback(model_args.model_name_or_path))

    # Print model
    if trainer.is_world_process_zero():
        print("=" * 15 + " Model " + "=" * 15)
        print(model)

    # Selectively freeze parameters
    freeze_params_hybrid(model, trainer, finetuning_args)

    # Training
    if training_args.do_train:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", category=UserWarning, module="torch.utils.checkpoint"
            )
            train_result = trainer.train(
                resume_from_checkpoint=training_args.resume_from_checkpoint
            )
        trainer.save_model()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()
        if trainer.is_world_process_zero() and finetuning_args.plot_loss:
            plot_loss(training_args.output_dir, keys=["loss", "eval_loss"])

    # Create model card
    create_modelcard_and_push(
        trainer, model_args, data_args, training_args, finetuning_args
    )
