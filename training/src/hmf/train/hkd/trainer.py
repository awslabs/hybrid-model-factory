import json
import os
from types import MethodType
from typing import TYPE_CHECKING, Any, Dict, Optional, Union

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, SequentialSampler
from transformers import Seq2SeqTrainer
from transformers.masking_utils import create_causal_mask
from typing_extensions import override

from ...extras import logging
from ...extras.constants import IGNORE_INDEX
from ...extras.packages import is_transformers_version_greater_than
from ..callbacks import SaveProcessorCallback
from ..trainer_utils import (
    SaveShardMixin,
    SequenceParallelBatchSampler,
    create_custom_optimizer,
    create_custom_scheduler,
)

if TYPE_CHECKING:
    from torch.utils.data import Dataset
    from transformers import PreTrainedTokenizer, ProcessorMixin
    from transformers.trainer import PredictionOutput

    from ...hparams import FinetuningArguments


logger = logging.get_logger(__name__)


class CustomTrainer(SaveShardMixin, Seq2SeqTrainer):
    r"""
    This trainer is used to train Hybrid models with fused Hybrid + Attention modules--typically
    for Stage 1 of our hybridization procedure. Stage 1 is a layerwise distillation procedure which
    aims to align the outputs of the teacher's (transformer) layers to the student's (hybrid) layers.
    We consider a 'layer' to be a decoder layer, consisting of a sequence mixer followed by an MLP.
    Our Stage 1 procedure minimizes the MSE between the decoder layer outputs of the SSM and Attention
    streams. This trainer assumes that the model being trained is a 'fused' architecture, with parallel
    streams for the teacher and student paths (see hybridization/).
    """

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

        # Force model_accepts_loss_kwargs to False for custom MSE loss. This ensures that the
        # base HF transformers trainer normalizes the loss by gradient_accumulation_steps
        # consistently (in the training_step function), so we do not need to normalize by
        # gradient accumulations in the compute_loss function in this file.
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

        # Which MSE loss to apply per layer
        self.lw_distill_target = finetuning_args.hybrid_lw_distill_target

        # Whether to apply MSE loss to LM head inputs
        self.distill_pre_lm_head = finetuning_args.hybrid_distill_pre_lm_head

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
    def training_step(self, model, inputs, *args, **kwargs):
        # TODO: sequence_parallel modes other than 'zigzag-ring' may not need dummy forward
        if not self._has_dummy_forwarded and model.sequence_parallel_group is not None:
            model.train()
            self._has_dummy_forwarded = True
        return super().training_step(model, inputs, *args, **kwargs)

    @override
    def _get_train_sampler(
        self, *args, **kwargs
    ) -> Optional["torch.utils.data.Sampler"]:
        """
        For SP with batch_size > 1, we use a custom BatchSampler via get_train_dataloader.
        For SP with batch_size = 1, SequentialSampler works correctly with accelerator distribution.
        """
        if (
            self.model.sequence_parallel_group is not None
            or self.finetuning_args.disable_shuffling
        ):
            return SequentialSampler(self.train_dataset)
        return super()._get_train_sampler(*args, **kwargs)

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

    def compute_lw_mse_loss(
        self, model: nn.Module, inputs: Dict[str, Any], **kwargs
    ) -> torch.Tensor:
        """
        Computes the layerwise distillation loss of an SSM + Attention fused model. Namely,
        for all decoder layers in the model--which are assumed to have both a (parallel) SSM
        and Attention module--we compute the MSE between the layer's SSM decoder output and
        the Attention decoder output. All parameters except those specific to the SSM should be frozen
        (handled in workflow.py). 

        If self.distill_pre_lm_head is True, we will also compute a MSE between the SSM inputs to the LM head, and the
        Attention inputs.

        Arguments:
            model: The Hybrid model with fused Attention and SSM layers.
            inputs: The original input dict to the model.

        Returns:
            mse_loss: Layerwise MSE loss.
        """
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        inputs_embeds = model.model.embed_tokens(input_ids)

        # Use position_ids from inputs if available (handles SP correctly), otherwise compute local position IDs
        if "position_ids" in inputs:
            position_ids = inputs["position_ids"]
            cache_position = position_ids[0]
        else:
            past_seen_tokens = 0
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )
            position_ids = cache_position.unsqueeze(0)

        # Create causal mask mapping for different attention types
        mask_kwargs = {
            "config": model.model.config,
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "cache_position": cache_position,
            "past_key_values": None,
            "position_ids": position_ids,
        }
        causal_mask_mapping = {"full_attention": create_causal_mask(**mask_kwargs)}

        hidden_states = inputs_embeds

        # Create position embeddings to be shared across the decoder layers
        position_embeddings = model.model.rotary_emb(hidden_states, position_ids)

        mse_loss = 0

        attn_hidden_states = hidden_states

        # Use attention-style masking
        causal_mask = causal_mask_mapping["full_attention"]
        for decoder_layer in model.model.layers:
            if model.model.gradient_checkpointing and model.model.training:
                layer_outputs = model.model._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attn_hidden_states,
                    causal_mask,
                    position_ids,
                    None,  # past_key_values (we do not use fused layers for inference)
                    False,  # output_attentions
                    False,  # use_cache
                    cache_position,
                    position_embeddings,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states=hidden_states,
                    attention_hidden_states=attn_hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_values=None,
                    output_attentions=False,
                    use_cache=False,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                )

            (
                hidden_states,  # SSM decoder layer outputs
                attn_hidden_states,  # Attention decoder layer outputs
                hybrid_mixer_out,
                attn_mixer_out,
                hybrid_mlp_out,
                attn_mlp_out,
            ) = layer_outputs

            if self.lw_distill_target == "decoder":
                # MSE loss between entire attention (post softmax and MLP) and entire Hybrid layer (post MLP)
                mse_loss += F.mse_loss(hidden_states, attn_hidden_states)
            elif self.lw_distill_target == "mixer":
                # In this case, we want to match the outputs of the hybrid mixer (such as mamba or bmojo) to those of the attention mixer only
                # hybrid_mixer_out <-> attn_mixer_out
                mse_loss += F.mse_loss(hybrid_mixer_out, attn_mixer_out)
            elif self.lw_distill_target == "residuals":
                # In this case, we want to match the activations that get added to residuals:
                # hybrid_mixer_out <-> attn_mixer_out (the first residual)
                # hybrid_mlp_out <-> attn_mlp_out (the second residual)

                # First residual
                mse_loss += F.mse_loss(hybrid_mixer_out, attn_mixer_out)

                # Second residual
                mse_loss += F.mse_loss(hybrid_mlp_out, attn_mlp_out)

            elif self.lw_distill_target == "all":
                # Here, we consider all three losses above

                # Decoder outputs
                mse_loss += F.mse_loss(hidden_states, attn_hidden_states)

                # First residual
                mse_loss += F.mse_loss(hybrid_mixer_out, attn_mixer_out)

                # Second residual
                mse_loss += F.mse_loss(hybrid_mlp_out, attn_mlp_out)

        # Normalize by number of layers
        mse_loss = mse_loss / len(model.model.layers)

        if self.distill_pre_lm_head:
            # LM Head
            hidden_states = model.model.norm(
                hidden_states
            )  # SSM hidden states before LM head

            attn_hidden_states = model.model.norm(
                attn_hidden_states
            )  # Attention hidden states before LM head
            mse_loss += F.mse_loss(hidden_states, attn_hidden_states)

        return mse_loss

    @override
    def compute_loss(
        self,
        model: nn.Module,
        inputs: Dict[str, Any],
        return_outputs: Optional[bool] = False,
        **kwargs,
    ) -> torch.Tensor:
        r"""
        Computes the layerwise MSE loss averaged over all layers in the model. This function
        assumes the model is a 'fused' Hybrid model, having parallel streams of both SSM and Attention
        layers. The MSE loss is computed between the decoder layers of the Attention and SSM streams.
        """

        if (
            model.sequence_parallel_group is None
        ):  # No sequence parallel

            # Layerwise MSE Loss
            loss = self.compute_lw_mse_loss(model, inputs, **kwargs)

        else:
            # Compute local losses first
            per_layer_mse_loss = self.compute_lw_mse_loss(model, inputs, **kwargs)

            # Aggregate losses across sequence parallel groups
            sp_group = model.sequence_parallel_group
            world_size = dist.get_world_size(sp_group)

            dist.all_reduce(per_layer_mse_loss, op=dist.ReduceOp.SUM, group=sp_group)
            loss = per_layer_mse_loss / world_size

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
