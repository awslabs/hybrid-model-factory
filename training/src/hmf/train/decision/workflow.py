"""
Workflow for training decision models (stage `decision`).

The model is a decision model checkpoint (see `hmf decision-init`) and the data is decision data
tokenized with stage `decision` (see data/decision_prompt.py). What trains depends on
finetuning_type:

    - lora: LoRA adapters on the backbone, and the pointer head.
    - full: all parameters, or with hybrid_learnable_params only those whose names contain one of
      its entries, e.g. `decision_head` to train the head alone on a frozen backbone.
"""

from typing import TYPE_CHECKING, Optional

from ...data import DecisionDataCollator, get_dataset, get_template_and_fix_tokenizer
from ...extras import logging
from ...extras.ploting import plot_loss
from ...model import load_model, load_tokenizer
from ..callbacks import PreserveTokenizerConfigCallback
from ..trainer_utils import create_modelcard_and_push, freeze_params_hybrid
from .metric import ComputeDecisionMetrics
from .trainer import DecisionTrainer


if TYPE_CHECKING:
    from transformers import Seq2SeqTrainingArguments, TrainerCallback

    from ...hparams import DataArguments, FinetuningArguments, ModelArguments


logger = logging.get_logger(__name__)


def _check_arguments(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
) -> None:
    """Rejects options the decision workflow does not support."""
    if model_args.sequence_parallel_size > 1:
        raise ValueError("Sequence parallelism is not supported for decision models.")
    if data_args.packing:
        raise ValueError("Packing is not supported for decision models.")
    if training_args.do_predict or training_args.predict_with_generate:
        raise ValueError("Prediction is not supported for decision models; use do_eval.")
    if finetuning_args.hybrid_learnable_params is not None and finetuning_args.finetuning_type in ("lora", "oft"):
        raise ValueError(
            "hybrid_learnable_params cannot be combined with LoRA: LoRA already trains the adapters and the "
            "pointer head. To train only some parameters, use finetuning_type: full with hybrid_learnable_params."
        )

    custom_optimizers = {
        "use_galore": finetuning_args.use_galore,
        "use_apollo": finetuning_args.use_apollo,
        "use_badam": finetuning_args.use_badam,
        "use_adam_mini": finetuning_args.use_adam_mini,
        "use_muon": finetuning_args.use_muon,
        "loraplus_lr_ratio": finetuning_args.loraplus_lr_ratio is not None,
    }
    enabled = [name for name, on in custom_optimizers.items() if on]
    if enabled:
        raise ValueError(f"{', '.join(enabled)} is not supported for decision models.")

    if finetuning_args.decision_checkpoint_min_tokens < 0:
        raise ValueError("decision_checkpoint_min_tokens must be at least 0.")
    if finetuning_args.decision_checkpoint_min_tokens > 0 and model_args.disable_gradient_checkpointing:
        raise ValueError(
            "decision_checkpoint_min_tokens needs gradient checkpointing, but disable_gradient_checkpointing is "
            "true. Remove one of them."
        )

    for name in ("decision_anchor_kl_weight", "decision_teacher_kl_weight"):
        if getattr(finetuning_args, name) < 0:
            raise ValueError(f"{name} must be at least 0, got {getattr(finetuning_args, name)}.")
        if getattr(finetuning_args, name) > 0 and data_args.decision_label_folder is None:
            raise ValueError(
                f"{name} > 0 needs anchor or teacher targets: set decision_label_folder to a label set of the "
                "tokenized data (see data/README.md, Decision labels)."
            )


# The KL weight that uses each target of a label set
KL_WEIGHTS = {
    "has_ref": ("decision_anchor_kl_weight", "anchor"),
    "has_teacher": ("decision_teacher_kl_weight", "teacher"),
}


def _check_label_set(train_dataset, data_args: "DataArguments", finetuning_args: "FinetuningArguments") -> None:
    """
    Checks the label set against the KL weights, since a weight above 0 needs rows with that target. A
    label set's targets whose weight is 0 are not used, which is logged.
    """
    if data_args.decision_label_folder is None or train_dataset is None:
        return
    for has_key, (weight_name, target) in KL_WEIGHTS.items():
        num_rows = sum(train_dataset[has_key])
        weight = getattr(finetuning_args, weight_name)
        if weight > 0 and num_rows == 0:
            raise ValueError(
                f"{weight_name} is {weight}, but the label set {data_args.decision_label_folder} has no "
                f"{target} targets."
            )
        if weight > 0:
            logger.info_rank0(f"{target.capitalize()} term: weight {weight}, on {num_rows:,} training rows.")
        elif num_rows:
            logger.info_rank0(
                f"The label set has {target} targets on {num_rows:,} rows, but {weight_name} is 0: unused."
            )


def mark_zero3_leaf_modules(model) -> None:
    """
    Under DeepSpeed ZeRO-3, marks PEFT's ModulesToSaveWrapper (which holds the trained copy of the
    pointer head) as a ZeRO-3 leaf module, so that DeepSpeed hooks the wrapper as one unit and gathers
    the head's weights together when it runs.

    Without it, ZeRO-3 hooks the wrapper's two copies of the head and then the wrapper itself. The
    wrapper passes attribute lookups it cannot answer to its trained copy, so DeepSpeed takes the
    copy's backward helper (post_bwd_fn) for the wrapper's own, and the backward fails with
    "'PointerHead' object has no attribute 'ds_grads_remaining'". The head is small, so gathering it
    as one unit costs nothing.
    """
    from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled

    if not is_deepspeed_zero3_enabled():
        return
    from peft.utils import ModulesToSaveWrapper

    if not any(isinstance(module, ModulesToSaveWrapper) for module in model.modules()):
        return
    from deepspeed.utils import set_z3_leaf_modules

    set_z3_leaf_modules(model, [ModulesToSaveWrapper])
    logger.info_rank0("DeepSpeed ZeRO-3: the trained head (PEFT's ModulesToSaveWrapper) is gathered as one unit.")


def _warn_if_augmentations_reused(trainer: DecisionTrainer, training_args: "Seq2SeqTrainingArguments") -> None:
    """Warns if training makes more passes over the rows than there are augmentations of them."""
    augmentations = trainer.train_dataset["augmentation"]
    num_augmentations = len(set(augmentations))
    rows_per_augmentation = len(augmentations) / num_augmentations

    global_batch_size = training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps
    global_batch_size *= training_args.world_size
    if training_args.max_steps > 0:
        passes = training_args.max_steps * global_batch_size / rows_per_augmentation
    else:
        passes = training_args.num_train_epochs

    if passes > num_augmentations + 1e-6:
        logger.warning_rank0(
            f"Training makes {passes:.2f} passes over the rows, but the data has {num_augmentations} "
            f"augmentation(s) of each row, so augmentations will be reused with the same option orders. Tokenize "
            f"with decision_num_augmentations >= {int(-(-passes // 1))} for fresh option orders in every pass."
        )


def run_decision(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
    callbacks: Optional[list["TrainerCallback"]] = None,
):
    _check_arguments(model_args, data_args, training_args, finetuning_args)

    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    dataset_module = get_dataset(template, model_args, data_args, training_args, stage="decision", **tokenizer_module)
    _check_label_set(dataset_module.get("train_dataset"), data_args, finetuning_args)
    model = load_model(tokenizer, model_args, finetuning_args, training_args.do_train)
    mark_zero3_leaf_modules(model)  # before the Trainer starts DeepSpeed
    if getattr(model.config, "model_type", None) != "hmf_decision":
        raise ValueError(
            f"{model_args.model_name_or_path} is not a decision model (model_type "
            f"{getattr(model.config, 'model_type', None)!r}). Convert it with `hmf decision-init` first."
        )

    trainer = DecisionTrainer(
        model=model,
        args=training_args,
        finetuning_args=finetuning_args,
        data_collator=DecisionDataCollator(
            tokenizer=tokenizer, pad_to_multiple_of=finetuning_args.decision_pad_to_multiple_of
        ),
        callbacks=callbacks,
        compute_metrics=ComputeDecisionMetrics(),
        **dataset_module,
        **tokenizer_module,
    )

    # Add callback to preserve original tokenizer_config.json
    if model_args.preserve_tokenizer_config:
        trainer.add_callback(PreserveTokenizerConfigCallback(model_args.model_name_or_path))

    # Selectively freeze parameters (e.g. hybrid_learnable_params set to decision_head trains the head alone)
    freeze_params_hybrid(model, trainer, finetuning_args)

    # Training
    if training_args.do_train:
        _warn_if_augmentations_reused(trainer, training_args)
        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        trainer.save_model()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()
        if trainer.is_world_process_zero() and finetuning_args.plot_loss:
            plot_loss(training_args.output_dir, keys=["loss", "eval_loss", "eval_accuracy"])

    # Evaluation
    if training_args.do_eval:
        metrics = trainer.evaluate(metric_key_prefix="eval")
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Create model card
    create_modelcard_and_push(trainer, model_args, data_args, training_args, finetuning_args)
