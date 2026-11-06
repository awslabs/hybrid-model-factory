# Training Decision Models with MLX

This package trains decision models with [MLX](https://github.com/ml-explore/mlx) on Apple silicon, as an alternative to the PyTorch trainer that runs on GPUs. A training config selects it with `decision_backend: mlx`, and everything else about the run stays as it is with PyTorch: the same `hmf train` command, the same configs and tokenized data, the same label sets, and the same output folder, whose adapter or model loads in PyTorch for labeling, evaluation, and serving. On a Mac, those other stages run in PyTorch on MPS, Apple silicon's GPU backend, so this package only replaces training.

[examples/decision/README.md](../../../../../examples/decision/README.md#part-3-training-on-a-mac) explains how to set up a Mac and run the whole pipeline, and lists the times and memory measured on one.

## What It Supports

The backend trains decision models whose backbone is Qwen3 or Qwen3.5, the models that [mlx-lm](https://github.com/ml-explore/mlx-lm) implements, with LoRA (`finetuning_type: lora`) or full fine-tuning (`finetuning_type: full`, optionally with `hybrid_learnable_params`, e.g., `decision_head` to train the head alone). It trains on one device, in one process, with the backbone in bfloat16 (`bf16`), float16 (`fp16`), or float32, and the trainable parameters in float32.

HMF's own hybrid layers (GKA, B'MOJO, Mamba2, and the primed hybrid models) are not supported, since mlx-lm does not implement them, and they need CUDA in PyTorch too. The backend also refuses the options that only the PyTorch trainer has, with an error that names them: DeepSpeed, sequence parallelism, more than one process, quantization, other optimizers than AdamW, schedules other than linear, cosine, constant, and constant with warmup, `load_best_model_at_end`, and `decision_save_optimizer` and `decision_resume_optimizer` (an MLX run keeps its optimizer's state in its own checkpoints, for `resume_from_checkpoint`).

## How It Differs from the PyTorch Trainer

Both trainers read the same batches. `get_dataset` loads the tokenized data with its label set, `DecisionAugmentationSampler` orders the rows, and `DecisionDataCollator` pads them, so with the same seed, both train on the same rows in the same order. From there, the MLX trainer does what the Hugging Face Trainer does for `DecisionTrainer`, with the same loss, the same three optimizer groups (the head's matrices at `decision_head_lr` with weight decay, its other parameters at the same rate without it, and the rest at `learning_rate`), the same learning rate schedules and warmup, gradient clipping by the global norm, gradient accumulation, evaluation metrics, logs, checkpoints, and output files, and it differs from the PyTorch trainer only in how the model runs, in the ways below.

- **The backbone is mlx-lm's.** [backbone.py](backbone.py) loads a decision checkpoint's backbone into mlx-lm's Qwen3 or Qwen3.5 text model, whose weights have the same names, and [model.py](model.py) adds the pointer head, which computes in float32, as in PyTorch.
- **Qwen3.5's Gated DeltaNet layers train with a chunked rule.** mlx-lm runs these layers with a Metal kernel that has no backward pass, so in training it falls back to a loop over every token, which takes about 1.9 seconds for the forward and backward passes of one layer on a 1,339-token prompt. [gated_delta.py](gated_delta.py) computes the same recurrence chunk by chunk with MLX operations, which MLX differentiates, in about 60 milliseconds, and leaves mlx-lm's kernel in place for evaluation.
- **Prompts are not padded to a multiple of a length.** The GPU configs pad to a multiple of 256 tokens, so that the Triton kernels of Gated DeltaNet compile for only a few lengths. MLX gains nothing from that, and padding added 45% to the tokens of the v19 data, so the Mac configs set `decision_pad_to_multiple_of: null`.
- **Gradient checkpointing can apply to the long batches only, with the same dropout masks in a checkpointed layer's two forward passes.** `decision_checkpoint_min_tokens` checkpoints only the batches longer than its number of padded tokens, as in PyTorch. A checkpointed layer runs its forward pass again in the backward pass, so each layer call draws one random key, from which its LoRA dropout draws the same masks both times, which also makes the masks independent of whether a layer is checkpointed.
- **MLX's cache of freed buffers is capped at 2 GB.** By default, MLX keeps freed buffers for reuse until the cache reaches its memory limit, and since the batches' shapes change at every step, a 0.6B run's footprint grew to 22 GB while its arrays never exceeded 6 GB. With the cap (`CACHE_LIMIT_GB` in [train.py](train.py)), the same run used 6.9 GB.
- **Gradient accumulation averages the gradients of each step's batches,** including those of a shorter last step of an epoch, as the PyTorch trainer does.

## Files

| file | contents |
| --- | --- |
| [backbone.py](backbone.py) | loading a decision checkpoint's backbone into mlx-lm, gradient checkpointing per layer, and the dropout that checkpointed layers use |
| [gated_delta.py](gated_delta.py) | the chunked Gated DeltaNet rule, and its installation in mlx-lm's Qwen3.5 layers for training |
| [model.py](model.py) | the decision model in MLX, with its pointer head |
| [loss.py](loss.py) | the decision loss, with the anchor and teacher terms |
| [lora.py](lora.py) | LoRA adapters, and which parameters each finetuning type trains |
| [checkpoint.py](checkpoint.py) | saving PEFT adapters and full models in the formats PyTorch reads, and loading adapters saved by either |
| [train.py](train.py) | the training loop, its argument checks, and `run_decision_mlx`, which `run_decision` calls for `decision_backend: mlx` |

## Tests

The tests in [tests/decision/mlx_backend/](../../../../../tests/decision/mlx_backend/) run on a Mac with tiny Qwen3 and Qwen3.5 models, and skip where MLX is not installed. They check the MLX code against the PyTorch code it mirrors, so that the two backends give the same models.

- The chunked Gated DeltaNet rule matches mlx-lm's loop over tokens in its outputs, its final state, and the gradients of all its inputs, with masks, several lengths and chunk sizes, and more value heads than key heads, and matches mlx-lm's Metal kernel in a whole model.
- The decision model's answers match PyTorch's `DecisionModel` on the same checkpoint, and the loss, its parts, and the gradients of every LoRA and head parameter match a PEFT model's.
- An adapter saved by MLX loads in PyTorch with the same answers, an adapter saved by PyTorch loads in MLX, and a fully fine-tuned model saved by MLX loads in PyTorch.
- Training the same starting adapter on the same data for 9 steps, with gradient accumulation, warmup, a cosine schedule, weight decay, two learning rates, and clipping, gives the same adapter with both backends, within 1% of each weight's update over the run, and the same logged losses, gradient norms, and learning rates.
- Resuming from a checkpoint gives the same adapter as an uninterrupted run, gradient checkpointing does not change the gradients (with dropout), and the options the backend refuses raise errors.

On the real Qwen3.5-2B, the gradients of one batch in float32 match PyTorch's with a median relative difference of 1.4% per tensor. The difference comes from small differences between mlx-lm's and transformers' implementations of the model's forward pass (e.g., where the epsilon of Gated DeltaNet's query and key normalization goes), which the tiny test models do not show.
