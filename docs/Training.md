# Training

This guide covers the practical aspects of running training jobs using Hybrid Model Factory (HMF): data preprocessing, launching single/multi-node training, Sequence Parallelism, and DeepSpeed configuration. For an overview of the Priming pipeline (Stages 0-2) and distillation configuration, see [Priming.md](Priming.md).

> **Supported training regimes:** HMF has been validated with pretraining, SFT, and MSE distillation. Additional regimes inherited from LlamaFactory (e.g., DPO, PPO, KTO) are available but untested—we recommend verifying they work as expected for your use case. Training regime implementations can be found in [`training/src/hmf/train/`](../training/src/hmf/train/).

## Tokenization

For large datasets, we recommend pre-tokenizing your data before training. This speeds up training startup and allows tokenized data to be reused across multiple training runs.

> **Note:** We recommend running tokenization on a single node.

### Quick Start

Tokenization can be run using our `hmf preprocess` command:

```bash
cd training
hmf preprocess examples/tokenization/sft.yaml --shard_size 20000 --parallel_jobs 100
```

Then, use the tokenized data in your training config by specifying `tokenized_path`.

```yaml
tokenized_path: /path/to/save/tokenized/sft_data_32k
```

### How It Works

The preprocessing pipeline runs in 4 phases:

1. **Align** - Load and normalize each dataset in parallel
2. **Merge & Shard** - Combine datasets and split into shards
3. **Tokenize** - Tokenize each shard in parallel
4. **Concatenate** - Merge tokenized shards into final output

### Configuration

To tokenize a dataset, first create a YAML config—similar to a training config—specifying the tokenization parameters. See [training/examples/tokenization/](../training/examples/tokenization/) for complete examples.

> **Warning:** We use a custom tokenization pipeline different from the standard LlamaFactory tokenizer. Be careful if porting over configs directly from LlamaFactory projects as some argument defaults for packing and merging samples will be different.

Below are the key parameters to specify in the config:

| Parameter | Description |
|-----------|-------------|
| `model_name_or_path` | Model/tokenizer to use (e.g., `Qwen/Qwen3-8B`) |
| `stage` | `pt` for pretraining, `sft` for supervised fine-tuning |
| `dataset` | Dataset name(s) from `dataset_info.json` (comma-separated for multiple) |
| `tokenized_path` | Output path for tokenized data |
| `cutoff_len` | Tokenized sequence length |
| `sequence_parallel_size` | Set >1 for Sequence Parallel tokenization (required for long-context training with SP) |
| `template` | Chat template (e.g., `qwen3` for SFT, `qwen3_nothink_pt` for pretraining). Note: some templates modify tokenizer settings (e.g., EOS token). See [Post-Training Checkpoint Configuration](#post-training-checkpoint-configuration) for details. |
| `packing` | Pack multiple sequences into each sample (`true`/`false`) |
| `enable_thinking` | Whether to add think tags during tokenization |

### Datasets

The `dataset` parameter must reference datasets defined in [training/data/dataset_info.json](../training/data/dataset_info.json). You can use HuggingFace datasets or local files. Multiple datasets can be combined with comma separation as follows:

```yaml
dataset: dataset1,dataset2
```

### CLI Options

```bash
hmf preprocess <config.yaml> [--shard_size N] [--parallel_jobs J] [--no-cleanup]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--shard_size` | 75000 | Examples per shard. Larger = fewer shards, more memory per worker |
| `--parallel_jobs` | 128 | Number of parallel tokenization workers |
| `--no-cleanup` | false | Keep temporary directories after completion, helpful for debugging purposes (cleanup is enabled by default) |

### Examples

#### Pretraining Data (Short Context)

```bash
# hmf preprocess examples/tokenization/pretraining.yaml --shard_size 1000 --parallel_jobs 100
```

```yaml
stage: pt
dataset: pg19-sample
cutoff_len: 8192
sequence_parallel_size: 1
template: qwen3_pt
packing: true
```

#### Pretraining Data with Sequence Parallelism (128K)

```bash
# hmf preprocess examples/tokenization/pretraining_sp.yaml --shard_size 1000 --parallel_jobs 100
```

```yaml
stage: pt
dataset: pg19-sample
cutoff_len: 131072
sequence_parallel_size: 8
template: qwen3_pt
packing: true
```

#### SFT (Short Context)

```bash
# hmf preprocess examples/tokenization/sft.yaml --shard_size 20000 --parallel_jobs 100
```

```yaml
stage: sft
dataset: tulu-3-sft-mixture
cutoff_len: 32768
sequence_parallel_size: 1
template: qwen3
packing: true
enable_thinking: false
```

#### SFT with Sequence Parallelism (128K)

```bash
# hmf preprocess examples/tokenization/sft_sp.yaml --shard_size 5000 --parallel_jobs 100
```

```yaml
stage: sft
dataset: tulu-3-sft-mixture
cutoff_len: 131072
sequence_parallel_size: 4
template: qwen3
packing: true
enable_thinking: false
```

#### Reasoning (Short Context)

```bash
# hmf preprocess examples/tokenization/reasoning.yaml --shard_size 20000 --parallel_jobs 100
```

```yaml
stage: sft
dataset: Nemotron-Instruction-Following-Chat-v1-chat,Nemotron-Agentic-v1
cutoff_len: 32768
sequence_parallel_size: 1
template: qwen3
packing: true
mix_strategy: interleave_fast
max_merged_samples: 100000 # max_merged_samples * interleave_probs = samples_per_dataset for interleave_fast method, defaults to the size of the concatenated dataset
interleave_probs: 0.5,0.5
filter: repetition_ngram5_repeat10_window100_maxrun50
packing_method: random
```

#### Reasoning with Sequence Parallelism (128K)

```bash
# hmf preprocess examples/tokenization/reasoning_sp.yaml --shard_size 5000 --parallel_jobs 100
```

```yaml
stage: sft
dataset: Nemotron-Instruction-Following-Chat-v1-chat,Nemotron-Agentic-v1
cutoff_len: 131072
sequence_parallel_size: 8
template: qwen3
packing: true
mix_strategy: interleave_fast
max_merged_samples: 100000 # max_merged_samples * interleave_probs = samples_per_dataset for interleave_fast method, defaults to the size of the concatenated dataset
interleave_probs: 0.5,0.5
filter: repetition_ngram5_repeat10_window100_maxrun50
packing_method: random
```

---

## Running Training

Training is launched using the `hmf train` command for single-node or `hmf train-multinode` for multi-node setups. The same command is used for all training stages:

| Stage | What it does | Example config |
|-------|-------------|----------------|
| Stage 1 (Distillation) | Trains SSM layers to match Attention via MSE loss using the fused architecture | [examples/priming/stage1/](../training/examples/priming/stage1/) |
| Stage 2 (Fine-tuning) | Fine-tunes the unfused Hybrid model (long-context, SFT, reasoning, etc.) | [examples/priming/stage2/](../training/examples/priming/stage2/) |

For details on the Priming pipeline, distillation objectives, and Stage 1/2 configuration, see [Priming.md](Priming.md).

### Single Node

```bash
cd training
hmf train examples/priming/stage2/hqwen3_8b_gka_sft_it.yaml
```

This automatically detects available GPUs and launches distributed training with `torchrun`.

### Multi-Node

Run on each node:

```bash
cd training
hmf train-multinode <config.yaml> <master_addr> <nnodes> <node_rank>
```

| Argument | Description |
|----------|-------------|
| `config.yaml` | Path to training config |
| `master_addr` | IP address of the master node (node 0) |
| `nnodes` | Total number of nodes |
| `node_rank` | Rank of this node (0 to nnodes-1) |

Example with 4 nodes:

```bash
# Node 0 (master)
hmf train-multinode examples/priming/stage2/hqwen3_8b_gka_long_ctx.yaml 10.0.0.1 4 0

# Node 1
hmf train-multinode examples/priming/stage2/hqwen3_8b_gka_long_ctx.yaml 10.0.0.1 4 1

# Node 2
hmf train-multinode examples/priming/stage2/hqwen3_8b_gka_long_ctx.yaml 10.0.0.1 4 2

# Node 3
hmf train-multinode examples/priming/stage2/hqwen3_8b_gka_long_ctx.yaml 10.0.0.1 4 3
```

---

## Post-Training Checkpoint Configuration

Training templates can modify your model's tokenizer and chat template in ways that are useful during tokenization/training but may be undesirable in the final checkpoint. This section covers how to ensure your saved checkpoints are configured correctly for inference.

### Preserving the Original Tokenizer Config

Some templates change tokenizer settings at training time. For example, `qwen3_nothink_pt` sets `replace_eos=True`, which swaps the EOS token from `<|im_end|>` to `<|endoftext|>`. This is needed for correct tokenization during pretraining-style training (e.g., Stage 1 distillation), but the trainer writes the modified tokenizer to saved checkpoints, which can change the model's behavior at inference time.

To handle this, set `preserve_tokenizer_config` in your training config:

```yaml
template: qwen3_nothink_pt
preserve_tokenizer_config: True
```

When enabled, the original `tokenizer_config.json` from `model_name_or_path` is copied into each checkpoint directory (and the final output), overwriting the modified version. The modified config is kept as `tokenizer_config.LlamaFactory.json` for reference.

`preserve_tokenizer_config` defaults to `True`. Set it to `False` if you intentionally want the template's tokenizer modifications to persist in your saved checkpoints.

### Disabling Thinking in the Chat Template (SFT-IT Models)

When training via SFT on instruction-tuning data without think tags (e.g., Qwen3 models fine-tuned on Tulu-sft-mixture), the saved checkpoint retains the base model's chat template, which supports thinking by default. At inference time, this means the model may attempt to "think" unless the caller explicitly passes `enable_thinking=False`.

To force thinking off unconditionally, modify the chat template in the saved checkpoint's `tokenizer_config.json`. The change is in the `add_generation_prompt` block at the end of the Jinja template.

The original template conditionally injects an empty think block:

```jinja
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n' }}
    {%- if enable_thinking is defined and enable_thinking is false %}
        {{- '<think>\n\n</think>\n\n' }}
    {%- endif %}
{%- endif %}
```

To force thinking off, remove the conditional and always inject the empty block:

```jinja
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n' }}
        {{- '<think>\n\n</think>\n\n' }}
{%- endif %}
```

This ensures that an empty `<think></think>` block is appended to the model's generation prompt by default, forcing the model to operate in non-thinking mode regardless of what the caller passes for `enable_thinking` in the tokenizer arguments (e.g., see the [Qwen3 Model Card](https://huggingface.co/Qwen/Qwen3-8B#switching-between-thinking-and-non-thinking-mode)).

---

## Sequence Parallelism

Sequence Parallelism (SP) splits long sequences across multiple GPUs, enabling training on contexts that wouldn't fit on a single GPU. Use SP when memory limitations require it—typically when training on very long sequences (64K+) or large models.

### Configuration

Set `sequence_parallel_size` in your training config:

```yaml
sequence_parallel_size: 8  # Split each sequence across 8 GPUs
cutoff_len: 131072         # 128K context
```

### Important: Tokenization Must Match

When using SP, your tokenized data must be prepared with the same `sequence_parallel_size`:

```yaml
# tokenization config
sequence_parallel_size: 8
cutoff_len: 131072
tokenized_path: /path/to/data_128k_sp8
```

```yaml
# training config
sequence_parallel_size: 8
cutoff_len: 131072
tokenized_path: /path/to/data_128k_sp8
```

See [training/examples/priming/stage2/hqwen3_8b_gka_long_ctx.yaml](../training/examples/priming/stage2/hqwen3_8b_gka_long_ctx.yaml) for a complete long-context training config with SP.

---

## DeepSpeed

Training uses DeepSpeed for distributed optimization. We recommend starting with ZeRO-0 and increasing if you run into memory issues.

| Stage | What it shards | Recommendation |
|-------|----------------|----------------|
| ZeRO-0 | Nothing | Default starting point |
| ZeRO-1 | Optimizer states | Try if ZeRO-0 OOMs |
| ZeRO-2 | Optimizer states + gradients | Good balance of memory and speed |
| ZeRO-3 | Optimizer states + gradients + model params | Large models and/or long contexts where ZeRO-2 is insufficient |

Set the DeepSpeed config in your training YAML:

```yaml
deepspeed: examples/deepspeed/ds_z0_config.json  # or ds_z1, ds_z2, ds_z3
```

See [training/examples/deepspeed/](../training/examples/deepspeed/) for config files.
