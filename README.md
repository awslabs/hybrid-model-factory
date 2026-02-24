# Hybrid Model Factory

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-3776AB.svg?logo=python&logoColor=white)](https://python.org)
[![arXiv Gated KalmaNet ](https://img.shields.io/badge/arXiv-GKA-b31b1b.svg)](https://arxiv.org/abs/2511.21016)
[![arXiv B'MOJO](https://img.shields.io/badge/arXiv-B'MOJO-b31b1b.svg)](https://arxiv.org/abs/2407.06324)
[![arXiv PICASO](https://img.shields.io/badge/arXiv-PICASO-b31b1b.svg)](https://arxiv.org/abs/2502.17605)
[![Models on HuggingFace](https://img.shields.io/badge/🤗_Model_Zoo-HuggingFace-yellow.svg)](https://huggingface.co/collections/amazon/primed-hybrid-models-collection)
[![Contributions Welcome](https://img.shields.io/badge/Contributions-Welcome-brightgreen.svg)](CONTRIBUTING.md)

An open-source toolkit for training, priming, and serving next-generation **Hybrid architectures** — models that combine Attention with State Space Models (SSMs).

Hybrid architectures include both Transformers and vanilla SSMs as special cases and modulate how each layer manages memory — some layers can retain precise recent context (Attention), while others maintain a compressed, fading summary of the unbounded past (SSMs). This makes Hybrids particularly well-suited for agentic workflows and long-running AI agents where context accumulates over long horizons but full-fidelity recall of every token at every model layer is prohibitively expensive and unnecessary.

Building Next Generation Hybrid models, however, spans the full AI software stack—from custom kernels and distributed long-context training to inference integration with frameworks like vLLM. Hybrid Model Factory wraps all of this into one toolkit:

- **Priming**: Bootstrap training of Hybrid architectures from a pre-trained Transformer's weights to preserve most of the base Transformer's knowledge without training from scratch.
- **Training**: Distill, fine-tune Hybrid models, and extend their context length (128K+) with built-in Sequence Parallelism.
- **Inference**: Deploy Hybrid Primed models through an optimized vLLM plugin.

---

**Contents**
- [Features](#features)
- [Primed Model Zoo](#primed-model-zoo)
- [Getting Started](#getting-started)
- [Tested Configurations](#tested-configurations)
- [Priming Pipeline Overview](#priming-pipeline-overview)
- [Supported Architectures](#supported-architectures)
- [Performance](#performance)
- [Research](#our-research)
- [Contributing](#contributing)
- [Citation](#citation)
- [Acknowledgments](#acknowledgments)
- [License](#license)

---

## Features

Hybrid Model Factory extends [LlamaFactory](https://github.com/hiyouga/LlamaFactory) with support for long context training and inference deployment of Hybrid models.

| Category | Details |
|----------|---------|
| Hybrid layer types | Gated KalmaNet (GKA), Gated DeltaNet (GDN), Mamba2, B'MOJO-F |
| Hybrid models | Qwen3Next, Qwen3.5, Primed Qwen3-GKA, Primed Qwen3-BMOJOF, Primed Ministral3-GDN, ... |
| Priming pipeline | Initialize (Stage 0) → Distill (Stage 1) → Fine-tune (Stage 2), with Qwen3, Qwen3-MoE, Qwen2.5, Llama3, Ministral3 as base Transformers |
| Sequence Parallelism | Universal SP implementation for any SSM layer (GKA, GDN, Mamba2, ...) to train on 128K+ contexts across multiple GPUs. |
| Inference | Optimized vLLM plugin for inference with continuous batching and tensor parallelism for Hybrid and Primed models (Qwen2.5/3/3-MoE,3-Next,3.5) |
| Context extension | Training-free 2-4× context extension (experimental HF feature) [Hybrid cache composition](https://arxiv.org/abs/2502.17605) |

---

## Primed Model Zoo

We release a model zoo of small-mid size Primed models built on top of Qwen3 to enable the community to experiment with Hybrid architectures. All our models are Primed from Qwen3 with a 50% Hybrid ratio and support up to 128K context. See [Hybrid Layer Types](#hybrid-layer-types) for details on each architecture.

To cover different modes of use we released two types of models:
- 💬 **Instruct**: Long-context instruction-tuned models
- 🧠 **Reasoning**: Long-context models with enhanced reasoning capabilities

### 8B Models (Primed from Qwen3-8B)

| Hybrid Type | Instruct 💬 | Reasoning 🧠 |
|-------------|-------------|--------------|
| B'MOJO-F| [🤗 BMOJOF-primed-HQwen3-8B-Instruct](https://huggingface.co/amazon/BMOJOF-primed-HQwen3-8B-Instruct) | — |
| GKA | [🤗 GKA-primed-HQwen3-8B-Instruct](https://huggingface.co/amazon/GKA-primed-HQwen3-8B-Instruct) | [🤗 GKA-primed-HQwen3-8B-Reasoner](https://huggingface.co/amazon/GKA-primed-HQwen3-8B-Reasoner) |
| GDN | [🤗 GDN-primed-HQwen3-8B-Instruct](https://huggingface.co/amazon/GDN-primed-HQwen3-8B-Instruct) | [🤗 GDN-primed-HQwen3-8B-Reasoner](https://huggingface.co/amazon/GDN-primed-HQwen3-8B-Reasoner) |
| Mamba2 | [🤗 Mamba2-primed-HQwen3-8B-Instruct](https://huggingface.co/amazon/Mamba2-primed-HQwen3-8B-Instruct) | — |

### 32B Models (Primed from Qwen3-32B)

| Hybrid Type | Instruct 💬 | Reasoning 🧠 |
|-------------|-------------|--------------|
| GKA | [🤗 GKA-primed-HQwen3-32B-Instruct](https://huggingface.co/amazon/GKA-primed-HQwen3-32B-Instruct) | [🤗 GKA-primed-HQwen3-32B-Reasoner](https://huggingface.co/amazon/GKA-primed-HQwen3-32B-Reasoner) |
| GDN | [🤗 GDN-primed-HQwen3-32B-Instruct](https://huggingface.co/amazon/GDN-primed-HQwen3-32B-Instruct) | - |

> **Note:** To keep the model zoo manageable, we only release additional variants beyond 8B Instruct for GKA and GDN. GKA for its benchmark performance and its ability to trade inference FLOPs for accuracy at test time by adjusting the iterations of its internal Chebyshev solver, and GDN for its community adoption.

---

## Getting Started

### Inference

Production deployment is handled by a [vLLM plugin](https://docs.vllm.ai/en/v0.15.1/design/plugin_system/) that registers all our custom Hybrid architectures automatically — no vLLM fork required.

**1. Build the Docker image:**

```bash
cd vllm-inference
docker build -t vllm-hybrid .
```

**2. Serve a model:**

```bash
docker run --rm --runtime nvidia --gpus all \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -p 8000:8000 \
  --ipc=host \
  vllm-hybrid \
  --model amazon/GKA-primed-HQwen3-8B-Reasoner \
  --enable-prefix-caching \
  --mamba-cache-mode align \
  --mamba-cache-dtype float32 \
  --mamba-ssm-cache-dtype float32 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --reasoning-parser qwen3
```

> For Instruct models, drop `--enable-auto-tool-choice`, `--tool-call-parser`, and `--reasoning-parser`.

**3. Query the server:**

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "amazon/GKA-primed-HQwen3-8B-Reasoner",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "What is linear attention in the context of LLMs?"}
    ]
  }'
```

See [docs/Inference.md](docs/Inference.md) for full serving configuration (including pip install without Docker), flags, and [examples/vllm-inference/](examples/vllm-inference/) for one-click launch scripts for A100 and H200 GPUs. For development and debugging, models can also be loaded directly with HuggingFace Transformers — see [docs/Inference.md](docs/Inference.md#huggingface-transformers-inference).

### Training

**1. Set up the environment (Docker):**

```bash
cd training/docker
docker build -t hmf-training -f Dockerfile .
docker run --gpus all -it --rm --network host --ipc=host hmf-training
```

**2. Install and run** (inside the container):

```bash
git clone https://github.com/awslabs/hybrid-model-factory.git
cd hybrid-model-factory/training
pip install -e .
hmf train examples/priming/stage2/hqwen3_8b_gka_sft_it.yaml
```

See [docs/Training.md](docs/Training.md) for detailed training documentation.

> For large datasets, we recommend pre-tokenizing your data before training. This speeds up training startup and allows data to be reused across multiple runs. See [docs/Training.md](docs/Training.md) for full tokenization options.

#### End-to-End Pipeline Scripts

For a complete end-to-end pipeline (Stage 0 → Stage 1 → Stage 2) that runs with a single script, see the hardware-specific examples:

- [`examples/training/8xA100/`](examples/training/8xA100/) — Qwen3-4B on 8× A100 (40GB)
- [`examples/training/8xH200/`](examples/training/8xH200/) — Qwen3-8B on 8× H200 (140GB)

Each directory contains a `setup_docker.sh` to build and launch the container, and a `run_pipeline.sh` that runs the full priming pipeline inside it. The pipeline configs are in [`training/examples/priming/full_pipeline/`](training/examples/priming/full_pipeline/).

All our code (including custom kernels and vLLM implementations) has been tested on A100 and H200 GPUs, with Sequence Parallel (SP) tested up to SP group size 8 for long-context training. Refer to [Tested Configurations](#tested-configurations) for a comprehensive list of hardware requirements for different model sizes and training stages.

---

## Priming Pipeline Overview

If you want to train a Hybrid model but have a limited compute budget, Priming lets you jumpstart training from pre-trained model weights (for example, an off-the-shelf Transformer) by replacing its Attention layers with more efficient SSM layers.

![Pipeline Overview](assets/figures/overview.png)


### Stage 0: Initialize a Hybrid architecture using the pre-trained Transformer's weights

Initializes a **fused Hybrid architecture** from a pre-trained Transformer's weights. The original Attention path (frozen teacher) and the new SSM path (trainable student) run in parallel, with shared parameters (embeddings, MLPs, layer norms) to reduce memory overhead.

![Fused Decoder Layer](assets/figures/fused.png)

```bash
cd training
hmf prime-init examples/priming/stage0/qwen3_8b_gka.yaml
```

The `hybrid_override_pattern` in the config specifies which layers become SSM layers:

| Pattern | Layer type |
|---------|------------|
| `*` | Standard Attention |
| `*DA` | Dual-path Attention (fused, for distillation) |
| `GKA` / `GKA*` | Gated KalmaNet (standard / fused) |
| `GDN` / `GDN*` | Gated DeltaNet (standard / fused) |
| `M2` / `M2*` | Mamba2 (standard / fused) |
| `BMF` / `BMF*` | B'MOJO-F (standard / fused) |
| `SWA` / `SWA*` | Sliding Window Attention (standard / fused) |

See [training/examples/priming/stage0/](training/examples/priming/stage0/) for configs for all layer types and [docs/Priming.md](docs/Priming.md) for the full configuration reference.

### Stage 1: Attention → SSM Distillation

Trains the SSM layers to match the Attention layers' behavior using MSE distillation. Attention weights are frozen; only SSM parameters are trained.

```bash
cd training
hmf train examples/priming/stage1/hqwen3_8b_gka.yaml
```

After distillation, convert the fused model to a standard Hybrid:

```bash
cd training
hmf prime-unfuse ./models/HQwen3-8B-GKA-Fused/Stage1/checkpoint-<STEPS>
```

### Stage 2: Fine-tuning

Fine-tune the Hybrid model for your target use case. Two common paths:

**Long-context continued pretraining** (extends context to 128K+ using Sequence Parallelism):

```bash
cd training

# Single node
hmf train examples/priming/stage2/hqwen3_8b_gka_long_ctx.yaml

# Multi-node (run on each node)
hmf train-multinode examples/priming/stage2/hqwen3_8b_gka_long_ctx.yaml <master_addr> <nnodes> <node_rank>
```

See [training/examples/priming/stage2/](training/examples/priming/stage2/) for all configs, [docs/Training.md](docs/Training.md) for multi-node training and DeepSpeed configuration, and [docs/SequenceParallel.md](docs/SequenceParallel.md) for details on the Sequence Parallelism implementation.


**Supervised fine-tuning (SFT):**

```bash
cd training

# Single node
hmf train examples/priming/stage2/hqwen3_8b_gka_sft_it.yaml

# Multi-node (run on each node)
hmf train-multinode examples/priming/stage2/hqwen3_8b_gka_sft_it.yaml <master_addr> <nnodes> <node_rank>
```

See [training/examples/priming/stage2/](training/examples/priming/stage2/) for all configs and [docs/Training.md](docs/Training.md) for multi-node training, Sequence Parallelism, and DeepSpeed configuration.

### Training-Free Context Length Extension

Hybrid models also support training-free context extension up to 2-4× their native size via experimental Hybrid state composition (e.g. [PICASO](https://arxiv.org/abs/2502.17605)). See [docs/StateComposition.md](docs/StateComposition.md) for usage.

---

## Supported Architectures

### Base Models

| Model Family | Supported Versions |
|--------------|--------------------|
| Qwen | Qwen3, Qwen3-MoE, Qwen2.5, Qwen3Next, Qwen3.5 |
| Llama | Llama3, Llama3.1 |
| Mistral | Ministral3 |

### Hybrid Layer Types

**Gated KalmaNet (GKA, pronounced "gee-ka")**: More expressive SSM than both Mamba2 and GDN. Inspired by Kalman filtering, it computes the state at each time-step based on the entire past rather than the instantaneous update rules used by other SSMs. This translates to strong performance on long-context and reasoning tasks. Uniquely among SSM layers, GKA supports variable compute at inference via its num_iter parameter — trade off latency vs. quality per deployment without retraining. [[Paper](https://arxiv.org/abs/2511.21016)]

**Gated DeltaNet (GDN)**: SSM with diagonal + low-rank transition dynamics. Extends Mamba2 with the Delta Update rule, improving expressiveness through gated state transitions while retaining Mamba2's efficiency. [[Paper](https://arxiv.org/abs/2412.06464)]

**Mamba2**: SSM with diagonal transition dynamics and input-dependent gating. Less expressive than GDN and GKA due to its diagonal structure, but well-supported by optimized kernels. [[Paper](https://arxiv.org/abs/2405.21060)]

**B'MOJO-F**: Hybrid layer that couples Attention (sliding window for local context) and SSMs (for global context) within the same layer. [[Paper](https://arxiv.org/abs/2407.06324)]

**Sliding Window Attention (SWA)**: Standard attention limited to a fixed window (e.g., 512 tokens). Useful as a baseline for comparing exact-but-bounded memory (SWA) against approximate-but-unbounded memory (SSMs).

For guidance on choosing the right layer type and hybridization ratio for your use case, see [docs/LayerSelection.md](docs/LayerSelection.md).

---

## Performance

Results on our Primed Hybrid 8B and 32B models using different layer types. All models use a 50% Hybrid ratio (half of Attention layers replaced with SSM). 
Our Primed Hybrid models closely match Transformer (Qwen3) performance while achieving up to 2× faster inference (see [below](#inference-efficiency)).

### 8B Long-context Instruct Models (Primed from Qwen3-8B)

#### Long-Context Benchmarks

Evaluated on [HELMET](https://github.com/princeton-nlp/HELMET), [MRCR](https://huggingface.co/datasets/openai/mrcr), and [BABILong](https://github.com/booydar/babilong) across context lengths from 8K to 128K, using a weighted average with geometrically increasing weights for longer contexts. 

![Long-Context Benchmarks](assets/figures/long_context_results_8B_models.png)

#### Short-Context NLP Benchmarks

Evaluations on Tulu3-dev from [OLMES](https://github.com/allenai/olmes). Each category in the table below averages the following Tulu3-dev subtasks: 
1. Math: GSM8K, MATH,
2. Knowledge: MMLU, PopQA, TruthfulQA,
3. Coding: HumanEval, HumanEval+, 
4. Reasoning: BigBenchHard
5. Instruction Following: IFEval.

| Model                            | Math | Knowledge | Coding | Reasoning | Instruction Following  | Average |
|----------------------------------|---|---|---|---|---|---|
| Qwen3-8B [Long] <sup>1</sup>                 | 64.56 | 49.75 | 91 | 76.27 | 74.49  | 71.21 |
| GKA-primed-HQwen3-8B-Instruct    |  64.15 | 47.90 | 90.46 | 72.60 | 70.98 | 69.22 |
| GDN-primed-HQwen3-8B-Instruct    |  59.54 | 48.41 | 91.18 | 72.97 | 73.57 | 69.13 |
| Mamba2-primed-HQwen3-8B-Instruct | 57.77 | 46.91 | 89.56 | 70.99 | 74.86 | 68.02 |
| BMOJOF-primed-HQwen3-8B-Instruct | 65.69 | 48.63 | 90.02 | 76.42 | 75.60 | 71.27 |

<sup>1</sup> *Qwen3-8B [Long]* is Qwen3-8B model from HF trained with our long-context priming data.

### 8B Long-context Reasoning Models (Primed from Qwen3-8B)
Evaluations on math reasoning (AIME24/25), science (GPQA), coding (LiveCodeBenchv5, Scicode), tool-calling (BFCLv3/v4), and instruction-following (IFBench).  Evaluations are done using the [Nemo Evaluator SDK](https://docs.nvidia.com/nemo/evaluator/latest/).  We have provided the evaluation configuration [examples/evaluation/nemo_reasoning_evals.yaml](https://github.com/awslabs/hybrid-model-factory/blob/main/examples/evaluation/nemo_reasoning_evals.yaml) for reproducibility.  Evaluations are done at 64K generation length.

| Model                         | AIME24 | AIME25 | GPQA | Live Code Bench-v5 | BFCLv4 (minus web-search) | BFCLv3 | IFBench | SciCode | Average |
|-------------------------------|------|-----------|--------|-----------|----------|------|----|-----|-----|
| Qwen3-8B (thinking, from HF)  | 78.67 | 71.0 | 57.77 | 57.94 | 68.30 | 66.46 | 31.60 | 10.63 | 55.29 |
| GKA-primed-HQwen3-8B-Reasoner | 82.00 | 73.67 | 61.81 | 63.10 | 66.47 | 62.20 | 38.96 | 6.41 | 56.82 |
| GDN-primed-HQwen3-8B-Reasoner | 82.00 | 73.33 | 61.49 | 62.94 | 63.27 | 57.44 | 37.80 | 2.50 | 55.10 |

*For BFCLv4, we remove the web-search subtask and weight each task by the number of entries (test examples) for that task:*  $`\text{Overall Accuracy} = \sum_{i} (\text{accuracy}_i \times \text{num\_entries}_i) / \sum_{i} \text{num\_entries}_i`$

### 32B Long-context Instruct Models (Primed from Qwen3-32B)

Benchmark descriptions are same as 8B Long-context Instruct models (see [above](#8b-long-context-instruct-models-primed-from-qwen3-8b)).


#### Long-Context Benchmarks

The plot below shows performance averaged over context lengths from 8K to 128K.

![Long-Context Benchmarks](assets/figures/long_context_results_32B_models.png)

#### Short-Context NLP Benchmarks

| Model                          | Math  | Knowledge | Coding | Reasoning | Instruction Following | Average |
|--------------------------------|-------|----------|--------|-----------|-----------------------|---------|
| Qwen3-32B [Long]<sup>2</sup>   | 74.43 | 54.47    | 94.54  | 82.89     | 81.52                 | 77.56   |
| GKA-primed-HQwen3-32B-Instruct | 74.02 | 53.95    | 93.43  | 80.31     | 78.74                 | 76.09   |
| GDN-primed-HQwen3-32B-Instruct | 73.65 | 54.35    | 94.40  | 80.99     | 79.3                  | 76.54   |

<sup>2</sup> *Qwen3-32B [Long]* is Qwen3-32B model from HF trained with our long-context priming data.

### 32B Long-context Reasoning Models (Primed from Qwen3-32B)

Benchmark descriptions are same as 8B Long-context Reasoning models (see [above](#8b-long-context-reasoning-models-primed-from-qwen3-8b)).

| Model  | AIME24 | AIME25 | GPQA | Live Code Bench-v5 | BFCLv4 (minus web-search) | BFCLv3 | IFBench | SciCode | Average |
|-------|------|-----------|--------|-----------|----------|------|----|-----|-----|
| Qwen3-32B (thinking, from HF)         | 86.33 | 70.00 | 65.40 | 64.44 | 69.30 | 69.57 | 32.61 | 15.94 | 59.20 |
| GKA-primed-HQwen3-32B-Reasoner             | 87.67 | 81.67 | 67.30 | 70.24 | 70.14 | 66.34 | 48.22 | 12.34 | 62.99 |


### Inference Efficiency

Sustained decode throughput (tokens/s) and mean TTFT on 8× H200 GPUs (TP=8), measured during pure decode with a saturated KV cache. Benchmarked with random data (without prefix-caching). GKA's `num_iter` controls solver iterations at inference — lower values trade a small amount of accuracy for faster speed. See the full [Inference guide](docs/Inference.md#performance-benchmarks) for methodology, TP×DP configurations, and reproducing instructions.

#### 8B Models

**Decode Throughput**

| Model | 16K | 32K | 64K | 128K |
|-------|-----|-----|-----|------|
| GKA-primed-HQwen3-8B (`num_iter=30`, default) | 15,892 (1.78×) | 9,159 (1.77×) | 5,173 (1.89×) | 2,736 (2.23×) |
| GKA-primed-HQwen3-8B (`num_iter=10`) | 17,261 (1.93×) | 9,668 (1.87×) | 5,359 (1.96×) | 2,801 (2.28×) |
| GDN-primed-HQwen3-8B | 17,479 (1.95×) | 10,080 (1.95×) | 5,521 (2.01×) | 2,863 (2.33×) |
| Mamba2-primed-HQwen3-8B | 16,844 (1.88×) | 9,966 (1.93×) | 5,460 (1.99×) | 2,825 (2.30×) |
| BMOJOF-primed-HQwen3-8B | 7,854 (0.88×) | 5,597 (1.08×) | 3,573 (1.30×) | 2,153 (1.75×) |
| Qwen3-8B (baseline) | 8,951 | 5,174 | 2,740 | 1,227 |

**TTFT** (N requests chosen to fill the Transformer's KV cache to capacity; Hybrid models serve the same N with memory to spare)

| Model | 16K | 32K | 64K | 128K |
|-------|-----|-----|-----|------|
| GKA-primed-HQwen3-8B (`num_iter=30`, default) | 35,013 ms (1.26×) | 38,502 ms (1.18×) | 44,893 ms (1.06×) | 53,606 ms (0.85×) |
| GKA-primed-HQwen3-8B (`num_iter=10`) | 33,008 ms (1.19×) | 36,334 ms (1.11×) | 42,076 ms (0.99×) | 51,404 ms (0.82×) |
| GDN-primed-HQwen3-8B | 27,805 ms (1.00×) | 30,975 ms (0.95×) | 36,151 ms (0.85×) | 46,389 ms (0.74×) |
| Mamba2-primed-HQwen3-8B | 28,668 ms (1.03×) | 31,405 ms (0.96×) | 36,666 ms (0.86×) | 46,618 ms (0.74×) |
| BMOJOF-primed-HQwen3-8B | 44,763 ms (1.61×) | 47,600 ms (1.46×) | 52,272 ms (1.23×) | 61,702 ms (0.98×) |
| Qwen3-8B (baseline) | 27,736 ms | 32,661 ms | 42,462 ms | 62,922 ms |

#### 32B Models

**Decode Throughput**

| Model | 16K | 32K | 64K | 128K |
|-------|-----|-----|-----|------|
| GKA-primed-HQwen3-32B (`num_iter=30`, default) | 6,810 (1.29×) | 4,152 (1.45×) | 2,385 (1.82×) | 1,168 (1.99×) |
| GKA-primed-HQwen3-32B (`num_iter=10`) | 7,778 (1.47×) | 4,534 (1.58×) | 2,537 (1.94×) | 1,200 (2.05×) |
| GDN-primed-HQwen3-32B | 8,133 (1.53×) | 4,876 (1.70×) | 2,688 (2.06×) | 1,238 (2.11×) |
| Qwen3-32B (baseline) | 5,299 | 2,865 | 1,308 | 586 |

**TTFT** (N requests chosen to fill the Transformer's KV cache to capacity; Hybrid models serve the same N with memory to spare)

| Model | 16K | 32K | 64K | 128K |
|-------|-----|-----|-----|------|
| GKA-primed-HQwen3-32B (`num_iter=30`, default) | 52,053 ms (1.32×) | 58,613 ms (1.21×) | 68,241 ms (1.05×) | 84,935 ms (0.90×) |
| GKA-primed-HQwen3-32B (`num_iter=10`) | 48,560 ms (1.23×) | 55,039 ms (1.13×) | 64,766 ms (0.99×) | 81,410 ms (0.86×) |
| GDN-primed-HQwen3-32B | 42,492 ms (1.08×) | 48,417 ms (1.00×) | 57,525 ms (0.88×) | 73,145 ms (0.77×) |
| Qwen3-32B (baseline) | 39,421 ms | 48,527 ms | 65,104 ms | 94,479 ms |

---

## Our Research

- [Gated KalmaNet: A Fading Memory Layer Through Test-Time Ridge Regression](https://arxiv.org/abs/2511.21016) (CVPR 2026)
- [PICASO: Permutation-Invariant Context Composition with State Space Models](https://arxiv.org/abs/2502.17605) (ICLR 2025)
- [B'MOJO: Hybrid State Space Realizations of Foundation Models with Eidetic and Fading Memory](https://arxiv.org/abs/2407.06324) (NeurIPS 2024)

### Related Research

- [Marconi: Prefix Caching for the Era of Hybrid LLMs](https://arxiv.org/abs/2411.19379) (MLSys 2025)
- [Expansion Span: Combining Fading Memory and Retrieval in Hybrid State Space Models](https://arxiv.org/abs/2412.13328) (NeuS 2025)
- [Maximally-Informative Retrieval for State Space Model Generation](https://arxiv.org/abs/2506.12149) (ArXiv 2506.12149)

---

## Tested Configurations

| Setup | Hardware | Notes |
|-------|----------|-------|
| Training Stage 1: Distillation (4B, 8K) | 8× A100 40GB | ZeRO-2 |
| Training Stage 2: Fine-tuning (4B, 64K) | 8× A100 40GB | ZeRO-3, Sequence Parallelism |
| Training Stage 1: Distillation (8B, 8K) | 8× H200 140GB | ZeRO-0 |
| Training Stage 2: Fine-tuning (8B, 128K) | 8× H200 140GB | ZeRO-2, Sequence Parallelism |
| Inference (8B, TP=1) | 1× A100 40GB or H200 | |
| Inference (8B, TP=8) | 8× A100 40GB or 8× H200 | |
| Inference (32B, TP=1) | 1× H200 | |
| Inference (32B, TP=8) | 8× H200 | |
| Inference (32B, TP=8) | 8× A100 40GB | TP=8 required (32B does not fit at TP=1) |

Lower-cost setups (LoRA, quantization) are expected to work but are not yet validated. Contributions welcome.

---

## Contributing

We welcome contributions: bug fixes, new Hybrid layer types, documentation improvements, and benchmark results. See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.


---

## Citation

If you use this library in your research, please cite:

```bibtex
@software{hybrid_model_factory,
  title = {Hybrid Model Factory},
  author  = {Nunez* Elvis and Kaul* Prannay and Chattopadhyay* Aditya and Becker* Evan and Bowman* Ben and Zancato* Luca and Thomas David and Xia Wei and Soatto Stefano},
  year = {2026},
  url = {https://github.com/awslabs/hybrid-model-factory}
}
```
\* Key contributors (equal contribution)


---

## Acknowledgments

This codebase builds upon:

- [LlamaFactory](https://github.com/hiyouga/LLaMA-Factory) — base training framework
- [360-LlamaFactory](https://github.com/Qihoo360/360-LLaMA-Factory) — sequence parallel tokenization and attention SP
- [Flash Linear Attention](https://github.com/sustcsonglin/flash-linear-attention) — efficient Linear Attention kernels
- [Transformers](https://github.com/huggingface/transformers) — model architectures and utilities

For running Hybrid models on AWS Trainium, see [State Space Models Neuron](https://github.com/awslabs/state-space-models-neuron), our earlier project that provides NKI-optimized Mamba2 kernels, Hybrid model (Mamba2Hybrid) training with tensor parallelism, and HuggingFace-compatible checkpoints on the Trainium/NeuronX stack. Hybrid Model Factory builds on the lessons learned there and extends the toolkit to GPU with additional architectures (GKA, GDN, B'MOJO-F), a full priming pipeline, and vLLM-based inference.

---

## License

This project is licensed under Apache 2.0, see the [LICENSE](LICENSE) file for details.
