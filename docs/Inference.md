# Inference

This guide covers how to deploy Hybrid models for inference.

| Framework | Use Case |
|-----------|----------|
| [vLLM Plugin](#vllm-plugin) (Recommended) | Production serving, benchmarking, and deployment. Optimized Triton kernels, continuous batching, prefix caching, and tensor parallelism. |
| [HuggingFace Transformers](#huggingface-transformers-inference) | Developing custom Hybrid layers and debugging model behavior. Runs the same code as the training pipeline — no separate optimized kernels. |

**Contents**
- [Prerequisites](#prerequisites)
- [vLLM Plugin](#vllm-plugin)
  - [Docker (Recommended)](#docker-recommended)
  - [pip install](#pip-install)
- [Serving Configuration](#serving-configuration)
  - [Recommended Flags for Hybrid Models](#recommended-flags-for-hybrid-models)
  - [Reasoning Models](#reasoning-models)
  - [State Cache Precision](#state-cache-precision)
  - [Tensor Parallelism](#tensor-parallelism)
  - [GKA Compute Control](#gka-compute-control-num_iter)
  - [Other Useful Flags](#other-useful-flags)
- [Supported Architectures](#supported-architectures)
- [Performance Benchmarks](#performance-benchmarks)
  - [Why Primed Hybrid Models Are Faster](#why-primed-hybrid-models-are-faster)
  - [Decode Throughput](#decode-throughput-tp8)
  - [Aggregate Throughput (TP × DP)](#aggregate-throughput-on-8-gpus-tp-x-dp)
  - [TTFT at Same Batch Size](#ttft-at-same-batch-size-tp8)
  - [GKA Compute Control Benchmarks](#gka-compute-control-benchmarks)
  - [Reproducing These Numbers](#reproducing-these-numbers)
- [HuggingFace Transformers Inference](#huggingface-transformers-inference)
- [Training-Free Context Extension](#training-free-context-extension)
- [Troubleshooting](#troubleshooting)
- [Compatibility](#compatibility)

---

## Prerequisites

- **CUDA GPU**
- **Python 3.10+**

> **Note:** This plugin is built for and tested on **vLLM v0.15.1**. Our Docker image pins this by pulling from the official vLLM public image (`vllm/vllm-openai:v0.15.1`).

For general vLLM deployment guidance using Docker (shared memory, GPU runtime, networking, etc.), see the [vLLM Docker documentation](https://docs.vllm.ai/en/v0.15.1/deployment/docker/).

---

## vLLM Plugin

Production deployment is handled by a [vLLM plugin](https://docs.vllm.ai/en/v0.15.1/design/plugin_system/) that registers all our custom Hybrid architectures automatically — no vLLM fork required.

### Docker (Recommended)

**1. Build the image**

The provided [`Dockerfile`](../vllm-inference/Dockerfile) layers the plugin on top of the official `vllm/vllm-openai:v0.15.1` image. The entrypoint, ports, and CLI are identical to stock vLLM.

```bash
cd vllm-inference
docker build -t vllm-hybrid .
```

**2. Serve a model**

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
  --reasoning-parser qwen3 \
  [additional engine args]
```

You can add any other [engine args](https://docs.vllm.ai/en/v0.15.1/configuration/engine_args/) you need. For Instruct models, drop `--enable-auto-tool-choice`, `--tool-call-parser`, and `--reasoning-parser`.

> **Important:** `--ipc=host` is required for shared memory access when using tensor parallelism. See the [vLLM Docker documentation](https://docs.vllm.ai/en/v0.15.1/deployment/docker/) for details.

> **Tip:** For one-click launch scripts for A100 and H200 GPUs, see [`examples/vllm-inference/`](../examples/vllm-inference/).

**3. Query the server**

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

**4. Interactive shell (development)**

For debugging or running ad-hoc scripts inside the container:

```bash
docker run --rm --runtime nvidia --gpus all -it \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  --ipc=host \
  --entrypoint="" \
  vllm-hybrid \
  bash
```

### pip install

Requires an environment with `vllm==0.15.1` already installed:

```bash
cd vllm-inference
pip install -e .
```

Alternatively, install with the `vllm` extra to pull the compatible vLLM version automatically:

```bash
pip install -e ".[vllm]"
```

> **Important:** The `.[vllm]` extra installs `vllm==0.15.1` and its dependencies, which may overwrite packages already present in your environment.

The plugin registers itself automatically via Python `entry_points` — no code changes or imports needed. Use vLLM as normal — Hybrid architectures are available immediately.

```bash
vllm serve \
  --model amazon/GKA-primed-HQwen3-8B-Reasoner \
  --enable-prefix-caching \
  --mamba-cache-mode align \
  --mamba-cache-dtype float32 \
  --mamba-ssm-cache-dtype float32 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --reasoning-parser qwen3 \
  [additional engine args]
```

You can add any other [engine args](https://docs.vllm.ai/en/v0.15.1/configuration/engine_args/) you need. Query the server with the same `curl` command shown [above](#docker-recommended).

---

## Serving Configuration

### Recommended Flags for Hybrid Models

The following flags are specific to our Hybrid models and are already included in the [provided commands](#docker-recommended) above.

> **Note:** Several of these flags contain "mamba" in their names (i.e., `--mamba-cache-mode`, `--mamba-cache-dtype`, `--mamba-ssm-cache-dtype`). These are vLLM's engine args for configuring fixed-size SSM layer behavior and apply to **all** Hybrid layer types (GKA, GDN, Mamba2) — not just Mamba2.

| Flag | Description | Without it |
|------|-------------|------------|
| `--enable-prefix-caching` | Enables automatic prefix caching. Hybrid models require explicit enabling of this in vLLM v0.15.1 | No prefix caching — reduced throughput for shared-prefix workloads |
| `--mamba-cache-mode align` | Caches SSM state only at PagedAttention block boundaries — the most memory-efficient strategy for Hybrids and the only mode that supports all Hybrid layer types (GKA, GDN, Mamba2) | Defaults to `all`, which caches at every block boundary (more memory, not supported for all layer types) |
| `--mamba-cache-dtype float32` | Sets all GKA and GDN caches (conv and SSM) to `float32`. For Mamba2, sets only the conv-state cache | Lower precision — quality degrades on long sequences |
| `--mamba-ssm-cache-dtype float32` | Sets the Mamba2 SSM-state cache to `float32` (GKA/GDN SSM caches are already controlled by `--mamba-cache-dtype`) | Lower precision for Mamba2 — quality degrades on long sequences |

### Reasoning Models

Reasoning models (e.g., `GKA-primed-HQwen3-8B-Reasoner`) require additional flags for tool calling and reasoning extraction. Add these to your `docker run` (or `vllm serve`) command, in addition to the [recommended flags](#recommended-flags-for-hybrid-models) above:

```bash
# docker run ... / vllm serve ...
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --reasoning-parser qwen3
```

| Flag | Description |
|------|-------------|
| `--enable-auto-tool-choice` | Allows the model to generate tool calls autonomously |
| `--tool-call-parser hermes` | Parses tool calls using the Hermes format |
| `--reasoning-parser qwen3` | Extracts `<think>...</think>` reasoning content from model output |

Thinking mode is enabled by default — the model reasons inside `<think>...</think>` before responding. To turn thinking off, pass `chat_template_kwargs` in your request body:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "amazon/GKA-primed-HQwen3-8B-Reasoner",
    "messages": [
      {"role": "user", "content": "What is Linear Attention in the context of LLMs?"}
    ],
    "chat_template_kwargs": {"enable_thinking": false}
  }'
```

### State Cache Precision

Setting both cache dtypes to `float32` is recommended for all Hybrid models. This is consistent with the recommendation for other hybrid models such as [NVIDIA Nemotron-Nano](https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-9B-v2#use-it-with-vllm).

### Tensor Parallelism

Tensor parallelism is supported for all architectures. Add `--tensor-parallel-size` to your `docker run` (or `vllm serve`) command, in addition to the [recommended flags](#recommended-flags-for-hybrid-models) above:

```bash
# docker run ... / vllm serve ...
  --tensor-parallel-size 2
```

Depending on your workload, running multiple model replicas at lower TP (e.g., TP1×DP8 or TP2×DP4) may yield higher aggregate throughput than a single high-TP replica. See the [TP × DP table](#aggregate-throughput-on-8-gpus-tp-x-dp) for measured configurations.

### GKA Compute Control (`num_iter`)

GKA layers use an iterative solver. The iteration count (`num_iter`, default **30**) trades off compute/quality vs inference speed. Override it at serving time by adding `--hf-overrides` to your `docker run` (or `vllm serve`) command, in addition to the [recommended flags](#recommended-flags-for-hybrid-models) above:

```bash
# docker run ... / vllm serve ...
  --hf-overrides '{"gka_overrides": {"num_iter": 10}}'
```

> **Note:** `gka_overrides` is a separate top-level key rather than a direct edit to `gka_config`. This is because vLLM's `--hf-overrides` replaces top-level config keys rather than merging nested dicts. The plugin merges `gka_overrides` into `gka_config` at model init time, avoiding accidental overwrites of non-specified keys.

### Other Useful Flags

| Flag | Description |
|------|-------------|
| `--max-model-len <N>` | Override the maximum sequence length |
| `--served-model-name <name>` | Alias for the model in the API (replaces the HF tag/local model path) |
| `--gpu-memory-utilization <f>` | Fraction of GPU memory to use (default: 0.9) |

---

## Supported Architectures

### Registered Model Architectures

All architectures run on the vLLM V1 engine with prefix caching and chunked prefill support.

| Architecture | Base Model | Notes |
|---|---|---|
| `HybridQwen2ForCausalLM` | Qwen-2 / Qwen-2.5 | Dense |
| `HybridQwen3ForCausalLM` | Qwen-3 | Dense |
| `HybridQwen3MoeForCausalLM` | Qwen-3 MoE | Sparse MoE with expert parallelism |

### Supported Layer Types

Each model can use an arbitrary per-layer mix of the following layer types, configured via the `hybrid_override_pattern` in the model's `config.json`:

| Symbol | Layer Type | Description |
|---|---|---|
| `*` | Attention | Standard full causal attention |
| `GKA` | Gated KalmaNet | SSM layer inspired by Kalman filtering |
| `GDN` | Gated DeltaNet | SSM layer with delta rule updates |
| `M2` | Mamba2 | Mamba-2 state space model layer |
| `BMF` | B'MOJO-F | Hybrid layer combining eidetic (attention) and fading (SSM) memory |
| `SWA` | Sliding Window Attention | Attention limited to a fixed window |

> **Important:** GKA, GDN, and Mamba2 cannot be mixed with each other within a single model. See [Troubleshooting](#cannot-mix-gka-gdn-and-mamba2) for details.

### The `hybrid_override_pattern`

The layer composition is defined by the `hybrid_override_pattern` field in the model's `config.json`. This is a dash-separated string where each segment specifies the layer type at that index:

```
*-*-GKA-*-*-GKA-*-*-GKA-*-*-GKA-...
```

In this example, layers 0, 1, 3, 4, ... are standard Attention and layers 2, 5, 8, 11, ... are GKA. All released models in the [Primed Model Zoo](../README.md#primed-model-zoo) come with this pattern pre-configured — no manual editing needed.

---

## Performance Benchmarks

### Why Primed Hybrid Models Are Faster

Each Primed Hybrid model is derived from a base Transformer by replacing a portion of its Attention layers with SSM layers that maintain a fixed-size recurrent state instead of a growing KV cache. At a 50% Hybrid ratio, roughly half the KV cache (which grows linearly with sequence length) is replaced with fixed-size SSM state. The practical benefits:

- **Higher throughput at long contexts** — less memory spent on KV cache means more memory for batching, which translates to higher tokens/sec.
- **More concurrent sequences** — with ~50% KV cache reduction, you can serve roughly 2x as many concurrent sequences before hitting memory limits.
- **Growing advantage with context length** — at long contexts, Attention dominates the forward pass while SSM layers remain negligible in cost. Since the Hybrid model makes roughly half as many Attention calls as the base Transformer, the throughput advantage grows with context length.

### Decode Throughput (TP=8)

Sustained decode tokens/s on 8× H200 GPUs, measured during pure decode (no concurrent prefill). Each model runs with a saturated KV cache. All benchmarks use random data to isolate raw throughput (no prefix-caching benefits). Our Hybrid models have a 50% Hybrid ratio.

> **Note:** Model names in the tables below (e.g., "GKA-primed-HQwen3-8B") are shorthand — the same numbers apply to both the Instruct and Reasoner variants where available. See the [Primed Model Zoo](../README.md#primed-model-zoo) for full model names.

#### 8B Models

| Model | 16K | 32K | 64K | 128K |
|-------|-----|-----|-----|------|
| GKA-primed-HQwen3-8B (`num_iter=30`, default) | 15,892 (1.78x) | 9,159 (1.77x) | 5,173 (1.89x) | 2,736 (2.23x) |
| GKA-primed-HQwen3-8B (`num_iter=10`) | 17,261 (1.93x) | 9,668 (1.87x) | 5,359 (1.96x) | 2,801 (2.28x) |
| GDN-primed-HQwen3-8B | 17,479 (1.95x) | 10,080 (1.95x) | 5,521 (2.01x) | 2,863 (2.33x) |
| Mamba2-primed-HQwen3-8B | 16,844 (1.88x) | 9,966 (1.93x) | 5,460 (1.99x) | 2,825 (2.30x) |
| BMOJOF-primed-HQwen3-8B | 7,854 (0.88x) | 5,597 (1.08x) | 3,573 (1.30x) | 2,153 (1.75x) |
| Qwen3-8B (baseline) | 8,951 | 5,174 | 2,740 | 1,227 |

Hybrid models achieve up to ~2.3× sustained decode throughput over the Transformer baseline, with the advantage growing at longer contexts. GKA, GDN, and Mamba2 all reach ~2.3× at 128K. The speedup comes from Hybrid models replacing half the KV cache with fixed-size SSM state, freeing memory for larger batches. GKA throughput can be further improved by reducing `num_iter` — see [Compute Control Benchmarks](#gka-compute-control-benchmarks) for details.

B'MOJO-F surpasses the Transformer from 32K onward and reaches 1.75× at 128K; its lower throughput relative to other Hybrid models reflects three forward passes per hybrid layer, which we expect to improve with future kernel optimizations.

#### 32B Models

| Model | 16K | 32K | 64K | 128K |
|-------|-----|-----|-----|------|
| GKA-primed-HQwen3-32B (`num_iter=30`, default) | 6,810 (1.29x) | 4,152 (1.45x) | 2,385 (1.82x) | 1,168 (1.99x) |
| GKA-primed-HQwen3-32B (`num_iter=10`) | 7,778 (1.47x) | 4,534 (1.58x) | 2,537 (1.94x) | 1,200 (2.05x) |
| GDN-primed-HQwen3-32B | 8,133 (1.53x) | 4,876 (1.70x) | 2,688 (2.06x) | 1,238 (2.11x) |
| Qwen3-32B (baseline) | 5,299 | 2,865 | 1,308 | 586 |

GKA 32B reaches up to 1.99× at 128K at the default `num_iter=30`, with higher throughput available by reducing `num_iter` — see [Compute Control Benchmarks](#gka-compute-control-benchmarks). The throughput advantage grows with context length, consistent with the 8B results.

### Aggregate Throughput on 8 GPUs (TP x DP)

Sustained decode tokens/s (× DP replicas) across all 8 H200 GPUs, varying the tensor-parallelism / data-parallelism split.

#### 8B Models

**GKA-primed-HQwen3-8B**

| Config | 16K | 32K | 64K | 128K |
|--------|-----|-----|-----|------|
| TP1 × DP8 | 17,640 | 9,976 | 4,791 | 2,444 |
| TP2 × DP4 | 17,880 | 10,411 | 5,106 | 2,625 |
| TP4 × DP2 | 17,366 | 10,326 | 5,222 | 2,731 |
| TP8 × DP1 | 15,892 | 9,159 | 5,173 | 2,736 |

The optimal TP × DP split depends on the workload (context length, batch size, latency requirements). Users should benchmark their specific use case to determine the best configuration.

#### 32B Models

**GKA-primed-HQwen3-32B**

| Config | 16K | 32K | 64K | 128K |
|--------|-----|-----|-----|------|
| TP2 × DP4 | 6,083 | 3,406 | 1,850 | 1,081 |
| TP4 × DP2 | 6,744 | 4,085 | 2,323 | 1,099 |
| TP8 × DP1 | 6,810 | 4,152 | 2,385 | 1,168 |

### TTFT at Same Batch Size (TP=8)

Mean TTFT (`mean_ttft_ms` from `vllm bench serve`) on 8× H200 GPUs. Batch size (N) is set to saturate the Transformer's KV cache — Hybrid models serve the same N with memory to spare. Ratios below 1.0 indicate faster prefill than the Transformer baseline.

#### 8B Models

| Model | 16K | 32K | 64K | 128K |
|-------|-----|-----|-----|------|
| GKA-primed-HQwen3-8B (`num_iter=30`, default) | 35,013 ms (1.26x) | 38,502 ms (1.18x) | 44,893 ms (1.06x) | 53,606 ms (0.85x) |
| GKA-primed-HQwen3-8B (`num_iter=10`) | 33,008 ms (1.19x) | 36,334 ms (1.11x) | 42,076 ms (0.99x) | 51,404 ms (0.82x) |
| GDN-primed-HQwen3-8B | 27,805 ms (1.00x) | 30,975 ms (0.95x) | 36,151 ms (0.85x) | 46,389 ms (0.74x) |
| Mamba2-primed-HQwen3-8B | 28,668 ms (1.03x) | 31,405 ms (0.96x) | 36,666 ms (0.86x) | 46,618 ms (0.74x) |
| BMOJOF-primed-HQwen3-8B | 44,763 ms (1.61x) | 47,600 ms (1.46x) | 52,272 ms (1.23x) | 61,702 ms (0.98x) |
| Qwen3-8B (baseline) | 27,736 ms | 32,661 ms | 42,462 ms | 62,922 ms |

All Hybrid models deliver faster prefill than the Transformer at long contexts. The TTFT advantage comes from cheaper prefill compute — at long contexts, Attention dominates total prefill cost, so replacing Attention layers with SSM layers yields large savings. At 128K, GKA achieves 0.85× TTFT (15% faster) at default `num_iter=30`. GDN and Mamba2 reach 0.74× (26% faster) at 128K. GKA TTFT can be further improved by reducing `num_iter` — see [Compute Control Benchmarks](#gka-compute-control-benchmarks).

B'MOJO-F has higher TTFT relative to other Hybrid models, converging toward the Transformer at long contexts (0.98× at 128K); this reflects three forward passes per hybrid layer, which we expect to improve with future kernel optimizations. All TTFT results are under chunked prefill settings; we would expect larger TTFT benefits without chunked prefill.

#### 32B Models

| Model | 16K | 32K | 64K | 128K |
|-------|-----|-----|-----|------|
| GKA-primed-HQwen3-32B (`num_iter=30`, default) | 52,053 ms (1.32x) | 58,613 ms (1.21x) | 68,241 ms (1.05x) | 84,935 ms (0.90x) |
| GKA-primed-HQwen3-32B (`num_iter=10`) | 48,560 ms (1.23x) | 55,039 ms (1.13x) | 64,766 ms (0.99x) | 81,410 ms (0.86x) |
| GDN-primed-HQwen3-32B | 42,492 ms (1.08x) | 48,417 ms (1.00x) | 57,525 ms (0.88x) | 73,145 ms (0.77x) |
| Qwen3-32B (baseline) | 39,421 ms | 48,527 ms | 65,104 ms | 94,479 ms |

GKA 32B achieves 0.90× TTFT (10% faster) at 128K with default `num_iter=30`. The crossover point where Hybrid TTFT beats the Transformer is around 64K, consistent with 8B results. Reducing `num_iter` further improves TTFT — see [Compute Control Benchmarks](#gka-compute-control-benchmarks).

### GKA Compute Control Benchmarks

GKA's iterative Chebyshev solver allows trading compute for speed at serving time by reducing `num_iter` (default **30**). See [GKA Compute Control](#gka-compute-control-num_iter) for how to set this flag. The tables below show the impact on decode throughput and TTFT.

#### Decode Throughput

**8B Models**

| Model | 16K | 32K | 64K | 128K |
|-------|-----|-----|-----|------|
| GKA-primed-HQwen3-8B (`num_iter=30`, default) | 15,892 (1.78x) | 9,159 (1.77x) | 5,173 (1.89x) | 2,736 (2.23x) |
| GKA-primed-HQwen3-8B (`num_iter=10`) | 17,261 (1.93x) | 9,668 (1.87x) | 5,359 (1.96x) | 2,801 (2.28x) |
| GKA-primed-HQwen3-8B (`num_iter=5`) | 17,606 (1.97x) | 9,770 (1.89x) | 5,399 (1.97x) | 2,811 (2.29x) |
| GKA-primed-HQwen3-8B (`num_iter=1`) | 17,485 (1.95x) | 9,780 (1.89x) | 5,413 (1.98x) | 2,812 (2.29x) |

**32B Models**

| Model | 16K | 32K | 64K | 128K |
|-------|-----|-----|-----|------|
| GKA-primed-HQwen3-32B (`num_iter=30`, default) | 6,810 (1.29x) | 4,152 (1.45x) | 2,385 (1.82x) | 1,168 (1.99x) |
| GKA-primed-HQwen3-32B (`num_iter=10`) | 7,778 (1.47x) | 4,534 (1.58x) | 2,537 (1.94x) | 1,200 (2.05x) |
| GKA-primed-HQwen3-32B (`num_iter=5`) | 8,039 (1.52x) | 4,621 (1.61x) | 2,569 (1.96x) | 1,206 (2.06x) |
| GKA-primed-HQwen3-32B (`num_iter=1`) | 8,177 (1.54x) | 4,678 (1.63x) | 2,593 (1.98x) | 1,210 (2.06x) |

Most of the gain comes from reducing 30 → 10; further reductions yield diminishing returns as the solver becomes a smaller fraction of total decode time. The `num_iter` effect is more pronounced at 32B than 8B — at 16K, reducing from 30 to 1 improves throughput by 20% (1.29× → 1.54×). `num_iter` has a larger relative impact at shorter context lengths where GKA's forward-pass time is a meaningful fraction of total decode time; at longer contexts, memory bandwidth dominates and the difference narrows.

#### TTFT

**8B Models**

| Model | 16K | 32K | 64K | 128K |
|-------|-----|-----|-----|------|
| GKA-primed-HQwen3-8B (`num_iter=30`, default) | 35,013 ms (1.26x) | 38,502 ms (1.18x) | 44,893 ms (1.06x) | 53,606 ms (0.85x) |
| GKA-primed-HQwen3-8B (`num_iter=10`) | 33,008 ms (1.19x) | 36,334 ms (1.11x) | 42,076 ms (0.99x) | 51,404 ms (0.82x) |
| GKA-primed-HQwen3-8B (`num_iter=5`) | 32,318 ms (1.17x) | 35,690 ms (1.09x) | 41,490 ms (0.98x) | 50,752 ms (0.81x) |
| GKA-primed-HQwen3-8B (`num_iter=1`) | 31,741 ms (1.14x) | 35,716 ms (1.09x) | 39,963 ms (0.94x) | 50,232 ms (0.80x) |

**32B Models**

| Model | 16K | 32K | 64K | 128K |
|-------|-----|-----|-----|------|
| GKA-primed-HQwen3-32B (`num_iter=30`, default) | 52,053 ms (1.32x) | 58,613 ms (1.21x) | 68,241 ms (1.05x) | 84,935 ms (0.90x) |
| GKA-primed-HQwen3-32B (`num_iter=10`) | 48,560 ms (1.23x) | 55,039 ms (1.13x) | 64,766 ms (0.99x) | 81,410 ms (0.86x) |
| GKA-primed-HQwen3-32B (`num_iter=5`) | 47,958 ms (1.22x) | 54,320 ms (1.12x) | 63,826 ms (0.98x) | 80,369 ms (0.85x) |
| GKA-primed-HQwen3-32B (`num_iter=1`) | 46,726 ms (1.19x) | 53,061 ms (1.09x) | 62,645 ms (0.96x) | 79,321 ms (0.84x) |

At shorter contexts, reducing `num_iter` has a bigger relative impact on TTFT. At 16K the gap between `num_iter=30` and `num_iter=1` is meaningful (1.26× → 1.14× for 8B), whereas at 128K it is smaller (0.85× → 0.80×) — at long contexts, Attention prefill dominates and tuning GKA's solver iterations has limited effect.

### Reproducing These Numbers

With a vLLM server already running (see [Docker](#docker-recommended) or [pip install](#pip-install)), run [`vllm bench serve`](https://docs.vllm.ai/en/v0.15.1/cli/bench/serve/) against it:

```bash
docker exec <container-name> \
  vllm bench serve \
    --model <served-model-name> \
    --tokenizer <model-path> \
    --dataset-name random \
    --random-input-len 14336 \
    --random-output-len 2048 \
    --num-prompts <N> \
    --max-concurrency <N> \
    --request-rate inf \
    --ignore-eos
```

`--model` is the `--served-model-name` from `vllm serve` (defaults to the HuggingFace model path if not specified). For the numbers above, `N` was chosen to saturate the KV cache at each sequence length (input + output).

---

## HuggingFace Transformers Inference

> **Note:** This path is for **developing custom Hybrid layers and debugging model behavior only**. It runs the same model implementations as the training pipeline (`hmf.model.hybrid_zoo`), with no optimized kernels or vLLM-specific abstractions. For production serving, use the [vLLM Plugin](#vllm-plugin).

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import hmf.model.hybrid_zoo.models.model_register  # Register Hybrid architectures

model = AutoModelForCausalLM.from_pretrained(
    "amazon/GKA-primed-HQwen3-8B-Reasoner", trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained("amazon/GKA-primed-HQwen3-8B-Reasoner")

messages = [{"role": "user", "content": "What is linear attention in the context of LLMs?"}]
prompt = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
)
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=256)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

Requires the training package to be installed (`cd training && pip install -e .`).

---

## Training-Free Context Extension

Hybrid models support training-free context-length extension up to 4x their native context size via [PICASO (Permutation-Invariant Context Composition)](https://arxiv.org/abs/2502.17605). This works by composing SSM states from independently processed chunks.

See [StateComposition.md](StateComposition.md) for usage and supported composition types.

---

## Troubleshooting

### Quality degrades on long sequences

Set the SSM state cache to `float32`:

```bash
--mamba-cache-dtype float32 --mamba-ssm-cache-dtype float32
```

This is the most common issue. Without these flags, accumulated numerical error in the recurrent state causes quality degradation at long contexts.

### B'MOJO-F models: KV cache assertion error

B'MOJO-F layers have two attention modules per decoder layer (s-stream and c-stream). The plugin automatically patches vLLM to handle this by setting `num_attn_module=2` during KV cache initialization. If you see assertion errors related to `extract_layer_index`, ensure the plugin is properly installed and loaded before model initialization.

### Triton allocator errors at TP=1

At TP=1, vLLM may not initialize the Triton memory allocator (which is normally set as a side effect of FlashInfer at TP>1). The plugin handles this automatically by setting a custom allocator on import. If you see Triton allocation errors, verify the plugin is installed correctly in the serving environment.

### `assert num_cache_lines >= batch`

You may see an error like:

```
File ".../causal_conv1d.py", line 1160, in causal_conv1d_update
    assert num_cache_lines >= batch
```

This happens when the CUDA graph capture size exceeds the SSM state cache size. Reduce the capture size until the error disappears, e.g.:

```bash
--max-cudagraph-capture-size 128  # default is 512
```

See [vllm-project/vllm#34571](https://github.com/vllm-project/vllm/pull/34571) for details.

### OOM during inference despite sufficient profiled memory

Hybrid SSM layers allocate intermediate tensors (chunk states, conv buffers, solver intermediates) inside their kernels that are not fully captured by vLLM's memory profiler. The profiler runs a dummy forward pass to estimate GPU memory usage, but the SSM profile paths may exit early — before reaching the actual kernel allocations — so these tensors are not accounted for. The profiler therefore overestimates available KV-cache memory, and the model OOMs at runtime when the real SSM kernels run.

This can happen with any Hybrid model (GKA, GDN, or Mamba2), though it is most pronounced with Mamba2 due to the larger intermediate tensors in `mamba_chunk_scan_combined_varlen` and `selective_state_update`. The issue is more likely on smaller-capacity GPUs (e.g., A100 40GB) where memory headroom is tighter.

**Fix:** reduce `--gpu-memory-utilization` so that the untracked headroom is large enough:

```bash
--gpu-memory-utilization 0.8  # default is 0.9; adjust as needed
```

The required reduction depends on model size, sequence length, and GPU memory capacity. If you still see OOM, lower it further.

### Cannot mix GKA, GDN, and Mamba2

GKA, GDN, and Mamba2 each have incompatible state shapes: GDN uses a 2-tuple state (conv + recurrent), GKA uses a 3-tuple state (conv + h_kk + h_kv), and Mamba2 uses a differently-shaped 2-tuple state. Therefore, a single model cannot contain two (or more) types of SSM layer.

---

## Compatibility

| Component | Version |
|-----------|---------|
| vLLM | `0.15.1` (V1 engine only) |
| CUDA | Required (Triton kernels for GKA, GDN, B'MOJO-F) |
| Python | 3.10+ |
