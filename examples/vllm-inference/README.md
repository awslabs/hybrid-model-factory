# vLLM Inference Examples

One-click scripts to build the Docker image and serve a Hybrid model. The provided scripts default to GKA models. To serve other Hybrid types (GDN, Mamba2, BMOJOF), change the model name in the script.

## Usage

From the repository root:

```bash
bash examples/vllm-inference/serve_8b_reasoner_h200.sh
```

Each script builds the Docker image and starts serving. Configuration variables can be overridden via environment variables:

```bash
# Serve 8B Instruct on GPU 7, port 8080
GPUS=7 PORT=8080 bash examples/vllm-inference/serve_8b_instruct_h200.sh

# Serve 8B Reasoner on GPUs 6 and 7 with TP=2
GPUS=6,7 TP=2 bash examples/vllm-inference/serve_8b_reasoner_h200.sh
```

| Variable   | Description                                       | Example              |
|------------|---------------------------------------------------|----------------------|
| `GPUS`     | GPU IDs (passed to `NVIDIA_VISIBLE_DEVICES`)      | `0`, `6,7`, `all`    |
| `TP`       | Tensor parallelism size                           | `1`, `2`, `8`        |
| `PORT`     | Host port for the API server                      | `8000`, `8080`       |
| `HF_CACHE` | HuggingFace cache directory for model weights     | `~/.cache/huggingface` |

## Available Scripts

### A100 40GB

| Script | Model | Default TP |
|--------|-------|------------|
| `serve_8b_instruct_a100.sh` | GKA-primed-HQwen3-8B-Instruct | 1 |
| `serve_8b_reasoner_a100.sh` | GKA-primed-HQwen3-8B-Reasoner | 1 |
| `serve_32b_instruct_a100.sh` | GKA-primed-HQwen3-32B-Instruct | 8 |
| `serve_32b_reasoner_a100.sh` | GKA-primed-HQwen3-32B-Reasoner | 8 |

> 32B models at bf16 do not fit on a single A100 40GB — TP=8 only.

### H200 140GB

| Script | Model | Default TP |
|--------|-------|------------|
| `serve_8b_instruct_h200.sh` | GKA-primed-HQwen3-8B-Instruct | 1 |
| `serve_8b_reasoner_h200.sh` | GKA-primed-HQwen3-8B-Reasoner | 1 |
| `serve_32b_instruct_h200.sh` | GKA-primed-HQwen3-32B-Instruct | 1 |
| `serve_32b_reasoner_h200.sh` | GKA-primed-HQwen3-32B-Reasoner | 1 |
