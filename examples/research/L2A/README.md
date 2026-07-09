# L2A: Learning When to Attend

Official implementation for:

[Learning When to Attend: Conditional Memory Access for Long-Context LLMs](https://arxiv.org/abs/2603.17484)

Sakshi Choudhary, Aditya Chattopadhyay, Luca Zancato, Elvis Nunez, Matthew Trager, Wei Xia, Stefano Soatto


## Overview

L2A extends the effective context length of LLMs by replacing standard Attention
layers with L2A layers that perform *conditional* long-range memory access. Each
L2A layer consists of three modules:

1. **Local Attention** (Sliding Window Attention): processes nearby context efficiently
2. **Router**: a learned linear projection + sigmoid that decides per-token whether
   Global Attention is needed. Outputs binary decision $d_t$.
3. **Global Attention** (conditional): full causal attention invoked only for tokens where $d_t=1$.

On Qwen models, we found that L2A allows ~80% of tokens to skip Global Attention entirely, yielding up to
2× training throughput improvements while matching standard long-context training
performance within 3%.

We provide a custom Triton kernel which efficiently implements the
conditional Global Attention on GPUs.

The sections below describe:
- **Model architecture** (`model/`): L2A model definition for Qwen3
- **Model construction** (`construct/`): Scripts to convert a base Qwen3 model into an L2A model
- **Training** (`train/`): Training workflow with sparsity regularization and sequence parallelism

> **Note:** This implementation currently supports Qwen3 as the base model.


## Prerequisites

This example uses the same environment as HMF. Build and enter the Docker container:

```bash
cd training/docker
docker build -t hmf-training -f Dockerfile .
docker run --gpus all -it --rm --network host --ipc=host hmf-training
```

Then install HMF inside the container:

```bash
# From the repo root
cd training
pip install -e .
```


## Project Structure

```
L2A/
├── configs/                    # Example configuration files
│   ├── construct_qwen3_8b.yaml    # Model construction config
│   └── train_l2a_qwen3_8b.yaml   # Training config
├── model/                      # Model architecture (registered with HF AutoModel)
│   ├── model_register.py       # Auto-registers L2A Qwen3 with AutoModel
│   ├── layers/
│   │   └── qsparse_prefill.py # Q-Sparse Triton kernel (conditional Global Attention)
│   └── l2a_qwen3/             # L2A Qwen3 (config + modeling + cache)
│       ├── configuration_qwen3.py
│       ├── modeling_qwen3.py
│       └── cache.py
├── construct/                  # Model construction (Base Qwen3 → L2A Qwen3)
│   ├── l2a_model.py           # CLI entry point
│   ├── l2a_construct_utils.py # Weight copying utilities
│   └── l2a_qwen3.py          # Qwen3-specific construction logic
└── train/                      # Training
    ├── train.py               # Entry point (registers models, launches training)
    ├── train_multinode.py     # Multi-node launcher
    └── l2a/
        ├── trainer.py         # CustomTrainer with sparsity regularization
        └── workflow.py        # Training orchestration
```


## Usage

### 1. Construct an L2A Model

Create a YAML config for the L2A model construction (see
`configs/construct_qwen3_8b.yaml` for a full example):

```yaml
base_model_name_or_path: Qwen/Qwen3-8B
l2a_initfrombase: true
sliding_window: 4095
sigmoid_linear: true
sigmoid_linear_zero_init: true
sigmoid_temp: 0.1
sig_input_type: hidden_linearproj
output_dir: ./models/L2A-Qwen3-8B
```

Key fields:
- `l2a_initfrombase`: Initialize both SWA and Global Attention weights from the base model's attention weights
- `sliding_window`: Window size for Local Attention (SWA)
- `sigmoid_linear`: Use a learned linear projection in the Router
- `sigmoid_linear_zero_init`: Zero-init the Router projection so all tokens attend globally at initialization

Then run:

```bash
cd examples/research/L2A/construct
python l2a_model.py --config ../configs/construct_qwen3_8b.yaml
```

This loads the base Qwen3, replaces all attention layers with L2A layers
(duplicating the attention weights for both Local and Global Attention paths),
and saves the resulting model.

### 2. Train

Training uses HMF's standard argument format. The entry point registers the L2A
models and uses a custom trainer with sparsity regularization to encourage the
Router to be selective (preventing it from always invoking Global Attention).

See `configs/train_l2a_qwen3_8b.yaml` for a full example (**NOTE**: this config is intended for 48 nodes; adjust global batch size accordingly if using a different number of nodes). Key L2A-specific fields:

```yaml
# Only train attention layers + norms + router (freeze FFN)
hybrid_learnable_params: self_attn,norm,sigmoid_linear

# Router sparsity regularization
lambda_reg: 0.00000001
lambda_reg_scheduler: linear
```

Launch training:

```bash
cd examples/research/L2A/train

# Single-node (auto-detects GPUs, launches torchrun if multi-GPU available)
python train.py ../configs/train_l2a_qwen3_8b.yaml

# Multi-node (run on each node)
python train_multinode.py ../configs/train_l2a_qwen3_8b.yaml MASTER_IP NNODES NODE_RANK
```

### 3. Inference

Inference is currently supported via HuggingFace Transformers (no vLLM integration yet).

L2A models are registered as custom HuggingFace architectures. To load them, you
need to add the L2A module to your Python path so that `model_register.py` can
register the model classes with `AutoModel` before calling `from_pretrained`:

```python
import sys
import os

# Add L2A root to path so model_register can be imported
sys.path.insert(0, os.path.join("path/to/examples/research/L2A"))

# This import registers L2A model types with HuggingFace AutoModel
import model.model_register  # noqa: F401

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "path/to/trained_l2a_model"

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(model_path)

model.eval()
inputs = tokenizer("Hello, how are you?", return_tensors="pt").to(model.device)
output_ids = model.generate(**inputs, max_new_tokens=100, use_cache=True)
print(tokenizer.decode(output_ids[0], skip_special_tokens=True))
```

During inference, the Router's per-token decisions allow skipping Global Attention
for the majority of tokens, reducing time-to-first-token. Layers with consistently
high sparsity can be "pruned" post-training (removing Global Attention entirely),
reducing KV cache memory by up to 50%.


## Citation

```bibtex
@inproceedings{choudharylearning,
  title={Learning When to Attend: Conditional Memory Access for Long-Context LLMs},
  author={Choudhary, Sakshi and Chattopadhyay, Aditya and Zancato, Luca and Nunez, Elvis and Trager, Matthew and Xia, Wei and Soatto, Stefano},
  journal={Forty-third International Conference on Machine Learning},
  year={2026}
}
```
