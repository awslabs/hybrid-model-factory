# HOPE: Higher-Order Pruning of Experts

A second-order expert pruning method for Mixture-of-Experts (MoE) language models, which accounts for pairwise expert interactions.

HOPE formulates expert selection as a binary quadratic program over an interaction matrix **F**, where each entry captures the joint contribution of an expert pair weighted by their router gate values and output norms. This goes beyond first-order methods that score experts independently.

## Installation

**Prerequisites:** PyTorch must be installed separately to match your CUDA driver. See [pytorch.org](https://pytorch.org/get-started/locally/) for instructions. For example:

```bash
# Example for CUDA 12.4
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

Then install HOPE:

```bash
git clone https://github.com/awslabs/hybrid-model-factory.git
cd hybrid-model-factory/examples/research/HOPE
pip install -e .
```

## Quick Start

HOPE uses a three-step pipeline: Calibrate, Solve, Prune

### 1. Calibrate: Collect the F-matrix

Run calibration prompts through your model to collect expert interaction statistics:

```bash
hope calibrate \
    --model-path /path/to/moe-model \
    --prompts prompts.txt \
    --out-path observations.h5
```

Where `prompts.txt` has one prompt (string) per line. Alternatively, use `--prompts` to specify a `.json` file which contains a list of strings, or a list of lists of ints (pre-tokenized token IDs).

This produces an HDF5 file which contains the statistics needed to construct the F matrix (expert usage and co-usage statistics).

### 2. Solve: Find the optimal pruning set

Solve the HOPE quadratic program to determine which experts to prune:

```bash
hope solve \
    --obs-path observations.h5 \
    --prune-frac 0.5 \
    --out-path pruneset.json
```

This materializes the F-matrix and solves the corresponding quadratic program to produce the actual prune-set (i.e. which experts to prune in each layer) as a JSON file. This is the main result of the HOPE algorithm.

### 3. Prune: Create the smaller model

Apply the pruning set to produce a smaller checkpoint:

```bash
hope prune \
    --model-path /path/to/moe-model \
    --pruneset-path pruneset.json \
    --out-path /path/to/pruned-model
```

The pruned model is a standard HuggingFace checkpoint and can be loaded directly:

```python
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("/path/to/pruned-model")
```

You may also prune your model with your own code instead of the script here (the main HOPE result is the prune-set JSON).

### First-order baselines

HOPE's calibration data also supports several first-order baselines (REAP, EAN, MAN, freq):

```bash
hope baselines \
    --obs-path observations.h5 \
    --prune-frac 0.5 \
    --method reap \
    --out-path pruneset_reap.json
```

## Python API

```python
from hope.calibrate import calibrate
from hope.solve import solve
from hope.prune import prune_model

# Step 1
calibrate("path/to/model", ["prompt 1", "prompt 2", ...], "obs.h5")

# Step 2
solve("obs.h5", prune_frac=0.5, out_path="pruneset.json")

# Step 3
prune_model("path/to/model", "pruneset.json", "path/to/pruned")
```

See `examples/quickstart.py` for a complete end-to-end example.

## Supported Models

HOPE's calibration step hooks into a HuggingFace MoE's expert Module. The observer requires that models use the stacked-experts layout, where all expert weights (in each MoE layer) are stored as a 3D tensor (`gate_up_proj` with shape `(num_experts, 2 * intermediate_size, hidden_size)` and `down_proj` with shape `(num_experts, hidden_size, intermediate_size)` on a single module). HOPE's code is tested on `transformers==5.2.0` and on the following model architectures:

- Qwen3 MoE (e.g. Qwen3-30B-A3B)
- Qwen3-Next
- Qwen3.5 MoE (e.g. Qwen3.5-35B-A3B, Qwen3.5-122B-A10B)
- GLM-4.5-Air

## Citation

```bibtex
@article{tseng2026hope,
    title={Higher-Order Pruning of Experts in Mixture-of-Experts Language Models},
    author={Tseng, Alex M. and Kaul, Prannay and Zancato, Luca and Xia, Wei and Soatto, Stefano},
    journal={arXiv preprint arXiv:2609.18916},
    year={2026},
    url={https://arxiv.org/abs/2609.18916},
}
```

## License

Apache-2.0. See [LICENSE](LICENSE).
