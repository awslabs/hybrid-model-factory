## Priming: Hybrid Models from Pre-trained Models

Priming uses a pre-trained Transformer's weights as an initial condition to bootstrap a new Hybrid architecture. The resulting model is its own artifact — not a modified Transformer, but a purpose-built Hybrid realization that inherits the base model's knowledge while gaining the efficiency benefits of State-Space Model (SSM) layers: smaller KV cache, faster long-context inference, and lower memory usage.

This guide walks through the full Priming pipeline: from architecture conversion to distillation to fine-tuning. It covers different base models (e.g. Qwen3, Qwen3-MoE, Qwen2.5, Llama3, Ministral3) and SSM layers (e.g., Gated KalmaNet, Gated DeltaNet, Mamba2, B'MOJO-F).

## Why Prime Models?

Training a strong Hybrid model from scratch requires massive compute and energy consumption.
Priming saves time and energy by starting from an existing Transformer checkpoint and training the new SSM layers to preserve the model's end-to-end quality despite operating with a different, more efficient memory hierarchy.

The name draws from *priming* in psychology — where prior exposure shapes subsequent learning. Here, a Transformer's pre-trained weights prime the Hybrid model, accelerating its training and preserving knowledge that would otherwise require learning from scratch.

## Pipeline Overview

Priming follows a three-stage recipe:
![Pipeline Overview](../assets/figures/overview.png)

| Stage | What it does | Trainable parameters |
|-------|-------------|---------------------|
| **Stage 0** | Initialize a Hybrid architecture using the pre-trained Transformer's weights; SSM layers are seeded from their corresponding Attention layers | N/A (no training) |
| **Stage 1** | Distills Attention outputs into SSM layers using MSE loss | SSM parameters only |
| **Stage 2** | Fine-tunes the Hybrid model for your target use case (long-context, SFT, reasoning, etc.) | Configurable (typically SSM + Attention + norms) |

---

## Stage 0: Transformer → Hybrid Conversion

Stage 0 initializes a **fused Hybrid architecture** from a pretrained Transformer. In this architecture, each replaced layer contains both the original Attention path and the new SSM path running in parallel:

- The **Attention path** (teacher) is frozen and provides supervisory signal during Stage 1.
- The **SSM path** (student) is the trainable component.

Shared parameters (embeddings, MLPs, layer norms, LM head) are inherited directly from the base Transformer, reducing memory overhead compared to loading separate teacher and student models.

![Fused Decoder Layer](../assets/figures/fused.png)

In the figure above, Attention Blocks (purple) process both streams through Attention, providing the supervisory signal. Hybrid Blocks (yellow) route each stream through its native component—the Attention stream through the Attention mixer, and the SSM stream through the SSM mixer (e.g., GKA)—so the student can learn from the teacher's representations via MSE loss (green arrows).

### SSM Weight Initialization

When replacing an Attention layer with an SSM layer, the Attention projection weights are transferred to initialize the SSM parameters. For example, for [GKA](https://arxiv.org/abs/2511.21016) and [GDN](https://arxiv.org/abs/2412.06464), the mapping is:

| Attention parameter | SSM parameter |
|-------------------|---------------|
| W_Q (query projection) | q_proj |
| W_K (key projection) | k_proj |
| W_V (value projection) | v_proj |
| W_O (output projection) | o_proj |
| 0.5 * (W_O^T + repeat_interleave(W_V)) | g_proj (gate) |

This initialization is motivated by the mathematical relationship between Attention and SSMs: removing the Softmax from Attention yields a recurrence that maps directly to the SSM formulation. Notably, we initialize gate projections (z for Mamba2) as a blend of the transposed output projection and the (GQA-expanded) value projection, which we found empirically to improve upon random initialization used in prior works [The Mamba in the Llama](https://arxiv.org/abs/2408.15237).

All other SSM-specific parameters (e.g., conv weights) are initialized randomly. 
See [hybrid_layer_init.py](../training/src/hmf/priming/hybrid_layer_init.py) for the full initialization logic for each layer type.

### Configuration

Create a YAML config specifying the base model, the layer pattern, and any layer-specific parameters:

```yaml
# Base model to convert
base_model_name_or_path: Qwen/Qwen3-8B

# Output directory for the converted model
output_dir: ./models/HQwen3-8B-GKA-Fused

# Layer pattern: specifies the type of each decoder layer (see Layer Pattern Syntax below)
hybrid_override_pattern: "*DA-GKA*-*DA-GKA*-*DA-GKA*-..."

# Layer-specific parameters (varies by layer type)
gka:
  use_forgetting_gate: true
  solver_type: "chebyshev"
  # ... see Layer Configuration Reference below
```

See [examples/priming/stage0/](../training/examples/priming/stage0/) for complete configs for all supported layer types.

### Layer Pattern Syntax

The `hybrid_override_pattern` string specifies the layer type for each decoder layer, separated by hyphens (`-`). The number of entries must match the number of decoder layers in the base model.

**For Stage 0/1 (fused architecture):**

| Pattern | Description |
|---------|-------------|
| `*` | Standard Attention (kept as-is) |
| `*DA` | Dual-path Attention (fused) |
| `GKA*` | Fused GKA + Attention |
| `GDN*` | Fused GDN + Attention |
| `M2*` | Fused Mamba2 + Attention |
| `BMF*` | Fused B'MOJO-F + Attention |
| `SWA*` | Fused SWA + Attention |
| `GDN>GKA` | Fused GDN → GKA (SSM-to-SSM distillation) |

**For Stage 2 (standard/unfused architecture):**

| Pattern | Description |
|---------|-------------|
| `*` | Standard Attention |
| `GKA` | Gated KalmaNet |
| `GDN` | Gated DeltaNet |
| `M2` | Mamba2 |
| `BMF` | B'MOJO-F |
| `SWA` | Sliding Window Attention |

**Example** — 36-layer model, 50% Hybrid ratio, fused (Stage 0/1):
```
*DA-GKA*-*DA-GKA*-*DA-GKA*-*DA-GKA*-*DA-GKA*-*DA-GKA*-*DA-GKA*-*DA-GKA*-*DA-GKA*-*DA-GKA*-*DA-GKA*-*DA-GKA*-*DA-GKA*-*DA-GKA*-*DA-GKA*-*DA-GKA*-*DA-GKA*-*DA-GKA*
```

**Example** — same model, unfused (Stage 2):
```
*-GKA-*-GKA-*-GKA-*-GKA-*-GKA-*-GKA-*-GKA-*-GKA-*-GKA-*-GKA-*-GKA-*-GKA-*-GKA-*-GKA-*-GKA-*-GKA-*-GKA-*-GKA
```

**Hybrid ratio** is the fraction of Attention layers replaced with SSM layers. A 50% ratio on a 36-layer model means 18 Attention + 18 SSM layers. However, not all Attention layers are equally suited for replacement by SSMs. Consequently, we develop a scoring algorithm to identify which layers are "best" for priming. For more details on how we perform layer selection, see [docs/LayerSelection.md](LayerSelection.md).

### Running Stage 0

```bash
cd training
hmf prime-init examples/priming/stage0/qwen3_8b_gka.yaml
```

The script will:
1. Load the base Transformer model
2. Construct the fused Hybrid architecture with the specified layer pattern
3. Transfer shared weights (embeddings, MLPs, layer norms, LM head) from the base model
4. Initialize SSM layers using the Attention weight mapping described above
5. Save the fused model and tokenizer to `output_dir`

---

## Stage 1: Attention → SSM Distillation

Stage 1 trains the SSM layers to match the Attention layers' behavior using **MSE distillation**. Using the fused architecture from Stage 0, both paths process the same input in parallel. The Attention weights are **frozen**, i.e., only the SSM-specific parameters are trained.

By default (`hybrid_distill_pre_lm_head: true`), the MSE loss is applied on the **final hidden states before the LM head**, comparing the SSM path's output to the Attention path's output after all decoder layers:

```
L = ||h_SSM - h_Attention||²     (computed after the final layer norm, before the LM head)
```

This end-to-end objective gives the SSM layers freedom to develop their own internal representations, i.e., they don't need to match the teacher at every layer, only at the final output.

Optionally, **layerwise MSE** can be enabled via `hybrid_lw_distill_target` to add per-layer supervision at different granularities:

| `hybrid_lw_distill_target` | What it matches per layer |
|---|---|
| `none` (default) | No per-layer loss—only the pre-LM-head loss is used |
| `mixer` | Sequence mixer outputs (Attention vs SSM mixer) |
| `decoder` | Full decoder layer outputs (after mixer + MLP) |
| `residuals` | Both residual contributions (mixer output + MLP output) |
| `all` | All of the above combined |

These can be combined with `hybrid_distill_pre_lm_head: true` for both per-layer and end-to-end supervision simultaneously.

### Configuration

```yaml
### Model ###
model_name_or_path: ./models/HQwen3-8B-GKA-Fused

### Distillation ###
stage: hkd                        # Specify our distillation trainer for fused Hybrid models
hybrid_lw_distill_target: none
hybrid_distill_pre_lm_head: true  # Apply MSE on final hidden states (before LM head)
hybrid_learnable_params: gka      # Train only SSM parameters (use the layer type name)

### Training ###
per_device_train_batch_size: 1
gradient_accumulation_steps: 4
learning_rate: 1.0e-4
lr_scheduler_type: constant_with_warmup
warmup_steps: 720
max_steps: 19000

### Output ###
output_dir: ./models/HQwen3-8B-GKA-Fused/Stage1
```

Key parameters:
- `stage: hkd` — enables the hybrid knowledge distillation training mode
- `hybrid_learnable_params` — comma-separated list of parameter name substrings to train (e.g., `gka`, `mamba`, `gdn`). All other parameters are frozen.
- `hybrid_distill_pre_lm_head: true` — applies MSE loss on the final hidden states before the LM head
- `hybrid_lw_distill_target` — which intermediate outputs to align. Options: `none`, `decoder`, `mixer`, `residuals`, `all`

See [examples/priming/stage1/](../training/examples/priming/stage1/) for complete configs.

### Running Stage 1

Single node:
```bash
cd training
hmf train examples/priming/stage1/hqwen3_8b_gka.yaml
```

Multi-node (run on each node):
```bash
cd training
hmf train-multinode examples/priming/stage1/hqwen3_8b_gka.yaml <master_addr> <nnodes> <node_rank>
```

### Converting Fused to Standard (Unfusing)

After distillation, convert the fused model to a standard Hybrid model by removing the parallel Attention paths:

```bash
cd training
hmf prime-unfuse ./models/HQwen3-8B-GKA-Fused/Stage1/checkpoint-<STEPS>
```

Replace `<STEPS>` with the actual checkpoint step number (e.g., `checkpoint-19000`).

This performs the following conversions:
- Fused SSM layers become standard SSM layers (e.g., `GKA*` becomes `GKA`)
- Dual-path Attention becomes standard Attention (`*DA` becomes `*`)
- Fused SSM-to-SSM layers become the target SSM (e.g., `GDN>GKA` becomes `GKA`)

The output is saved to a `_unfused/` subdirectory by default. You can specify a custom output path with `--save_dir`.

---

## Stage 2: Fine-tuning

After Stage 1, the unfused Hybrid model can be fine-tuned for the target use case. Common paths include:

### Long-Context Continued Pretraining

Extends the model's effective context length (e.g., to 128K) using continued pretraining on long-context data with Sequence Parallelism:

```yaml
### Model ###
model_name_or_path: ./models/HQwen3-8B-GKA-Fused/Stage1/checkpoint-<STEPS>_unfused
config_overrides_json: '{"rope_theta": 5000000, "max_position_embeddings": 131072}'

### Training mode ###
stage: pt

### Train only sequence mixing params ###
hybrid_learnable_params: gka,norm,self_attn

### Context length ###
cutoff_len: 131072
sequence_parallel_size: 8         # Use SP8 for 128K context
```

```bash
cd training
hmf train examples/priming/stage2/hqwen3_8b_gka_long_ctx.yaml
```

See [docs/SequenceParallel.md](SequenceParallel.md) for details on Sequence Parallelism configuration.

### Supervised Fine-Tuning (Instruction Tuning)

Standard SFT on instruction-following data:

```yaml
### Model ###
model_name_or_path: ./models/HQwen3-8B-GKA-Fused/Stage1/checkpoint-<STEPS>_unfused

### Training mode ###
stage: sft

### Training ###
learning_rate: 5.0e-5
lr_scheduler_type: linear
warmup_ratio: 0.03
num_train_epochs: 2.0
cutoff_len: 32768
```

```bash
cd training
hmf train examples/priming/stage2/hqwen3_8b_gka_sft_it.yaml
```

### Reasoning SFT

Fine-tuning on reasoning traces (e.g., from [OpenThoughts3](https://arxiv.org/abs/2506.04178)):

```yaml
### Model ###
model_name_or_path: ./models/HQwen3-8B-GKA-Fused/Stage1/checkpoint-<STEPS>_unfused

### Training mode ###
stage: sft

### Training ###
learning_rate: 5.0e-5
lr_scheduler_type: cosine
warmup_ratio: 0.1
num_train_epochs: 1.0
cutoff_len: 32768
drop_exceed_length_data: true
```

```bash
cd training
hmf train examples/priming/stage2/hqwen3_8b_gka_sft_reasoning.yaml
```

### Multi-Node Training (Stage 2)

For any Stage 2 config, multi-node training is supported via the launcher script:

```bash
cd training
hmf train-multinode examples/priming/stage2/your_config.yaml <master_addr> <nnodes> <node_rank>
```

See [examples/priming/stage2/](../training/examples/priming/stage2/) for all available Stage 2 configs.

---

## Supported Base Models

The Priming pipeline currently supports the following Transformer architectures:

| Model Family | Supported Versions |
|--------------|--------------------|
| Qwen | Qwen3, Qwen3-MoE, Qwen2.5, Qwen3Next, Qwen3.5 |
| Llama | Llama3, Llama3.1 |
| Mistral | Ministral3 |

---

## Layer Configuration Reference

During Stage 0, the following layer-specific parameters can be specified in the YAML config. Each block is nested under the layer type key.

### Gated KalmaNet (GKA, pronounced "gee-ka")

```yaml
gka:
  use_alpha_connection: true   # Alpha connection for residual paths
  use_v_conv: true             # Apply convolution to value vectors
  use_forgetting_gate: true    # Forgetting gate mechanism
  gla_rescale: true            # GLA-style rescaling
  solver_type: "chebyshev"     # Solver: Currently, we only support "chebyshev"
  bp_lambda: true              # Backpropagate through lambda parameters
  num_iter: 30                 # Iterations for iterative solvers
  ridge_strength: 0.02         # Ridge regression regularization
  use_gate: true               # Output gating with learned projection
  conv_size: 4                 # Convolution kernel size
  norm_eps: 1.0e-6             # RMSNorm epsilon
  use_forgetting_gate_kk: true # Forgetting gate for key-key interactions
  use_beta_gate: true          # Beta gating for keys and values
  chunk_size: 64               # Triton block size for chunked computation
```

### Gated DeltaNet (GDN)

```yaml
gdn:
  use_gate: true          # Use output gating (FusedRMSNormGated)
  use_short_conv: true    # Apply short convolutions to q/k/v projections
  allow_neg_eigval: false # Allow negative eigenvalues (scales beta by 2)
  conv_size: 4            # Convolution kernel size
  conv_bias: false        # Use bias in short convolutions
  norm_eps: 1.0e-5        # RMSNorm epsilon
```

### Mamba2

```yaml
mamba2:
  use_qk_norm: true        # Apply QK (CB) normalization
  use_pos_emb: false       # Use positional embeddings for B and C
```

### B'MOJO-F

```yaml
bmojo:
  window_size: 2048       # Sliding window size for local context (total window = 2 x window_size)
  tie_attn_weights: true  # Tie projection weights for in-context and fading tokens
  ssm_mixer: "gka"        # SSM mixer type: "gka", "gdn", "mamba2"
```

For B'MOJO-F models, additional parameters for the chosen SSM layer can be provided. For example, a B'MOJO-F model with a GKA sequence mixer may look like:

```yaml
bmojo:
  window_size: 2048
  tie_attn_weights: true
  ssm_mixer: "gka"
gka:
  use_alpha_connection: true
  use_v_conv: true
  use_forgetting_gate: true
  gla_rescale: true
  solver_type: "chebyshev"
  bp_lambda: true
  num_iter: 30
  ridge_strength: 0.02
  use_gate: true
  conv_size: 4
  norm_eps: 1.0e-6
  use_forgetting_gate_kk: true
  use_beta_gate: true
  chunk_size: 64
```

### Sliding Window Attention (SWA)

```yaml
swa:
  window_size: 512  # Attention window size (required, no default)
```

See [hybrid_dataclasses.py](../training/src/hmf/model/hybrid_zoo/layers/hybrid_dataclasses.py) for the full source of all configuration options.

---

## End-to-End Example: Priming Qwen3-8B with GKA

Here's the complete workflow to prime Qwen3-8B into a Hybrid model with GKA layers:

```bash
cd training

# Stage 0: Convert Transformer to Fused Hybrid
hmf prime-init examples/priming/stage0/qwen3_8b_gka.yaml

# Stage 1: Distill Attention to SSM
hmf train examples/priming/stage1/hqwen3_8b_gka.yaml

# Unfuse: Convert fused model to standard Hybrid
hmf prime-unfuse ./models/HQwen3-8B-GKA-Fused/Stage1/checkpoint-19000

# Stage 2: Fine-tune for your use case (pick one or both; if both, run A first)

# Option A: Extend context to 128K via continued pretraining
hmf train examples/priming/stage2/hqwen3_8b_gka_long_ctx.yaml

# Option B: Supervised fine-tuning for instruction following
# If running after A, ensure model_name_or_path in the yaml points to A's output checkpoint.
hmf train examples/priming/stage2/hqwen3_8b_gka_sft_it.yaml
```