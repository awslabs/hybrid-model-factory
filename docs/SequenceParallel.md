# Sequence Parallel (SP)

Hybrid Model Factory implements sequence parallelism to support training Hybrid models on long context lengths (128K+) by distributing sequence chunks across multiple GPUs. We implement two SP variants for SSM layers, used in conjunction with the Attention layers in the Hybrid model:

1. P2P-SP: An efficient implementation that computes SSM states in parallel for each chunk (on each GPU), then communicates the final state to the next chunk to account for the missing contribution. Currently supports Mamba2, Gated KalmaNet, and B'MOJO-F layers.

2. Universal-SP (USP): Despite its efficiency, P2P-SP requires knowledge of specialized kernels implementing the SSM layer. USP removes this requirement by gathering the entire sequence on each GPU before the SSM's forward pass, then scattering the output back. The result is a plug-and-play SP interface that works with any sequence-mixing layer (not just SSMs) enabling long-context training with minimal effort. Currently used for Gated DeltaNet layers.

This document covers:
- [How SP works](#how-sp-works)
- [Quick start](#quick-start)
- [Configuration reference](#configuration-reference)
- [Architecture overview](#architecture-overview)
- [Adding SP support for a new layer](#adding-sp-support-for-a-new-layer)
- [Numerical verification](#numerical-verification)

---

## Why Sequence Parallel?

As training context length increases, the memory footprint of network activations grows linearly and quickly causes OOM errors. Other parallelism strategies (TP, PP) shard model weights but cannot scale indefinitely with context length. SP shards the input activations along the sequence dimension, so each GPU only stores a fraction of the sequence. This allows scaling to arbitrarily long contexts given enough GPUs.

Depending on the ratio of SSM vs Attention layers SP on a Hybrid model provides a 1.5–2× training throughput improvement over equally-sized Transformers at long contexts (≥64K), with gains increasing at longer sequences.

---

## How SP Works

### Zig-Zag Sharding Pattern

To maximize throughput, our SP implementation uses a zig-zag pattern to distribute sequence chunks across GPUs. The input sequence is split into `2 × SP_size` chunks, and each GPU receives two discontiguous chunks. This balances the compute/communication load for Attention layers (which benefit from zig-zag) while being neutral for SSM layers.

Example with `SP=4` (8 chunks total):
```
GPU 0: chunks [0, 7]
GPU 1: chunks [1, 6]
GPU 2: chunks [2, 5]
GPU 3: chunks [3, 4]
```

The data preprocessing splits each training sample into this zig-zag layout before training begins (see `sequence_parallel_mode: "zigzag-ring"` in config).

### Attention Layers

Attention layers use [ring flash attention](https://github.com/zhuzilin/ring-flash-attention) (zigzag-ring mode) to compute attention across distributed chunks. The implementation monkey-patches the HuggingFace `transformers` flash attention forward to route through the distributed attention backend.

### State-Space Model (SSM) Layer

SSM layers (Mamba2, GKA, GDN, etc.) have a sequential dependency along the sequence dimension through two operations:

1. **1D Convolutions** — Each token depends on the previous `d_conv - 1` tokens. When the sequence is sharded, the first tokens of each chunk need context from the preceding chunk on another GPU.

2. **Recurrent State (SSM kernel)** — The SSM output at any position depends on the hidden state from the previous position. When chunks are on different GPUs, the final hidden state of each chunk must be communicated to the next.

All other operations (projections, activations, gating) are token-independent and run in parallel without communication.

### The P2P-SP Algorithm for SSMs

The algorithm uses the **Chunked Parallel Form** — a hybrid between the fully recurrent and fully parallel SSM formulations. The key insight is that the state update is linear, so we can decouple the computation into:

1. **Compute with zero initial state** (parallel, no communication needed)
2. **Correct for the actual initial state** (requires state from previous chunk)

**Forward pass:**
```
# On all GPUs in parallel:
y, final_state = SSM(x, A, B, C, D, initial_state=0)

# Sequential state passing via P2P:
for each chunk (in sequence order):
    prev_state = receive final_state from previous chunk's GPU
    update final_state with contribution from prev_state
    send updated final_state to next chunk's GPU

# On all GPUs in parallel:
y += correction_from_prev_state(prev_state)
```

**Backward pass:** The backward is a "reversal" of the forward — gradients flow in the opposite direction via P2P communication. Custom `torch.autograd.Function` classes handle the distributed gradient propagation since PyTorch's autograd does not natively handle distributed operations.

**Why the sequential state passing is not a bottleneck:** The SSM state is typically comparable in size to 256–512 tokens of KV cache, making its communication cost over NVLink negligible compared to the full KV cache transfers required by Attention layers at long context lengths.

#### 1D Convolution Handling

For the 1D convolutions, each GPU needs the last `d_conv - 1` tokens from the preceding chunk. The `reorder_for_ssm_p2p` operation handles this:

1. Each GPU pads its chunks to create space for incoming tokens
2. P2P send/receive exchanges the boundary tokens between GPUs
3. The convolution runs on the padded input
4. The padding is stripped from the output

Since each GPU holds two discontiguous chunks (zig-zag pattern), some boundary exchanges happen within the same GPU (no P2P needed) while others require cross-GPU communication.

### The USP Algorithm for SSMs

USP takes a simpler approach: instead of passing states between GPUs, each GPU gathers the full sequence, runs the SSM on it locally, then scatters the output back to the zig-zag layout.

**Forward pass:**
```
# On all GPUs:
x_full = ZigZagGatherScatter(x_local)    # gather full sequence from all GPUs

# On all GPUs in parallel (redundant compute, but no state dependencies):
y_full = SSM(x_full)

# On all GPUs:
y_local = ZigZagScatter(y_full)           # scatter back to local chunks
```

**Backward pass:** `ZigZagGatherScatter` and `ZigZagScatter` are `torch.autograd.Function` subclasses, so the backward pass is handled automatically — scatter in forward becomes gather in backward and vice versa.

**Trade-off:** USP trades extra communication bandwidth (all-gather + scatter of the full sequence) and redundant compute for implementation simplicity. There is no need for custom state-passing logic, conv boundary handling, or output correction. This makes it the recommended starting point when adding SP support for a new layer type.

**Note:** USP can be further optimized by distributing heads across GPUs instead of replicating the full computation, taking advantage of the head-wise independence in many existing SSM layers.

---

## Quick Start

### Single-Node Training with SP

```yaml
# In your training config YAML:
sequence_parallel_size: 8    # Number of GPUs for SP (must divide total GPUs)
```

```bash
cd training
hmf train examples/priming/stage2/hqwen3_8b_gka_long_ctx.yaml
```

### Multi-Node Training with SP

```bash
cd training

# On each node:
hmf train-multinode \
  examples/priming/stage2/hqwen3_8b_gka_long_ctx.yaml \
  <master_addr> <nnodes> <node_rank>
```

### Example Config (128K context, SP=8)

```yaml
### Model ###
model_name_or_path: ./models/HQwen3-8B-GKA/checkpoint

### Training ###
stage: pt
cutoff_len: 131072
sequence_parallel_size: 8
per_device_train_batch_size: 1
gradient_accumulation_steps: 2
pure_bf16: true
bf16: true
flash_attn: fa2

# Optimizer
learning_rate: 5.0e-5
lr_scheduler_type: cosine
max_steps: 4480
max_grad_norm: 1.0

# Checkpointing
output_dir: ./models/HQwen3-8B-GKA/checkpoint-long-ctx

### DeepSpeed ###
deepspeed: examples/deepspeed/ds_z2_config.json
```

The `sequence_parallel_size` parameter controls how many GPUs participate in SP. The total number of GPUs must be divisible by this value. The remaining GPUs are used for data parallelism.

---

## Configuration Reference

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `sequence_parallel_size` | int | 1 | Number of GPUs for SP. Values > 1 enable SP. Must divide total GPU count. |
| `sequence_parallel_mode` | str | `"zigzag-ring"` | SP mode. `"zigzag-ring"` (recommended). |

SP composes with other parallelism strategies:
- **DeepSpeed ZeRO** (Any Stage): Shards model parameters, optimizer states, and gradients across data-parallel ranks.

---

## Architecture Overview

The SP implementation is split across several modules:

```
training/src/hmf/
├── model/model_utils/
│   ├── allgather_sp_helper.py      # ZigZag gather/scatter autograd functions
│   └── sequence_parallel.py        # Attention SP (ring attention, Ulysses)
├── model/hybrid_zoo/layers/
│   ├── mamba/
│   │   ├── mamba2.py               # Mamba2 layer with P2P-SP in forward()
│   │   └── sequence_parallel/mamba2/
│   │       └── mamba_mixer_cp_utils.py  # utils for P2P-SP for Mamba2
│   ├── gated_kalmanet/
│   │   ├── gka.py                  # GKA layer with P2P-SP in forward()
│   │   └── sequence_parallel/
│   │       └── gka_sp_utils.py     # utils P2P-SP for GKA (kk, h_kv states)
│   ├── gated_deltanet/
│   │   └── gdn.py                  # GDN layer with USP in forward()
│   └── ...
├── data/
│   ├── data_utils.py               # preprocess_sp_dataset (zigzag/ulysses splitting)
│   └── processor/sequence_parallel.py  # sp_split for data loading
└── train/pt/workflow.py            # SP group initialization & recursive assignment
```

> **Note:** Ulysses SP is currently not supported for Hybrid models.

### Two SP Strategies for SSM Layers

The codebase implements two strategies for handling SP in SSM layers. They differ in communication pattern:

**Strategy 1: P2P-SP (Mamba2, GKA)**

Each GPU computes the SSM with zero initial state, then passes the final hidden state to the next GPU via point-to-point communication. The receiving GPU corrects its output and state using the received state. This is communication-efficient since only the compact hidden state is transferred.

Key functions:
- `reorder_for_ssm_p2p()` — Exchanges conv boundary tokens between GPUs
- `state_passing_p2p()` / `state_passing_gka_p2p()` — P2P hidden state communication
- `mamba_state_passing_fwd()` / `gka_state_passing_fwd()` — State correction math

**Strategy 2: Universal-SP (GDN)**

Each GPU gathers the full sequence via `ZigZagGatherScatter`, runs the SSM kernel on the full sequence locally, then scatters the output back. This is simpler to implement but uses more communication bandwidth. It enables fast prototyping of new SSM layers (or more generally new sequence-mixing layers) for long-context training, without being bottlenecked by implementation details.

Key functions:
- `ZigZagGatherScatter.apply()` — Gathers chunks from all GPUs into contiguous sequence
- `ZigZagScatter.apply()` — Scatters output back to zig-zag layout

---

## Adding SP Support for a New Layer

To add SP support for a new SSM layer, you need to:

1. **Choose a strategy** (P2P-SP or USP)
2. **Implement the SP logic in your layer's `forward()` method**
3. **Write a numerical verification test**

### Option A: USP Strategy (Simpler)

This is the easiest path. Your layer gathers the full sequence, computes normally, then scatters back. Use this if your layer already has an efficient fused kernel.

```python
from hmf.model.model_utils.allgather_sp_helper import ZigZagGatherScatter, ZigZagScatter

class MyNewLayer(nn.Module):
    def __init__(self, ...):
        super().__init__()
        self.sequence_parallel_group = None  # Set by the framework at init

    def forward(self, hidden_states, **kwargs):
        # --- SP: Gather full sequence ---
        if self.sequence_parallel_group is not None:
            sp_size = dist.get_world_size(group=self.sequence_parallel_group)
            sp_rank = dist.get_rank(group=self.sequence_parallel_group)
            hidden_states = ZigZagGatherScatter.apply(
                hidden_states,
                self.sequence_parallel_group,
                sp_rank, sp_size, sp_size * 2,
            )  # [B, L_full, D]

        # --- Your layer's normal computation ---
        output = my_ssm_kernel(hidden_states, ...)

        # --- SP: Scatter back to local chunks ---
        if self.sequence_parallel_group is not None:
            output = ZigZagScatter.apply(
                output,
                self.sequence_parallel_group,
                sp_rank, sp_size, sp_size * 2,
            )  # [B, L_local, D]

        return output
```

The `ZigZagGatherScatter` and `ZigZagScatter` are `torch.autograd.Function` subclasses, so gradients flow correctly through them automatically.

### Option B: P2P Strategy (More Efficient)

This requires more work but is more communication-efficient. You need to implement:

1. **Conv boundary exchange** — Use `reorder_for_ssm_p2p` if your layer has 1D convolutions
2. **State passing** — A custom `torch.autograd.Function` that sends/receives hidden states via P2P
3. **Output correction** — Apply the received previous state to correct your layer's output

Here is the general pattern, following how Mamba2 and GKA implement it:

```python
from hmf.model.hybrid_zoo.layers.mamba.sequence_parallel.mamba2.mamba_mixer_cp_utils import (
    reorder_for_ssm_p2p,
)

class MyNewLayer(nn.Module):
    def __init__(self, ...):
        super().__init__()
        self.sequence_parallel_group = None
        self.d_conv = 4  # your conv kernel size

    def forward(self, hidden_states, **kwargs):
        if self.sequence_parallel_group is not None:
            sp_size = dist.get_world_size(group=self.sequence_parallel_group)
            sp_rank = dist.get_rank(group=self.sequence_parallel_group)

            # Step 1: Exchange conv boundary tokens via P2P
            hidden_states = reorder_for_ssm_p2p(
                hidden_states,
                self.sequence_parallel_group,
                torch.cuda.Stream(),
                sp_size, sp_rank, self.d_conv,
            )  # [2*B, L/(2*SP) + d_conv-1, D]

        # Step 2: Run projections and convolutions on padded input
        x = self.proj(hidden_states)
        x = self.conv1d(x)

        if self.sequence_parallel_group is not None:
            # Remove conv padding
            x = x[:, self.d_conv - 1:, :].contiguous()

        # Step 3: Run SSM kernel with zero initial state
        y, final_state = my_ssm_kernel(x, initial_state=None, output_final_state=True)

        if self.sequence_parallel_group is not None:
            # Step 4: Pass states via P2P and correct output
            prev_state = my_state_passing_p2p(
                final_state, decay_factors,
                sp_rank, sp_size, self.sequence_parallel_group, batch_size,
            )
            y = y + compute_correction(prev_state, ...)

            # Step 5: Recombine the two zigzag chunks
            b_sp, l, d = y.shape
            y = y.view(b_sp // 2, l * 2, d)  # [B, L/SP, D]

        return y
```

**Implementing `my_state_passing_p2p`:** You need a `torch.autograd.Function` that:
- In `forward`: sends `final_state` to the next rank and receives from the previous rank via `dist.isend`/`dist.irecv`
- In `backward`: sends gradients in the reverse direction

See `State_Passing_P2P` in `mamba_mixer_cp_utils.py` or `State_Passing_GKA_P2P` in `gka_sp_utils.py` for complete implementations. The core pattern is:

```python
class MyStatePassing(torch.autograd.Function):
    @staticmethod
    def forward(ctx, final_state, decay, sp_rank, sp_size, sp_group, bs):
        # Even ranks send first, odd ranks receive first (avoids deadlocks)
        prev_state = torch.zeros_like(final_state)

        if sp_rank > 0:
            # Receive from previous rank
            dist.recv(prev_state, src=prev_rank, group=sp_group)

        # Update final_state with contribution from prev_state
        updated_state = final_state + decay * prev_state

        if sp_rank < sp_size - 1:
            # Send to next rank
            dist.send(updated_state, dst=next_rank, group=sp_group)

        ctx.save_for_backward(decay, prev_state)
        return prev_state

    @staticmethod
    def backward(ctx, grad_output):
        # Reverse direction: send grad to previous rank, receive from next
        ...
```

**Key mathematical insight:** For SSMs that have a linear state update one can "compute the zero state, and then correct it". Concretely, if your SSM computes:

```
y_t = C_t @ s_t
s_t = A_t @ s_{t-1} + B_t @ x_t
```

Then the output decomposes as:
```
y_t = y_t(s_0=0) + C_t @ (A_t:1) @ s_0
```

where `A_t:1 = A_t @ A_{t-1} @ ... @ A_1` is the cumulative decay. This linearity holds for Mamba2, GDN, GKA, and other linear attention variants.

### The `sequence_parallel_group` Attribute

The framework automatically sets `self.sequence_parallel_group` on every module in the model via `set_sequence_parallel_group_recursive()` during training initialization. When `sequence_parallel_size=1`, this attribute is `None`, so your SP code paths are skipped entirely.

Your layer does not need to handle group creation — just check `if self.sequence_parallel_group is not None` to decide whether to run SP logic.

---

## Numerical Verification

Each layer includes a test that verifies SP correctness by comparing outputs and gradients between:
- **SP mode**: The layer running with `sequence_parallel_group` set, processing local chunks
- **No-SP mode**: The same layer running on the full sequence on a single GPU

The test framework lives in each layer's `tests/` directory. The pattern is:

```bash
# Run with torchrun (needs multiple GPUs)
cd training/src/hmf/model/hybrid_zoo/layers/<your_layer>/tests
torchrun --nproc-per-node 8 test_<layer>.py \
  --seqlen 8192 --dtype bf16 --model_config /path/to/config.json
```

The test:
1. Creates a random input tensor and broadcasts it to all GPUs
2. **Local (SP) run**: Extracts zig-zag chunks for each rank, runs forward + backward with SP enabled, gathers outputs and reduces gradients
3. **Global (no-SP) run**: Runs forward + backward on the full sequence without SP
4. Compares outputs and parameter gradients using ℓ∞ norm and relative error

Expected thresholds (following Transformer Engine conventions):
- BF16 activations: ℓ∞ norm ≈ 3.5e-2
- FP32 activations: ℓ∞ norm ≈ 1e-3

To add a test for your new layer, follow the existing pattern in `layers/gated_kalmanet/tests/` or `layers/gated_deltanet/tests/`. The shared utilities (`utils.py`) provide `get_local_result()` and `get_global_result()` helpers that handle the boilerplate.

---

## FAQ

**Q: Can I use SP with packed/variable-length sequences (`cu_seqlens`)?**
A: Not currently. SP requires fixed-length sequences. All layers assert that `cu_seqlens is None` when SP is active.

**Q: What SP sizes are supported?**
A: Any power of 2 that divides the total GPU count. Common values include 2, 4, and 8. The sequence length must be divisible by `2 × SP_size`.

**Q: Does SP work with DeepSpeed?**
A: Yes. SP can be composed with DeepSpeed ZeRO. See the example configs in `training/examples/deepspeed/`.

**Q: Which SP mode should I use, `zigzag-ring` or `ulysses`?** 
A: `zigzag-ring` is the default and recommended mode. It provides better load balancing for Hybrid models. `ulysses` is available as an alternative for attention layers but is not currently supported for SSM layers.

**Q: How does SP interact with the data pipeline?**
A: The data pipeline splits each sample into zig-zag chunks during preprocessing (via `preprocess_sp_dataset` in `data_utils.py`). Each GPU's dataloader yields only its assigned chunks. This means tokenized datasets must be prepared with the target SP size--you cannot change SP size without re-tokenizing.
