#!/usr/bin/env bash
# Decision models under sequence parallelism, against the same models without it (see
# decision_sp_parity.py). Run on a node with 8 GPUs. The script changes into training/ itself, so it
# can be run from any directory. Each run prints a PASS/FAIL table.
set -euo pipefail
cd "$(dirname "$0")/../../.."
SCRIPT=tests/decision/model/decision_sp_parity.py

torchrun --nproc-per-node 2 $SCRIPT                  # SP 2
torchrun --nproc-per-node 4 $SCRIPT                  # SP 4
torchrun --nproc-per-node 4 $SCRIPT --sp_size 2      # 2 SP groups of 2 (SP with data parallelism)
torchrun --nproc-per-node 8 $SCRIPT --sp_size 4 --seq_lens 1024,2048

# Exact fp32 check of the attention-free backbones (GKA's solver amplifies bf16 rounding)
TRITON_F32_DEFAULT=ieee NVIDIA_TF32_OVERRIDE=0 torchrun --nproc-per-node 2 $SCRIPT --dtype fp32
TRITON_F32_DEFAULT=ieee NVIDIA_TF32_OVERRIDE=0 torchrun --nproc-per-node 4 $SCRIPT --dtype fp32
