#!/bin/bash
# =============================================================================
# Priming Pipeline: Stage 0 → Stage 1 → Stage 2
#
# Runs on a single 8×H200 node. Run from inside the Docker container (see setup_docker.sh).
#
# To resume, comment out stages that have already completed.
# =============================================================================

set -euo pipefail

# Resolve repo root (3 levels up from this script)
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

# ---------------------
# Install Hybrid Model Factory
# ---------------------
cd "${REPO_ROOT}/training"
pip install -e .

# ---------------------
# Stage 0: Convert Transformer → Fused Hybrid
# ---------------------

hmf prime-init examples/priming/full_pipeline/8xH200/qwen3_8b_stage0.yaml

# ---------------------
# Tokenize data for Stage 1 (8K pretraining data)
# ---------------------

hmf preprocess examples/priming/full_pipeline/8xH200/tokenize_pretrain.yaml \
    --shard_size 1000 \
    --parallel_jobs 100

# ---------------------
# Stage 1: Distill Attention → SSM
# ---------------------

hmf train examples/priming/full_pipeline/8xH200/qwen3_8b_stage1.yaml

# ---------------------
# Unfuse the Stage 1 checkpoint
# ---------------------

hmf prime-unfuse ./models/HQwen3-8B-GKA-Fused/Stage1/checkpoint-10

# ---------------------
# Tokenize data for Stage 2 (Long Context)
# ---------------------

# Long-context (128K, SP8):
hmf preprocess examples/priming/full_pipeline/8xH200/tokenize_long_ctx.yaml --shard_size 1000 --parallel_jobs 100

# ---------------------
# Stage 2: Fine-tuning
# ---------------------

# Long-context continued pretraining:
hmf train examples/priming/full_pipeline/8xH200/qwen3_8b_stage2.yaml

echo "Priming pipeline complete"
echo "Stage 0 (fused model): $(pwd)/models/HQwen3-8B-GKA-Fused"
echo "Stage 1 (distilled model): $(pwd)/models/HQwen3-8B-GKA-Fused/Stage1/checkpoint-10_unfused"
echo "Stage 2 (long context model): $(pwd)/models/HQwen3-8B-GKA-Fused/Stage1/checkpoint-10_unfused/Long-Ctx/checkpoint-10"