#!/usr/bin/env bash
# Serve GKA-primed-HQwen3-8B-Instruct on H200 140GB GPUs
set -euo pipefail

# ─── Configuration ──────────────────────────────────────────────────────────
MODEL="amazon/GKA-primed-HQwen3-8B-Instruct"
IMAGE="vllm-hybrid"
PORT="${PORT:-8000}"
TP="${TP:-1}"                        # Tensor parallelism: 1 or 8
GPUS="${GPUS:-all}"                  # GPU IDs: "0", "6,7", or "all"
HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"
# ────────────────────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DOCKERFILE_DIR="$(cd "$SCRIPT_DIR/../../vllm-inference" && pwd)"

# Build the Hybrid Model Factory vLLM plugin image
echo "Building $IMAGE from $DOCKERFILE_DIR ..."
docker build -t "$IMAGE" "$DOCKERFILE_DIR"

# Hybrid-specific flags:
#   --mamba-cache-mode align    — SSM state caching aligned to PagedAttention block boundaries
#   --mamba-cache-dtype float32 — full-precision state caches (important for long-context quality)
#   --mamba-ssm-cache-dtype float32 — same for Mamba2 SSM state
echo "Serving $MODEL on TP=$TP ..."
docker run --rm --runtime nvidia -e NVIDIA_VISIBLE_DEVICES="$GPUS" \
  -v "$HF_CACHE":/root/.cache/huggingface \
  -p "$PORT":8000 \
  --ipc=host \
  "$IMAGE" \
  --model "$MODEL" \
  --tensor-parallel-size "$TP" \
  --enable-prefix-caching \
  --mamba-cache-mode align \
  --mamba-cache-dtype float32 \
  --mamba-ssm-cache-dtype float32
