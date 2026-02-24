#!/usr/bin/env bash
# Serve GKA-primed-HQwen3-8B-Reasoner on A100 40GB GPUs
set -euo pipefail

# ─── Configuration ──────────────────────────────────────────────────────────
MODEL="amazon/GKA-primed-HQwen3-8B-Reasoner"
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

# A100 at TP=1 requires a smaller cudagraph capture size
EXTRA_FLAGS=()
if [ "$TP" -eq 1 ]; then
  EXTRA_FLAGS+=(--max-cudagraph-capture-size 128)
fi

# Hybrid-specific flags:
#   --mamba-cache-mode align    — SSM state caching aligned to PagedAttention block boundaries
#   --mamba-cache-dtype float32 — full-precision state caches (important for long-context quality)
#   --mamba-ssm-cache-dtype float32 — same for Mamba2 SSM state
#
# Reasoner model flags:
#   --enable-auto-tool-choice / --tool-call-parser hermes — enable tool calling
#   --reasoning-parser qwen3 — extract <think>...</think> reasoning content
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
  --mamba-ssm-cache-dtype float32 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --reasoning-parser qwen3 \
  "${EXTRA_FLAGS[@]}"
