#!/bin/bash
# Build and launch the HMF training Docker container.

set -euo pipefail

# Resolve repo root (3 levels up from this script)
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

# Build the image
docker build -t hmf-training -f "${REPO_ROOT}/training/docker/Dockerfile" "${REPO_ROOT}/training/docker"

docker run --gpus all -it --rm \
    --network host \
    --privileged \
    --ipc=host \
    -v "${REPO_ROOT}":/workspace \
    -w /workspace/examples/training/8xH200 \
    hmf-training

# After running, the docker will land in /workspace/examples/training/8xH200.
# Next, inside of the docker, run: bash run_pipeline.sh
