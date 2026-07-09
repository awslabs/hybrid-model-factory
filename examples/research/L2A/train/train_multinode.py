"""
L2A Multi-Node Training Launcher.

Mimics `hmf train-multinode` behavior: launches torchrun across multiple nodes.

Usage:
    # On each node:
    python train_multinode.py path/to/config.yaml MASTER_ADDR NNODES NODE_RANK [--master-port PORT]

    # Example (2 nodes):
    # Node 0: python train_multinode.py configs/train_l2a_qwen3_8b.yaml 10.0.0.1 2 0
    # Node 1: python train_multinode.py configs/train_l2a_qwen3_8b.yaml 10.0.0.1 2 1
"""

import argparse
import os
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Launch multi-node L2A training.",
        usage="python train_multinode.py config master_addr nnodes node_rank [--master-port PORT] [--nproc-per-node N]"
    )
    parser.add_argument("config", help="Path to the YAML training config file")
    parser.add_argument("master_addr", help="IP address of the master node")
    parser.add_argument("nnodes", type=int, help="Total number of nodes")
    parser.add_argument("node_rank", type=int, help="Rank of this node (0 to nnodes-1)")
    parser.add_argument("--master-port", type=int, default=29500, help="Port for communication (default: 29500)")
    parser.add_argument("--nproc-per-node", type=int, default=None, help="GPUs per node (default: auto-detect)")
    args = parser.parse_args()

    # Auto-detect GPUs if not specified
    if args.nproc_per_node is None:
        import torch
        args.nproc_per_node = torch.cuda.device_count() if torch.cuda.is_available() else 1

    # Path to train.py (same directory)
    train_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train.py")

    cmd = (
        f"torchrun "
        f"--nnodes {args.nnodes} "
        f"--node_rank {args.node_rank} "
        f"--nproc_per_node {args.nproc_per_node} "
        f"--master_addr {args.master_addr} "
        f"--master_port {args.master_port} "
        f"{train_script} {args.config}"
    )

    env = os.environ.copy()
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    env["TORCH_NCCL_AVOID_RECORD_STREAMS"] = "1"

    print(f"Launching: {cmd}")
    result = subprocess.run(cmd.split(), env=env)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
