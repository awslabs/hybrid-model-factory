"""
L2A Training Entry Point.

Mimics `hmf train` behavior: auto-detects multi-GPU and launches torchrun
if needed, registers L2A models, and runs the L2A training workflow.

Usage:
    # Single or multi-GPU (auto-detected)
    python train.py path/to/config.yaml

    # Explicit multi-GPU via deepspeed
    deepspeed --num_gpus=8 train.py path/to/config.yaml

    # Explicit multi-GPU via torchrun
    torchrun --nproc_per_node=8 train.py path/to/config.yaml
"""

import os
import sys
import subprocess
from copy import deepcopy

# Add L2A root to path so we can import model/ and train/ packages
_L2A_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TRAIN_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _L2A_ROOT)
sys.path.insert(0, _TRAIN_DIR)

# Register L2A models with HuggingFace AutoModel
import model.model_register  # noqa: F401


def _get_device_count():
    """Get the number of available GPUs."""
    import torch
    if torch.cuda.is_available():
        return torch.cuda.device_count()
    return 1


def _run_training():
    """Run the L2A training workflow."""
    from hmf.hparams import get_train_args, read_args
    from hmf.train.callbacks import LogCallback, PissaConvertCallback, ReporterCallback
    from l2a import run_l2a

    args = read_args()

    # Extract L2A-specific keys before HMF parses (it doesn't know about them)
    l2a_keys = {}
    if isinstance(args, dict):
        for key in ["lambda_reg", "lambda_reg_scheduler"]:
            if key in args:
                l2a_keys[key] = args.pop(key)

    model_args, data_args, training_args, finetuning_args, generating_args = get_train_args(args)

    # Inject L2A-specific args into training_args so the trainer can access them
    training_args.lambda_reg = l2a_keys.get("lambda_reg", 0.0)
    training_args.lambda_reg_scheduler = l2a_keys.get("lambda_reg_scheduler", "constant")

    callbacks = [LogCallback()]
    if finetuning_args.pissa_convert:
        callbacks.append(PissaConvertCallback())
    callbacks.append(ReporterCallback(model_args, data_args, finetuning_args, generating_args))

    from l2a.trainer import L2ASparsityCallback
    callbacks.append(L2ASparsityCallback())

    run_l2a(model_args, data_args, training_args, finetuning_args, callbacks)

    import torch.distributed as dist
    try:
        if dist.is_initialized():
            dist.destroy_process_group()
    except Exception:
        pass


def main():
    """Main entry point — handles torchrun launch if multi-GPU detected."""
    from hmf.extras.misc import find_available_port, get_device_count

    # If we're already inside a torchrun/deepspeed worker, just run training
    if os.environ.get("LOCAL_RANK") is not None:
        _run_training()
        return

    # Check if multi-GPU launch is needed
    num_devices = _get_device_count()
    if num_devices > 1:
        # Launch via torchrun
        nnodes = os.getenv("NNODES", "1")
        node_rank = os.getenv("NODE_RANK", "0")
        nproc_per_node = os.getenv("NPROC_PER_NODE", str(num_devices))
        master_addr = os.getenv("MASTER_ADDR", "127.0.0.1")
        master_port = os.getenv("MASTER_PORT", str(find_available_port()))

        env = deepcopy(os.environ)
        env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        env["TORCH_NCCL_AVOID_RECORD_STREAMS"] = "1"

        cmd = (
            f"torchrun --nnodes {nnodes} --node_rank {node_rank} "
            f"--nproc_per_node {nproc_per_node} "
            f"--master_addr {master_addr} --master_port {master_port} "
            f"{__file__} {' '.join(sys.argv[1:])}"
        )
        process = subprocess.run(cmd.split(), env=env)
        sys.exit(process.returncode)
    else:
        # Single GPU or deepspeed/torchrun already handling distribution
        _run_training()


if __name__ == "__main__":
    main()
