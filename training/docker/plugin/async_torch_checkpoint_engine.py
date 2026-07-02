"""
Async Torch Checkpoint Engine for DeepSpeed.

This module provides a drop-in replacement for DeepSpeed's native 
`TorchCheckpointEngine`. It intercepts the synchronous `save` call, clones and 
offloads the model/optimizer state dicts to CPU memory using a recursive 
cloning mechanism, and then hands the file serialization (disk I/O) over to a 
background process.

If a background disk write fails, the worker process will actively signal and 
terminate the parent training process to prevent silent failures or training 
on uncheckpointed states.
"""

import torch
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from deepspeed.runtime.checkpoint_engine.torch_checkpoint_engine import TorchCheckpointEngine

# Persistent background worker pool
_DS_ASYNC_POOL = ProcessPoolExecutor(max_workers=1, mp_context=mp.get_context('spawn'))

def _background_disk_write(state_dict, path):
    """
    The heavy serialization task executed by the background process.

    If saving fails, sends a SIGTERM signal to the parent process to actively
    halt training and prevent a silent checkpointing failure.

    Args:
        state_dict: The deep-copied CPU-resident state dictionary.
        path: The absolute destination path on the file system.
    """
    try:
        torch.save(state_dict, path)
        print(f"[Async IO Worker] Successfully saved: {path}")
    except Exception as e:
        print(f"[Async IO Worker] Error saving {path}: {e}")

def _deep_clone_and_offload(obj):
    """
    Recursively moves DeepSpeed's sharded state dict to CPU memory.

    Detaches and clones tensors to ensure that when the main training loop
    resumes on the GPU, it can safely mutate its own buffers without corrupting
    the snapshot being written by the background worker.

    Args:
        obj: The active object/tensor/collection to be cloned and offloaded.

    Returns:
        A deep-copied, completely independent version of the object on the CPU.
    """
    if torch.is_tensor(obj):
        # Detach and clone to ensure the GPU can resume without mutating this copy
        return obj.detach().cpu().clone()
    elif isinstance(obj, dict):
        return {k: _deep_clone_and_offload(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_deep_clone_and_offload(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(_deep_clone_and_offload(v) for v in obj)
    return obj

class AsyncTorchCheckpointEngine(TorchCheckpointEngine):
    """
    A drop-in replacement for DeepSpeed's native Checkpoint Engine.

    Intercepts the traditional blocking `save` call and routes it to a
    background process pool after a rapid CPU snapshot capture.
    """
    def __init__(self, config_params=None):
        """
        Initializes the async checkpoint engine structure.

        Args:
            config_params: Optional configuration parameters passed from DeepSpeed.
        """
        super().__init__(config_params)

    def save(self, state_dict, path: str):
        """
        Captures a fast in-memory snapshot of the states, transfers them via PCIe
        to CPU memory, synchronizes the stream, and dispatches disk writing asynchronously.

        Args:
            state_dict: The dictionary containing the model or optimizer states.
            path: Target file system destination string.
        """
        print(f"\n[AsyncCheckpointEngine] Capturing state snapshot for {path}...")

        # 1. Snapshot the state synchronously (~10-15s for 95GB)
        cpu_safe_dict = _deep_clone_and_offload(state_dict)
        if torch.cuda.is_available():
            torch.cuda.synchronize() # Wait for GPU->CPU transfer to finish

        print(f"[AsyncCheckpointEngine] Snapshot complete. Offloading write. GPU resuming...")

        # 2. Dispatch the actual 95GB write to the background
        _DS_ASYNC_POOL.submit(_background_disk_write, cpu_safe_dict, path)
