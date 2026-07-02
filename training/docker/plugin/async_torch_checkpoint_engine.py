import torch
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from deepspeed.runtime.checkpoint_engine.torch_checkpoint_engine import TorchCheckpointEngine

# Persistent background worker pool
_DS_ASYNC_POOL = ProcessPoolExecutor(max_workers=1, mp_context=mp.get_context('spawn'))

def _background_disk_write(state_dict, path):
    """The heavy serialization task executed by the background process."""
    try:
        torch.save(state_dict, path)
        print(f"[Async IO Worker] Successfully saved: {path}")
    except Exception as e:
        print(f"[Async IO Worker] Error saving {path}: {e}")

def _deep_clone_and_offload(obj):
    """Recursively moves DeepSpeed's sharded state dict to pinned CPU memory."""
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
    Intercepts the final `save` call and routes it to a background thread.
    """
    def __init__(self, config_params=None):
        super().__init__(config_params)

    def save(self, state_dict, path: str):
        print(f"\n[AsyncCheckpointEngine] Capturing state snapshot for {path}...")

        # 1. Snapshot the state synchronously (~10-15s for 95GB)
        cpu_safe_dict = _deep_clone_and_offload(state_dict)
        if torch.cuda.is_available():
            torch.cuda.synchronize() # Wait for GPU->CPU transfer to finish

        print(f"[AsyncCheckpointEngine] Snapshot complete. Offloading write. GPU resuming...")

        # 2. Dispatch the actual 95GB write to the background
        _DS_ASYNC_POOL.submit(_background_disk_write, cpu_safe_dict, path)
