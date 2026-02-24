# Ensure Triton allocator is set for kernels that need runtime memory allocation.
# This is needed because vLLM's FLA ops (solve_tril, etc.) require an allocator
# but vLLM doesn't set one. With TP>1, FlashInfer sets it as a side effect,
# but with TP=1 it's never set, causing failures in eager mode.

from vllm.triton_utils import triton
import torch


def _triton_alloc(size: int, align: int, stream):
    return torch.empty(size, dtype=torch.int8, device="cuda")


if hasattr(triton, "set_allocator"):
    # Check if allocator is the default NullAllocator
    current = triton.runtime._allocation._allocator.get()
    if type(current).__name__ == "NullAllocator":
        triton.set_allocator(_triton_alloc)
