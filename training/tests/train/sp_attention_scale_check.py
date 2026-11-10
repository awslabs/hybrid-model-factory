"""
Checks that sequence parallel attention (new_flash_attn_forward in model/model_utils/sequence_parallel.py) uses
the scale of the attention scores that it is given, as flash attention does on one GPU. Gemma 4 scales its scores
by 1, since its queries and keys are normalized, while the kernels' default is 1/sqrt(head_dim). For each scale
(the default, 1, and 0.3) and each mode (zigzag-ring, and Ulysses on exactly 2 ranks), every rank's output must
match its part of flash attention's output over the whole sequence. Run with torchrun on 2 or more GPUs:

    torchrun --nproc-per-node 2 tests/train/sp_attention_scale_check.py
"""

import datetime
import os

import torch
import torch.distributed as dist
from flash_attn import flash_attn_func

from hmf.model.hybrid_zoo.models.decision.modeling_decision import zigzag_local
from hmf.model.model_utils.sequence_parallel import new_flash_attn_forward

SEQ_LEN, HEADS, KV_HEADS, HEAD_DIM = 1024, 4, 2, 64
# bf16 outputs, whose rounding differs between one kernel call and the ring's partial results
TOLERANCE = 2e-2


def _local(tensor: torch.Tensor, mode: str, rank: int, world: int) -> torch.Tensor:
    """A rank's part of a [batch, seq_len, ...] tensor: two zig-zag chunks, or one contiguous chunk (Ulysses)."""
    if mode == "zigzag-ring":
        return zigzag_local(tensor, rank, world).contiguous()
    local = tensor.shape[1] // world
    return tensor[:, rank * local : (rank + 1) * local].contiguous()


def _relative_error(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a.float() - b.float()).abs().max() / b.float().abs().max()).item()


def main() -> int:
    dist.init_process_group("nccl", timeout=datetime.timedelta(minutes=10))
    rank, world = dist.get_rank(), dist.get_world_size()
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0)))
    torch.cuda.set_device(device)
    if world < 2:
        raise SystemExit("Needs 2 or more ranks.")

    gen = torch.Generator().manual_seed(0)
    shape_q, shape_kv = (1, SEQ_LEN, HEADS, HEAD_DIM), (1, SEQ_LEN, KV_HEADS, HEAD_DIM)
    query, key, value = (
        torch.randn(shape, generator=gen).to(device, torch.bfloat16) for shape in (shape_q, shape_kv, shape_kv)
    )

    results = []
    # The scales must change the output for the check to mean anything
    default_out = flash_attn_func(query, key, value, causal=True)
    sensitive = _relative_error(flash_attn_func(query, key, value, causal=True, softmax_scale=1.0), default_out) > 0.1
    results.append(("scale 1 changes flash attention's output", sensitive, ""))
    for mode in ("zigzag-ring", "ulysses"):
        if mode == "ulysses" and world != 2:
            results.append((f"{mode} (skipped, needs exactly 2 ranks)", True, ""))
            continue
        for scale in (None, 1.0, 0.3):
            expected = _local(flash_attn_func(query, key, value, causal=True, softmax_scale=scale), mode, rank, world)
            try:
                out = new_flash_attn_forward(
                    _local(query, mode, rank, world), _local(key, mode, rank, world),
                    _local(value, mode, rank, world), None, softmax_scale=scale, group=dist.group.WORLD, mode=mode,
                )
                error = _relative_error(out, expected)
                results.append((f"{mode} scale {scale}", error < TOLERANCE, f"relative error {error:.2e}"))
            except Exception as exc:  # noqa: BLE001  # report the error, such as a kernel that lacks the argument
                results.append((f"{mode} scale {scale}", False, f"{type(exc).__name__}: {exc}"))

    passed = torch.tensor([all(ok for _, ok, _ in results)], device=device, dtype=torch.int)
    dist.all_reduce(passed, op=dist.ReduceOp.MIN)
    if rank == 0:
        print(f"\nSP attention scale check: world {world}")
        for name, ok, message in results:
            print(f"  {'PASS' if ok else 'FAIL'}  {name:<44} {message}")
        print("ALL PASSED" if passed.item() else "SOME CHECKS FAILED (on rank 0 or another rank)")
    dist.barrier()
    dist.destroy_process_group()
    return 0 if passed.item() else 1


if __name__ == "__main__":
    raise SystemExit(main())
