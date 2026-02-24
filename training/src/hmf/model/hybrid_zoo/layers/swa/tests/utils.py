"""
Shared utilities for SWA sequence parallel tests.

Reuses the same distributed testing patterns as the GDN SP tests:
- Zig-zag local extraction for SP input distribution
- Gather and reconstruct for comparing SP vs. single-GPU outputs
- Stats printing and JSON logging

The key difference from GDN tests is that SWA requires position_embeddings
(cos/sin from RoPE) in addition to hidden_states, so the get_local_result
and get_global_result functions handle that.
"""

import datetime
import random
import argparse
from pathlib import Path
import json
from typing import Tuple

import torch
import torch.distributed as dist

from transformers import AutoConfig

from hmf.train.pt.workflow import set_sequence_parallel_group_recursive
from hmf.model.hybrid_zoo.models.model_register import register_hybrid_models
from hmf.model.hybrid_zoo.models.hybrid_qwen3.modeling_hybrid_qwen3 import (
    Qwen3RotaryEmbedding,
    Qwen3SWAttention,
)


def set_seed(rank, seed=42):
    seed = rank + seed
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def log(msg, a, rank0_only=False):
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    if rank0_only:
        if rank == 0:
            print(
                f"{msg}: max {a.abs().max().item():.3g}, mean {a.abs().mean().item():.3g}",
                flush=True,
            )
        return

    for i in range(world_size):
        if i == rank:
            if rank == 0:
                print(f"{msg}:")
            print(
                f"[{rank}] max {a.abs().max().item():.3g}, mean {a.abs().mean().item():.3g}",
                flush=True,
            )
        dist.barrier()


def extract_local(value, rank, world_size, dim=1):
    """Extract zig-zag local chunks for a given rank."""
    value_chunks = value.chunk(2 * world_size, dim=dim)
    local_value = torch.cat(
        [value_chunks[rank], value_chunks[2 * world_size - rank - 1]], dim=dim
    )
    return local_value.contiguous()


def reconstruct_from_local(local_tensors, world_size, dim=1):
    """Reconstruct global tensor from zig-zag local tensors."""
    all_chunks = [None] * (2 * world_size)

    for rank, local_tensor in enumerate(local_tensors):
        chunk_size = local_tensor.size(dim) // 2
        chunks = local_tensor.split(chunk_size, dim=dim)

        all_chunks[rank] = chunks[0]
        all_chunks[2 * world_size - rank - 1] = chunks[1]

    return torch.cat(all_chunks, dim=dim).contiguous()


def gather_from_local(local_tensor, rank, world_size, verbose=False):
    """Gather local tensors from all ranks and reconstruct global tensor on rank 0."""
    if rank == 0:
        gather_list = [torch.zeros_like(local_tensor) for _ in range(world_size)]
        dist.gather(local_tensor, gather_list, dst=0)
    else:
        dist.gather(local_tensor, None, dst=0)

    if rank == 0:
        return reconstruct_from_local(gather_list, world_size, dim=1)


def print_stats(a, b, tensor_name, logs_tensors_diffs, values=False, rank=-1):
    if values:
        print(f"#### Values rank {rank} ####")
        print(f"Rank: {rank} {tensor_name} {a}")
        print(f"Rank: {rank} {tensor_name} {b}")
        print(f"#### Values rank {rank} ####")

    max_diff = (a - b).abs().max()
    rel_diff = float(torch.norm(a - b) / torch.norm(a))

    print(f"{tensor_name} max diff: {max_diff:.4f}")
    print(f"{tensor_name} relative diff: {rel_diff * 100:.2f} %")
    logs_tensors_diffs[tensor_name] = {"max_diff": max_diff, "rel_diff": rel_diff}


def print_logs(logs_dict):
    print("\n##############")
    print("Printing Logs:")
    for tensor_name, v in logs_dict.items():
        print(f"{tensor_name} max diff: {v['max_diff']:.4f}")
        print(f"{tensor_name} relative diff: {v['rel_diff'] * 100:.2f} %")


def init_sp_group(sp_size):
    assert dist.is_initialized()
    world_size = dist.get_world_size()
    assert (
        world_size % sp_size == 0
    ), "Total number of GPUs must be a multiple of sequence_parallel_size."

    sp_group_num = world_size // sp_size
    sp_ranks_list = [
        list(range(i * sp_size, i * sp_size + sp_size)) for i in range(sp_group_num)
    ]

    sp_groups = [dist.new_group(sp_ranks_this) for sp_ranks_this in sp_ranks_list]

    global_rank_this = dist.get_rank()
    sp_idx = global_rank_this // sp_size
    return sp_groups[sp_idx]


def init_sp_distr():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    # Same seed so randomly initialized models across GPUs have the same weights
    set_seed(0)

    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")

    if rank == 0:
        print("Remark: we are using CP = World Size!!!")
    cp_size = world_size
    sp_group = init_sp_group(cp_size)
    return rank, world_size, device, cp_size, sp_group


def make_position_embeddings(
    config, seqlen: int, device: torch.device, dtype: torch.dtype
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate RoPE position embeddings (cos, sin) for a given sequence length.

    Returns:
        Tuple of (cos, sin), each of shape [1, seqlen, head_dim].
    """
    rotary_emb = Qwen3RotaryEmbedding(config, device=device)
    # Dummy hidden states just for dtype/device
    dummy = torch.zeros(1, seqlen, config.hidden_size, device=device, dtype=dtype)
    position_ids = torch.arange(seqlen, device=device).unsqueeze(0)
    cos, sin = rotary_emb(dummy, position_ids)
    return cos, sin


def get_local_result(
    dtype, input_tensor, dout_tensor, distr_configs, config, state_dict_path=None
):
    """
    Run SWA forward/backward with sequence parallelism and gather results.

    Each rank gets its zig-zag chunk of the input and position embeddings,
    runs the SP-enabled SWA forward, then gathers outputs and gradients
    back to rank 0 for comparison.
    """
    rank, world_size, device, cp_size, sp_group = distr_configs
    batch_size, seqlen, hidden_dim = input_tensor.shape
    assert seqlen % (2 * world_size) == 0

    # Create model
    local_model = Qwen3SWAttention(config=config, layer_idx=0)
    set_sequence_parallel_group_recursive(local_model, sp_group)

    if state_dict_path is not None:
        local_model.load_state_dict(torch.load(state_dict_path), strict=True)

    local_model.to(device, dtype)
    state_dict = local_model.state_dict()

    # Generate global position embeddings and broadcast
    cos, sin = make_position_embeddings(config, seqlen, device, dtype)
    dist.broadcast(cos, src=0)
    dist.broadcast(sin, src=0)

    # Broadcast and extract local input
    input_tensor = input_tensor.to(device=device, dtype=dtype)
    dist.broadcast(input_tensor, src=0)
    local_input = extract_local(input_tensor, rank, world_size).detach().clone()
    local_input.requires_grad = True

    # Extract local position embeddings
    local_cos = extract_local(cos, rank, world_size).detach().clone()
    local_sin = extract_local(sin, rank, world_size).detach().clone()

    # Extract local dout
    dout_tensor = dout_tensor.to(device=device, dtype=dtype)
    dist.broadcast(dout_tensor, src=0)
    local_dout = extract_local(dout_tensor, rank, world_size).detach().clone()

    del input_tensor, dout_tensor

    # Forward (SWA returns 3-tuple: attn_output, attn_weights, None)
    local_out, _, _ = local_model(
        hidden_states=local_input,
        position_embeddings=(local_cos, local_sin),
        attention_mask=None,
    )

    if rank == 0:
        print("##### Computed forward pass local #####")

    # Backward
    local_out.backward(local_dout)

    if rank == 0:
        print("##### Computed backward pass local #####")

    # All-reduce parameter gradients
    for name, p in local_model.named_parameters():
        if p.grad is None:
            continue
        dist.all_reduce(p.grad)

    # Gather outputs
    gathered_out = gather_from_local(local_out, rank, world_size)
    if gathered_out is not None:
        gathered_out = gathered_out.cpu()

    gathered_input_grad = gather_from_local(local_input.grad, rank, world_size)
    if gathered_input_grad is not None:
        gathered_input_grad = gathered_input_grad.cpu()

    param_grads = {}
    param_grads["input_tensor"] = gathered_input_grad

    for name, p in local_model.named_parameters():
        if p.grad is None:
            print(f"Grad on {name} is None")
            continue
        param_grads[name] = p.grad.cpu()

    dist.destroy_process_group()

    return state_dict, gathered_out, param_grads


def get_global_result(
    dtype, input_tensor, dout_tensor, distr_configs, config, state_dict
):
    """
    Run SWA forward/backward on a single GPU (no SP) as the reference.
    """
    rank, world_size, device, cp_size, sp_group = distr_configs

    # Create model without SP
    global_model = Qwen3SWAttention(config=config, layer_idx=0)
    global_model.load_state_dict(state_dict)
    global_model.to(device, dtype)

    batch_size, seqlen, hidden_dim = input_tensor.shape

    # Generate position embeddings
    cos, sin = make_position_embeddings(config, seqlen, device, dtype)

    input_tensor = input_tensor.to(device=device, dtype=dtype)

    # Forward
    global_out, _, _ = global_model(
        hidden_states=input_tensor, position_embeddings=(cos, sin), attention_mask=None
    )

    if rank == 0:
        print("##### Computed forward pass global #####")

    # Backward
    global_out.backward(dout_tensor.to(device=device, dtype=dtype))

    if rank == 0:
        print("##### Computed backward pass global #####")

    param_grads = {}
    param_grads["input_tensor"] = input_tensor.grad.cpu()

    for name, p in global_model.named_parameters():
        if p.grad is None:
            print(f"Grad on {name} is None")
            continue
        param_grads[name] = p.grad.cpu()

    dist.destroy_process_group()
    return global_out.detach().cpu(), param_grads


def str2dtype(s: str) -> torch.dtype:
    s = s.lower()
    if s in ("fp32", "float32", "f32"):
        return torch.float32
    if s in ("bf16", "bfloat16", "bf32"):
        return torch.bfloat16
    if s in ("fp16", "float16", "f16", "half"):
        return torch.float16
    raise argparse.ArgumentTypeError(
        f"Unsupported dtype '{s}'. Choose from fp32, bf16, fp16."
    )


def get_argparser():
    parser = argparse.ArgumentParser(description="ZigZag-SWA distributed sanity-check")

    parser.add_argument("--exp_name", type=str, default="swa", help="Experiment name")
    parser.add_argument("--batch_size", type=int, default=1, help="Global batch size")
    parser.add_argument(
        "--hidden_dim", type=int, default=4096, help="Model hidden dimension"
    )
    parser.add_argument("--seqlen", type=int, default=4096, help="Sequence length")
    parser.add_argument(
        "--init_state_dict_path",
        type=str,
        default=None,
        help="Path to state_dict for loading SWA parameters",
    )
    parser.add_argument(
        "--sp",
        type=int,
        default=-1,
        help="Sequence parallel [template for logging only, "
        "control SP size with --nproc-per-node]",
    )
    parser.add_argument("--head_dim", type=int, default=128, help="Per-head dimension")
    parser.add_argument(
        "--dtype",
        type=str,
        default="bf16",
        choices=["fp32", "bf16", "fp16"],
        help="Computation precision",
    )
    parser.add_argument(
        "--model_config", type=str, required=True, help="Path to model config.json"
    )

    args = parser.parse_args()
    dtype = str2dtype(args.dtype)
    return args, dtype


def to_builtin(o):
    if isinstance(o, torch.Tensor):
        return o.item() if o.numel() == 1 else o.tolist()
    if isinstance(o, dict):
        return {k: to_builtin(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [to_builtin(x) for x in o]
    return o


def logs_to_json(args, logs, exp_name="swa"):
    record = {
        "timestamp": datetime.datetime.utcnow().isoformat(timespec="seconds"),
        "run_params": {**vars(args), "dtype": str(args.dtype)},
        "metrics": to_builtin(logs),
    }

    out_dir = Path("results") / exp_name
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = f"bs{args.batch_size}_seq{args.seqlen}_sp{args.sp}_hd{args.head_dim}_{args.dtype}.json"
    with open(out_dir / fname, "w") as f:
        json.dump(record, f, indent=2)

    print(f"Wrote results to {out_dir / fname}", flush=True)
