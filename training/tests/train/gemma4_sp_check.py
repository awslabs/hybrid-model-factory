"""
Checks sequence parallelism (SP) for hybrid Gemma 4 on GPUs, beyond what tests/model/test_hybrid_gemma4_sp.py checks
on the CPU: flash attention 2 with Gemma 4's window, SDPA's memory-efficient kernel with heads of 512, and GKA, GDN,
and Mamba2 layers, whose kernels need CUDA. Every rank's outputs, and the gradients of its inputs (or, summed over
the ranks, of the model's parameters), are compared with those of the whole sequence on one GPU:

    A. The attention functions at Gemma 4 E2B's sizes in bf16: sliding window attention with heads of 256, one
       key/value head, and a window of 512, and full attention with heads of 512, against flash attention 2 and
       SDPA over the whole sequence.
    B. The tiny hybrid Gemma 4 of tests/model/test_hybrid_gemma4.py in float32, with GKA, GDN, and Mamba2 at
       sliding, full, source, and sharing layers, HMF's SWA, and the two streams of Stage 1.
    C. With --model, a real hybrid Gemma 4 (such as a primed E2B) at --seq-len tokens, in --dtype (bf16 by
       default), with each decoder layer's difference, to find the first layer that diverges.

Part B, and part C in float32, need TRITON_F32_DEFAULT=ieee and NVIDIA_TF32_OVERRIDE=0 (otherwise they're skipped),
and --parts chooses the parts to run (such as --parts C).

Run from training/ with torchrun on 2 or more GPUs:

    TRITON_F32_DEFAULT=ieee NVIDIA_TF32_OVERRIDE=0 PYTHONPATH=src torchrun --nproc-per-node 2 \
        tests/train/gemma4_sp_check.py
    PYTHONPATH=src torchrun --nproc-per-node 2 tests/train/gemma4_sp_check.py --model /path/to/primed/E2B \
        --seq-len 16384
"""

import argparse
import datetime
import os
import sys
import traceback
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from flash_attn import flash_attn_func
from torch.nn.attention import SDPBackend, sdpa_kernel

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "model"))
from test_hybrid_gemma4 import TINY, VOCAB  # noqa: E402

import hmf.model.hybrid_zoo.models.model_register  # noqa: E402,F401  # Registers the HMF model families
from hmf.model.hybrid_zoo.models.decision.modeling_decision import zigzag_local  # noqa: E402
from hmf.model.hybrid_zoo.models.hybrid_gemma4.configuration_hybrid_gemma4 import HybridGemma4TextConfig  # noqa: E402
from hmf.model.hybrid_zoo.models.hybrid_gemma4.modeling_hybrid_gemma4 import HybridGemma4ForCausalLM  # noqa: E402
from hmf.model.hybrid_zoo.models.hybrid_gemma4.sequence_parallel import (  # noqa: E402
    sp_full_attention,
    sp_windowed_attention,
)
from hmf.train.sft.workflow import set_sequence_parallel_group_recursive  # noqa: E402

# The relative tolerances of bf16 kernels against each other (parts A and C), and of float32 with ieee Triton
# kernels (part B)
BF16_TOLERANCE = 2e-2
FLOAT32_TOLERANCE = 1e-3
TINY_SEQ_LEN = 1024
# Part C in bf16 reports a difference beyond BF16_TOLERANCE as a warning, which doesn't fail the run. Its fp32 run is
# the check of correctness: in fp32, an all-'*' Gemma 4 E2B agrees with and without SP within 2e-5 at every layer,
# while in bf16 its middle layers, whose fp32 differences are the largest too, magnify rounding of any size beyond
# BF16_TOLERANCE. A GKA-primed E2B can only be checked in bf16: as tested on H200 GPUs, GKA's backward kernel at
# E2B's heads of 128 needs more shared memory in fp32 than they have.
REAL_MODEL_BF16_WARNINGS_ONLY = True


def _local(tensor, rank, world, dim=1):
    return zigzag_local(tensor.transpose(1, dim), rank, world).transpose(1, dim).contiguous()


def _error(a, b):
    """The largest difference between a and b, relative to the largest magnitude of b, or to 1 if that is smaller."""
    return ((a.float() - b.float()).abs().max() / b.float().abs().max().clamp_min(1.0)).item()


def _attention_case(rank, world, group, device, full):
    """Part A: returns the differences of one attention function's output and input gradients from the kernel's."""
    gen = torch.Generator().manual_seed(0)
    length, heads, head_dim = 2 * world * 1024, 8, 512 if full else 256
    shapes = [(1, heads, length, head_dim), (1, 1, length, head_dim), (1, 1, length, head_dim)]
    query, key, value = (torch.randn(s, generator=gen).to(device, torch.bfloat16) for s in shapes)
    upstream = torch.randn(shapes[0], generator=gen).to(device, torch.bfloat16)

    inputs = [t.clone().requires_grad_() for t in (query, key, value)]
    if full:
        with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
            expected = F.scaled_dot_product_attention(
                inputs[0], inputs[1].expand(-1, heads, -1, -1), inputs[2].expand(-1, heads, -1, -1),
                is_causal=True, scale=1.0,
            )
    else:
        expected = flash_attn_func(*(t.transpose(1, 2) for t in inputs), softmax_scale=1.0, causal=True,
                                   window_size=(511, 0)).transpose(1, 2)
    (expected * upstream).sum().backward()

    local = [_local(t, rank, world, dim=2).requires_grad_() for t in (query, key, value)]
    out = sp_full_attention(*local, 1.0, group) if full else sp_windowed_attention(*local, 512, 1.0, group)
    (out * _local(upstream, rank, world, dim=2)).sum().backward()

    errors = {"output": _error(out, _local(expected, rank, world, dim=2))}
    for name, t, ref in zip(("query", "key", "value"), local, inputs):
        errors[f"{name} gradient"] = _error(t.grad, _local(ref.grad, rank, world, dim=2))
    return errors, BF16_TOLERANCE


def _gradients(model, group=None):
    grads = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad = param.grad.detach().float().clone()
            if group is not None:
                dist.all_reduce(grad, group=group)
            grads[name] = grad
    model.zero_grad(set_to_none=True)
    return grads


def _compare_model(model, rank, world, group, ids, weights, stage1=False, backbone=False, per_layer=False):
    """
    Returns the largest differences of the outputs and summed parameter gradients under SP from one GPU's. The
    outputs are the logits, every layer's six outputs of Stage 1 (stage1), or the backbone's final hidden states
    (backbone), which keep a real model's comparison clear of its logits over the whole vocabulary. With per_layer,
    it also returns each decoder layer's difference, to find the first layer that diverges.
    """
    positions = torch.arange(ids.shape[1], device=ids.device).expand(ids.shape[0], -1)
    mask = torch.ones_like(ids)
    layer_outputs = []
    hooks = [
        layer.register_forward_hook(lambda module, args, out: layer_outputs.append(out.detach()))
        for layer in (model.model.layers if per_layer else [])
    ]

    def run(ids, positions, mask, weights):
        if stage1:
            outputs = list(model.model.stage1_layer_outputs(ids, mask, positions))
            (outputs[-1][0] * weights[..., : outputs[-1][0].shape[-1]]).sum().backward()
            return torch.cat([out for layer in outputs for out in layer], dim=-1)
        if backbone:
            hidden = model.model(input_ids=ids, position_ids=positions, use_cache=False).last_hidden_state
            (hidden.float() * weights).sum().backward()
            return hidden
        logits = model(input_ids=ids, position_ids=positions, use_cache=False).logits
        (logits.float() * weights).sum().backward()
        return logits

    expected = run(ids, positions, mask, weights).detach()
    expected_grads = _gradients(model)
    expected_layers, layer_outputs[:] = list(layer_outputs), []
    set_sequence_parallel_group_recursive(model, group)
    try:
        out = run(*(_local(t, rank, world) for t in (ids, positions, mask, weights)))
        grads = _gradients(model, group)
    finally:
        set_sequence_parallel_group_recursive(model, None)
        for hook in hooks:
            hook.remove()
    errors = {
        "outputs": _error(out, _local(expected, rank, world)),
        "parameter gradients": max(_error(grads[n], expected_grads[n]) for n in expected_grads),
    }
    if per_layer:
        errors["layers"] = [_error(o, _local(e, rank, world)) for o, e in zip(layer_outputs, expected_layers)]
    return errors


def _tiny_case(rank, world, group, device, pattern, stage1=False, **extra):
    """Part B: returns the differences of the tiny model in float32 under SP from those on one GPU."""
    torch.manual_seed(0)
    config = HybridGemma4TextConfig(**{**TINY, "max_position_embeddings": 4096}, hybrid_override_pattern=pattern,
                                    **extra)
    config._attn_implementation = "sdpa"
    model = HybridGemma4ForCausalLM(config).to(device).train()
    gen = torch.Generator().manual_seed(1)
    ids = torch.randint(3, VOCAB, (2, TINY_SEQ_LEN), generator=gen).to(device)
    weights = torch.randn(2, TINY_SEQ_LEN, VOCAB, generator=gen).to(device)
    return _compare_model(model, rank, world, group, ids, weights, stage1), FLOAT32_TOLERANCE


def _real_model_case(rank, world, group, device, path, seq_len, dtype):
    """Part C: returns the differences of a real hybrid Gemma 4 under SP from those on one GPU, layer by layer."""
    model = HybridGemma4ForCausalLM.from_pretrained(path, dtype=dtype, attn_implementation="sdpa")
    model = model.to(device).train()
    gen = torch.Generator().manual_seed(1)
    ids = torch.randint(3, model.config.vocab_size, (1, seq_len), generator=gen).to(device)
    weights = torch.randn(1, seq_len, model.config.hidden_size, generator=gen).to(device)
    errors = _compare_model(model, rank, world, group, ids, weights, backbone=True, per_layer=True)
    if dtype == torch.bfloat16:
        return errors, BF16_TOLERANCE, REAL_MODEL_BF16_WARNINGS_ONLY
    return errors, FLOAT32_TOLERANCE


SWA = {"swa_config": {"window_size": 3}}
# The tiny model's layers: sliding 0 to 3 and 5 to 8, full 4 and 9; layers 6 to 9 share the keys and values of
# layer 5 (sliding) and layer 4 (full)
TINY_CASES = {
    "GKA at a sliding layer and at both sources": ("GKA-*-*-*-GKA-GKA-*-*-*-*", {}),
    "GKA at sharing layers, SWA at a sharing layer": ("*-*-*-*-*-*-GKA-SWA-*-GKA", SWA),
    "GDN and Mamba2 at sliding and full layers": ("GDN-M2-*-*-GDN-*-*-*-M2-*", {}),
    "Stage 1 with GKA* at both sources and SWA* at a sharing layer": (
        "*DA-GKA*-*DA-*DA-GKA*-GKA*-*DA-SWA*-*DA-*DA", {**SWA, "stage1": True}
    ),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None, help="a hybrid Gemma 4 checkpoint for part C")
    parser.add_argument("--seq-len", type=int, default=16384)
    parser.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16", help="part C's dtype")
    parser.add_argument("--parts", default="ABC", help="the parts to run, such as C")
    args = parser.parse_args()
    # Float32 is exact enough for the comparisons only with ieee Triton kernels and without TF32
    exact_float32 = os.environ.get("TRITON_F32_DEFAULT") == "ieee" and os.environ.get("NVIDIA_TF32_OVERRIDE") == "0"

    dist.init_process_group("nccl", timeout=datetime.timedelta(minutes=20))
    rank, world = dist.get_rank(), dist.get_world_size()
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0)))
    torch.cuda.set_device(device)
    group = dist.group.WORLD

    float32_skip = "skipped: float32 needs TRITON_F32_DEFAULT=ieee and NVIDIA_TF32_OVERRIDE=0"
    cases = {}
    if "A" in args.parts:
        cases["A. sliding window attention, bf16"] = lambda: _attention_case(rank, world, group, device, full=False)
        cases["A. full attention with heads of 512, bf16"] = lambda: _attention_case(
            rank, world, group, device, full=True
        )
    if "B" in args.parts:
        for name, (pattern, extra) in TINY_CASES.items():
            extra = dict(extra)
            stage1 = extra.pop("stage1", False)
            cases[f"B. {name}"] = (
                (lambda p=pattern, e=extra, s=stage1: _tiny_case(rank, world, group, device, p, s, **e))
                if exact_float32 else float32_skip
            )
    if "C" in args.parts and args.model:
        dtype = getattr(torch, args.dtype)
        cases[f"C. {Path(args.model).name} at {args.seq_len} tokens, {args.dtype}"] = (
            (lambda: _real_model_case(rank, world, group, device, args.model, args.seq_len, dtype))
            if dtype == torch.bfloat16 or exact_float32 else float32_skip
        )

    results = []
    for name, case in cases.items():
        if isinstance(case, str):
            results.append((name, None, case))
            continue
        try:
            errors, tolerance, *warning_only = case()
            layers = errors.pop("layers", None)
            failed = {key: value for key, value in errors.items() if value > tolerance}
            message = ", ".join(f"{key} {value:.1e}" for key, value in errors.items())
            if layers is not None:
                first = next((i for i, value in enumerate(layers) if value > tolerance), None)
                message += f"; first layer above the tolerance: {first}; layers " + " ".join(
                    f"{i}:{value:.0e}" for i, value in enumerate(layers)
                )
            ok = not failed or ("warn" if warning_only and warning_only[0] else False)
            results.append((name, ok, message))
        except Exception as error:  # noqa: BLE001  # report every case
            traceback.print_exc()
            results.append((name, False, f"{type(error).__name__}: {error}"))
        torch.cuda.empty_cache()

    # A skip (None) or a warning ("warn") doesn't fail the run
    passed = torch.tensor([all(ok is not False for _, ok, _ in results)], device=device, dtype=torch.int)
    dist.all_reduce(passed, op=dist.ReduceOp.MIN)
    if rank == 0:
        print(f"\nHybrid Gemma 4 SP check: world {world}, relative errors (largest over the outputs or gradients)")
        for name, ok, message in results:
            status = "SKIP" if ok is None else "WARN" if ok == "warn" else "PASS" if ok else "FAIL"
            print(f"  {status}  {name:<66} {message}")
        print("ALL PASSED" if passed.item() else "SOME CHECKS FAILED (on rank 0 or another rank)")
    dist.barrier()
    dist.destroy_process_group()
    return 0 if passed.item() else 1


if __name__ == "__main__":
    raise SystemExit(main())
