"""
Checks, on causal LMs, the check that the SFT and pre-training trainers run at their first batch, to
make sure that every layer of the backbone supports sequence parallelism (SP). The check is
check_sequence_parallel_backbone, in model/model_utils/sequence_parallel_check.py. Run with torchrun
on 2 or more GPUs:

    torchrun --nproc-per-node 2 tests/train/sp_backbone_check.py

The test models are small causal LMs with random weights, with two to four layers and a hidden size
of 128. With zigzag-ring, stock Qwen3 and an HMF hybrid with GKA must pass. Stock Qwen3.5 must be
rejected for its Gated DeltaNet layers, Qwen3 with a sliding window for its sliding-window attention,
and Gemma 2 for its attention softcapping (and for the sliding window of its even layers). With
Ulysses, which this test runs only on exactly 2 ranks, stock Qwen3 must still pass, and the HMF hybrid must be
rejected for its GKA layer, which supports zigzag-ring only. For each rejected model, the error that
the trainers raise must name the layers and say how to fix it. For every model, the check must run
exactly one forward pass in eval mode, as the SFT and pre-training trainers rely on it as their first
forward pass.

A model whose layers cannot be imported is skipped.
"""

import datetime
import os
import traceback
from types import SimpleNamespace
from typing import Callable, Dict, List, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn

import hmf.model.hybrid_zoo.models.model_register  # noqa: F401  # Registers HMF's models
from hmf.model.hybrid_zoo.models.decision.modeling_decision import zigzag_local
from hmf.model.model_utils.sequence_parallel import apply_sequence_parallel
from hmf.model.model_utils.sequence_parallel_check import (
    NO_SEQUENCE_PARALLELISM,
    SLIDING_WINDOW,
    SOFTCAPPING,
    SP_ATTENTION,
    ZIGZAG_RING_ONLY,
    check_sequence_parallel_backbone,
    describe_backbone_problems,
    sequence_parallel_backbone_problems,
)
from hmf.train.sft.workflow import set_sequence_parallel_group_recursive


VOCAB_SIZE = 64
SEQ_LEN = 512
# Ulysses splits the attention heads across the ranks, so the test models have enough heads to split
# (4 attention heads and 2 key-value heads)
SIZES = {"hidden_size": 128, "intermediate_size": 256, "num_attention_heads": 4, "num_key_value_heads": 2,
         "head_dim": 32, "vocab_size": VOCAB_SIZE, "tie_word_embeddings": True}


# Models


def _qwen3(**kwargs):
    from transformers import Qwen3Config

    return Qwen3Config(num_hidden_layers=2, **SIZES, **kwargs)


def _qwen3_5():
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

    return Qwen3_5TextConfig(
        num_hidden_layers=4, linear_num_key_heads=2, linear_num_value_heads=2, linear_key_head_dim=64,
        linear_value_head_dim=64, layer_types=["linear_attention"] * 3 + ["full_attention"], **SIZES,
    )


def _gemma2():
    from transformers import Gemma2Config

    # Gemma 2 alternates sliding-window and full attention, starting with a sliding window on layer 0
    return Gemma2Config(num_hidden_layers=2, attn_logit_softcapping=50.0, sliding_window=64,
                        query_pre_attn_scalar=32, **SIZES)


def _hmf_gka():
    from hmf.model.hybrid_zoo.models.hybrid_qwen3.configuration_hybrid_qwen3 import HybridQwen3Config

    return HybridQwen3Config(num_hidden_layers=2, hybrid_override_pattern="GKA-*", **SIZES)


# For each model and mode, the problems that sequence_parallel_backbone_problems must report, as {reason: layers}
CASES: List[Tuple[str, Callable, str, Dict[str, List[int]]]] = [
    ("qwen3", _qwen3, "zigzag-ring", {}),
    ("hmf_gka", _hmf_gka, "zigzag-ring", {}),
    ("qwen3_5", _qwen3_5, "zigzag-ring", {NO_SEQUENCE_PARALLELISM: [0, 1, 2]}),
    ("qwen3_sliding", lambda: _qwen3(use_sliding_window=True, sliding_window=64, max_window_layers=0),
     "zigzag-ring", {SLIDING_WINDOW: [0, 1]}),
    ("gemma2", _gemma2, "zigzag-ring", {SOFTCAPPING: [0, 1], SLIDING_WINDOW: [0]}),
    ("qwen3", _qwen3, "ulysses", {}),
    ("hmf_gka", _hmf_gka, "ulysses", {ZIGZAG_RING_ONLY: [0]}),
]


def build_model(config_fn: Callable, device: torch.device) -> nn.Module:
    """A causal LM with random weights, the same on every rank, that runs ring attention."""
    from transformers import AutoModelForCausalLM

    config = config_fn()
    config._attn_implementation = SP_ATTENTION
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(config).to(device=device, dtype=torch.bfloat16)
    for tensor in list(model.parameters()) + list(model.buffers()):
        dist.broadcast(tensor.data, src=0)
    return model


def make_inputs(mode: str, sp_rank: int, sp_size: int, device: torch.device) -> Dict[str, torch.Tensor]:
    """
    One rank's part of a batch of two sequences, as the SP data pipeline prepares it. Zigzag-ring
    gives each rank two zig-zag chunks, and Ulysses gives it one contiguous chunk.
    """
    gen = torch.Generator().manual_seed(0)
    input_ids = torch.randint(2, VOCAB_SIZE, (2, SEQ_LEN), generator=gen)
    position_ids = torch.arange(SEQ_LEN).expand(2, -1)
    if mode == "zigzag-ring":
        input_ids, position_ids = zigzag_local(input_ids, sp_rank, sp_size), zigzag_local(position_ids, sp_rank, sp_size)
    else:
        local = SEQ_LEN // sp_size
        input_ids = input_ids[:, sp_rank * local : (sp_rank + 1) * local]
        position_ids = position_ids[:, sp_rank * local : (sp_rank + 1) * local]
    return {"input_ids": input_ids.to(device), "position_ids": position_ids.contiguous().to(device)}


def run_case(name: str, config_fn: Callable, mode: str, expected: Dict[str, List[int]], group, sp_size: int,
             device: torch.device) -> Tuple[bool, str]:
    model = build_model(config_fn, device)
    set_sequence_parallel_group_recursive(model, group)
    model.train()  # as in the trainers, which check in train mode
    inputs = make_inputs(mode, dist.get_rank(group), sp_size, device)

    forward_passes = []

    def run_forward():
        forward_passes.append(model.training)
        return model(**inputs)

    # Called as the SFT and pre-training trainers call it, since they rely on its forward pass in eval mode
    problems = sequence_parallel_backbone_problems(model, run_forward, mode, always_run_forward=True)
    if forward_passes != [False]:
        return False, f"expected one forward pass in eval mode, got {len(forward_passes)} (training: {forward_passes})"
    found: Dict[str, List[int]] = {}
    for index, _, reason in problems:
        found.setdefault(reason, []).append(index)
    lines = describe_backbone_problems(problems)
    if found != expected:
        return False, f"found {found}, expected {expected}: {lines}"
    if not model.training:
        return False, "the check left the model in eval mode"

    # The error the trainers raise names the layers, and says how to fix it
    try:
        check_sequence_parallel_backbone(model, run_forward, mode, always_run_forward=True)
    except ValueError as error:
        message = str(error)
        if not expected:
            return False, f"raised an error, although nothing is wrong: {message}"
        fix = "zigzag-ring" if ZIGZAG_RING_ONLY in found else "hmf prime-init"
        if any(line not in message for line in lines) or fix not in message:
            return False, f"the error does not name the layers or does not mention {fix!r}: {message}"
        return True, f"rejected: {lines}"
    if expected:
        return False, "no error raised"
    return True, "no problems"


def main() -> int:
    dist.init_process_group("nccl", timeout=datetime.timedelta(minutes=10))
    rank, world = dist.get_rank(), dist.get_world_size()
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0)))
    torch.cuda.set_device(device)
    if world < 2:
        raise SystemExit("Needs 2 or more ranks.")

    results = []
    for mode in ("zigzag-ring", "ulysses"):
        if mode == "ulysses" and world != 2:
            results.append(("(all)", mode, True, "skipped (needs exactly 2 ranks)"))
            continue
        try:
            group = apply_sequence_parallel(SimpleNamespace(sequence_parallel_size=world, sequence_parallel_mode=mode))
        except ImportError as error:
            results.append(("(all)", mode, True, f"skipped ({error})"))
            continue
        for name, config_fn, case_mode, expected in CASES:
            if case_mode != mode:
                continue
            try:
                passed, message = run_case(name, config_fn, mode, expected, group, world, device)
            except ImportError as error:
                passed, message = True, f"skipped ({error})"
            except Exception as error:
                passed, message = False, f"{type(error).__name__}: {error}"
                traceback.print_exc()
            results.append((name, mode, passed, message))
            torch.cuda.empty_cache()

    passed = torch.tensor([all(r[2] for r in results)], device=device, dtype=torch.int)
    dist.all_reduce(passed, op=dist.ReduceOp.MIN)
    if rank == 0:
        print(f"\nSP backbone check on causal LMs: world {world}")
        for name, mode, ok, message in results:
            print(f"  {'PASS' if ok else 'FAIL'}  {name:<14} {mode:<12} {message}")
        print("ALL PASSED" if passed.item() else "SOME CHECKS FAILED (on rank 0 or another rank)")
    dist.barrier()
    dist.destroy_process_group()
    return 0 if passed.item() else 1


if __name__ == "__main__":
    raise SystemExit(main())
