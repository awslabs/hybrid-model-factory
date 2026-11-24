"""
Tests sequence parallelism (SP) for hybrid Gemma 4's attention (models/hybrid_gemma4/sequence_parallel.py): with
the sequence split in HMF's zigzag layout across 2 and 4 processes, each process's outputs, and the gradients of its
inputs (or, summed over the processes, of the model's parameters), must match those of the whole sequence in one
process. The attention functions are tested alone, with grouped-query attention, several scales and windows,
windows that reach back exactly one chunk, and chunks too short for the window, and then inside the tiny hybrid
Gemma 4 of test_hybrid_gemma4.py: its sliding and full layers, the layers that share keys and values, HMF's SWA at
plain, sharing, and source layers, and the two streams of Stage 1. GKA, GDN, and Mamba2 need CUDA, so a GPU check
covers them. The tests run on the CPU in float64, with gloo, whose processes meet through a file and talk over the
loopback interface only.
"""

import datetime
import os
import platform
import sys
import traceback
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_hybrid_gemma4 import TINY, VOCAB  # noqa: E402

from hmf.model.hybrid_zoo.models.decision.modeling_decision import zigzag_local  # noqa: E402
from hmf.model.hybrid_zoo.models.hybrid_gemma4.configuration_hybrid_gemma4 import HybridGemma4TextConfig  # noqa: E402
from hmf.model.hybrid_zoo.models.hybrid_gemma4.modeling_hybrid_gemma4 import (  # noqa: E402
    Gemma4RMSNorm,
    HybridGemma4ForCausalLM,
)
from hmf.model.hybrid_zoo.models.hybrid_gemma4.sequence_parallel import (  # noqa: E402
    sp_full_attention,
    sp_windowed_attention,
)
from hmf.train.sft.workflow import set_sequence_parallel_group_recursive  # noqa: E402

TOLERANCE = 1e-9
# Long enough for chunks of 4 tokens at 4 processes, since the tiny model's window of 4 reaches back 3 tokens
SEQ_LEN = 32


def _local(tensor: torch.Tensor, rank: int, world: int, dim: int = 1) -> torch.Tensor:
    """Returns a process's two zigzag chunks of tensor along dim."""
    return zigzag_local(tensor.transpose(1, dim), rank, world).transpose(1, dim)


def _error(a: torch.Tensor, b: torch.Tensor) -> float:
    """
    The largest difference between a and b, relative to the largest magnitude of b, or to 1 if that is smaller (as
    for the query and key gradients of a window of 1, which are zero).
    """
    return ((a - b).abs().max() / b.abs().max().clamp_min(1.0)).item()


def _norm_in_input_dtype(self, hidden_states: torch.Tensor) -> torch.Tensor:
    """Runs Gemma4RMSNorm.forward without its cast to float32, so that a float64 model computes in float64."""
    normed_output = self._norm(hidden_states)
    return normed_output * self.weight if self.with_scale else normed_output


def _reference_attention(query, key, value, scale, window=None):
    """Returns the causal attention over the whole sequence, with a window if given, as in transformers' Gemma 4."""
    length = query.shape[2]
    positions = torch.arange(length)
    mask = positions[None, :] <= positions[:, None]
    if window is not None:
        mask &= positions[:, None] - positions[None, :] < window
    n_rep = query.shape[1] // key.shape[1]
    key, value = key.repeat_interleave(n_rep, dim=1), value.repeat_interleave(n_rep, dim=1)
    return F.scaled_dot_product_attention(query, key, value, attn_mask=mask, scale=scale)


def _attention_case(rank, world, group, kv_heads, scale, window, chunk_len=5):
    """Returns the largest differences of one attention function's output and input gradients from the reference."""
    gen = torch.Generator().manual_seed(0)
    length = 2 * world * chunk_len
    query = torch.randn(2, 4, length, 8, generator=gen, dtype=torch.float64)
    key, value = (torch.randn(2, kv_heads, length, 8, generator=gen, dtype=torch.float64) for _ in range(2))
    upstream = torch.randn(2, 4, length, 8, generator=gen, dtype=torch.float64)

    inputs = [t.clone().requires_grad_() for t in (query, key, value)]
    expected = _reference_attention(*inputs, scale, window)
    (expected * upstream).sum().backward()

    local = [_local(t, rank, world, dim=2).clone().requires_grad_() for t in (query, key, value)]
    if window is None:
        out = sp_full_attention(*local, scale, group)
    else:
        out = sp_windowed_attention(*local, window, scale, group)
    (out * _local(upstream, rank, world, dim=2)).sum().backward()

    errors = {"output": _error(out, _local(expected, rank, world, dim=2))}
    for name, t, ref in zip(("query", "key", "value"), local, inputs):
        errors[f"{name} gradient"] = _error(t.grad, _local(ref.grad, rank, world, dim=2))
    return errors


def _model(pattern, **extra):
    torch.manual_seed(0)
    config = HybridGemma4TextConfig(**TINY, hybrid_override_pattern=pattern, **extra)
    # SDPA, since eager attention computes its softmax in float32 even for float64 inputs
    config._attn_implementation = "sdpa"
    return HybridGemma4ForCausalLM(config).double().train()


def _model_inputs():
    gen = torch.Generator().manual_seed(1)
    ids = torch.randint(3, VOCAB, (2, SEQ_LEN), generator=gen)
    weights = torch.randn(2, SEQ_LEN, VOCAB, generator=gen, dtype=torch.float64)
    return ids, weights


def _parameter_gradients(model, group=None):
    grads = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad = param.grad.clone()
            if group is not None:
                dist.all_reduce(grad, group=group)
            grads[name] = grad
    model.zero_grad()
    return grads


def _model_case(rank, world, group, pattern, **extra):
    """
    Returns the largest differences of a model's logits under SP, and of its parameters' gradients summed over the
    processes, from those of the whole sequence in one process.
    """
    model = _model(pattern, **extra)
    ids, weights = _model_inputs()
    expected = model(input_ids=ids, use_cache=False).logits
    (expected * weights).sum().backward()
    expected_grads = _parameter_gradients(model)

    set_sequence_parallel_group_recursive(model, group)
    positions = torch.arange(SEQ_LEN).expand(2, -1)
    logits = model(input_ids=_local(ids, rank, world), position_ids=_local(positions, rank, world),
                   use_cache=False).logits
    (logits * _local(weights, rank, world)).sum().backward()
    grads = _parameter_gradients(model, group)
    set_sequence_parallel_group_recursive(model, None)

    return {
        "logits": _error(logits, _local(expected, rank, world)),
        "parameter gradients": max(_error(grads[n], expected_grads[n]) for n in expected_grads),
        "parameters with gradients": float(set(grads) != set(expected_grads)),
    }


def _stage1_case(rank, world, group, pattern, **extra):
    """
    Returns the largest differences of a Stage 1 model's six outputs at every layer under SP, and of the
    gradients of its parameters through the hybrid stream, from those of the whole sequence in one process.
    """
    model = _model(pattern, **extra)
    ids, weights = _model_inputs()
    hidden_weights = weights[..., : TINY["hidden_size"]]
    attention_mask = torch.ones_like(ids)

    expected = list(model.model.stage1_layer_outputs(ids, attention_mask))
    (expected[-1][0] * hidden_weights).sum().backward()
    expected_grads = _parameter_gradients(model)

    set_sequence_parallel_group_recursive(model, group)
    positions = torch.arange(SEQ_LEN).expand(2, -1)
    outputs = list(model.model.stage1_layer_outputs(
        _local(ids, rank, world), _local(attention_mask, rank, world), _local(positions, rank, world)
    ))
    (outputs[-1][0] * _local(hidden_weights, rank, world)).sum().backward()
    grads = _parameter_gradients(model, group)
    set_sequence_parallel_group_recursive(model, None)

    return {
        "layer outputs": max(
            _error(out, _local(ref, rank, world))
            for layer_out, layer_ref in zip(outputs, expected) for out, ref in zip(layer_out, layer_ref)
        ),
        "parameter gradients": max(_error(grads[n], expected_grads[n]) for n in expected_grads),
    }


def _too_short_chunk_case(rank, world, group):
    """
    Returns {"refused": 0.0} if sp_windowed_attention refuses chunks shorter than window - 1 with an error that
    names the window, and {"refused": 1.0} otherwise, so that a failure shows as a difference above the tolerance.
    """
    tensor = torch.zeros(1, 1, 2 * 3, 8, dtype=torch.float64)
    try:
        sp_windowed_attention(tensor, tensor, tensor, 5, 1.0, group)
    except ValueError as error:
        return {"refused": float("window - 1 = 4" not in str(error))}
    return {"refused": 1.0}


# The tiny model's layers: sliding 0 to 3 and 5 to 8, full 4 and 9; layers 6 to 9 share the keys and values of
# layer 5 (sliding) and layer 4 (full)
SWA = {"swa_config": {"window_size": 3}}
CASES = {
    "full attention": lambda r, w, g: _attention_case(r, w, g, kv_heads=1, scale=1.0, window=None),
    "full attention with two key/value heads and scale 0.3": lambda r, w, g: _attention_case(
        r, w, g, kv_heads=2, scale=0.3, window=None
    ),
    "window of 1": lambda r, w, g: _attention_case(r, w, g, kv_heads=1, scale=1.0, window=1),
    "window of 3": lambda r, w, g: _attention_case(r, w, g, kv_heads=2, scale=1.0, window=3),
    "window reaching back one whole chunk": lambda r, w, g: _attention_case(r, w, g, kv_heads=1, scale=0.3, window=6),
    "chunks shorter than the window": _too_short_chunk_case,
    "Gemma 4 (every layer '*')": lambda r, w, g: _model_case(r, w, g, "-".join(["*"] * 10)),
    "SWA at a sliding, a full, and a sharing layer": lambda r, w, g: _model_case(
        r, w, g, "*-SWA-*-*-SWA-*-*-SWA-*-*", **SWA
    ),
    "SWA at both sources and a sharing full layer": lambda r, w, g: _model_case(
        r, w, g, "*-*-*-*-SWA-SWA-*-*-*-SWA", **SWA
    ),
    "Stage 1 ('*DA', and 'SWA*' at a source and a sharing layer)": lambda r, w, g: _stage1_case(
        r, w, g, "*DA-*DA-*DA-*DA-*DA-SWA*-*DA-SWA*-*DA-*DA", **SWA
    ),
}


def _worker(rank, world, store_path, results_dir):
    # Gemma's norms compute in float32, which would hide differences smaller than float32's rounding
    Gemma4RMSNorm.forward = _norm_in_input_dtype
    os.environ["GLOO_SOCKET_IFNAME"] = "lo0" if platform.system() == "Darwin" else "lo"
    dist.init_process_group(
        "gloo", init_method=f"file://{store_path}", rank=rank, world_size=world,
        timeout=datetime.timedelta(seconds=120),
    )
    group = dist.group.WORLD
    lines = []
    for name, case in CASES.items():
        try:
            errors = case(rank, world, group)
            failed = {key: value for key, value in errors.items() if value > TOLERANCE}
            if failed:
                lines.append(f"{name}: {failed}")
        except Exception:  # noqa: BLE001  # record the error, so the test reports every case
            lines.append(f"{name}: {traceback.format_exc()}")
    Path(results_dir, f"{rank}.txt").write_text("\n".join(lines))
    dist.destroy_process_group()


@pytest.mark.parametrize("world", [2, 4])
def test_sequence_parallel_attention_matches_the_whole_sequence_in_one_process(world, tmp_path):
    mp.spawn(_worker, args=(world, str(tmp_path / "store"), str(tmp_path)), nprocs=world, join=True)
    failures = [f"rank {rank}: {line}" for rank in range(world)
                for line in Path(tmp_path, f"{rank}.txt").read_text().splitlines() if line]
    assert not failures, "\n".join(failures)
