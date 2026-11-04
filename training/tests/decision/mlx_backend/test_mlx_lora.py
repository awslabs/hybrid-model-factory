"""
Tests for the LoRA adapters and trainable parameters of the MLX backend (hmf/train/decision/mlx/lora.py), with tiny
Qwen3 and Qwen3.5 decision models: the targets of lora_target all, that a new adapter leaves the answers unchanged,
merging, the scale with rsLoRA, and which parameters each finetuning type trains.
"""

import pytest


pytest.importorskip("mlx.core", reason="needs MLX (Apple silicon)")
pytest.importorskip("mlx_lm", reason="needs mlx-lm")

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten
from transformers import AutoModel

from hmf.model.model_utils.misc import find_all_linear_modules
from hmf.train.decision.mlx.lora import attach_lora, lora_layers, lora_scale, lora_targets, merge_lora, set_trainable
from hmf.train.decision.mlx.model import load_decision_model


SOURCES = ["qwen3", "qwen3_5"]


def _inputs(batch: dict) -> tuple:
    return tuple(mx.array(batch[name].numpy()) for name in ("input_ids", "opt_idx", "answer_idx"))


def randomize_lora(model, seed: int = 0) -> None:
    """Gives every adapter a random B, so that the adapters change the answers."""
    mx.random.seed(seed)
    for _, layer in lora_layers(model):
        layer.lora_b = 0.1 * mx.random.normal(layer.lora_b.shape)


@pytest.mark.parametrize("source", SOURCES)
def test_all_targets_match_pytorch(decision_model_dir, source):
    model_dir = decision_model_dir(source)
    expected = find_all_linear_modules(AutoModel.from_pretrained(model_dir), freeze_vision_tower=True)

    assert lora_targets(load_decision_model(model_dir, dtype="float32"), ["all"]) == sorted(expected)
    assert lora_targets(load_decision_model(model_dir, dtype="float32"), ["v_proj", "q_proj"]) == ["q_proj", "v_proj"]


@pytest.mark.parametrize("source", SOURCES)
def test_a_new_adapter_leaves_the_answers_unchanged(decision_model_dir, batch, source):
    model = load_decision_model(decision_model_dir(source), dtype="float32")
    before = np.array(model(*_inputs(batch))[0])

    paths = attach_lora(model, lora_targets(model, ["all"]), rank=4, alpha=8)

    assert paths and all(path.startswith("layers.") for path in paths)
    for _, layer in lora_layers(model):
        assert layer.lora_a.dtype == mx.float32 and layer.lora_b.dtype == mx.float32
        assert layer.scale == 2.0
    np.testing.assert_allclose(np.array(model(*_inputs(batch))[0]), before, atol=1e-6)


@pytest.mark.parametrize("source", SOURCES)
def test_merging_keeps_the_answers(decision_model_dir, batch, source):
    model = load_decision_model(decision_model_dir(source), dtype="float32")
    attach_lora(model, ["q_proj", "down_proj"], rank=4, alpha=8)
    randomize_lora(model)
    with_adapters = np.array(model(*_inputs(batch))[0])

    count = merge_lora(model)

    assert count == len([1 for path, _ in tree_flatten(model.parameters()) if path.endswith("q_proj.weight")]) + 2
    assert not lora_layers(model)
    np.testing.assert_allclose(np.array(model(*_inputs(batch))[0]), with_adapters, atol=1e-5)


def test_lora_scale():
    assert lora_scale(16, 32) == 2.0
    assert lora_scale(16, 32, use_rslora=True) == 8.0


def test_no_matching_target(decision_model_dir):
    model = load_decision_model(decision_model_dir("qwen3"), dtype="float32")
    with pytest.raises(ValueError, match="No linear layer"):
        attach_lora(model, ["nothing"], rank=4, alpha=8)


@pytest.mark.parametrize("source", SOURCES)
def test_lora_trains_the_adapters_and_the_head(decision_model_dir, source):
    model = load_decision_model(decision_model_dir(source), dtype="bfloat16")
    attach_lora(model, lora_targets(model, ["all"]), rank=4, alpha=8)

    trainable = set_trainable(model, "lora")

    assert trainable == sorted(path for path, _ in tree_flatten(model.trainable_parameters()))
    assert all(path.endswith(("lora_a", "lora_b")) or path.startswith("decision_head.") for path in trainable)
    assert sum(path.startswith("decision_head.") for path in trainable) == 6
    assert all(value.dtype == mx.float32 for _, value in tree_flatten(model.trainable_parameters()))
    frozen = dict(tree_flatten(model.parameters()))
    assert frozen["model.embed_tokens.weight"].dtype == mx.bfloat16


def test_full_fine_tuning(decision_model_dir):
    model = load_decision_model(decision_model_dir("qwen3"), dtype="bfloat16")
    every = sorted(path for path, _ in tree_flatten(model.parameters()))

    assert set_trainable(model, "full") == every
    assert all(value.dtype == mx.float32 for _, value in tree_flatten(model.parameters()))


def test_head_only(decision_model_dir):
    model = load_decision_model(decision_model_dir("qwen3"), dtype="bfloat16")

    trainable = set_trainable(model, "full", hybrid_learnable_params="decision_head")

    assert len(trainable) == 6 and all(path.startswith("decision_head.") for path in trainable)
    partial = set_trainable(model, "full", hybrid_learnable_params="decision_head, mlp")
    assert any(".mlp." in path for path in partial) and not any("self_attn" in path for path in partial)


@pytest.mark.parametrize(
    "finetuning_type, kwargs, message",
    [
        ("freeze", {}, "supports finetuning_type"),
        ("lora", {}, "needs adapters"),
        ("lora", {"hybrid_learnable_params": "mlp"}, "cannot be combined"),
        ("full", {"hybrid_learnable_params": "nothing"}, "No parameter"),
    ],
)
def test_set_trainable_errors(decision_model_dir, finetuning_type, kwargs, message):
    model = load_decision_model(decision_model_dir("qwen3"), dtype="float32")
    with pytest.raises(ValueError, match=message):
        set_trainable(model, finetuning_type, **kwargs)
