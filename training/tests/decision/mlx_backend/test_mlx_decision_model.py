"""
Tests for the decision model of the MLX backend (hmf/train/decision/mlx/backbone.py and model.py), with tiny Qwen3 and
Qwen3.5 decision models: that it answers as the PyTorch DecisionModel does on the same checkpoint, that Qwen3.5 gives
the same answers in training mode (the chunked Gated DeltaNet rule) as in evaluation mode (mlx-lm's Metal kernel), the
dtypes and the frozen backbone, the head's dropout, and the checkpoints it refuses.
"""

import pytest


pytest.importorskip("mlx.core", reason="needs MLX (Apple silicon)")
pytest.importorskip("mlx_lm", reason="needs mlx-lm")

import json
import shutil

import mlx.core as mx
import numpy as np
import torch
from mlx import nn
from mlx.utils import tree_flatten
from transformers import AutoModel

from hmf.train.decision.mlx.backbone import _layer_pattern, _rope_settings, load_backbone
from hmf.train.decision.mlx.model import MASK_VALUE, load_decision_model


SOURCES = ["qwen3", "qwen3_5"]


def _mlx_inputs(batch: dict) -> tuple:
    return tuple(mx.array(batch[name].numpy()) for name in ("input_ids", "opt_idx", "answer_idx"))


def _torch_outputs(model_dir: str, batch: dict):
    model = AutoModel.from_pretrained(model_dir, dtype=torch.float32).eval()
    with torch.no_grad():
        return model(**batch)


@pytest.mark.parametrize("source", SOURCES)
def test_answers_match_pytorch(decision_model_dir, batch, source):
    model_dir = decision_model_dir(source)
    expected = _torch_outputs(model_dir, batch)

    logits, log_probs = load_decision_model(model_dir, dtype="float32")(*_mlx_inputs(batch))

    real = batch["opt_idx"].numpy() >= 0
    np.testing.assert_allclose(np.array(logits), expected.logits.numpy(), atol=1e-4)
    np.testing.assert_allclose(np.array(log_probs)[real], expected.log_probs.numpy()[real], atol=1e-4)
    assert np.all(np.array(logits)[~real] == MASK_VALUE)
    assert np.all(np.isneginf(np.array(log_probs)[~real]))


@pytest.mark.parametrize("source", SOURCES)
def test_bfloat16_backbone(decision_model_dir, batch, source):
    model_dir = decision_model_dir(source)
    model = load_decision_model(model_dir, dtype="bfloat16")

    dtypes = {path: value.dtype for path, value in tree_flatten(model.parameters())}
    for path, dtype in dtypes.items():
        if path.startswith("decision_head.") or path.endswith("A_log"):
            assert dtype == mx.float32, path
        else:
            assert dtype == mx.bfloat16, path
    _, log_probs = model(*_mlx_inputs(batch))
    _, log_probs_fp32 = load_decision_model(model_dir, dtype="float32")(*_mlx_inputs(batch))
    real = batch["opt_idx"].numpy() >= 0
    assert np.array(log_probs).dtype == np.float32
    np.testing.assert_allclose(np.array(log_probs)[real], np.array(log_probs_fp32)[real], atol=5e-2)


@pytest.mark.parametrize("source", SOURCES)
def test_only_the_head_is_trainable(decision_model_dir, source):
    model = load_decision_model(decision_model_dir(source), dtype="float32")

    trainable = [path for path, _ in tree_flatten(model.trainable_parameters())]

    assert trainable and all(path.startswith("decision_head.") for path in trainable)


def test_training_mode_matches_evaluation_mode(decision_model_dir, batch):
    model = load_decision_model(decision_model_dir("qwen3_5"), dtype="float32")  # the head has no dropout
    inputs = _mlx_inputs(batch)

    eval_logits, _ = model(*inputs)
    model.train()
    train_logits, _ = model(*inputs)

    np.testing.assert_allclose(np.array(train_logits), np.array(eval_logits), atol=1e-4)


def test_head_dropout_applies_in_training_mode_only(decision_model_dir, batch):
    model = load_decision_model(decision_model_dir("qwen3"), dtype="float32")
    model.decision_head.dropout = nn.Dropout(0.5)
    model.eval()  # a new module starts in training mode
    inputs = _mlx_inputs(batch)

    eval_logits = [np.array(model(*inputs)[0]) for _ in range(2)]
    model.train()
    train_logits = np.array(model(*inputs)[0])

    np.testing.assert_array_equal(eval_logits[0], eval_logits[1])
    assert not np.allclose(train_logits, eval_logits[0])


def test_refuses_a_model_that_is_not_a_decision_model(decision_model_dir):
    source_dir = decision_model_dir("qwen3").replace("/decision", "/source")

    with pytest.raises(ValueError, match="not a decision model"):
        load_backbone(source_dir)


def test_refuses_an_unsupported_backbone(decision_model_dir, tmp_path):
    model_dir = tmp_path / "decision"
    shutil.copytree(decision_model_dir("qwen3"), model_dir)
    config = json.loads((model_dir / "config.json").read_text())
    config["text_config"]["model_type"] = "llama"
    (model_dir / "config.json").write_text(json.dumps(config))

    with pytest.raises(ValueError, match="supports decision models on"):
        load_backbone(str(model_dir))


def test_refuses_an_unknown_dtype(decision_model_dir):
    with pytest.raises(ValueError, match="no dtype"):
        load_backbone(decision_model_dir("qwen3"), dtype="int8")


def test_layer_pattern():
    layer_types = ["linear_attention", "linear_attention", "linear_attention", "full_attention"] * 2
    assert _layer_pattern({"layer_types": layer_types})["full_attention_interval"] == 4
    assert _layer_pattern({"hidden_size": 8}) == {"hidden_size": 8}
    with pytest.raises(ValueError, match="every k layers"):
        _layer_pattern({"layer_types": ["full_attention", "linear_attention", "linear_attention", "full_attention"]})


def test_rope_settings():
    default = {"rope_parameters": {"rope_type": "default", "rope_theta": 1e6}}
    assert _rope_settings(default) == {**default, "rope_theta": 1e6, "rope_scaling": None}
    yarn = {"rope_parameters": {"rope_type": "yarn", "rope_theta": 1e6, "factor": 4.0}}
    assert _rope_settings(yarn)["rope_scaling"] == yarn["rope_parameters"]
    assert _rope_settings({"rope_theta": 1e4}) == {"rope_theta": 1e4}
