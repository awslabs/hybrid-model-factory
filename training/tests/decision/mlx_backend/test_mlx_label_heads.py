"""
Tests for the label heads of decision models in MLX (LabelHead and LabelPointerHead in hmf/train/decision/mlx/),
with tiny Qwen3 and Qwen3.5 decision models converted with each label head. They check that MLX gives the same answers
as PyTorch, that the rows of a "label" or "label_pointer" head stay frozen while those of a "label_trained" head train,
that LoRA training gives the same adapter as the PyTorch DecisionTrainer, which PEFT then loads, and that full
fine-tuning saves a model that loads in PyTorch. Two more tests check the softcap of the logits, which no Qwen model
uses, and that a head refuses more options than it has rows.
"""

import pytest


pytest.importorskip("mlx.core", reason="needs MLX (Apple silicon)")
pytest.importorskip("mlx_lm", reason="needs mlx-lm")

import json
import os
import shutil

import mlx.core as mx
import numpy as np
import torch
from mlx.utils import tree_flatten
from peft import PeftModel
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from test_mlx_train import NUM_ROWS, _adapter, _config, _starting_adapter, decision_row
from transformers import AutoModel

from hmf.model.hybrid_zoo.models.decision.modeling_decision import LabelHead as TorchLabelHead
from hmf.train.decision.mlx.checkpoint import load_adapter
from hmf.train.decision.mlx.lora import attach_lora, lora_targets, set_trainable
from hmf.train.decision.mlx.model import LabelHead, LabelPointerHead, load_decision_model
from hmf.train.tuner import run_exp


SOURCES = ["qwen3", "qwen3_5"]
HEAD_TYPES = ["label", "label_trained", "label_pointer"]
# Each head type's frozen rows are listed by their parameter names in the MLX model
FROZEN_ROWS = {"label": ["decision_head.rows"], "label_trained": [], "label_pointer": ["decision_head.label.rows"]}


@pytest.fixture
def dataset_dir(tmp_path) -> str:
    """The tiny decision dataset of test_mlx_train.py, whose options a label head reads as A, B, and so on."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "toy.jsonl").write_text("".join(json.dumps(decision_row(i)) + "\n" for i in range(NUM_ROWS)))
    (data_dir / "dataset_info.json").write_text(
        json.dumps({"toy_decision": {"file_name": "toy.jsonl", "formatting": "decision"}})
    )
    return str(data_dir)


def _inputs(batch: dict) -> tuple:
    return tuple(mx.array(batch[name].numpy()) for name in ("input_ids", "opt_idx", "answer_idx"))


def _real(batch: dict) -> np.ndarray:
    return batch["opt_idx"].numpy() >= 0


def _torch_log_probs(model, batch: dict) -> np.ndarray:
    with torch.no_grad():
        return model(**batch).log_probs.numpy()


def _perturbed_copy(model_dir: str, target) -> str:
    """
    Copies a decision model with noise added to every weight of its head, so that the head's rows and its pointer's
    query projection, which hmf decision-init sets to 0 for a label pointer head, all change the answers.
    """
    shutil.copytree(model_dir, target)
    path = os.path.join(target, "model.safetensors")
    weights = load_file(path)
    generator = torch.Generator().manual_seed(0)
    for name, value in weights.items():
        if name.startswith("decision_head."):
            weights[name] = value + 0.1 * torch.randn(value.shape, generator=generator, dtype=value.dtype)
    save_file(weights, path, metadata={"format": "pt"})
    return str(target)


@pytest.mark.parametrize("source", SOURCES)
@pytest.mark.parametrize("head_type", HEAD_TYPES)
def test_answers_match_pytorch(decision_model_dir, batch, tmp_path, source, head_type):
    model_dir = _perturbed_copy(decision_model_dir(source, head_type), tmp_path / "decision")
    expected = _torch_log_probs(AutoModel.from_pretrained(model_dir, dtype=torch.float32).eval(), batch)

    model = load_decision_model(model_dir, dtype="float32")

    head_class = LabelPointerHead if head_type == "label_pointer" else LabelHead
    assert isinstance(model.decision_head, head_class)
    log_probs = np.array(model(*_inputs(batch))[1])
    np.testing.assert_allclose(log_probs[_real(batch)], expected[_real(batch)], atol=1e-4)


def test_softcap_matches_pytorch():
    # Qwen's LM heads have no softcap, so the tiny models above never use it
    hidden_size, num_labels, cap = 16, 6, 2.0
    torch.manual_seed(0)
    torch_head = TorchLabelHead(hidden_size, num_labels, softcap=cap)
    torch_head.copy_rows(torch.randn(num_labels, hidden_size))
    # The states are large enough that the cap changes the logits
    answer_states = 3 * torch.randn(4, hidden_size)
    option_mask = torch.tensor([[True] * 5, [True] * 3 + [False] * 2, [True] * 5, [True] * 2 + [False] * 3])
    expected = torch_head(answer_states, None, option_mask).numpy()

    head = LabelHead(hidden_size, num_labels, softcap=cap)
    head.load_weights([("rows", mx.array(torch_head.rows.numpy()))])
    logits = np.array(head(mx.array(answer_states.numpy()), None, mx.array(option_mask.numpy())))

    real = option_mask.numpy()
    np.testing.assert_allclose(logits[real], expected[real], atol=1e-5)
    assert np.abs(logits[real]).max() <= cap and np.abs(logits[real]).max() > 0.9 * cap


def test_a_label_head_refuses_more_options_than_rows():
    head = LabelHead(16, 3)

    with pytest.raises(ValueError, match="at most 3 options, got 4"):
        head(mx.zeros((2, 16)), None, mx.ones((2, 4), dtype=mx.bool_))


@pytest.mark.parametrize("head_type", HEAD_TYPES)
def test_label_rows_stay_frozen_except_with_label_trained(decision_model_dir, head_type):
    model = load_decision_model(decision_model_dir("qwen3", head_type), dtype="float32")
    head_paths = {path for path in dict(tree_flatten(model.parameters())) if path.startswith("decision_head.")}

    attach_lora(model, lora_targets(model, ["all"]), rank=4, alpha=8)
    with_lora = set(set_trainable(model, "lora"))
    assert with_lora & head_paths == head_paths - set(FROZEN_ROWS[head_type])

    full = set(set_trainable(load_decision_model(decision_model_dir("qwen3", head_type)), "full"))
    assert not full & set(FROZEN_ROWS[head_type])
    # A label head has no weights but its frozen rows, so training it alone is refused (see
    # test_a_label_head_alone_has_nothing_to_train)
    if head_type != "label":
        head_model = load_decision_model(decision_model_dir("qwen3", head_type))
        head_alone = set_trainable(head_model, "full", "decision_head")
        assert set(head_alone) == head_paths - set(FROZEN_ROWS[head_type])


def _source_head(model_dir: str) -> dict:
    """The head's weights of a decision model, by their names in a PEFT adapter."""
    with safe_open(os.path.join(model_dir, "model.safetensors"), "pt") as f:
        return {f"base_model.model.{key}": f.get_tensor(key) for key in f.keys() if key.startswith("decision_head.")}


# The pointer head's key bias adds the same <bias, q> to every option's logit, which the softmax ignores, so its
# gradient is 0 up to rounding, and Adam turns that rounding noise into updates that differ between frameworks
NULL_DIRECTION = "k.bias"


@pytest.mark.parametrize("source", SOURCES)
@pytest.mark.parametrize("head_type", HEAD_TYPES)
def test_lora_training_matches_pytorch(decision_model_dir, dataset_dir, batch, tmp_path, source, head_type):
    model_dir = decision_model_dir(source, head_type)
    _starting_adapter(model_dir, tmp_path / "start")
    common = dict(
        adapter_name_or_path=str(tmp_path / "start"), max_steps=6, gradient_accumulation_steps=2, warmup_steps=1,
        weight_decay=0.1, max_grad_norm=1.0, decision_option_labels="letters",
    )

    run_exp(_config(model_dir, dataset_dir, tmp_path / "torch", backend="torch", use_cpu=True, **common))
    run_exp(_config(model_dir, dataset_dir, tmp_path / "mlx", backend="mlx", **common))

    start, torch_adapter, mlx_adapter = (_adapter(tmp_path / name) for name in ("start", "torch", "mlx"))
    assert set(mlx_adapter) == set(torch_adapter)
    for key, value in torch_adapter.items():
        if not key.endswith(NULL_DIRECTION):
            update = np.abs(value.numpy() - start[key].numpy()).max()
            np.testing.assert_allclose(mlx_adapter[key].numpy(), value.numpy(), atol=1e-2 * update + 1e-7, err_msg=key)
    # The run saves the letter labels that the label head reads, which evaluation and serving then use
    with open(tmp_path / "mlx" / "decision_prompt.json") as f:
        assert json.load(f)["option_labels"] == "letters"
    # The frozen rows are saved with the adapter unchanged, and the trained ones have changed
    source_head = _source_head(model_dir)
    for key in source_head:
        if "rows" in key:
            unchanged = torch.equal(mlx_adapter[key], source_head[key])
            assert unchanged == (head_type != "label_trained"), key

    # PEFT loads the adapter that MLX saved, with the answers that MLX gives
    base = AutoModel.from_pretrained(model_dir, dtype=torch.float32)
    peft_model = PeftModel.from_pretrained(base, str(tmp_path / "mlx"))
    expected = _torch_log_probs(peft_model.eval(), batch)
    model = load_decision_model(model_dir, dtype="float32")
    load_adapter(model, str(tmp_path / "mlx"))
    log_probs = np.array(model(*_inputs(batch))[1])
    np.testing.assert_allclose(log_probs[_real(batch)], expected[_real(batch)], atol=1e-4)


def test_a_label_head_alone_has_nothing_to_train(decision_model_dir):
    model = load_decision_model(decision_model_dir("qwen3", "label"))

    with pytest.raises(ValueError, match="No parameter of the model is trainable"):
        set_trainable(model, "full", "decision_head")


@pytest.mark.parametrize("head_type", ["label_trained", "label_pointer"])
def test_full_fine_tuning_of_the_head_loads_in_pytorch(decision_model_dir, dataset_dir, batch, tmp_path, head_type):
    model_dir = decision_model_dir("qwen3", head_type)
    output_dir = tmp_path / "output"

    run_exp(
        _config(
            model_dir, dataset_dir, output_dir, finetuning_type="full", hybrid_learnable_params="decision_head",
            max_steps=2, decision_option_labels="letters",
        )
    )

    saved = load_file(str(output_dir / "model.safetensors"))
    source = load_file(os.path.join(model_dir, "model.safetensors"))
    assert set(saved) == set(source)
    for key in source:
        if "rows" in key:
            assert torch.equal(saved[key], source[key]) == (head_type == "label_pointer"), key
        elif not key.startswith("decision_head."):
            # The backbone is frozen, so it is saved unchanged
            assert torch.equal(saved[key], source[key]), key
    trained = AutoModel.from_pretrained(str(output_dir), dtype=torch.float32).eval()
    model = load_decision_model(str(output_dir), dtype="float32")
    log_probs = np.array(model(*_inputs(batch))[1])
    np.testing.assert_allclose(log_probs[_real(batch)], _torch_log_probs(trained, batch)[_real(batch)], atol=1e-4)
