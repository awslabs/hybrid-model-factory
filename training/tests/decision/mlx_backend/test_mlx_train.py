"""
Tests for training decision models with MLX (hmf/train/decision/mlx/train.py, decision_backend: mlx), with tiny Qwen3
and Qwen3.5 decision models and a tiny decision dataset: the learning rate schedule, gradient checkpointing, that a
run trains the same adapter as the PyTorch DecisionTrainer does from the same start, a run end to end with
evaluations and checkpoints, resuming, training the head alone, and the options the MLX backend refuses.
"""

import pytest


pytest.importorskip("mlx.core", reason="needs MLX (Apple silicon)")
pytest.importorskip("mlx_lm", reason="needs mlx-lm")

import json
import os

import mlx.core as mx
import numpy as np
import torch
from mlx import nn
from mlx.utils import tree_flatten
from peft import LoraConfig, get_peft_model
from safetensors import safe_open
from transformers import AutoModel, get_scheduler

from hmf.eval.decision.predictor import DecisionPredictor
from hmf.train.decision.mlx.backbone import KeyedDropout, set_gradient_checkpointing
from hmf.train.decision.mlx.lora import attach_lora, lora_layers, lora_targets, set_trainable
from hmf.train.decision.mlx.loss import decision_loss
from hmf.train.decision.mlx.model import load_decision_model
from hmf.train.decision.mlx.train import SCHEDULERS, lr_factor
from hmf.train.tuner import run_exp


NUM_ROWS = 28  # 7 batches of 4 rows, so with 2 batches per step, the last step of an epoch has one batch


def decision_row(i: int) -> dict:
    """A decision row, whose kind cycles through noul, choice, and score."""
    kind = ("noul", "choice", "score")[i % 3]
    if kind == "noul":
        options, label = [["false", None], ["true", "it holds"]], i % 2
    elif kind == "choice":
        options, label = [["refund", "return it"], ["ignore", None], ["escalate", "to a human"]], i % 3
    else:
        options, label = [["0", "low"], ["1", "mid"], ["2", "high"], ["3", "top"]], i % 4
    return {
        "kind": kind,
        "state": f"state {' '.join(['words'] * (1 + i % 7))} {i}",
        "instructions": f"question {i}?",
        "instruction_variants": [f"other question {i}?"],
        "options": options,
        "label": label,
        "task": ("a", "b")[i % 2],
    }


@pytest.fixture
def dataset_dir(tmp_path) -> str:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "toy.jsonl").write_text("".join(json.dumps(decision_row(i)) + "\n" for i in range(NUM_ROWS)))
    (data_dir / "dataset_info.json").write_text(
        json.dumps({"toy_decision": {"file_name": "toy.jsonl", "formatting": "decision"}})
    )
    return str(data_dir)


def _config(model_dir: str, dataset_dir: str, output_dir, backend: str = "mlx", **overrides) -> dict:
    config = {
        "model_name_or_path": model_dir,
        "stage": "decision",
        "do_train": True,
        "finetuning_type": "lora",
        "lora_target": "all",
        "lora_rank": 4,
        "dataset": "toy_decision",
        "dataset_dir": dataset_dir,
        "cutoff_len": 512,
        "decision_num_augmentations": 2,
        "decision_pad_to_multiple_of": 16,
        "per_device_train_batch_size": 4,
        "learning_rate": 1e-3,
        "decision_head_lr": 1e-2,
        "save_strategy": "no",
        "logging_steps": 1,
        "output_dir": str(output_dir),
        "overwrite_output_dir": True,
        "overwrite_cache": True,
        "report_to": "none",
        "seed": 0,
        "decision_backend": backend,
    }
    config.update(overrides)
    return config


def _adapter(path) -> dict:
    with safe_open(os.path.join(path, "adapter_model.safetensors"), "pt") as f:
        return {key: f.get_tensor(key) for key in f.keys()}


def _log_history(output_dir) -> list:
    with open(os.path.join(output_dir, "trainer_state.json")) as f:
        return json.load(f)["log_history"]


@pytest.mark.parametrize("scheduler", SCHEDULERS)
def test_learning_rate_schedule_matches_transformers(scheduler):
    optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=1.0)
    schedule = get_scheduler(scheduler, optimizer, num_warmup_steps=3, num_training_steps=10)
    for step in range(10):
        assert lr_factor(step, 10, 3, scheduler) == pytest.approx(schedule.get_last_lr()[0]), step
        optimizer.step()
        schedule.step()


@pytest.mark.parametrize("source", ["qwen3", "qwen3_5"])
def test_gradient_checkpointing_gives_the_same_gradients(decision_model_dir, batch, source):
    # With LoRA dropout, which mx.checkpoint's rerun of a layer would draw again without the layer call's key
    model = load_decision_model(decision_model_dir(source), dtype="float32")
    attach_lora(model, lora_targets(model, ["all"]), rank=4, alpha=8, dropout=0.3)
    set_trainable(model, "lora")
    for _, layer in lora_layers(model):  # adapters that change the answers, so that their dropout matters
        layer.lora_b = 0.1 * mx.random.normal(layer.lora_b.shape)
    model.train()
    inputs = {name: mx.array(batch[name].numpy()) for name in ("input_ids", "opt_idx", "answer_idx")}
    label_dist = (mx.array(batch["opt_idx"].numpy()) >= 0).astype(mx.float32)
    targets = {"label_dist": label_dist / label_dist.sum(-1, keepdims=True), "weight": mx.ones((3,))}

    def loss_fn(model):
        return decision_loss(model(**inputs)[1], targets)[0]

    def gradients(checkpointing: bool, seed: int) -> dict:
        set_gradient_checkpointing(model.model, checkpointing)
        mx.random.seed(seed)
        loss, grads = nn.value_and_grad(model, loss_fn)(model)
        return {"loss": loss, **dict(tree_flatten(grads))}

    plain, checkpointed, other_masks = gradients(False, 0), gradients(True, 0), gradients(False, 1)
    for path, grad in plain.items():
        np.testing.assert_allclose(np.array(checkpointed[path]), np.array(grad), atol=1e-6, err_msg=path)
    assert float(other_masks["loss"]) != pytest.approx(float(plain["loss"]), abs=1e-6)  # dropout is active


def test_keyed_dropout_uses_the_forward_masks_in_the_rerun():
    dropout = KeyedDropout(0.5)
    weights, inputs = mx.ones((64,)), mx.ones((64,))

    class Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.weights = weights
            self.dropout = dropout

        def __call__(self, x):
            return self.dropout(x) * self.weights

    class Decoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = [Layer()]

    decoder = Decoder()
    decoder.train()
    set_gradient_checkpointing(decoder, True)
    for seed in range(10):
        mx.random.seed(seed)
        # With weights of 1, the gradient of the sum of each output with respect to the weights is the output itself
        out, grad = nn.value_and_grad(decoder, lambda m: m.layers[0](inputs).sum())(decoder)
        mx.random.seed(seed)
        np.testing.assert_array_equal(np.array(grad["layers"][0]["weights"]), np.array(decoder.layers[0](inputs)))


def _starting_adapter(model_dir: str, adapter_dir) -> None:
    """A PyTorch adapter on every linear layer, with random A and B, from which both backends start."""
    torch.manual_seed(1)
    model = AutoModel.from_pretrained(model_dir, dtype=torch.float32)
    targets = lora_targets(load_decision_model(model_dir, dtype="float32"), ["all"])
    lora = LoraConfig(
        r=4, lora_alpha=8, target_modules=targets, modules_to_save=["decision_head"], init_lora_weights=False,
        task_type=None,
    )
    get_peft_model(model, lora).save_pretrained(str(adapter_dir))


# The pointer head's key bias adds the same <bias, q> to every option's logit, which the softmax ignores, so its
# gradient is 0 up to rounding. Adam turns that rounding noise into sizable updates, which differ between frameworks.
NULL_DIRECTION = "decision_head.k.bias"


@pytest.mark.parametrize("source", ["qwen3", "qwen3_5"])
def test_training_matches_pytorch(decision_model_dir, dataset_dir, tmp_path, source):
    # With gradient accumulation, whose short last step of an epoch both backends average over its own batches
    model_dir = decision_model_dir(source)
    _starting_adapter(model_dir, tmp_path / "start")
    common = dict(
        adapter_name_or_path=str(tmp_path / "start"), max_steps=9, gradient_accumulation_steps=2, warmup_steps=2,
        lr_scheduler_type="cosine", weight_decay=0.1, max_grad_norm=1.0, train_sampling_strategy="group_by_length",
    )

    run_exp(_config(model_dir, dataset_dir, tmp_path / "torch", backend="torch", use_cpu=True, **common))
    run_exp(_config(model_dir, dataset_dir, tmp_path / "mlx", backend="mlx", **common))

    start, torch_adapter, mlx_adapter = (_adapter(tmp_path / name) for name in ("start", "torch", "mlx"))
    assert set(mlx_adapter) == set(torch_adapter)
    for key, value in torch_adapter.items():
        if not key.endswith(NULL_DIRECTION):
            update = np.abs(value.numpy() - start[key].numpy()).max()
            # Within 1% of the parameter's update over the run, which fp32 rounding in Gated DeltaNet reaches
            np.testing.assert_allclose(mlx_adapter[key].numpy(), value.numpy(), atol=1e-2 * update, err_msg=key)
    torch_logs = [entry for entry in _log_history(tmp_path / "torch") if "loss" in entry]
    mlx_logs = [entry for entry in _log_history(tmp_path / "mlx") if "loss" in entry]
    assert [entry["step"] for entry in mlx_logs] == [entry["step"] for entry in torch_logs]
    for torch_entry, mlx_entry in zip(torch_logs, mlx_logs):
        for name in ("loss", "grad_norm", "learning_rate", "head_learning_rate", "epoch"):
            assert mlx_entry[name] == pytest.approx(torch_entry[name], rel=1e-3, abs=1e-4), (name, mlx_entry["step"])


def test_accumulation_averages_the_batches(decision_model_dir, dataset_dir, tmp_path):
    # In order, 7 batches of 4 rows in steps of 2 batches cover the rows as 4 batches of 8 do (the last one has 4
    # rows either way), and every row has weight 1, so both give the same gradients
    model_dir = decision_model_dir("qwen3")
    common = dict(max_steps=6, train_sampling_strategy="sequential", max_grad_norm=1.0)

    run_exp(_config(model_dir, dataset_dir, tmp_path / "accumulated", gradient_accumulation_steps=2, **common))
    run_exp(_config(model_dir, dataset_dir, tmp_path / "large", per_device_train_batch_size=8, **common))

    accumulated, large = _adapter(tmp_path / "accumulated"), _adapter(tmp_path / "large")
    for key, value in large.items():
        if not key.endswith(NULL_DIRECTION):
            np.testing.assert_allclose(accumulated[key].numpy(), value.numpy(), atol=1e-5, err_msg=key)
    accumulated_logs = [entry for entry in _log_history(tmp_path / "accumulated") if "loss" in entry]
    large_logs = [entry for entry in _log_history(tmp_path / "large") if "loss" in entry]
    for a, b in zip(accumulated_logs, large_logs):
        assert a["loss"] == pytest.approx(b["loss"], abs=1e-4)
        assert a["grad_norm"] == pytest.approx(b["grad_norm"], rel=1e-4)


def test_training_end_to_end(decision_model_dir, dataset_dir, tmp_path):
    model_dir = decision_model_dir("qwen3_5")
    output_dir = tmp_path / "output"

    run_exp(
        _config(
            model_dir, dataset_dir, output_dir, do_eval=True, val_size=0.25, max_steps=4, eval_strategy="steps",
            eval_steps=2, save_strategy="steps", save_steps=2, save_total_limit=1, bf16=True,
        )
    )

    # The adapter, with the head, and the tokenizer, and only the last checkpoint
    assert os.path.isfile(output_dir / "adapter_config.json") and not os.path.exists(output_dir / "model.safetensors")
    assert os.path.isfile(output_dir / "tokenizer.json")
    assert sorted(name for name in os.listdir(output_dir) if name.startswith("checkpoint-")) == ["checkpoint-4"]
    assert os.path.isfile(output_dir / "checkpoint-4" / "mlx_optimizer.safetensors")

    logs = _log_history(output_dir)
    train_logs = [entry for entry in logs if "loss" in entry]
    assert [entry["step"] for entry in train_logs] == [1, 2, 3, 4]
    for entry in train_logs:
        assert entry["head_learning_rate"] == pytest.approx(10 * entry["learning_rate"])
        assert entry["padded_tokens_per_second"] > 0 and entry["peak_memory_gb"] > 0
    assert [entry["step"] for entry in logs if "eval_loss" in entry] == [2, 4]

    with open(output_dir / "eval_results.json") as f:
        eval_results = json.load(f)
    expected = {"eval_loss", "eval_accuracy", "eval_nll", "eval_brier", "eval_ece", "eval_choice_accuracy"}
    assert expected <= set(eval_results)
    with open(output_dir / "all_results.json") as f:
        assert {"train_loss", "eval_loss"} <= set(json.load(f))

    # PyTorch's predictor loads the adapter
    predictor = DecisionPredictor(model_dir, str(output_dir), device="cpu", calibration=None)
    probabilities = predictor.probabilities([decision_row(i) for i in range(3)])
    assert all(sum(probs) == pytest.approx(1.0, abs=1e-5) for probs in probabilities)


def test_resuming_matches_an_uninterrupted_run(decision_model_dir, dataset_dir, tmp_path):
    model_dir = decision_model_dir("qwen3")
    run_exp(_config(model_dir, dataset_dir, tmp_path / "straight", max_steps=6, gradient_accumulation_steps=2))
    run_exp(_config(
        model_dir, dataset_dir, tmp_path / "resumed", max_steps=6, gradient_accumulation_steps=2,
        save_strategy="steps", save_steps=3,
    ))
    checkpoint = tmp_path / "resumed" / "checkpoint-3"
    (tmp_path / "resumed" / "checkpoint-6").rename(tmp_path / "unused")

    run_exp(_config(
        model_dir, dataset_dir, tmp_path / "resumed", max_steps=6, gradient_accumulation_steps=2,
        resume_from_checkpoint=str(checkpoint),
    ))

    straight, resumed = _adapter(tmp_path / "straight"), _adapter(tmp_path / "resumed")
    for key, value in straight.items():
        np.testing.assert_allclose(resumed[key].numpy(), value.numpy(), atol=1e-6, err_msg=key)
    steps = [entry["step"] for entry in _log_history(tmp_path / "resumed") if "loss" in entry]
    assert steps == [1, 2, 3, 4, 5, 6]


def test_training_the_head_alone(decision_model_dir, dataset_dir, tmp_path):
    model_dir = decision_model_dir("qwen3_5")
    output_dir = tmp_path / "output"

    run_exp(_config(
        model_dir, dataset_dir, output_dir, finetuning_type="full", hybrid_learnable_params="decision_head",
        max_steps=2,
    ))

    with safe_open(str(output_dir / "model.safetensors"), "pt") as f:
        saved = {key: f.get_tensor(key) for key in f.keys()}
    source = {}
    for name in os.listdir(model_dir):
        if name.endswith(".safetensors"):
            with safe_open(os.path.join(model_dir, name), "pt") as f:
                source.update({key: f.get_tensor(key) for key in f.keys()})
    assert set(saved) == set(source)
    assert all(torch.equal(saved[key], source[key]) for key in source if not key.startswith("decision_head."))
    assert not torch.equal(saved["decision_head.q.weight"], source["decision_head.q.weight"])
    AutoModel.from_pretrained(str(output_dir))  # loads in PyTorch


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"finetuning_type": "freeze"}, "finetuning_type freeze"),
        ({"lr_scheduler_type": "polynomial"}, "lr_scheduler_type polynomial"),
        ({"optim": "sgd"}, "optim sgd"),
        ({"use_dora": True}, "DoRA"),
        ({"decision_save_optimizer": True}, "decision_save_optimizer"),
    ],
)
def test_unsupported_options_are_refused(decision_model_dir, dataset_dir, tmp_path, overrides, message):
    with pytest.raises(ValueError, match=message):
        run_exp(_config(decision_model_dir("qwen3"), dataset_dir, tmp_path / "output", max_steps=1, **overrides))
