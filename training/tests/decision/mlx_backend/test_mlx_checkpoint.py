"""
Tests for the checkpoints of the MLX backend (hmf/train/decision/mlx/checkpoint.py), with tiny Qwen3 and Qwen3.5
decision models: that an adapter saved by MLX loads in PyTorch and an adapter saved by PyTorch loads in MLX with the
same answers, that the loss and the gradients of every adapter and head parameter match PyTorch's on the same batch,
and that a fully fine-tuned model saved by MLX loads in PyTorch with the same answers.
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
from peft import LoraConfig, PeftModel, get_peft_model
from safetensors import safe_open
from transformers import AutoModel

from hmf.train.decision.mlx.checkpoint import load_adapter, save_adapter, save_full_model
from hmf.train.decision.mlx.lora import attach_lora, lora_layers, lora_targets, set_trainable
from hmf.train.decision.mlx.loss import decision_loss
from hmf.train.decision.mlx.model import load_decision_model
from hmf.train.decision.trainer import decision_loss as torch_decision_loss


SOURCES = ["qwen3", "qwen3_5"]


def _inputs(batch: dict) -> tuple:
    return tuple(mx.array(batch[name].numpy()) for name in ("input_ids", "opt_idx", "answer_idx"))


def _real(batch: dict) -> np.ndarray:
    return batch["opt_idx"].numpy() >= 0


def _torch_log_probs(model, batch: dict) -> np.ndarray:
    with torch.no_grad():
        return model(**batch).log_probs.numpy()


def _perturb_head(model, seed: int = 0) -> None:
    mx.random.seed(seed)
    head = tree_flatten(model.decision_head.parameters())
    noisy = [(name, value + 0.1 * mx.random.normal(value.shape)) for name, value in head]
    model.decision_head.load_weights(noisy)


@pytest.mark.parametrize("source", SOURCES)
def test_an_mlx_adapter_loads_in_pytorch(decision_model_dir, batch, tmp_path, source):
    model_dir = decision_model_dir(source)
    model = load_decision_model(model_dir, dtype="float32")
    targets = lora_targets(model, ["all"])
    attach_lora(model, targets, rank=4, alpha=8, use_rslora=True)
    mx.random.seed(0)
    for _, layer in lora_layers(model):
        layer.lora_b = 0.1 * mx.random.normal(layer.lora_b.shape)
    _perturb_head(model)
    expected = np.array(model(*_inputs(batch))[1])

    save_adapter(model, str(tmp_path), targets, rank=4, alpha=8, use_rslora=True, base_model_name_or_path=model_dir)

    config = json.loads((tmp_path / "adapter_config.json").read_text())
    assert config["modules_to_save"] == ["decision_head"] and config["task_type"] is None
    torch_model = PeftModel.from_pretrained(AutoModel.from_pretrained(model_dir, dtype=torch.float32), str(tmp_path))
    log_probs = _torch_log_probs(torch_model.eval(), batch)
    np.testing.assert_allclose(log_probs[_real(batch)], expected[_real(batch)], atol=1e-4)


def _torch_adapter(model_dir: str, adapter_dir: str, targets) -> PeftModel:
    """A PyTorch adapter with random A and B on the decision model, and a perturbed head, saved to adapter_dir."""
    torch.manual_seed(2)
    model = AutoModel.from_pretrained(model_dir, dtype=torch.float32)
    lora = LoraConfig(
        r=4, lora_alpha=8, target_modules=list(targets), modules_to_save=["decision_head"], init_lora_weights=False,
        task_type=None,
    )
    peft_model = get_peft_model(model, lora)
    with torch.no_grad():
        for name, param in peft_model.named_parameters():
            if "modules_to_save" in name:
                param.add_(0.1 * torch.randn_like(param))
    peft_model.save_pretrained(adapter_dir)
    return peft_model


@pytest.mark.parametrize("source", SOURCES)
def test_a_pytorch_adapter_loads_in_mlx(decision_model_dir, batch, tmp_path, source):
    model_dir = decision_model_dir(source)
    model = load_decision_model(model_dir, dtype="float32")
    peft_model = _torch_adapter(model_dir, str(tmp_path), lora_targets(model, ["all"]))
    expected = _torch_log_probs(peft_model.eval(), batch)

    config = load_adapter(model, str(tmp_path))

    assert config["r"] == 4
    log_probs = np.array(model(*_inputs(batch))[1])
    np.testing.assert_allclose(log_probs[_real(batch)], expected[_real(batch)], atol=1e-4)


def _targets(batch: dict) -> dict:
    """Training targets for the batch: labels, weights, and an anchor for two of the three rows."""
    num_options = (batch["opt_idx"] >= 0).sum(-1)
    width = batch["opt_idx"].shape[1]
    label_dist = torch.zeros(len(num_options), width)
    label_dist[torch.arange(len(num_options)), torch.tensor([1, 2, 0])] = 1.0
    ref_probs = torch.zeros(len(num_options), width)
    for row, n in enumerate(num_options.tolist()):
        ref_probs[row, :n] = torch.softmax(torch.arange(n, dtype=torch.float32), -1)
    return {
        "label_dist": label_dist,
        "weight": torch.tensor([1.0, 2.0, 0.5]),
        "ref_probs": ref_probs,
        "has_ref": torch.tensor([True, False, True]),
    }


def _peft_name(mlx_path: str) -> str:
    """The name of a trainable MLX parameter in a PEFT model of the same decision model."""
    if mlx_path.startswith("decision_head."):
        return "base_model.model.decision_head.modules_to_save.default." + mlx_path[len("decision_head."):]
    module, name = mlx_path.rsplit(".", 1)
    return f"base_model.model.{module}.lora_{name[-1].upper()}.default.weight"


@pytest.mark.parametrize("source", SOURCES)
def test_loss_and_gradients_match_pytorch(decision_model_dir, batch, tmp_path, source):
    model_dir = decision_model_dir(source)
    model = load_decision_model(model_dir, dtype="float32")
    peft_model = _torch_adapter(model_dir, str(tmp_path), lora_targets(model, ["all"]))
    load_adapter(model, str(tmp_path))
    set_trainable(model, "lora")
    model.train()
    targets = _targets(batch)
    mlx_targets = {name: mx.array(value.numpy()) for name, value in targets.items()}

    def loss_fn(model):
        _, log_probs = model(*_inputs(batch))
        return decision_loss(log_probs, mlx_targets, anchor_kl_weight=0.3)[0]

    loss, grads = nn.value_and_grad(model, loss_fn)(model)
    peft_model.train()
    torch_loss, _ = torch_decision_loss(peft_model(**batch).log_probs, targets, anchor_kl_weight=0.3)
    torch_loss.backward()

    assert float(loss) == pytest.approx(torch_loss.item(), abs=1e-5)
    torch_grads = {name: param.grad for name, param in peft_model.named_parameters() if param.requires_grad}
    mlx_grads = dict(tree_flatten(grads))
    assert {_peft_name(path) for path in mlx_grads} == set(torch_grads)
    for path, grad in mlx_grads.items():
        expected = torch_grads[_peft_name(path)].numpy()
        actual = np.array(grad).T if path.endswith(("lora_a", "lora_b")) else np.array(grad)
        np.testing.assert_allclose(actual, expected, atol=1e-4, rtol=1e-3, err_msg=path)  # gradients are of order 1


@pytest.mark.parametrize("source", SOURCES)
def test_a_fully_fine_tuned_model_loads_in_pytorch(decision_model_dir, batch, tmp_path, source):
    model_dir = decision_model_dir(source)
    model = load_decision_model(model_dir, dtype="float32")
    trainable = set_trainable(model, "full", hybrid_learnable_params="mlp,norm,decision_head")
    mx.random.seed(3)
    trained = tree_flatten(model.trainable_parameters())
    model.load_weights([(path, value + 0.05 * mx.random.normal(value.shape)) for path, value in trained], strict=False)
    expected = np.array(model(*_inputs(batch))[1])
    config = json.loads(open(os.path.join(model_dir, "config.json")).read())

    save_full_model(model, str(tmp_path), model_dir, config, dtype="float32")

    assert any("norm" in path and path.startswith("model.") for path in trainable)
    torch_model = AutoModel.from_pretrained(str(tmp_path), dtype=torch.float32).eval()
    log_probs = _torch_log_probs(torch_model, batch)
    np.testing.assert_allclose(log_probs[_real(batch)], expected[_real(batch)], atol=1e-4)


def test_frozen_parameters_are_copied_unchanged(decision_model_dir, tmp_path):
    model_dir = decision_model_dir("qwen3_5")
    model = load_decision_model(model_dir, dtype="bfloat16")
    set_trainable(model, "full", hybrid_learnable_params="decision_head")
    config = json.loads(open(os.path.join(model_dir, "config.json")).read())

    save_full_model(model, str(tmp_path), model_dir, config, dtype="bfloat16")

    source = {}
    for name in os.listdir(model_dir):
        if name.endswith(".safetensors"):
            with safe_open(os.path.join(model_dir, name), "pt") as f:
                source.update({key: f.get_tensor(key) for key in f.keys()})
    with safe_open(str(tmp_path / "model.safetensors"), "pt") as f:
        saved = {key: f.get_tensor(key) for key in f.keys()}
    assert set(saved) == set(source)
    for key, value in source.items():
        if not key.startswith("decision_head."):
            assert torch.equal(saved[key], value), key  # bit for bit, the source checkpoint's float32


def test_save_full_model_refuses_adapters(decision_model_dir, tmp_path):
    model = load_decision_model(decision_model_dir("qwen3"), dtype="float32")
    attach_lora(model, ["q_proj"], rank=4, alpha=8)
    with pytest.raises(ValueError, match="LoRA adapters"):
        save_full_model(model, str(tmp_path), decision_model_dir("qwen3"), {}, dtype="float32")
