"""
Tests for LoRA (and head-only) training of decision models, on tiny decision models converted
from a stock Transformer (Qwen3) and from a Gated DeltaNet + Attention model (Qwen3.5); see
conftest.py. They check what trains, that LoRA starts as a no-op, that training leaves the backbone
unchanged, and that the adapter (with the pointer head) is saved on its own, reloads exactly, can
be trained further, and merges into a standalone decision checkpoint with `hmf export`.
"""

import glob
import json
import os
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from peft import PeftModel
from safetensors import safe_open
from transformers import AutoModel, TrainingArguments, get_scheduler

from hmf.hparams.parser import _parse_train_args
from hmf.model import load_model, load_tokenizer
from hmf.model.hybrid_zoo.models.decision.modeling_decision import DecisionModel
from hmf.train.decision.trainer import MODEL_INPUTS, DecisionTrainer
from hmf.train.trainer_utils import freeze_params_hybrid
from hmf.train.tuner import export_model


SOURCES = ["qwen3", "qwen3_5"]
NUM_STEPS = 3


def _load(model_dir: str, device: str, **overrides):
    """Loads a decision model for training as the decision workflow does, with LoRA by default."""
    args = {
        "model_name_or_path": model_dir,
        "stage": "decision",
        "do_train": True,
        "finetuning_type": "lora",
        "lora_rank": 4,
        "disable_gradient_checkpointing": True,
        "output_dir": "unused",
        "report_to": "none",
    }
    args.update(overrides)
    model_args, _, _, finetuning_args, _ = _parse_train_args(args)
    tokenizer = load_tokenizer(model_args)["tokenizer"]
    model = load_model(tokenizer, model_args, finetuning_args, is_trainable=True)
    return model.to(device), tokenizer, finetuning_args


def _logits(model, batch) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        return model(**{name: batch[name] for name in MODEL_INPUTS}).logits


def _train(model, batch, num_steps: int = NUM_STEPS) -> None:
    model.train()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-2)
    for _ in range(num_steps):
        out = model(**{name: batch[name] for name in MODEL_INPUTS})
        F.nll_loss(out.log_probs, batch["label"]).backward()
        optimizer.step()
        optimizer.zero_grad()


def _snapshot(model) -> dict:
    return {name: param.detach().clone() for name, param in model.named_parameters()}


def _tolerance(dtype: torch.dtype) -> dict:
    return {"atol": 2e-2, "rtol": 2e-2} if dtype == torch.bfloat16 else {"atol": 1e-4, "rtol": 1e-4}


@pytest.fixture(params=SOURCES)
def source(request) -> str:
    return request.param


def test_trainable_parameters(source, decision_model_dir, lora_targets, device_and_dtype):
    model, _, _ = _load(decision_model_dir(source), device_and_dtype[0], lora_target=lora_targets[source])
    trainable = [name for name, param in model.named_parameters() if param.requires_grad]

    assert isinstance(model, PeftModel)
    # Only LoRA matrices and the head's trained copy train; the backbone and the original head are frozen
    assert all("lora_" in name or "decision_head.modules_to_save" in name for name in trainable), trainable
    for target in lora_targets[source].split(","):
        assert any(f".{target}.lora_A" in name for name in trainable), f"no LoRA on {target}"
    head = [name for name in trainable if "decision_head" in name]
    assert len(head) == 6  # the norm's weight and bias, and the q and k weights and biases
    assert not any("decision_head" in name and "lora_" in name for name in trainable)
    assert all(not p.requires_grad for n, p in model.named_parameters() if "decision_head.original_module" in n)


def test_lora_target_all_skips_the_head(source, decision_model_dir, device_and_dtype):
    model, _, _ = _load(decision_model_dir(source), device_and_dtype[0], lora_target="all")
    trainable = [name for name, param in model.named_parameters() if param.requires_grad]

    assert any("lora_" in name for name in trainable)
    assert not any("decision_head" in name and "lora_" in name for name in trainable)


def test_optimizer_learning_rates(source, decision_model_dir, lora_targets, device_and_dtype, tmp_path):
    # The real optimizer on a LoRA decision model, with the head's trained copy at decision_head_lr, the
    # LoRA matrices at learning_rate, and the schedule scaling both by the same factor
    model, _, _ = _load(decision_model_dir(source), device_and_dtype[0], lora_target=lora_targets[source])
    names = {id(param): name for name, param in model.named_parameters()}
    trainer = SimpleNamespace(
        optimizer=None,
        model=model,
        args=TrainingArguments(output_dir=str(tmp_path), learning_rate=1e-4, weight_decay=0.01, report_to="none"),
        finetuning_args=SimpleNamespace(decision_head_lr=1e-3),
    )

    optimizer = DecisionTrainer.create_optimizer(trainer)

    groups = {
        (group["lr"], group["weight_decay"]): sorted(names[id(p)] for p in group["params"])
        for group in optimizer.param_groups
    }
    head_prefix = "base_model.model.decision_head.modules_to_save.default."
    assert set(groups) == {(1e-3, 0.01), (1e-3, 0.0), (1e-4, 0.0)}
    assert groups[(1e-3, 0.01)] == sorted(head_prefix + name for name in ("q.weight", "k.weight"))
    assert groups[(1e-3, 0.0)] == sorted(
        head_prefix + name for name in ("norm.weight", "norm.bias", "q.bias", "k.bias")
    )
    assert groups[(1e-4, 0.0)] and all("lora_" in name for name in groups[(1e-4, 0.0)])
    trainable = {name for name, param in model.named_parameters() if param.requires_grad}
    assert {name for group in groups.values() for name in group} == trainable

    # Through warmup and the cosine decay, the head's learning rate stays 10x the adapters'
    scheduler = get_scheduler("cosine", optimizer, num_warmup_steps=2, num_training_steps=10)
    for _ in range(8):
        optimizer.step()
        scheduler.step()
        lrs = {round(group["initial_lr"], 6): group["lr"] for group in optimizer.param_groups}
        assert lrs[0.001] == pytest.approx(10 * lrs[0.0001])
        assert lrs[0.0001] > 0


def test_lora_starts_as_noop(source, decision_model_dir, lora_targets, device_and_dtype, make_batch):
    device, dtype = device_and_dtype
    model, tokenizer, _ = _load(decision_model_dir(source), device, lora_target=lora_targets[source])
    base = AutoModel.from_pretrained(decision_model_dir(source), dtype=dtype).to(device)
    batch = make_batch(tokenizer)

    torch.testing.assert_close(_logits(model, batch), _logits(base, batch))


def test_training_updates_only_lora_and_head(source, decision_model_dir, lora_targets, device_and_dtype, make_batch):
    model, tokenizer, _ = _load(decision_model_dir(source), device_and_dtype[0], lora_target=lora_targets[source])
    batch = make_batch(tokenizer)
    before = _snapshot(model)

    _train(model, batch)

    for name, param in model.named_parameters():
        changed = not torch.equal(param, before[name])
        if param.requires_grad:
            if "lora_B" in name or "decision_head.modules_to_save" in name:
                assert changed, f"{name} did not train"
        else:
            assert not changed, f"frozen {name} changed"


def _trained_adapter(source, decision_model_dir, lora_targets, device, make_batch, tmp_path):
    """Trains a LoRA adapter for a few steps and saves it; returns its directory, logits, and batch."""
    model, tokenizer, _ = _load(decision_model_dir(source), device, lora_target=lora_targets[source])
    batch = make_batch(tokenizer)
    _train(model, batch)
    adapter_dir = str(tmp_path / "adapter")
    model.save_pretrained(adapter_dir)
    return adapter_dir, _logits(model, batch), batch


def test_adapter_saved_separately_with_the_head(
    source, decision_model_dir, lora_targets, device_and_dtype, make_batch, tmp_path
):
    adapter_dir, _, _ = _trained_adapter(
        source, decision_model_dir, lora_targets, device_and_dtype[0], make_batch, tmp_path
    )

    files = set(os.listdir(adapter_dir))
    assert {"adapter_config.json", "adapter_model.safetensors"} <= files
    assert not any(name.startswith("model") and name.endswith(".safetensors") for name in files)  # no full weights
    with safe_open(os.path.join(adapter_dir, "adapter_model.safetensors"), framework="pt") as f:
        keys = list(f.keys())
    assert all("lora_" in key or "decision_head" in key for key in keys), keys
    assert any("decision_head" in key and key.endswith("q.weight") for key in keys)
    with open(os.path.join(adapter_dir, "adapter_config.json")) as f:
        adapter_config = json.load(f)
    assert adapter_config["task_type"] is None
    assert adapter_config["base_model_name_or_path"] == decision_model_dir(source)
    assert "decision_head" in adapter_config["modules_to_save"]


def test_adapter_reloads_exactly(source, decision_model_dir, lora_targets, device_and_dtype, make_batch, tmp_path):
    device, dtype = device_and_dtype
    adapter_dir, trained_logits, batch = _trained_adapter(
        source, decision_model_dir, lora_targets, device, make_batch, tmp_path
    )

    base = AutoModel.from_pretrained(decision_model_dir(source), dtype=dtype)
    reloaded = PeftModel.from_pretrained(base, adapter_dir).to(device)

    torch.testing.assert_close(_logits(reloaded, batch), trained_logits)


def test_training_continues_a_saved_adapter(
    source, decision_model_dir, lora_targets, device_and_dtype, make_batch, tmp_path
):
    # With adapter_name_or_path, training loads the adapter and keeps training it (LLaMA-Factory's resume
    # path). The head trains from the adapter's copy, at decision_head_lr, and the adapter keeps its own rank
    device, dtype = device_and_dtype
    adapter_dir, trained_logits, batch = _trained_adapter(
        source, decision_model_dir, lora_targets, device, make_batch, tmp_path
    )
    model, _, _ = _load(decision_model_dir(source), device, adapter_name_or_path=adapter_dir, lora_rank=8)

    # It starts where the saved adapter ended, with the same trainable parameters as a new adapter
    torch.testing.assert_close(_logits(model, batch), trained_logits)
    trainable = [name for name, param in model.named_parameters() if param.requires_grad]
    assert all("lora_" in name or "decision_head.modules_to_save" in name for name in trainable), trainable
    assert len([name for name in trainable if "decision_head" in name]) == 6
    assert all(param.shape[0] == 4 for name, param in model.named_parameters() if "lora_A" in name)

    names = {id(param): name for name, param in model.named_parameters()}
    trainer = SimpleNamespace(
        optimizer=None,
        model=model,
        args=TrainingArguments(output_dir=str(tmp_path / "unused"), learning_rate=1e-4, report_to="none"),
        finetuning_args=SimpleNamespace(decision_head_lr=1e-3),
    )
    optimizer = DecisionTrainer.create_optimizer(trainer)
    head_groups = [g for g in optimizer.param_groups if any("decision_head" in names[id(p)] for p in g["params"])]
    assert {group["lr"] for group in head_groups} == {1e-3}

    before = _snapshot(model)
    _train(model, batch)
    for name, param in model.named_parameters():
        if "lora_B" in name or "decision_head.modules_to_save" in name:
            assert not torch.equal(param, before[name]), f"{name} did not train"

    # Saved again, it reloads exactly
    continued_dir = str(tmp_path / "continued")
    model.save_pretrained(continued_dir)
    base = AutoModel.from_pretrained(decision_model_dir(source), dtype=dtype)
    reloaded = PeftModel.from_pretrained(base, continued_dir).to(device)
    torch.testing.assert_close(_logits(reloaded, batch), _logits(model, batch))


def test_training_a_new_adapter_on_a_saved_one(
    source, decision_model_dir, lora_targets, device_and_dtype, make_batch, tmp_path
):
    # With create_new_adapter, the saved adapter (and its head) is merged in, and a new adapter of the
    # config's rank trains on top, its head's copy starting from the saved head
    device, dtype = device_and_dtype
    adapter_dir, trained_logits, batch = _trained_adapter(
        source, decision_model_dir, lora_targets, device, make_batch, tmp_path
    )
    model, _, _ = _load(
        decision_model_dir(source), device, adapter_name_or_path=adapter_dir, create_new_adapter=True,
        lora_rank=8, lora_target=lora_targets[source],
    )

    torch.testing.assert_close(_logits(model, batch), trained_logits, **_tolerance(dtype))
    assert all(param.shape[0] == 8 for name, param in model.named_parameters() if "lora_A" in name)
    head = model.base_model.model.decision_head
    for name, param in head.modules_to_save["default"].named_parameters():
        torch.testing.assert_close(param, head.original_module.get_parameter(name))
        assert param.requires_grad


def test_export_merges_into_a_decision_checkpoint(
    source, decision_model_dir, lora_targets, device_and_dtype, make_batch, tmp_path
):
    device, dtype = device_and_dtype
    adapter_dir, trained_logits, batch = _trained_adapter(
        source, decision_model_dir, lora_targets, device, make_batch, tmp_path
    )
    export_dir = str(tmp_path / "export")

    export_model(
        {
            "model_name_or_path": decision_model_dir(source),
            "adapter_name_or_path": adapter_dir,
            "finetuning_type": "lora",
            "stage": "decision",
            "export_dir": export_dir,
            "infer_dtype": "bfloat16" if dtype == torch.bfloat16 else "float32",
        }
    )

    # A plain decision checkpoint, with no adapter and the head in float32
    keys = {}
    for path in glob.glob(os.path.join(export_dir, "*.safetensors")):
        with safe_open(path, framework="pt") as f:
            keys.update({key: f.get_slice(key).get_dtype() for key in f.keys()})
    assert all(key.startswith(("model.", "decision_head.")) for key in keys), sorted(keys)[:10]
    assert not any("lora_" in key or "modules_to_save" in key for key in keys)
    assert {dtype for key, dtype in keys.items() if key.startswith("decision_head.")} == {"F32"}

    exported = AutoModel.from_pretrained(export_dir, dtype=dtype).to(device)
    assert isinstance(exported, DecisionModel)
    torch.testing.assert_close(_logits(exported, batch), trained_logits, **_tolerance(dtype))


def test_head_only_training(source, decision_model_dir, device_and_dtype, make_batch):
    device, dtype = device_and_dtype
    model, tokenizer, finetuning_args = _load(
        decision_model_dir(source),
        device,
        finetuning_type="full",
        hybrid_learnable_params="decision_head",
        pure_bf16=dtype == torch.bfloat16,  # keep the backbone in its dtype
    )
    freeze_params_hybrid(model, None, finetuning_args, verbose=False)
    batch = make_batch(tokenizer)
    before = _snapshot(model)

    assert {name for name, p in model.named_parameters() if p.requires_grad} == {
        f"decision_head.{name}" for name in ("norm.weight", "norm.bias", "q.weight", "q.bias", "k.weight", "k.bias")
    }
    _train(model, batch)
    for name, param in model.named_parameters():
        changed = not torch.equal(param, before[name])
        assert changed == name.startswith("decision_head."), name
