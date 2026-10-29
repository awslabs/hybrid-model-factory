"""
Tests for decision model training (train/decision/), covering the collator, the augmentation sampler, the loss
(cross-entropy, and the anchor and teacher KL terms), the optimizer's parameter groups, the evaluation
metrics, the workflow's argument checks, and an end-to-end LoRA run through run_exp on a tiny decision
model (fixtures in conftest.py). Training with a label set end to end is in
tests/decision/data/test_decision_label.py.
"""

import json
import math
import os
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn
from transformers import TrainingArguments

from hmf.data import DecisionDataCollator
from hmf.model.hybrid_zoo.models.decision.modeling_decision import PointerHead, masked_log_softmax
from hmf.train.decision import workflow
from hmf.train.decision.metric import ComputeDecisionMetrics, expected_calibration_error
from hmf.train.decision.trainer import DecisionAugmentationSampler, DecisionTrainer, decision_loss, forward_kl
from hmf.train.tuner import run_exp


# Collator


def _feature(input_ids, opt_idx, label, kind):
    num_options = len(opt_idx)
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "opt_idx": opt_idx,
        "answer_idx": len(input_ids) - 1,
        "label": label,
        "label_dist": [float(k == label) for k in range(num_options)],
        "option_order": list(range(num_options)),
        "weight": 0.5,
        "kind": kind,
        "task": "t",
        "source": "s",
        "row_index": 0,
        "augmentation": 0,
    }


def test_collator():
    features = [_feature([5, 6, 7], [0, 1], 1, "noul"), _feature([5, 6, 7, 8, 9], [1, 2, 3], 2, "choice")]
    batch = DecisionDataCollator(tokenizer=SimpleNamespace(pad_token_id=1))(features)

    assert set(batch) == {
        "input_ids", "attention_mask", "opt_idx", "answer_idx", "label", "label_dist", "weight", "kind_id"
    }
    assert batch["input_ids"].tolist() == [[5, 6, 7, 1, 1], [5, 6, 7, 8, 9]]
    assert batch["attention_mask"].tolist() == [[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]]
    assert batch["opt_idx"].tolist() == [[0, 1, -1], [1, 2, 3]]
    assert batch["label_dist"].tolist() == [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    assert batch["answer_idx"].tolist() == [2, 4]
    assert batch["label"].tolist() == [1, 2]
    assert batch["weight"].tolist() == [0.5, 0.5]
    assert batch["kind_id"].tolist() == [0, 1]  # DECISION_KINDS = (noul, choice, score)


def test_collator_pad_to_multiple_of():
    batch = DecisionDataCollator(tokenizer=SimpleNamespace(pad_token_id=1), pad_to_multiple_of=8)(
        [_feature([5, 6, 7], [0, 1], 0, "noul")]
    )
    assert batch["input_ids"].shape == (1, 8)


# Sampler


def test_collator_with_label_set():
    features = [_feature([5, 6, 7], [0, 1], 1, "noul"), _feature([5, 6, 7, 8, 9], [1, 2, 3], 2, "choice")]
    features[0].update(ref_probs=[0.3, 0.7], has_ref=True, teacher_probs=[], has_teacher=False)
    features[1].update(ref_probs=[0.2, 0.3, 0.5], has_ref=True, teacher_probs=[0.1, 0.1, 0.8], has_teacher=True)

    batch = DecisionDataCollator(tokenizer=SimpleNamespace(pad_token_id=1))(features)

    # Padded with 0 to the widest row, like label_dist; a row without the target is all 0
    torch.testing.assert_close(batch["ref_probs"], torch.tensor([[0.3, 0.7, 0.0], [0.2, 0.3, 0.5]]))
    torch.testing.assert_close(batch["teacher_probs"], torch.tensor([[0.0, 0.0, 0.0], [0.1, 0.1, 0.8]]))
    assert batch["has_ref"].tolist() == [True, True]
    assert batch["has_teacher"].tolist() == [False, True]


def test_sampler_one_augmentation_per_epoch():
    augmentations = [0, 1] * 10  # 10 rows, 2 augmentations each
    sampler = DecisionAugmentationSampler(augmentations, batch_size=4, seed=0)
    orders = []
    for epoch in range(4):
        sampler.set_epoch(epoch)
        order = list(sampler)
        orders.append(order)
        assert len(sampler) == len(order) == 10
        assert sorted(order) == [i for i, r in enumerate(augmentations) if r == epoch % 2]

    assert orders[0] != orders[2]  # the same augmentation, reshuffled
    sampler.set_epoch(2)
    assert list(sampler) == orders[2]  # reproducible
    other = DecisionAugmentationSampler(augmentations, batch_size=4, seed=1)
    assert list(other) != orders[0]


def test_sampler_rejects_gaps_in_augmentations():
    with pytest.raises(ValueError, match="numbered from 0"):
        DecisionAugmentationSampler([0, 2, 0, 2], batch_size=2, seed=0)


def _padding(order, lengths, batch_size):
    batches = [order[i : i + batch_size] for i in range(0, len(order), batch_size)]
    return sum(max(lengths[i] for i in batch) * len(batch) - sum(lengths[i] for i in batch) for batch in batches)


def test_sampler_groups_by_length():
    rng = np.random.default_rng(0)
    lengths = rng.integers(10, 2000, size=30).tolist()
    sampler = DecisionAugmentationSampler([0] * 30, batch_size=4, seed=0, lengths=lengths, sort_window=2)
    order = list(sampler)
    batches = [order[i : i + 4] for i in range(0, 30, 4)]

    assert sorted(order) == list(range(30))
    assert max(lengths[i] for i in batches[0]) == max(lengths)  # the longest batch goes first
    assert [len(b) for b in batches] == [4] * 7 + [2]  # the short batch goes last
    assert all(lengths[b[i]] >= lengths[b[i + 1]] for b in batches for i in range(len(b) - 1))
    ungrouped = list(DecisionAugmentationSampler([0] * 30, batch_size=4, seed=0))
    assert _padding(order, lengths, 4) < _padding(ungrouped, lengths, 4)


def test_sampler_without_shuffle():
    sampler = DecisionAugmentationSampler([0, 1] * 5, batch_size=2, seed=0, shuffle=False)
    sampler.set_epoch(1)
    assert list(sampler) == [1, 3, 5, 7, 9]  # augmentation 1, in dataset order


# Loss


def test_loss():
    logits = torch.tensor([[2.0, 0.5, -1.0], [0.3, -0.2, 0.0], [1.0, 1.0, 0.0]])
    option_mask = torch.tensor([[True, True, False], [True, True, True], [True, True, True]])
    log_probs = masked_log_softmax(logits, option_mask)
    inputs = {
        "label_dist": torch.tensor([[0.0, 1.0, 0.0], [0.05, 0.9, 0.05], [1.0, 0.0, 0.0]]),
        "weight": torch.tensor([1.0, 2.0, 0.0]),
    }

    loss, parts = decision_loss(log_probs, inputs)

    per_row = [-log_probs[0, 1], -(0.05 * log_probs[1, 0] + 0.9 * log_probs[1, 1] + 0.05 * log_probs[1, 2])]
    expected = (per_row[0] * 1.0 + per_row[1] * 2.0) / 3.0  # the weight-0 row does not count
    assert torch.isfinite(loss)
    torch.testing.assert_close(loss, expected)
    assert set(parts) == {"ce_loss"}

    inputs["weight"] = torch.zeros(3)
    assert decision_loss(log_probs, inputs)[0].item() == 0.0


def _kl(p, log_q):
    """KL(p || q) of one row, over the options where p > 0."""
    return sum(pk * (math.log(pk) - lq) for pk, lq in zip(p, log_q) if pk > 0)


def test_forward_kl():
    logits = torch.tensor([[2.0, 0.5, -1.0], [0.3, -0.2, 0.0], [1.0, 1.0, 0.0]], requires_grad=True)
    option_mask = torch.tensor([[True, True, False], [True, True, True], [True, True, True]])
    log_probs = masked_log_softmax(logits, option_mask)
    # Row 0 has 2 options (the third is padding, with target 0 and model -inf); row 1 has a target of exactly 0
    target = torch.tensor([[0.4, 0.6, 0.0], [0.0, 0.5, 0.5], [0.2, 0.2, 0.6]])
    rows = torch.tensor([True, True, False])

    kl = forward_kl(target, log_probs, rows)

    q = log_probs.detach().tolist()
    expected = (_kl([0.4, 0.6, 0.0], q[0]) + _kl([0.0, 0.5, 0.5], q[1])) / 2  # row 2 is not selected
    assert kl.item() == pytest.approx(expected, rel=1e-5)
    kl.backward()
    assert torch.isfinite(logits.grad).all()  # zero targets and padding give no NaN

    # The KL of a distribution with itself is 0
    assert forward_kl(log_probs.detach().exp(), log_probs.detach(), torch.ones(3, dtype=torch.bool)).item() == (
        pytest.approx(0.0, abs=1e-6)
    )


def _labeled_inputs():
    return {
        "label_dist": torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]),
        "weight": torch.tensor([1.0, 1.0, 1.0]),
        "ref_probs": torch.tensor([[0.5, 0.5, 0.0], [0.2, 0.3, 0.5], [0.0, 0.0, 0.0]]),
        "has_ref": torch.tensor([True, True, False]),
        "teacher_probs": torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.1, 0.1, 0.8]]),
        "has_teacher": torch.tensor([False, False, True]),
    }


def test_decision_loss_with_kl_terms():
    logits = torch.tensor([[2.0, 0.5, -1.0], [0.3, -0.2, 0.0], [1.0, 1.0, 0.0]])
    option_mask = torch.tensor([[True, True, False], [True, True, True], [True, True, True]])
    log_probs = masked_log_softmax(logits, option_mask)
    inputs = _labeled_inputs()
    ce_loss = decision_loss(log_probs, inputs)[0]

    loss, parts = decision_loss(log_probs, inputs, anchor_kl_weight=0.3, teacher_kl_weight=1.0)

    q = log_probs.tolist()
    anchor_kl = (_kl([0.5, 0.5, 0.0], q[0]) + _kl([0.2, 0.3, 0.5], q[1])) / 2  # the rows with an anchor
    teacher_kl = _kl([0.1, 0.1, 0.8], q[2])  # the one row with a teacher gets the full weight
    assert loss.item() == pytest.approx(ce_loss.item() + 0.3 * anchor_kl + 1.0 * teacher_kl, rel=1e-5)
    assert parts["ce_loss"].item() == pytest.approx(ce_loss.item())
    assert parts["anchor_kl"].item() == pytest.approx(anchor_kl, rel=1e-5)
    assert parts["teacher_kl"].item() == pytest.approx(teacher_kl, rel=1e-5)


def test_decision_loss_without_kl_terms():
    log_probs = masked_log_softmax(torch.randn(3, 3), torch.ones(3, 3, dtype=torch.bool))
    inputs = _labeled_inputs()
    ce_loss = decision_loss(log_probs, {"label_dist": inputs["label_dist"], "weight": inputs["weight"]})[0]

    # With weights of 0, the label set's targets are not used
    loss, parts = decision_loss(log_probs, inputs)
    assert loss.item() == ce_loss.item() and set(parts) == {"ce_loss"}

    # A batch without rows that have the target adds nothing for that term
    inputs["has_teacher"] = torch.zeros(3, dtype=torch.bool)
    loss, parts = decision_loss(log_probs, inputs, teacher_kl_weight=1.0)
    assert loss.item() == ce_loss.item() and "teacher_kl" not in parts


# Optimizer


class _TinyDecisionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.decision_head = PointerHead(8, dim=4)
        self.adapter = nn.Linear(8, 8)
        self.frozen = nn.Linear(8, 8).requires_grad_(False)


@pytest.mark.parametrize("head_lr", [1e-3, None])
def test_optimizer_groups(tmp_path, head_lr):
    model = _TinyDecisionModel()
    trainer = SimpleNamespace(
        optimizer=None,
        model=model,
        args=TrainingArguments(output_dir=str(tmp_path), learning_rate=1e-4, weight_decay=0.01, report_to="none"),
        finetuning_args=SimpleNamespace(decision_head_lr=head_lr),
    )

    optimizer = DecisionTrainer.create_optimizer(trainer)

    groups = {frozenset(id(p) for p in g["params"]): (g["lr"], g["weight_decay"]) for g in optimizer.param_groups}
    head = model.decision_head
    expected_head_lr = head_lr or 1e-4
    assert groups == {
        frozenset({id(head.q.weight), id(head.k.weight)}): (expected_head_lr, 0.01),
        frozenset({id(head.q.bias), id(head.k.bias), id(head.norm.weight), id(head.norm.bias)}): (
            expected_head_lr, 0.0
        ),
        frozenset({id(model.adapter.weight), id(model.adapter.bias)}): (1e-4, 0.0),
    }


# Metrics


def test_metrics():
    rows = [  # (probabilities, label, kind_id)
        ([0.7, 0.3], 0, 0),
        ([0.2, 0.5, 0.3], 2, 1),
        ([0.1, 0.6, 0.2, 0.1], 1, 2),
        ([0.9, 0.1], 1, 0),
    ]
    stats = [
        [int(np.argmax(p)), label, kind, math.log(p[label]), max(p), sum(x * x for x in p)] for p, label, kind in rows
    ]

    metrics = ComputeDecisionMetrics()(SimpleNamespace(predictions=np.array(stats)))

    correct = np.array([np.argmax(p) == label for p, label, _ in rows], dtype=float)
    brier = [sum((x - (k == label)) ** 2 for k, x in enumerate(p)) for p, label, _ in rows]
    assert metrics["accuracy"] == pytest.approx(0.5)
    assert metrics["nll"] == pytest.approx(np.mean([-math.log(p[label]) for p, label, _ in rows]))
    assert metrics["brier"] == pytest.approx(np.mean(brier))
    assert metrics["ece"] == pytest.approx(expected_calibration_error(np.array([max(p) for p, _, _ in rows]), correct))
    assert (metrics["noul_accuracy"], metrics["choice_accuracy"], metrics["score_accuracy"]) == (0.5, 0.0, 1.0)
    assert metrics["noul_brier"] == pytest.approx(np.mean([brier[0], brier[3]]))


def test_expected_calibration_error():
    # Two bins, confidence 0.95 with accuracy 1 (gap 0.05) and confidence 0.55 with accuracy 0 (gap 0.55)
    confidence = np.array([0.95, 0.95, 0.55, 0.55])
    correct = np.array([1.0, 1.0, 0.0, 0.0])
    assert expected_calibration_error(confidence, correct) == pytest.approx(0.5 * 0.05 + 0.5 * 0.55)


# Workflow checks


def _workflow_args(**overrides):
    model_args = SimpleNamespace(
        sequence_parallel_size=overrides.pop("sequence_parallel_size", 1),
        disable_gradient_checkpointing=overrides.pop("disable_gradient_checkpointing", False),
    )
    data_args = SimpleNamespace(
        packing=overrides.pop("packing", None), decision_label_folder=overrides.pop("decision_label_folder", None)
    )
    training_args = SimpleNamespace(do_predict=overrides.pop("do_predict", False), predict_with_generate=False)
    finetuning_args = SimpleNamespace(
        hybrid_learnable_params=None, finetuning_type="lora", use_galore=False, use_apollo=False, use_badam=False,
        use_adam_mini=False, use_muon=False, loraplus_lr_ratio=None, decision_anchor_kl_weight=0.0,
        decision_teacher_kl_weight=0.0, decision_checkpoint_min_tokens=0,
    )
    for name, value in overrides.items():
        setattr(finetuning_args, name, value)
    return model_args, data_args, training_args, finetuning_args


def test_check_arguments_accepts_defaults():
    workflow._check_arguments(*_workflow_args())
    workflow._check_arguments(*_workflow_args(finetuning_type="full", hybrid_learnable_params="decision_head"))
    workflow._check_arguments(*_workflow_args(decision_label_folder="labels_a", decision_teacher_kl_weight=1.0))
    workflow._check_arguments(*_workflow_args(decision_checkpoint_min_tokens=4096))  # with checkpointing on


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"sequence_parallel_size": 2}, "Sequence parallelism"),
        ({"packing": True}, "Packing"),
        ({"do_predict": True}, "Prediction"),
        ({"hybrid_learnable_params": "decision_head"}, "cannot be combined with LoRA"),
        ({"use_galore": True}, "use_galore"),
        ({"loraplus_lr_ratio": 16.0}, "loraplus_lr_ratio"),
        ({"decision_anchor_kl_weight": 0.3}, "needs anchor or teacher targets: set decision_label_folder"),
        ({"decision_label_folder": "labels_a", "decision_teacher_kl_weight": -1.0}, "must be at least 0"),
        ({"decision_checkpoint_min_tokens": -1}, "decision_checkpoint_min_tokens must be at least 0"),
        ({"decision_checkpoint_min_tokens": 4096, "disable_gradient_checkpointing": True}, "Remove one of them"),
    ],
)
def test_check_arguments_rejects(overrides, message):
    with pytest.raises(ValueError, match=message):
        workflow._check_arguments(*_workflow_args(**overrides))


def test_check_label_set(monkeypatch):
    infos = []
    monkeypatch.setattr(workflow.logger, "info_rank0", infos.append)
    train_dataset = {"has_ref": [True, False, True], "has_teacher": [False, False, False]}
    data_args = SimpleNamespace(decision_label_folder="labels_a")

    # A weight above 0 needs rows with that target
    finetuning_args = SimpleNamespace(decision_anchor_kl_weight=0.3, decision_teacher_kl_weight=1.0)
    with pytest.raises(ValueError, match="has no teacher targets"):
        workflow._check_label_set(train_dataset, data_args, finetuning_args)

    # Targets whose weight is 0 are not used, which is logged
    finetuning_args = SimpleNamespace(decision_anchor_kl_weight=0.0, decision_teacher_kl_weight=0.0)
    workflow._check_label_set(train_dataset, data_args, finetuning_args)
    assert any("anchor targets on 2 rows" in info and "unused" in info for info in infos)


@pytest.mark.parametrize(
    "max_steps, epochs, warns",
    [(0, 2.0, False), (0, 3.0, True), (5, 1.0, False), (6, 1.0, True)],
)
def test_warns_if_augmentations_reused(monkeypatch, max_steps, epochs, warns):
    warnings = []
    monkeypatch.setattr(workflow.logger, "warning_rank0", warnings.append)
    trainer = SimpleNamespace(train_dataset={"augmentation": [0, 1] * 20})  # 20 rows, 2 augmentations
    training_args = SimpleNamespace(
        per_device_train_batch_size=4, gradient_accumulation_steps=2, world_size=1,  # global batch of 8
        max_steps=max_steps, num_train_epochs=epochs,
    )

    workflow._warn_if_augmentations_reused(trainer, training_args)

    # 5 steps of 8 rows = 40 rows = 2 passes over 20 rows; 6 steps = 2.4 passes
    assert bool(warnings) == warns


# End to end


def test_run_decision_end_to_end(monkeypatch, tmp_path, decision_model_dir, decision_dataset_dir, lora_targets,
                                 device_and_dtype):
    # Record the augmentation each training epoch reads
    augmentations_used = []
    order = DecisionAugmentationSampler._order

    def recording_order(self):
        augmentations_used.append(self.epoch % self.num_augmentations)
        return order(self)

    monkeypatch.setattr(DecisionAugmentationSampler, "_order", recording_order)

    output_dir = tmp_path / "output"
    run_exp(
        {
            "model_name_or_path": decision_model_dir("qwen3"),
            "stage": "decision",
            "do_train": True,
            "do_eval": True,
            "finetuning_type": "lora",
            "lora_target": lora_targets["qwen3"],
            "lora_rank": 4,
            "dataset": "toy_decision",
            "dataset_dir": decision_dataset_dir,
            "cutoff_len": 512,
            "val_size": 0.2,
            "decision_num_augmentations": 2,
            "num_train_epochs": 3,
            "per_device_train_batch_size": 4,
            "learning_rate": 1e-3,
            "decision_head_lr": 1e-2,
            "train_sampling_strategy": "group_by_length",
            "eval_strategy": "epoch",
            "save_strategy": "no",
            "logging_steps": 1,
            "output_dir": str(output_dir),
            "overwrite_output_dir": True,
            "overwrite_cache": True,
            "report_to": "none",
            "bf16": device_and_dtype[0] == "cuda",
            "seed": 0,
        }
    )

    # Epochs 0, 1, and 2 read augmentations 0, 1, and 0, so set_epoch reaches the sampler
    assert augmentations_used == [0, 1, 0]

    # The adapter is saved on its own, with the head
    assert os.path.isfile(output_dir / "adapter_config.json")
    assert os.path.isfile(output_dir / "adapter_model.safetensors")
    assert not os.path.exists(output_dir / "model.safetensors")

    # Both learning rates are logged, the adapters' as learning_rate and the head's (10x) separately
    with open(output_dir / "trainer_state.json") as f:
        train_logs = [entry for entry in json.load(f)["log_history"] if "loss" in entry]
    assert train_logs and all("head_learning_rate" in entry for entry in train_logs)
    for entry in train_logs:
        assert entry["head_learning_rate"] == pytest.approx(10 * entry["learning_rate"])

    with open(output_dir / "eval_results.json") as f:
        eval_results = json.load(f)
    assert {"eval_loss", "eval_accuracy", "eval_nll", "eval_brier", "eval_ece"} <= set(eval_results)
    assert any(key.startswith(("eval_noul_", "eval_choice_", "eval_score_")) for key in eval_results)
    assert all(math.isfinite(value) for key, value in eval_results.items() if key.startswith("eval_"))
