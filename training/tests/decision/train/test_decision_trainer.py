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
import random
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


def _order_before_steps(sampler):
    """The order of grouping by length before steps were taken into account, which shuffled single batches."""
    rng = random.Random(sampler.seed * 1_000_003 + sampler.epoch)
    rows = list(sampler._epoch_rows())
    rng.shuffle(rows)
    group_size = sampler.batch_size * sampler.sort_window
    batches = []
    for start in range(0, len(rows), group_size):
        group = sorted(rows[start : start + group_size], key=lambda i: -sampler.lengths[i])
        batches += [group[i : i + sampler.batch_size] for i in range(0, len(group), sampler.batch_size)]
    last = batches.pop() if len(batches[-1]) < sampler.batch_size else None
    longest = max(range(len(batches)), key=lambda b: sampler.lengths[batches[b][0]]) if batches else None
    first = [batches.pop(longest)] if longest is not None else []
    rng.shuffle(batches)
    order = first + batches + ([last] if last is not None else [])
    return [i for batch in order for i in batch]


def _steps(order, batch_size, data_parallel_size):
    """The rows of each training step, as the devices take consecutive batches of the order in turn."""
    batches = [order[i : i + batch_size] for i in range(0, len(order), batch_size)]
    return [batches[i : i + data_parallel_size] for i in range(0, len(batches), data_parallel_size)]


def _waiting(order, lengths, batch_size, data_parallel_size):
    """The rows' worth of time spent per epoch, as each step lasts as long as its longest row."""
    steps = _steps(order, batch_size, data_parallel_size)
    return sum(max(lengths[i] for batch in step for i in batch) for step in steps)


def _sampler(num_rows, batch_size, data_parallel_size, sort_window, seed=0, num_augmentations=1):
    lengths = np.random.default_rng(seed).integers(10, 4000, size=num_rows * num_augmentations).tolist()
    augmentations = [a for _ in range(num_rows) for a in range(num_augmentations)]
    sampler = DecisionAugmentationSampler(augmentations, batch_size=batch_size, seed=seed, lengths=lengths,
                                          sort_window=sort_window, data_parallel_size=data_parallel_size)
    return sampler, lengths


SAMPLER_GRID = [
    # (rows, batch size, data parallel size, sort window): even and uneven row counts, a data parallel size that
    # does and does not divide the sort window, one above it, and more devices than batches
    (240, 4, 8, 50), (241, 4, 8, 50), (1000, 8, 8, 10), (997, 4, 3, 10), (500, 2, 16, 5), (77, 4, 2, 3),
    (30, 4, 16, 50), (5, 4, 8, 50), (3, 4, 2, 50), (1, 1, 1, 1),
]


@pytest.mark.parametrize("num_rows, batch_size, data_parallel_size, sort_window", SAMPLER_GRID)
def test_sampler_in_steps_serves_every_row_of_each_epoch_once(num_rows, batch_size, data_parallel_size, sort_window):
    sampler, _ = _sampler(num_rows, batch_size, data_parallel_size, sort_window, num_augmentations=2)
    for epoch in range(4):
        sampler.set_epoch(epoch)
        order = list(sampler)
        assert len(order) == len(sampler) == num_rows
        assert sorted(order) == list(range(epoch % 2, 2 * num_rows, 2))


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("num_rows, batch_size, sort_window", [(240, 4, 50), (241, 4, 7), (77, 2, 3), (3, 4, 50)])
def test_sampler_with_one_device_keeps_the_order_of_shuffling_single_batches(seed, num_rows, batch_size, sort_window):
    sampler, _ = _sampler(num_rows, batch_size, 1, sort_window, seed=seed, num_augmentations=2)
    for epoch in range(3):
        sampler.set_epoch(epoch)
        assert list(sampler) == _order_before_steps(sampler)


@pytest.mark.parametrize("num_rows, batch_size, data_parallel_size, sort_window", SAMPLER_GRID)
def test_sampler_gives_each_step_consecutive_batches_of_one_sorted_group(num_rows, batch_size, data_parallel_size,
                                                                        sort_window):
    sampler, lengths = _sampler(num_rows, batch_size, data_parallel_size, sort_window)
    steps = _steps(list(sampler), batch_size, data_parallel_size)
    for step in steps:
        rows = [i for batch in step for i in batch]
        full = len(step) == data_parallel_size and all(len(batch) == batch_size for batch in step)
        if full:
            # The batches of a full step are consecutive in their group's order, from the longest row
            assert all(lengths[rows[k]] >= lengths[rows[k + 1]] for k in range(len(rows) - 1))
    # Every step but the last is full
    assert all(len(step) == data_parallel_size and all(len(b) == batch_size for b in step) for step in steps[:-1])


# A step lasts as long as its longest row, which is its block's first, so with several blocks per sorted group,
# steps wait less than steps of batches from different groups. With 1 device the two orders are the same, and with
# few devices today's steps are short already, so only a strict improvement is checked in general.
@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("batch_size, data_parallel_size, sort_window",
                         [(4, 8, 50), (8, 2, 10), (4, 3, 10), (2, 4, 50), (4, 16, 64), (1, 32, 128)])
def test_sampler_in_steps_waits_less_than_shuffling_single_batches(seed, batch_size, data_parallel_size, sort_window):
    assert -(-sort_window // data_parallel_size) >= 4  # several blocks per group
    sampler, lengths = _sampler(2000, batch_size, data_parallel_size, sort_window, seed=seed)
    in_steps = _waiting(list(sampler), lengths, batch_size, data_parallel_size)
    single_batches = _waiting(_order_before_steps(sampler), lengths, batch_size, data_parallel_size)
    assert in_steps < single_batches


# For 8 devices and the default sort window, a group holds 7 blocks. With lengths drawn uniformly, a step's longest
# row is then on average about (7 + 1) / (2 * 7) = 0.57 of its group's longest, while a step of 8 batches from
# different groups nearly always holds a row close to it, so waiting should drop to about 0.6, below the bound of 0.8.
@pytest.mark.parametrize("seed", range(5))
def test_sampler_in_steps_waits_much_less_with_8_devices_and_the_default_window(seed):
    sampler, lengths = _sampler(2000, 4, 8, 50, seed=seed)
    in_steps = _waiting(list(sampler), lengths, 4, 8)
    single_batches = _waiting(_order_before_steps(sampler), lengths, 4, 8)
    assert in_steps < 0.8 * single_batches


@pytest.mark.parametrize("num_rows, batch_size, data_parallel_size, sort_window", SAMPLER_GRID)
def test_sampler_puts_the_longest_row_of_the_full_steps_first(num_rows, batch_size, data_parallel_size, sort_window):
    sampler, lengths = _sampler(num_rows, batch_size, data_parallel_size, sort_window)
    steps = _steps(list(sampler), batch_size, data_parallel_size)
    full = [step for step in steps if len(step) == data_parallel_size and all(len(b) == batch_size for b in step)]
    if full:
        assert steps[0] == full[0]
        assert max(lengths[i] for b in full[0] for i in b) == max(lengths[i] for s in full for b in s for i in b)


def test_sampler_in_steps_shuffles_its_steps_across_epochs_and_seeds():
    sampler, _ = _sampler(2000, 4, 8, 50, num_augmentations=2)
    orders = []
    for epoch in range(4):
        sampler.set_epoch(epoch)
        orders.append(list(sampler))
    assert orders[0] != orders[2]  # the same augmentation, reshuffled
    sampler.set_epoch(2)
    assert list(sampler) == orders[2]  # reproducible
    other, _ = _sampler(2000, 4, 8, 50, num_augmentations=2)
    other.seed = 1
    assert list(other) != orders[0]


def test_sampler_keeps_the_partial_block_and_the_short_last_batch_last():
    # 241 rows in batches of 4 on 8 devices: a first group of 56 batches (7 blocks), and a last group of 17 rows,
    # that is, 4 batches (a partial block) and a short batch of 1
    sampler, _ = _sampler(241, 4, 8, 50)
    order = list(sampler)
    batches = [order[i : i + 4] for i in range(0, len(order), 4)]
    assert [len(b) for b in batches] == [4] * 60 + [1]
    rows = list(range(241))
    random.Random(sampler.seed * 1_000_003 + sampler.epoch).shuffle(rows)  # the sampler's own shuffle
    last_group = set(rows[224:])
    assert {i for batch in batches[-5:] for i in batch} == last_group


def test_sampler_without_lengths_ignores_the_data_parallel_size():
    one = DecisionAugmentationSampler([0] * 50, batch_size=4, seed=3)
    eight = DecisionAugmentationSampler([0] * 50, batch_size=4, seed=3, data_parallel_size=8)
    assert list(one) == list(eight)


def test_sampler_rejects_a_data_parallel_size_below_one():
    with pytest.raises(ValueError, match="data_parallel_size"):
        DecisionAugmentationSampler([0, 0], batch_size=1, seed=0, data_parallel_size=0)


@pytest.mark.parametrize("sort_window", [1, 50])
@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("num_rows", [1, 5, 63, 64, 1001])
@pytest.mark.parametrize("sp_size", [1, 2])
@pytest.mark.parametrize("data_parallel_size", [1, 2, 3, 4, 7, 8, 16, 32, 64])
def test_sampler_serves_every_row_and_its_length_for_any_number_of_devices(data_parallel_size, sp_size, num_rows,
                                                                           batch_size, sort_window):
    lengths = np.random.default_rng(num_rows).integers(10, 4000, size=num_rows).tolist()
    sampler = DecisionAugmentationSampler([0] * num_rows, batch_size=batch_size, seed=0, lengths=lengths,
                                          sort_window=sort_window, repeat=sp_size,
                                          data_parallel_size=data_parallel_size)
    order = sampler._order()
    assert sorted(order) == list(range(num_rows))
    served = list(sampler)
    assert len(served) == len(sampler)
    assert set(served) == set(range(num_rows))
    # Under sequence parallelism, each batch comes sp_size times in a row, one copy per rank of the group
    batches = [served[i : i + batch_size] for i in range(0, len(served), batch_size)]
    for start in range(0, len(batches), sp_size):
        assert batches[start : start + sp_size] == [batches[start]] * sp_size


@pytest.mark.parametrize("num_rows", [1, 5, 63, 1001])
@pytest.mark.parametrize("sp_size, data_parallel_size", [(1, 1), (1, 3), (1, 8), (2, 4), (1, 16), (4, 4), (1, 64)])
def test_every_rank_gets_whole_batches_and_all_rows_are_trained(sp_size, data_parallel_size, num_rows):
    batch_size = 4
    lengths = np.random.default_rng(num_rows).integers(10, 4000, size=num_rows).tolist()
    sampler = DecisionAugmentationSampler([0] * num_rows, batch_size=batch_size, seed=0, lengths=lengths,
                                          repeat=sp_size, data_parallel_size=data_parallel_size)
    num_processes = sp_size * data_parallel_size
    by_rank = _batches_by_rank(sampler, batch_size, num_processes)
    # accelerate gives every rank the same number of whole batches, so no rank waits at a step that the others
    # skip. With one rank nothing waits, and the last batch can be short
    assert len({len(batches) for batches in by_rank}) == 1
    whole = [batch for batches in by_rank for batch in (batches if num_processes > 1 else batches[:-1])]
    assert all(len(batch) == batch_size for batch in whole)
    assert {i for batches in by_rank for batch in batches for i in batch} == set(range(num_rows))


def _batches_by_rank(sampler, batch_size, num_processes):
    """The batches of each rank, as HF's Trainer builds the training data loader and accelerate shards it."""
    from accelerate.data_loader import prepare_data_loader
    from torch.utils.data import DataLoader

    dataset = list(range(len(sampler.lengths) if sampler.lengths is not None else len(sampler)))
    by_rank = []
    for rank in range(num_processes):
        loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
        sharded = prepare_data_loader(loader, num_processes=num_processes, process_index=rank, put_on_device=False)
        by_rank.append([batch.tolist() for batch in sharded])
    return by_rank


@pytest.mark.parametrize("batch_size, data_parallel_size, sort_window", [(4, 8, 50), (4, 3, 10), (2, 4, 5)])
def test_accelerate_gives_the_devices_of_a_step_the_batches_of_one_block(batch_size, data_parallel_size, sort_window):
    sampler, lengths = _sampler(400, batch_size, data_parallel_size, sort_window)
    expected = _steps(list(sampler), batch_size, data_parallel_size)
    by_rank = _batches_by_rank(sampler, batch_size, data_parallel_size)
    full = [s for s in expected if len(s) == data_parallel_size and all(len(b) == batch_size for b in s)]
    for number, step in enumerate(full):
        assert [by_rank[rank][number] for rank in range(data_parallel_size)] == step


@pytest.mark.parametrize("sp_size, data_parallel_size", [(2, 2), (2, 4), (4, 2)])
def test_sequence_parallel_ranks_share_a_batch_and_a_step_trains_one_block(sp_size, data_parallel_size):
    batch_size = 2
    lengths = np.random.default_rng(0).integers(10, 4000, size=200).tolist()
    sampler = DecisionAugmentationSampler([0] * 200, batch_size=batch_size, seed=0, lengths=lengths, sort_window=6,
                                          repeat=sp_size, data_parallel_size=data_parallel_size)
    blocks = _steps(sampler._order(), batch_size, data_parallel_size)
    by_rank = _batches_by_rank(sampler, batch_size, sp_size * data_parallel_size)
    full = [s for s in blocks if len(s) == data_parallel_size and all(len(b) == batch_size for b in s)]
    for number, block in enumerate(full):
        batches = [by_rank[rank][number] for rank in range(sp_size * data_parallel_size)]
        # Ranks 0 to sp_size - 1 form the first sequence parallel group, and so on
        groups = [batches[g * sp_size : (g + 1) * sp_size] for g in range(data_parallel_size)]
        assert all(group == [group[0]] * sp_size for group in groups)
        assert [group[0] for group in groups] == block


@pytest.mark.parametrize("gradient_accumulation_steps", [1, 2, 4])
def test_each_micro_step_of_gradient_accumulation_trains_one_block(gradient_accumulation_steps):
    # The data loader serves micro-batches in order whatever gradient_accumulation_steps is, so each micro-step's
    # batches are one block of data_parallel_size batches
    batch_size, data_parallel_size = 4, 4
    sampler, _ = _sampler(400, batch_size, data_parallel_size, 10)
    blocks = _steps(list(sampler), batch_size, data_parallel_size)
    by_rank = _batches_by_rank(sampler, batch_size, data_parallel_size)
    micro_steps = list(zip(*by_rank))
    for step in range(len(micro_steps) // gradient_accumulation_steps):
        for micro in range(gradient_accumulation_steps):
            number = step * gradient_accumulation_steps + micro
            if len(blocks[number]) == data_parallel_size and all(len(b) == batch_size for b in blocks[number]):
                assert list(micro_steps[number]) == blocks[number]


@pytest.mark.parametrize("world_size, sp_size, expected", [(1, 1, 1), (8, 1, 8), (8, 2, 4), (8, 8, 1), (16, 4, 4)])
def test_trainer_gives_the_sampler_the_data_parallel_size(world_size, sp_size, expected):
    trainer = SimpleNamespace(
        train_dataset={"augmentation": [0] * 8, "input_ids": [[1, 2]] * 8},
        args=SimpleNamespace(train_sampling_strategy="random", per_device_train_batch_size=2, data_seed=None, seed=0,
                             world_size=world_size),
        finetuning_args=SimpleNamespace(disable_shuffling=False),
        _sequence_parallel_size=lambda: sp_size,
    )
    sampler = DecisionTrainer._get_train_sampler(trainer)
    assert sampler.data_parallel_size == expected and sampler.repeat == sp_size


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
        sequence_parallel_mode=overrides.pop("sequence_parallel_mode", "zigzag-ring"),
        disable_gradient_checkpointing=overrides.pop("disable_gradient_checkpointing", False),
    )
    data_args = SimpleNamespace(
        packing=overrides.pop("packing", None), decision_label_folder=overrides.pop("decision_label_folder", None)
    )
    training_args = SimpleNamespace(
        do_predict=overrides.pop("do_predict", False), predict_with_generate=False,
        deepspeed=overrides.pop("deepspeed", None),
        resume_from_checkpoint=overrides.pop("resume_from_checkpoint", None),
    )
    finetuning_args = SimpleNamespace(
        hybrid_learnable_params=None, finetuning_type="lora", use_galore=False, use_apollo=False, use_badam=False,
        use_adam_mini=False, use_muon=False, loraplus_lr_ratio=None, decision_anchor_kl_weight=0.0,
        decision_teacher_kl_weight=0.0, decision_checkpoint_min_tokens=0, decision_save_optimizer=False,
        decision_resume_optimizer=None,
    )
    for name, value in overrides.items():
        setattr(finetuning_args, name, value)
    return model_args, data_args, training_args, finetuning_args


def test_check_arguments_accepts_defaults():
    workflow._check_arguments(*_workflow_args())
    workflow._check_arguments(*_workflow_args(finetuning_type="full", hybrid_learnable_params="decision_head"))
    workflow._check_arguments(*_workflow_args(decision_label_folder="labels_a", decision_teacher_kl_weight=1.0))
    workflow._check_arguments(*_workflow_args(decision_checkpoint_min_tokens=4096))  # with checkpointing on
    workflow._check_arguments(*_workflow_args(sequence_parallel_size=2))  # zigzag-ring
    workflow._check_arguments(*_workflow_args(decision_save_optimizer=True, decision_resume_optimizer="earlier"))


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"sequence_parallel_size": 2, "sequence_parallel_mode": "ulysses"}, "zigzag-ring only"),
        ({"packing": True}, "Packing"),
        ({"do_predict": True}, "Prediction"),
        ({"hybrid_learnable_params": "decision_head"}, "cannot be combined with LoRA"),
        ({"use_galore": True}, "use_galore"),
        ({"loraplus_lr_ratio": 16.0}, "loraplus_lr_ratio"),
        ({"decision_anchor_kl_weight": 0.3}, "needs anchor or teacher targets: set decision_label_folder"),
        ({"decision_label_folder": "labels_a", "decision_teacher_kl_weight": -1.0}, "must be at least 0"),
        ({"decision_checkpoint_min_tokens": -1}, "decision_checkpoint_min_tokens must be at least 0"),
        ({"decision_checkpoint_min_tokens": 4096, "disable_gradient_checkpointing": True}, "Remove one of them"),
        ({"decision_save_optimizer": True, "deepspeed": "ds.json"}, "do not support DeepSpeed"),
        ({"decision_resume_optimizer": "earlier", "deepspeed": "ds.json"}, "do not support DeepSpeed"),
        ({"decision_resume_optimizer": "earlier", "resume_from_checkpoint": "ckpt"}, "use one of them"),
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
    "max_steps, epochs, warns, world_size, sp_size",
    [
        (0, 2.0, False, 1, 1), (0, 3.0, True, 1, 1), (5, 1.0, False, 1, 1), (6, 1.0, True, 1, 1),
        # The 2 ranks of a sequence parallel group share their rows, so the global batch is still 8
        (5, 1.0, False, 2, 2), (6, 1.0, True, 2, 2),
    ],
)
def test_warns_if_augmentations_reused(monkeypatch, max_steps, epochs, warns, world_size, sp_size):
    warnings = []
    monkeypatch.setattr(workflow.logger, "warning_rank0", warnings.append)
    trainer = SimpleNamespace(train_dataset={"augmentation": [0, 1] * 20})  # 20 rows, 2 augmentations
    training_args = SimpleNamespace(
        per_device_train_batch_size=4, gradient_accumulation_steps=2, world_size=world_size,  # global batch of 8
        max_steps=max_steps, num_train_epochs=epochs,
    )

    workflow._warn_if_augmentations_reused(trainer, training_args, sp_size)

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


def _train_with_accumulation(tmp_path, model_dir, lora_target, batch_size, accumulation):
    """Trains a tiny decision model for an epoch over 28 rows in order, and returns its adapter and logged norms."""
    from safetensors.torch import load_file

    data_dir = tmp_path / f"data_{batch_size}_{accumulation}"
    data_dir.mkdir()
    rows = [  # 3 steps of 8 rows, then one of 4, all of weight 1
        {
            "kind": "choice",
            "state": f"state {' '.join(['words'] * (1 + i % 7))} {i}",
            "instructions": f"question {i}?",
            "options": [["refund", "return it"], ["ignore", None], ["escalate", "to a human"]],
            "label": i % 3,
            "task": "a",
        }
        for i in range(28)
    ]
    (data_dir / "toy.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    (data_dir / "dataset_info.json").write_text(
        json.dumps({"toy_decision": {"file_name": "toy.jsonl", "formatting": "decision"}})
    )
    output_dir = tmp_path / f"output_{batch_size}_{accumulation}"
    run_exp(
        {
            "model_name_or_path": model_dir,
            "stage": "decision",
            "do_train": True,
            "finetuning_type": "lora",
            "lora_target": lora_target,
            "lora_rank": 4,
            "lora_dropout": 0.0,
            "dataset": "toy_decision",
            "dataset_dir": str(data_dir),
            "cutoff_len": 512,
            "decision_num_augmentations": 1,
            "num_train_epochs": 1,
            "per_device_train_batch_size": batch_size,
            "gradient_accumulation_steps": accumulation,
            "learning_rate": 1e-3,
            "decision_head_lr": 1e-2,
            "max_grad_norm": 1.0,
            "train_sampling_strategy": "sequential",
            "save_strategy": "no",
            "logging_steps": 1,
            "output_dir": str(output_dir),
            "overwrite_output_dir": True,
            "overwrite_cache": True,
            "report_to": "none",
            "use_cpu": True,
            "seed": 0,
        }
    )
    with open(output_dir / "trainer_state.json") as f:
        norms = [entry["grad_norm"] for entry in json.load(f)["log_history"] if "grad_norm" in entry]
    return load_file(str(output_dir / "adapter_model.safetensors")), norms


def test_gradient_accumulation_matches_a_larger_batch(tmp_path, decision_model_dir, lora_targets, device_and_dtype):
    # Without DeepSpeed, Accelerate's backward divides the loss by gradient_accumulation_steps after the Trainer
    # has, so DecisionTrainer.training_step undoes the second division. Then 4 rows accumulated twice train as 8
    # rows at once, including the last step of the epoch, which has a single batch of 4 (the Trainer divides it
    # by 1, as its group is short).
    if device_and_dtype[0] != "cpu":
        pytest.skip("compared in float32 on the CPU")
    model_dir, target = decision_model_dir("qwen3"), lora_targets["qwen3"]
    accumulated, accumulated_norms = _train_with_accumulation(tmp_path, model_dir, target, 4, 2)
    batched, batched_norms = _train_with_accumulation(tmp_path, model_dir, target, 8, 1)

    assert len(accumulated_norms) == len(batched_norms) == 4
    assert accumulated_norms == pytest.approx(batched_norms, rel=1e-4)
    assert accumulated.keys() == batched.keys()
    for name in accumulated:
        # The k bias of the head shifts every option's logit equally, so its true gradient is 0, and Adam turns
        # rounding noise in it into steps of different signs from run to run
        if "decision_head" in name and "k.bias" in name:
            continue
        torch.testing.assert_close(accumulated[name], batched[name], atol=1e-5, rtol=1e-4, msg=name)


# Resuming the optimizer's state


def _choice_rows(start: int, stop: int) -> list:
    """Choice rows numbered start to stop - 1, all of weight 1, the same whichever run they are in."""
    return [
        {
            "kind": "choice",
            "state": f"state {' '.join(['words'] * (1 + i % 7))} {i}",
            "instructions": f"question {i}?",
            "options": [["refund", "return it"], ["ignore", None], ["escalate", "to a human"]],
            "label": i % 3,
            "task": "a",
        }
        for i in range(start, stop)
    ]


def _train_on(tmp_path, name, model_dir, lora_target, rows, **overrides):
    """
    Trains a tiny decision model for an epoch over rows in order, on the CPU, with the options in a fixed order and
    a constant learning rate without warmup, and returns its output_dir. With options shuffled, a row's order would
    depend on its position in its own data file, so the same row would be shown differently in a split run.
    """
    data_dir = tmp_path / f"data_{name}"
    data_dir.mkdir()
    (data_dir / "toy.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    (data_dir / "dataset_info.json").write_text(
        json.dumps({"toy_decision": {"file_name": "toy.jsonl", "formatting": "decision"}})
    )
    output_dir = tmp_path / f"output_{name}"
    args = {
        "model_name_or_path": model_dir,
        "stage": "decision",
        "do_train": True,
        "finetuning_type": "lora",
        "lora_target": lora_target,
        "lora_rank": 4,
        "lora_dropout": 0.0,
        "dataset": "toy_decision",
        "dataset_dir": str(data_dir),
        "cutoff_len": 512,
        "decision_num_augmentations": 1,
        "decision_shuffle_options": False,
        "num_train_epochs": 1,
        "per_device_train_batch_size": 4,
        "learning_rate": 1e-3,
        "decision_head_lr": 1e-2,
        "lr_scheduler_type": "constant",
        "warmup_steps": 0,
        "train_sampling_strategy": "sequential",
        "save_strategy": "no",
        "logging_steps": 1,
        "output_dir": str(output_dir),
        "overwrite_output_dir": True,
        "overwrite_cache": True,
        "report_to": "none",
        "use_cpu": True,
        "seed": 0,
    }
    args.update(overrides)
    run_exp(args)
    return output_dir


def _adapter_and_logs(output_dir):
    from safetensors.torch import load_file

    with open(output_dir / "trainer_state.json") as f:
        logs = [entry for entry in json.load(f)["log_history"] if "grad_norm" in entry]
    return load_file(str(output_dir / "adapter_model.safetensors")), logs


def _assert_adapters_close(a, b):
    assert a.keys() == b.keys()
    for name in a:
        # The k bias of the head has a true gradient of 0, so Adam turns rounding noise in it into steps of
        # different signs from run to run (see test_gradient_accumulation_matches_a_larger_batch)
        if "decision_head" in name and "k.bias" in name:
            continue
        torch.testing.assert_close(a[name], b[name], atol=1e-5, rtol=1e-4, msg=name)


def test_resumed_optimizer_continues_training_as_one_run(tmp_path, decision_model_dir, lora_targets, device_and_dtype):
    # Training on 16 rows at once, or on the first 8 and then the last 8, continuing the same adapter from the
    # saved optimizer state, gives the same adapter and the same logged gradient norms. Without the optimizer
    # state, the second run starts Adam over, and the adapter ends up elsewhere.
    if device_and_dtype[0] != "cpu":
        pytest.skip("compared in float32 on the CPU")
    model_dir, target = decision_model_dir("qwen3"), lora_targets["qwen3"]
    whole = _train_on(tmp_path, "whole", model_dir, target, _choice_rows(0, 16))
    first = _train_on(tmp_path, "first", model_dir, target, _choice_rows(0, 8), decision_save_optimizer=True)
    resumed = _train_on(tmp_path, "resumed", model_dir, target, _choice_rows(8, 16),
                        adapter_name_or_path=str(first), decision_resume_optimizer=str(first),
                        decision_save_optimizer=True)
    restarted = _train_on(tmp_path, "restarted", model_dir, target, _choice_rows(8, 16),
                          adapter_name_or_path=str(first))

    # The state is saved only when asked for
    assert (first / "decision_optimizer.pt").is_file() and (resumed / "decision_optimizer.pt").is_file()
    assert not (whole / "decision_optimizer.pt").exists() and not (restarted / "decision_optimizer.pt").exists()

    whole_adapter, whole_logs = _adapter_and_logs(whole)
    resumed_adapter, resumed_logs = _adapter_and_logs(resumed)
    _, first_logs = _adapter_and_logs(first)
    _assert_adapters_close(resumed_adapter, whole_adapter)
    assert len(whole_logs) == 4 and len(first_logs) == len(resumed_logs) == 2
    assert [e["grad_norm"] for e in first_logs + resumed_logs] == pytest.approx(
        [e["grad_norm"] for e in whole_logs], rel=1e-4
    )

    # Restarted with a fresh optimizer, the second half trains the adapter differently
    restarted_adapter, _ = _adapter_and_logs(restarted)
    lora = [name for name in whole_adapter if "lora_" in name]
    assert max((restarted_adapter[n] - whole_adapter[n]).abs().max().item() for n in lora) > 1e-4


def test_resumed_optimizer_takes_the_learning_rate_of_the_new_run(tmp_path, decision_model_dir, lora_targets,
                                                                  device_and_dtype):
    # Only each parameter's state is resumed, so a run with another learning rate trains at its own
    if device_and_dtype[0] != "cpu":
        pytest.skip("trained on the CPU")
    model_dir, target = decision_model_dir("qwen3"), lora_targets["qwen3"]
    first = _train_on(tmp_path, "first", model_dir, target, _choice_rows(0, 8), decision_save_optimizer=True)
    resumed = _train_on(tmp_path, "resumed", model_dir, target, _choice_rows(8, 16), learning_rate=5e-4,
                        decision_head_lr=5e-3, adapter_name_or_path=str(first), decision_resume_optimizer=str(first))
    _, logs = _adapter_and_logs(resumed)
    assert {entry["learning_rate"] for entry in logs} == {5e-4}
    assert {entry["head_learning_rate"] for entry in logs} == {5e-3}


def test_resumed_optimizer_needs_the_same_parameters(tmp_path, decision_model_dir, lora_targets, device_and_dtype):
    # A state saved for other parameters (here, LoRA on fewer layers) is refused, and so is a folder without one
    if device_and_dtype[0] != "cpu":
        pytest.skip("trained on the CPU")
    model_dir = decision_model_dir("qwen3")
    fewer = _train_on(tmp_path, "fewer", model_dir, "q_proj,v_proj", _choice_rows(0, 8), decision_save_optimizer=True)
    with pytest.raises(ValueError, match="belongs to other parameters"):
        _train_on(tmp_path, "mismatch", model_dir, lora_targets["qwen3"], _choice_rows(8, 16),
                  decision_resume_optimizer=str(fewer))
    with pytest.raises(FileNotFoundError, match="decision_save_optimizer"):
        _train_on(tmp_path, "missing", model_dir, lora_targets["qwen3"], _choice_rows(8, 16),
                  decision_resume_optimizer=str(tmp_path))
