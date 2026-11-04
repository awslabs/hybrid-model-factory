"""
Tests for the decision loss of the MLX backend (hmf/train/decision/mlx/loss.py), against the PyTorch trainer's
decision_loss: the loss, its parts, and its gradient with respect to the logits, with padded options, rows of weight
0, targets that only some rows have, and targets of 0.
"""

import pytest


pytest.importorskip("mlx.core", reason="needs MLX (Apple silicon)")
pytest.importorskip("mlx_lm", reason="needs mlx-lm")

import mlx.core as mx
import numpy as np
import torch

from hmf.model.hybrid_zoo.models.decision.modeling_decision import masked_log_softmax as torch_masked_log_softmax
from hmf.train.decision.mlx.loss import decision_loss
from hmf.train.decision.mlx.model import masked_log_softmax
from hmf.train.decision.trainer import decision_loss as torch_decision_loss


NUM_OPTIONS = [4, 3, 2, 4, 2, 3]


def _batch(has_ref, has_teacher) -> dict:
    """A batch of six rows with up to four options, in PyTorch."""
    generator = torch.Generator().manual_seed(0)
    rows, width = len(NUM_OPTIONS), max(NUM_OPTIONS)
    option_mask = torch.tensor([[k < n for k in range(width)] for n in NUM_OPTIONS])

    def distributions(zero_first: bool = False) -> torch.Tensor:
        probs = torch.rand(rows, width, generator=generator).masked_fill(~option_mask, 0.0)
        if zero_first:
            probs[0, 1] = 0.0  # a target of exactly 0 in a real option
        return probs / probs.sum(-1, keepdim=True)

    label_dist = torch.zeros(rows, width)
    label_dist[torch.arange(rows), torch.tensor([0, 2, 1, 3, 0, 1])] = 1.0
    label_dist[3] = torch.tensor([0.0, 0.05, 0.9, 0.05])  # a smoothed score target
    return {
        "logits": torch.randn(rows, width, generator=generator),
        "option_mask": option_mask,
        "label_dist": label_dist,
        "weight": torch.tensor([1.0, 0.5, 0.0, 2.0, 1.0, 1.0]),
        "ref_probs": distributions(zero_first=True),
        "has_ref": torch.tensor(has_ref),
        "teacher_probs": distributions(),
        "has_teacher": torch.tensor(has_teacher),
    }


def _to_mlx(batch: dict) -> dict:
    return {name: mx.array(value.numpy()) for name, value in batch.items()}


@pytest.mark.parametrize(
    "has_ref, has_teacher",
    [
        ([True, False, True, True, False, True], [False, True, False, False, True, False]),
        ([False] * 6, [True] * 6),
    ],
)
@pytest.mark.parametrize("weights", [(0.0, 0.0), (0.3, 1.0)])
def test_loss_matches_pytorch(has_ref, has_teacher, weights):
    batch = _batch(has_ref, has_teacher)
    anchor_kl_weight, teacher_kl_weight = weights

    def mlx_loss(logits):
        log_probs = masked_log_softmax(logits, mlx_batch["option_mask"])
        return decision_loss(log_probs, mlx_batch, anchor_kl_weight, teacher_kl_weight)

    mlx_batch = _to_mlx(batch)
    loss, parts = mlx_loss(mlx_batch["logits"])
    grad = mx.grad(lambda logits: mlx_loss(logits)[0])(mlx_batch["logits"])

    logits = batch["logits"].clone().requires_grad_(True)
    log_probs = torch_masked_log_softmax(logits, batch["option_mask"])
    expected_loss, expected_parts = torch_decision_loss(log_probs, batch, anchor_kl_weight, teacher_kl_weight)
    expected_loss.backward()

    assert float(loss) == pytest.approx(expected_loss.item(), abs=1e-6)
    assert set(parts) == set(expected_parts)
    for name, value in expected_parts.items():
        assert float(parts[name]) == pytest.approx(value.item(), abs=1e-6), name
    np.testing.assert_allclose(np.array(grad), logits.grad.numpy(), atol=1e-6)
    assert np.all(np.isfinite(np.array(grad)))
