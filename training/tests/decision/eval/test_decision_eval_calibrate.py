"""
Tests for fitting temperatures (hmf/eval/decision/calibrate.py), on synthetic answers whose right
temperature is known. The labels are drawn from softmax(logits / T_true), so the model's raw
probabilities, softmax(logits), are miscalibrated by T_true.
"""

import random

import pytest
import torch

from hmf.eval.decision.calibrate import (
    GRID_SIZE,
    ece_at,
    fit_temperature_ece,
    fit_temperature_nll,
    fit_temperatures,
    temperature_grid,
)


def _synthetic(num_rows: int, true_temperature: float, kind: str = "choice", num_options: int = 4, seed: int = 0):
    generator = torch.Generator().manual_seed(seed)
    logits = torch.randn(num_rows, num_options, generator=generator) * 3
    labels = torch.multinomial(torch.softmax(logits / true_temperature, dim=-1), 1, generator=generator).squeeze(-1)
    log_probs = torch.log_softmax(logits, dim=-1).tolist()
    return [kind] * num_rows, labels.tolist(), log_probs


def test_grid():
    grid = temperature_grid()
    assert len(grid) == GRID_SIZE and grid[0] == pytest.approx(0.25) and grid[-1] == pytest.approx(6.0)
    ratios = [b / a for a, b in zip(grid, grid[1:])]
    assert max(ratios) == pytest.approx(min(ratios))  # evenly spaced in log scale


@pytest.mark.parametrize("true_temperature", [0.6, 2.0])
def test_nll_fit_recovers_the_temperature(true_temperature):
    _, labels, log_probs = _synthetic(20000, true_temperature)
    assert fit_temperature_nll(log_probs, labels) == pytest.approx(true_temperature, rel=0.08)


@pytest.mark.parametrize("true_temperature", [0.6, 2.0])
def test_ece_fit_improves_calibration(true_temperature):
    kinds, labels, log_probs = _synthetic(4000, true_temperature)  # the ECE grid search is the slow part
    fitted = fit_temperature_ece(kinds, labels, log_probs)

    assert fitted in temperature_grid()
    assert ece_at(kinds, labels, log_probs, fitted) < ece_at(kinds, labels, log_probs, 1.0)
    # Over-confident raw probabilities (labels drawn from softer ones) get a temperature above 1, and
    # under-confident ones below
    assert (fitted > 1) == (true_temperature > 1)


def test_kinds_with_few_rows_use_the_overall_temperature():
    choice = _synthetic(600, 2.0, kind="choice")
    noul = _synthetic(50, 2.0, kind="noul", num_options=2, seed=1)
    kinds, labels, log_probs = (a + b for a, b in zip(choice, noul))

    overall, by_kind = fit_temperatures(kinds, labels, log_probs, min_rows=200)

    assert set(by_kind) == {"choice"}  # noul has 50 rows
    assert 0.25 <= overall <= 6.0


def test_temperatures_never_change_the_answer():
    _, _, log_probs = _synthetic(200, 1.0)
    for temperature in (0.3, 1.0, 4.0):
        scaled = torch.softmax(torch.tensor(log_probs) / temperature, dim=-1)
        assert scaled.argmax(-1).tolist() == torch.tensor(log_probs).argmax(-1).tolist()


def test_rows_with_different_numbers_of_options():
    # Rows with 2, 3, and 4 options together, where the padding of the shorter rows carries no probability
    random.seed(0)
    log_probs = [torch.log_softmax(torch.randn(n), -1).tolist() for n in [2, 3, 4] * 100]
    labels = [random.randrange(len(lp)) for lp in log_probs]
    temperature = fit_temperature_nll(log_probs, labels)
    assert 0.25 <= temperature <= 6.0
