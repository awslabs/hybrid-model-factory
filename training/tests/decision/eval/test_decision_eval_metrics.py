"""
Tests for the confidence of a decision model's answer and the metrics of a set of answers
(hmf/eval/decision/metrics.py), on cases computed by hand.
"""

import math

import pytest

from hmf.eval.decision.metrics import (
    Prediction,
    apply_temperature,
    choice_confidence,
    confidence,
    format_summary,
    predictions,
    score_confidence,
    summarize,
)


# Confidence


@pytest.mark.parametrize(
    "probs, expected",
    [
        ([1 / 3, 1 / 3, 1 / 3], 0.0),  # uniform
        ([0.0, 1.0, 0.0], 1.0),  # certain
        ([0.75, 0.25], 0.5),  # (2 * 0.75 - 1) / 1
        ([0.6, 0.2, 0.2], 0.4),  # (3 * 0.6 - 1) / 2
        ([1.0], 1.0),  # one option, so the answer is forced
    ],
)
def test_choice_confidence(probs, expected):
    assert choice_confidence(probs) == pytest.approx(expected)


def test_score_confidence():
    # One level, so no spread
    assert score_confidence([0.0, 1.0, 0.0], ordinal_smoothing=0.0) == pytest.approx(1.0)
    # Mass split between the two ends, the most spread there is
    assert score_confidence([0.5, 0.0, 0.5], ordinal_smoothing=0.0) == pytest.approx(0.0)
    # Mass split between neighbors is a confident "in between", unlike the top probability
    neighbors = [0.0, 0.5, 0.5, 0.0, 0.0]
    assert score_confidence(neighbors, ordinal_smoothing=0.0) == pytest.approx((2 - 0.5) / 2)
    assert choice_confidence(neighbors) == pytest.approx((5 * 0.5 - 1) / 4)
    # With ordinal smoothing, the spread a perfectly fitted model keeps is full confidence
    smoothed = [0.05, 0.9, 0.05]  # sigma = sqrt(0.1)
    assert score_confidence(smoothed, ordinal_smoothing=0.1) == pytest.approx(1.0)
    assert score_confidence(smoothed, ordinal_smoothing=0.0) == pytest.approx(1 - math.sqrt(0.1))


def test_confidence_by_kind():
    probs = [0.0, 0.5, 0.5, 0.0, 0.0]
    assert confidence("score", probs, 0.0) == score_confidence(probs, 0.0)
    assert confidence("choice", probs) == choice_confidence(probs)
    assert confidence("noul", [0.25, 0.75]) == choice_confidence([0.25, 0.75])


# Temperatures


def test_apply_temperature():
    log_probs = [math.log(0.7), math.log(0.2), math.log(0.1)]
    assert apply_temperature(log_probs, 1.0) == pytest.approx([0.7, 0.2, 0.1])
    soft, sharp = apply_temperature(log_probs, 3.0), apply_temperature(log_probs, 0.5)
    assert max(soft) < 0.7 < max(sharp)  # above 1 softens, below 1 sharpens
    for probs in (soft, sharp):
        assert probs.index(max(probs)) == 0 and sum(probs) == pytest.approx(1.0)  # the answer is unchanged


def test_predictions_with_temperatures_by_kind():
    log_probs = [[math.log(0.7), math.log(0.3)]] * 2
    preds = predictions(["noul", "choice"], ["t", "t"], [0, 0], log_probs, {"noul": 2.0}, default_temperature=0.5)
    assert preds[0].probs == pytest.approx(apply_temperature(log_probs[0], 2.0))  # its own
    assert preds[1].probs == pytest.approx(apply_temperature(log_probs[1], 0.5))  # the default


# Metrics


def _preds():
    return [
        Prediction("choice", "a", 0, [0.8, 0.1, 0.1]),  # right, confidence 0.7
        Prediction("choice", "a", 1, [0.6, 0.3, 0.1]),  # wrong, confidence 0.4
        Prediction("noul", "b", 1, [0.04, 0.96]),  # right, confidence 0.92
        Prediction("score", "b", 2, [0.0, 0.1, 0.9], ordinal_smoothing=0.0),  # right, expected level 1.9
    ]


def test_summary():
    preds = _preds()
    summary = summarize(preds)
    overall = summary["overall"]

    assert overall["n"] == 4 and overall["accuracy"] == pytest.approx(0.75)
    assert overall["nll"] == pytest.approx(-(math.log(0.8) + math.log(0.3) + math.log(0.96) + math.log(0.9)) / 4)
    brier = [0.2**2 + 0.1**2 + 0.1**2, 0.6**2 + 0.7**2 + 0.1**2, 0.04**2 * 2, 0.1**2 * 2]
    assert overall["brier"] == pytest.approx(sum(brier) / 4)
    assert overall["score_mae"] == pytest.approx(0.1)  # |1.9 - 2|
    assert overall["mean_confidence"] == pytest.approx(sum(p.confidence for p in preds) / 4)
    assert summary["by_kind"]["choice"]["n"] == 2 and summary["by_kind"]["choice"]["accuracy"] == 0.5
    assert set(summary["by_task"]) == {"a", "b"}

    # Bands by confidence: 0.4 is low, 0.7 mid, 0.92 high (and the score answer, 1 - 0.3 / 1 = 0.7, mid)
    assert summary["by_confidence_band"]["low(<0.5)"]["n"] == 1
    assert summary["by_confidence_band"]["mid(0.5-0.9)"]["n"] == 2
    assert summary["by_confidence_band"]["high(>=0.9)"]["n"] == 1


def test_ece():
    # The bins' |accuracy - mean confidence|, weighted by their size. The two right answers at
    # confidence 0.7 contribute 0.3 each, the wrong one at 0.4 contributes 0.4, and the right one at 0.92
    # contributes 0.08
    ece = summarize(_preds())["overall"]["ece"]
    assert ece == pytest.approx((0.3 + 0.4 + 0.08 + 0.3) / 4)
    # A certain, right answer is perfectly calibrated
    assert summarize([Prediction("choice", "a", 0, [1.0, 0.0])])["overall"]["ece"] == pytest.approx(0.0)


def test_empty_groups_and_format():
    summary = summarize(_preds())
    assert summary["by_kind"]["score"]["n"] == 1
    assert summarize([])["overall"] == {}
    text = format_summary(summary, title="toy")
    assert text.startswith("toy") and "by_kind" in text and "by_confidence_band" in text
