"""
The confidence of a decision model's answer, and the metrics of a set of answers.

Confidence is not the top probability. It is the number a decision server reports with each answer,
defined (as in Strands Decider) so that a threshold means the same thing whatever the number of
options:

    - noul and choice: the normalized top probability, (N * p_max - 1) / (N - 1), which is 0 for a
      uniform distribution and 1 for a certain one.
    - score: how tightly the probability mass clusters on the ordered levels,
      (sigma_max - sigma) / (sigma_max - sqrt(eps)), clipped to [0, 1], where sigma is the standard
      deviation of the level under the model's probabilities, sigma_max = (L - 1) / 2 the largest
      possible (half the mass on each end of L levels), and eps the ordinal smoothing of training.
      Training targets put eps of the mass on neighboring levels, so a model that fits them exactly
      has sigma = sqrt(eps) and gets confidence 1. Mass split between two neighboring levels is a
      confident "in between", not doubt.

The metrics of a set of answers (summarize), each overall, by kind, by task, and by confidence band:

    - accuracy: the fraction of answers whose most probable option is the label.
    - ece: the expected calibration error of the confidence, over 10 equal-width bins.
    - ece_top_probability: the same, with the top probability as the confidence (what the training
      logs report as eval_ece).
    - nll: the mean negative log-probability of the label.
    - brier: the mean over answers of sum_k (p_k - y_k)^2, with y the one-hot label.
    - mean_confidence: the mean confidence.
    - score_mae: for score answers, the mean absolute difference between the expected level and the
      label, in levels.
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import torch

from ...data.decision_prompt import DECISION_KINDS
from ...train.decision.metric import NUM_ECE_BINS, expected_calibration_error


# The ordinal smoothing of training (decision_ordinal_smoothing), the floor of score confidence
DEFAULT_ORDINAL_SMOOTHING = 0.1
# The confidence bands: [low, high) of each
CONFIDENCE_BANDS = {"low(<0.5)": (0.0, 0.5), "mid(0.5-0.9)": (0.5, 0.9), "high(>=0.9)": (0.9, 1.01)}

# A temperature for every kind, or one per kind
Temperature = Union[float, Dict[str, float]]


def choice_confidence(probs: Sequence[float]) -> float:
    """The confidence of a noul or choice answer, (N * p_max - 1) / (N - 1), in [0, 1]."""
    n = len(probs)
    if n <= 1:
        return 1.0
    return float(max(0.0, min(1.0, (n * max(probs) - 1.0) / (n - 1.0))))


def score_confidence(probs: Sequence[float], ordinal_smoothing: float = DEFAULT_ORDINAL_SMOOTHING) -> float:
    """
    The confidence of a score answer, (sigma_max - sigma) / (sigma_max - sigma_floor), in [0, 1],
    where sigma is the standard deviation of the levels, sigma_max = (L - 1) / 2 (the mass split
    between the two ends) and sigma_floor = sqrt(ordinal_smoothing).
    """
    n = len(probs)
    if n <= 1:
        return 1.0
    total = sum(probs)
    if total <= 0:
        return 0.0
    p = [x / total for x in probs]
    mean = sum(i * pi for i, pi in enumerate(p))
    sigma = sum(pi * (i - mean) ** 2 for i, pi in enumerate(p)) ** 0.5
    sigma_max = (n - 1) / 2.0
    sigma_floor = ordinal_smoothing**0.5 if ordinal_smoothing > 0 else 0.0
    if sigma_max <= sigma_floor:  # a floor at or above the ceiling (few levels, large smoothing)
        return 1.0 if sigma <= sigma_floor else 0.0
    return float(max(0.0, min(1.0, (sigma_max - sigma) / (sigma_max - sigma_floor))))


def confidence(kind: str, probs: Sequence[float], ordinal_smoothing: float = DEFAULT_ORDINAL_SMOOTHING) -> float:
    """The confidence of an answer of the given kind."""
    if kind == "score":
        return score_confidence(probs, ordinal_smoothing)
    return choice_confidence(probs)


@dataclass
class Prediction:
    """
    A decision model's answer to one row.

    Args:
        kind: One of DECISION_KINDS.
        task: The row's task.
        label: The index of the correct option, in original order.
        probs: The probability of each option, in original order.
        ordinal_smoothing: The floor of score confidence (see score_confidence).
    """

    kind: str
    task: str
    label: int
    probs: List[float]
    ordinal_smoothing: float = DEFAULT_ORDINAL_SMOOTHING

    @property
    def pred(self) -> int:
        return max(range(len(self.probs)), key=self.probs.__getitem__)

    @property
    def correct(self) -> bool:
        return self.pred == self.label

    @property
    def confidence(self) -> float:
        return confidence(self.kind, self.probs, self.ordinal_smoothing)


def temperature_for(temperature: Temperature, kind: str, default: float = 1.0) -> float:
    """The temperature of a kind, which is its own if one was fitted, else the default."""
    if isinstance(temperature, dict):
        return float(temperature.get(kind, default))
    return float(temperature)


def apply_temperature(log_probs: Sequence[float], temperature: float) -> List[float]:
    """The probabilities at a temperature, softmax(log_probs / temperature)."""
    return torch.softmax(torch.tensor(log_probs, dtype=torch.float64) / temperature, dim=-1).tolist()


def predictions(
    kinds: Sequence[str],
    tasks: Sequence[str],
    labels: Sequence[int],
    log_probs: Sequence[Sequence[float]],
    temperature: Temperature = 1.0,
    ordinal_smoothing: float = DEFAULT_ORDINAL_SMOOTHING,
    default_temperature: float = 1.0,
) -> List[Prediction]:
    """
    The answers to some rows at a temperature (one for every kind, or one per kind).

    Args:
        kinds: The kind of each row.
        tasks: The task of each row.
        labels: The label of each row.
        log_probs: The model's log-probabilities over each row's options, at temperature 1.
        temperature: A temperature, or one per kind. Defaults to 1.
        ordinal_smoothing: The floor of score confidence. Defaults to 0.1.
        default_temperature: The temperature of kinds without one of their own, when temperature is
            per kind. Defaults to 1.
    """
    results = []
    for kind, task, label, lp in zip(kinds, tasks, labels, log_probs):
        probs = apply_temperature(lp, temperature_for(temperature, kind, default_temperature))
        results.append(Prediction(kind, task, int(label), probs, ordinal_smoothing))
    return results


def _block(items: List[Prediction]) -> Dict[str, float]:
    """The metrics of a set of answers."""
    if not items:
        return {}
    correct = np.array([p.correct for p in items], dtype=float)
    conf = np.array([p.confidence for p in items])
    top = np.array([max(p.probs) for p in items])
    block = {
        "n": len(items),
        "accuracy": float(correct.mean()),
        "ece": expected_calibration_error(conf, correct, NUM_ECE_BINS),
        "ece_top_probability": expected_calibration_error(top, correct, NUM_ECE_BINS),
        "nll": float(np.mean([-math.log(max(1e-12, p.probs[p.label])) for p in items])),
        "brier": float(np.mean([sum((q - (k == p.label)) ** 2 for k, q in enumerate(p.probs)) for p in items])),
        "mean_confidence": float(conf.mean()),
    }
    scores = [p for p in items if p.kind == "score"]
    if scores:
        block["score_mae"] = float(np.mean([abs(sum(i * q for i, q in enumerate(p.probs)) - p.label) for p in scores]))
    return block


def summarize(preds: List[Prediction], by_task: bool = True) -> Dict[str, object]:
    """
    The metrics of a set of answers, overall, by kind, by task, and by confidence band. The bands show
    whether confidence separates right from wrong answers, in which case the accuracy of the high band
    is clearly above that of the low one.
    """
    result: Dict[str, object] = {
        "overall": _block(preds),
        "by_kind": {kind: _block([p for p in preds if p.kind == kind]) for kind in DECISION_KINDS},
    }
    if by_task:
        tasks = sorted({p.task for p in preds})
        result["by_task"] = {task: _block([p for p in preds if p.task == task]) for task in tasks}
    result["by_confidence_band"] = {
        band: _block([p for p in preds if low <= p.confidence < high]) for band, (low, high) in CONFIDENCE_BANDS.items()
    }
    return result


def format_summary(summary: Dict[str, object], title: Optional[str] = None) -> str:
    """The summary as text tables."""
    lines = [title] if title else []
    overall = summary["overall"]
    if overall:
        lines.append(
            f"overall  n={overall['n']:,}  accuracy={overall['accuracy']:.3f}  ece={overall['ece']:.3f}  "
            f"nll={overall['nll']:.3f}  brier={overall['brier']:.3f}"
        )
    for section in ("by_kind", "by_task", "by_confidence_band"):
        rows = {name: block for name, block in summary.get(section, {}).items() if block}
        if not rows:
            continue
        width = max(len(name) for name in rows)
        lines.append(f"\n{section}")
        lines.append(f"  {'group':<{width}}  {'n':>7}  {'accuracy':>8}  {'ece':>6}  {'mean conf':>9}")
        for name, block in rows.items():
            lines.append(
                f"  {name:<{width}}  {block['n']:>7,}  {block['accuracy']:>8.3f}  {block['ece']:>6.3f}  "
                f"{block['mean_confidence']:>9.3f}"
            )
    return "\n".join(lines)
