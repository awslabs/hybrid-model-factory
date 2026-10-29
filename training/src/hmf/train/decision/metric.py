"""
Evaluation metrics for decision models, computed from the per-row statistics that
DecisionTrainer.prediction_step returns. Each metric is reported over all rows and per kind:

    - accuracy: the fraction of rows whose most probable option is the label.
    - nll: the mean negative log-probability of the label.
    - brier: the mean over rows of sum_k (p_k - y_k)^2, with y the one-hot label.
    - ece: the expected calibration error of the top probability, over 10 equal-width bins.
"""

from typing import TYPE_CHECKING, Dict

import numpy as np

from ...data.decision_prompt import DECISION_KINDS


if TYPE_CHECKING:
    from transformers import EvalPrediction


NUM_ECE_BINS = 10


def expected_calibration_error(confidence: np.ndarray, correct: np.ndarray, num_bins: int = NUM_ECE_BINS) -> float:
    """
    Buckets the rows by confidence and averages, over buckets weighted by their size, the gap
    between mean confidence and accuracy.
    """
    bins = np.minimum((confidence * num_bins).astype(int), num_bins - 1)
    ece = 0.0
    for b in np.unique(bins):
        in_bin = bins == b
        ece += in_bin.mean() * abs(correct[in_bin].mean() - confidence[in_bin].mean())
    return float(ece)


def _metrics(prediction, label, log_prob_label, p_max, sum_p_sq) -> Dict[str, float]:
    correct = (prediction == label).astype(float)
    return {
        "accuracy": float(correct.mean()),
        "nll": float(-log_prob_label.mean()),
        # sum_k (p_k - y_k)^2 = sum_k p_k^2 - 2 p_label + 1
        "brier": float((sum_p_sq - 2 * np.exp(log_prob_label) + 1).mean()),
        "ece": expected_calibration_error(p_max, correct),
    }


class ComputeDecisionMetrics:
    """Computes accuracy, NLL, Brier score, and ECE over all rows and per kind."""

    def __call__(self, eval_preds: "EvalPrediction") -> Dict[str, float]:
        stats = np.asarray(eval_preds.predictions, dtype=np.float64)
        prediction, label, kind_id, log_prob_label, p_max, sum_p_sq = stats.T

        result = _metrics(prediction, label, log_prob_label, p_max, sum_p_sq)
        for k, kind in enumerate(DECISION_KINDS):
            rows = kind_id == k
            if rows.any():
                kind_metrics = _metrics(
                    prediction[rows], label[rows], log_prob_label[rows], p_max[rows], sum_p_sq[rows]
                )
                result.update({f"{kind}_{name}": value for name, value in kind_metrics.items()})
        return result
