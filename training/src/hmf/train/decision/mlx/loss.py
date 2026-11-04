"""
The loss of a batch of decision rows in MLX, which mirrors decision_loss and forward_kl of the PyTorch trainer
(train/decision/trainer.py), whose docstrings explain the terms and how they are averaged.
"""

from typing import Dict, Tuple

import mlx.core as mx


__all__ = ["decision_loss", "forward_kl"]


def forward_kl(target_probs: mx.array, log_probs: mx.array, rows: mx.array) -> mx.array:
    """
    The mean forward KL divergence KL(target || model) over the selected rows, which is the sum over each row's
    options of p_k (log p_k - log q_k), with p the target and q the model. Options where the target is exactly 0
    carry no weight in this sum and are skipped, so that they cannot give NaN, which covers padded options (target 0,
    model -inf) and targets rounded to 0.

    Args:
        target_probs: The target distributions, of shape [batch, num_options], 0 past each row's options.
        log_probs: The model's log-probabilities, of shape [batch, num_options], -inf past each row's options.
        rows: Which rows to average over, a boolean array of shape [batch].

    Returns:
        The mean KL over the selected rows (a scalar).
    """
    target = target_probs.astype(log_probs.dtype)
    keep = target > 0
    log_target = mx.where(keep, mx.log(mx.maximum(target, 1e-12)), 0.0)
    log_model = mx.where(keep, log_probs, 0.0)
    per_row = (target * (log_target - log_model)).sum(axis=-1)  # [batch]
    selected = rows.astype(per_row.dtype)
    return (per_row * selected).sum() / selected.sum()


def decision_loss(
    log_probs: mx.array,
    batch: Dict[str, mx.array],
    anchor_kl_weight: float = 0.0,
    teacher_kl_weight: float = 0.0,
) -> Tuple[mx.array, Dict[str, mx.array]]:
    """
    The loss of a batch of decision rows, which is ce_loss + anchor_kl_weight * anchor_kl + teacher_kl_weight *
    teacher_kl. The cross-entropy against label_dist is averaged over the rows with their weights, and each KL term
    is averaged over the rows that have its target (has_ref for the anchor, has_teacher for the teacher), without
    the row weights. A batch without any row that has a target adds nothing for that target's term.

    Args:
        log_probs: The model's log-probabilities over each row's options, of shape [batch, num_options], -inf past
            each row's options.
        batch: The collated batch, with label_dist and weight, and with a label set also ref_probs, has_ref,
            teacher_probs, and has_teacher (see DecisionDataCollator).
        anchor_kl_weight: The weight of the anchor term. Defaults to 0 (no anchor term).
        teacher_kl_weight: The weight of the teacher term. Defaults to 0 (no teacher term).

    Returns:
        The loss, and its parts (ce_loss, and anchor_kl and teacher_kl when they are computed), unweighted.
    """
    # Padded options are -inf in log_probs and 0 in label_dist, and 0 * -inf would be NaN
    finite_log_probs = mx.where(mx.isinf(log_probs), 0.0, log_probs)
    per_row = -(batch["label_dist"].astype(log_probs.dtype) * finite_log_probs).sum(axis=-1)
    weights = batch["weight"].astype(per_row.dtype)
    ce_loss = (per_row * weights).sum() / mx.maximum(weights.sum(), 1e-8)

    loss, parts = ce_loss, {"ce_loss": ce_loss}
    for name, weight, probs_key, has_key in (
        ("anchor_kl", anchor_kl_weight, "ref_probs", "has_ref"),
        ("teacher_kl", teacher_kl_weight, "teacher_probs", "has_teacher"),
    ):
        # Whether any row has the target is known from the batch alone, before the model runs
        if weight > 0 and has_key in batch and bool(batch[has_key].any()):
            kl = forward_kl(batch[probs_key], log_probs, batch[has_key])
            loss = loss + weight * kl
            parts[name] = kl
    return loss, parts
