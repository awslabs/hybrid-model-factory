"""
Calibration of a trained decision model, with one temperature per question kind that divides the
model's logits before the softmax, fitted on held-out rows. A temperature above 1 makes the model
less confident, and one below 1 more confident. It never changes which option is most probable, so
accuracy is unchanged, but it changes how far the confidence of the answers can be trusted.

The temperatures are fitted in the same way as in Strands Decider (the calibrate command of its
training/evaluate.py), so that they can be compared with the ones it reports. The fit has three
steps.

    1. The rows: the calibration half of a data file (see data.py), at most limit of them.
    2. An overall temperature, minimizing the NLL (golden-section search over [0.25, 6]).
    3. A temperature for each kind with at least MIN_ROWS_PER_KIND rows, minimizing the ECE of the
       confidence (see metrics.py). The ECE is a step function of the temperature, so the minimum is
       searched over a grid of GRID_SIZE values, evenly spaced in log scale over [0.25, 6].

The temperatures are saved in decision_calibration.json in the adapter's folder (see
predictor.py), with the metrics before and after.

    hmf decision-eval calibrate path/to/decision-model --adapter_name_or_path path/to/lora \\
        --data path/to/holdout_v5_norule.jsonl
"""

import json
import math
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from ...extras import logging
from ...train.decision.metric import NUM_ECE_BINS, expected_calibration_error
from .data import partition_rows, read_rows, sample_rows
from .metrics import DEFAULT_ORDINAL_SMOOTHING, predictions, summarize
from .predictor import DecisionPredictor, calibration_path


logger = logging.get_logger(__name__)


LOW, HIGH = 0.25, 6.0
NLL_ITERATIONS = 60
GRID_SIZE = 60
MIN_ROWS_PER_KIND = 200


def _padded(log_probs: Sequence[Sequence[float]]) -> torch.Tensor:
    """The rows' log-probabilities as one [rows, max options] tensor, -inf past each row's options."""
    width = max(len(lp) for lp in log_probs)
    padded = torch.full((len(log_probs), width), float("-inf"), dtype=torch.float64)
    for i, lp in enumerate(log_probs):
        padded[i, : len(lp)] = torch.tensor(lp, dtype=torch.float64)
    return padded


def fit_temperature_nll(
    log_probs: Sequence[Sequence[float]], labels: Sequence[int], low: float = LOW, high: float = HIGH,
    iterations: int = NLL_ITERATIONS,
) -> float:
    """The temperature minimizing the mean NLL of the labels, found by golden-section search over [low, high]."""
    padded, target = _padded(log_probs), torch.tensor(list(labels)).view(-1, 1)

    def nll(t: float) -> float:
        return float(-torch.log_softmax(padded / t, dim=-1).gather(1, target).mean())

    invphi = (math.sqrt(5) - 1) / 2
    a, b = low, high
    c, d = b - invphi * (b - a), a + invphi * (b - a)
    fc, fd = nll(c), nll(d)
    for _ in range(iterations):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - invphi * (b - a)
            fc = nll(c)
        else:
            a, c, fc = c, d, fd
            d = a + invphi * (b - a)
            fd = nll(d)
    return (a + b) / 2


def ece_at(
    kinds: Sequence[str], labels: Sequence[int], log_probs: Sequence[Sequence[float]], temperature: float,
    ordinal_smoothing: float = DEFAULT_ORDINAL_SMOOTHING,
) -> float:
    """The ECE of the rows' confidence at a temperature."""
    preds = predictions(kinds, [""] * len(kinds), labels, log_probs, temperature, ordinal_smoothing)
    confidence = np.array([p.confidence for p in preds])
    correct = np.array([p.correct for p in preds], dtype=float)
    return expected_calibration_error(confidence, correct, NUM_ECE_BINS)


def temperature_grid(low: float = LOW, high: float = HIGH, size: int = GRID_SIZE) -> List[float]:
    """size temperatures from low to high, evenly spaced in log scale."""
    return [low * (high / low) ** (i / (size - 1)) for i in range(size)]


def fit_temperature_ece(
    kinds: Sequence[str], labels: Sequence[int], log_probs: Sequence[Sequence[float]],
    ordinal_smoothing: float = DEFAULT_ORDINAL_SMOOTHING,
) -> float:
    """The temperature of the grid minimizing the ECE (the first one, if several do)."""
    return min(temperature_grid(), key=lambda t: ece_at(kinds, labels, log_probs, t, ordinal_smoothing))


def fit_temperatures(
    kinds: Sequence[str], labels: Sequence[int], log_probs: Sequence[Sequence[float]],
    ordinal_smoothing: float = DEFAULT_ORDINAL_SMOOTHING, min_rows: int = MIN_ROWS_PER_KIND,
) -> Tuple[float, Dict[str, float]]:
    """
    The overall temperature (by NLL), and one per kind with at least min_rows rows (by ECE).

    Returns:
        The overall temperature, and the temperature of each kind that has enough rows.
    """
    overall = fit_temperature_nll(log_probs, labels)
    by_kind = {}
    for kind in sorted(set(kinds)):
        rows = [i for i, k in enumerate(kinds) if k == kind]
        if len(rows) < min_rows:
            logger.info_rank0(f"{kind}: {len(rows)} rows, fewer than {min_rows}: it uses the overall temperature.")
            continue
        by_kind[kind] = fit_temperature_ece(
            [kinds[i] for i in rows], [labels[i] for i in rows], [log_probs[i] for i in rows], ordinal_smoothing
        )
    return overall, by_kind


def calibrate(
    model_name_or_path: str,
    data: str,
    adapter_name_or_path: Optional[str] = None,
    split: str = "calib",
    limit: int = 4000,
    seed: int = 0,
    output: Optional[str] = None,
    ordinal_smoothing: float = DEFAULT_ORDINAL_SMOOTHING,
    min_rows_per_kind: int = MIN_ROWS_PER_KIND,
    prompt_format: str = "strands",
    force_bos: bool = False,
    max_len: Optional[int] = 4096,
    device: Optional[str] = None,
    predictor: Optional[DecisionPredictor] = None,
) -> Dict[str, Any]:
    """
    Fits a trained decision model's temperatures on a data file, and saves them.

    Args:
        model_name_or_path: The decision model.
        data: A decision data file of held-out rows (e.g. Strands Decider's holdout_v5_norule.jsonl).
        adapter_name_or_path: The model's LoRA adapter. Defaults to None.
        split: The half of the file to fit on, which is "calib", "test", or "all" (see data.py). Defaults to
            "calib", the half evaluation does not score by default.
        limit: The most rows to fit on (0 for all). Defaults to 4000.
        seed: Seed of the split and of the sampling. Defaults to 0.
        output: Where to save the calibration. Defaults to decision_calibration.json in the
            adapter's folder, or in the model's folder if there is no adapter.
        ordinal_smoothing: The ordinal smoothing of training (decision_ordinal_smoothing), the floor
            of score confidence. Defaults to 0.1.
        min_rows_per_kind: The fewest rows a kind needs for a temperature of its own; kinds with fewer
            use the overall temperature. Defaults to 200.
        prompt_format: The prompt format of training (decision_prompt_format). Defaults to "strands".
        force_bos: Whether prompts start with the BOS token (decision_force_bos). Defaults to False.
        max_len: The longest prompt, in tokens; longer rows are skipped. Defaults to 4096.
        device: Where to run the model. Defaults to the first GPU if there is one, else the CPU.
        predictor: A DecisionPredictor to use instead of loading the model. Defaults to None.

    Returns:
        The calibration, with the temperatures and the metrics before and after.
    """
    rows = sample_rows(partition_rows(read_rows(data), split, seed), limit, seed)
    predictor = predictor or DecisionPredictor(
        model_name_or_path, adapter_name_or_path, device=device, prompt_format=prompt_format, force_bos=force_bos,
        max_len=max_len, calibration=None,
    )
    log_probs = predictor.log_probs(rows)
    kept = [(row, lp) for row, lp in zip(rows, log_probs) if lp is not None]
    if not kept:
        raise ValueError(f"No row of {data} ({split}) fits in max_len {max_len}.")
    kinds = [row["kind"] for row, _ in kept]
    tasks = [row.get("task", "unknown") for row, _ in kept]
    labels = [int(row["label"]) for row, _ in kept]
    kept_log_probs = [lp for _, lp in kept]

    overall, by_kind = fit_temperatures(kinds, labels, kept_log_probs, ordinal_smoothing, min_rows_per_kind)

    def overall_metrics(temperature, default=1.0) -> Dict[str, float]:
        preds = predictions(kinds, tasks, labels, kept_log_probs, temperature, ordinal_smoothing, default)
        return summarize(preds, by_task=False)["overall"]

    calibration = {
        "temperature": overall,
        "temperature_by_kind": by_kind,
        "ordinal_smoothing": ordinal_smoothing,
        "min_rows_per_kind": min_rows_per_kind,
        "data": os.path.abspath(data),
        "split": split,
        "limit": limit,
        "seed": seed,
        "num_rows": len(kept),
        "num_skipped": len(rows) - len(kept),
        "before": overall_metrics(1.0),
        "after_overall_temperature": overall_metrics(overall),
        # As the predictor applies them, where a kind without its own temperature gets the overall one
        "after": overall_metrics(by_kind or overall, default=overall),
    }
    output = output or calibration_path(model_name_or_path, adapter_name_or_path)
    if os.path.exists(output):
        logger.info_rank0(f"Replacing the calibration {output}.")
    with open(output, "w", encoding="utf-8") as f:
        json.dump(calibration, f, indent=2)

    print(f"overall temperature {overall:.4f}; by kind " + ", ".join(f"{k}={v:.4f}" for k, v in by_kind.items()))
    for name in ("before", "after_overall_temperature", "after"):
        m = calibration[name]
        print(f"  {name:<26} n={m['n']:,} accuracy={m['accuracy']:.3f} ece={m['ece']:.3f} nll={m['nll']:.3f}")
    print(f"Saved to {output}")
    return calibration


if __name__ == "__main__":
    import fire

    fire.Fire(calibrate)
