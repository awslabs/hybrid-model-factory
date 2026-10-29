"""
Evaluation of a trained decision model on a decision data file, which scores its answers at its
calibrated temperatures (see calibrate.py), overall and by kind, task, and confidence band (see
metrics.py). The rows are the test half of the file by default, which calibration does not fit on
(see data.py).

    # Held-out tasks, on the test half, as calibration fits on the other
    hmf decision-eval evaluate path/to/decision-model --adapter_name_or_path path/to/lora \\
        --data path/to/holdout_v5_norule.jsonl --limit 6000 --out report.json

    # An evaluation set, on every row (the task breakdown gives each source's accuracy)
    hmf decision-eval evaluate ... --data path/to/multistep_v14_eval.jsonl --split all --limit 0

Rows whose prompt is longer than max_len are skipped, not truncated, and counted in the report.
"""

import json
import os
from typing import Any, Dict, Optional

from .data import partition_rows, read_rows, sample_rows
from .metrics import DEFAULT_ORDINAL_SMOOTHING, format_summary, predictions, summarize
from .predictor import DecisionPredictor


def evaluate(
    model_name_or_path: str,
    data: str,
    adapter_name_or_path: Optional[str] = None,
    split: str = "test",
    limit: int = 4000,
    seed: int = 0,
    raw: bool = False,
    calibration: Optional[str] = "auto",
    out: Optional[str] = None,
    ordinal_smoothing: Optional[float] = None,
    prompt_format: str = "strands",
    force_bos: bool = False,
    max_len: Optional[int] = 4096,
    device: Optional[str] = None,
    predictor: Optional[DecisionPredictor] = None,
) -> Dict[str, Any]:
    """
    Evaluates a trained decision model on a data file.

    Args:
        model_name_or_path: The decision model.
        data: A decision data file.
        adapter_name_or_path: The model's LoRA adapter. Defaults to None.
        split: The half of the file to score, which is "test", "calib", or "all" (see data.py). Defaults to
            "test", the half calibration does not fit on.
        limit: The most rows to score (0 for all). Defaults to 4000.
        seed: Seed of the split and of the sampling. Defaults to 0.
        raw: Whether to score at temperature 1, ignoring the calibration. Defaults to False.
        calibration: "auto" (the model's saved calibration, if any), the path of a calibration file,
            or None. Defaults to "auto".
        out: A JSON file to write the report to. Defaults to None.
        ordinal_smoothing: The floor of score confidence (see metrics.py). Defaults to the
            calibration's, else 0.1.
        prompt_format: The prompt format of training (decision_prompt_format). Defaults to "strands".
        force_bos: Whether prompts start with the BOS token (decision_force_bos). Defaults to False.
        max_len: The longest prompt, in tokens; longer rows are skipped. Defaults to 4096.
        device: Where to run the model. Defaults to the first GPU if there is one, else the CPU.
        predictor: A DecisionPredictor to use instead of loading the model. Defaults to None.

    Returns:
        The report, with the metrics overall and by kind, task, and confidence band, and the temperatures.
    """
    rows = sample_rows(partition_rows(read_rows(data), split, seed), limit, seed)
    predictor = predictor or DecisionPredictor(
        model_name_or_path, adapter_name_or_path, device=device, prompt_format=prompt_format, force_bos=force_bos,
        max_len=max_len, calibration=calibration,
    )
    calibrated = predictor.calibration is not None and not raw
    if ordinal_smoothing is None:
        ordinal_smoothing = (predictor.calibration or {}).get("ordinal_smoothing", DEFAULT_ORDINAL_SMOOTHING)

    log_probs = predictor.log_probs(rows)
    kept = [(row, lp) for row, lp in zip(rows, log_probs) if lp is not None]
    kinds = sorted({row["kind"] for row, _ in kept})
    temperatures = {kind: predictor.temperature(kind) if calibrated else 1.0 for kind in kinds}
    preds = predictions(
        [row["kind"] for row, _ in kept],
        [row.get("task", "unknown") for row, _ in kept],
        [int(row["label"]) for row, _ in kept],
        [lp for _, lp in kept],
        temperatures,
        ordinal_smoothing,
    )

    report = {
        **summarize(preds),
        "data": os.path.abspath(data),
        "split": split,
        "limit": limit,
        "seed": seed,
        "num_rows": len(rows),
        "num_skipped": len(rows) - len(kept),
        "calibrated": calibrated,
        "temperature_by_kind": temperatures,
        "ordinal_smoothing": ordinal_smoothing,
    }
    print(format_summary(report, title=f"{data} ({split}, {len(kept):,} rows, {report['num_skipped']} over max_len)"))
    if out is not None:
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"Report written to {out}")
    return report


if __name__ == "__main__":
    import fire

    fire.Fire(evaluate)
