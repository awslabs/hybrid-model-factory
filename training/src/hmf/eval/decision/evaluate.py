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

    # A pair file, whose rows come in adjacent pairs, on every pair
    hmf decision-eval evaluate ... --data path/to/flips_v20_eval.jsonl --pairs

Rows whose prompt is longer than max_len are skipped, not truncated, and counted in the report. If
the calibration holds a gate (see gate.py), the report also has what the gate takes of the rows and
how often those answers are right, which checks that its thresholds hold on rows they were not fitted on.

With pairs, the file's rows come in adjacent pairs, and the report also has the pair metrics (see
metrics.py). A pair file is scored whole, since the split and the sampling would separate the two rows
of a pair, and a pair with a row longer than max_len is left out whole.
"""

import json
import os
from typing import Any, Dict, Optional, Union

from .data import partition_rows, read_rows, sample_rows
from .gate import evaluate_gate, format_gate
from .metrics import DEFAULT_ORDINAL_SMOOTHING, format_summary, pair_metrics, predictions, summarize
from .predictor import AUTO, DecisionPredictor


def evaluate(
    model_name_or_path: str,
    data: str,
    adapter_name_or_path: Optional[str] = None,
    split: Optional[str] = None,
    limit: Optional[int] = None,
    seed: int = 0,
    raw: bool = False,
    calibration: Optional[str] = "auto",
    out: Optional[str] = None,
    ordinal_smoothing: Optional[float] = None,
    prompt_format: str = AUTO,
    force_bos: Union[bool, str] = AUTO,
    max_len: Union[int, None, str] = AUTO,
    option_labels: str = AUTO,
    device: Optional[str] = None,
    predictor: Optional[DecisionPredictor] = None,
    pairs: bool = False,
    cuda_graphs: bool = False,
) -> Dict[str, Any]:
    """
    Evaluates a trained decision model on a data file.

    Args:
        model_name_or_path: The decision model.
        data: A decision data file.
        adapter_name_or_path: The model's LoRA adapter. Defaults to None.
        split: The half of the file to score, which is "test", "calib", or "all" (see data.py). Defaults to
            "test", the half calibration does not fit on, and to "all" for a pair file, which allows no other.
        limit: The most rows to score (0 for all). Defaults to 4000, and to 0 for a pair file, which allows no
            other.
        seed: Seed of the split and of the sampling. Defaults to 0.
        raw: Whether to score at temperature 1, ignoring the calibration. Defaults to False.
        calibration: "auto" (the model's saved calibration, if any), the path of a calibration file,
            or None. Defaults to "auto".
        out: A JSON file to write the report to. Defaults to None.
        ordinal_smoothing: The floor of score confidence (see metrics.py). Defaults to the
            calibration's, else 0.1.
        prompt_format: The prompt format of training (decision_prompt_format). Defaults to "auto",
            the model's decision_prompt.json, else "strands" (see predictor.py).
        force_bos: Whether prompts start with the BOS token (decision_force_bos). Defaults to "auto",
            the model's, else False.
        max_len: The longest prompt, in tokens, beyond which rows are skipped. Defaults to "auto",
            the model's, else 4096.
        option_labels: How the options are labeled, "numbers" or "letters" (decision_option_labels).
            Defaults to "auto", the model's, else the labels its head reads.
        device: Where to run the model. Defaults to the first GPU if there is one, else the CPU.
        predictor: A DecisionPredictor to use instead of loading the model. Defaults to None.
        pairs: Whether the file's rows come in adjacent pairs, which adds the pair metrics. Defaults to False.
        cuda_graphs: Whether the model answers through CUDA graphs (see graphs.py). Defaults to False.

    Returns:
        The report, with the metrics overall and by kind, task, and confidence band, the temperatures, and
        with pairs, the pair metrics.
    """
    if pairs:
        split, limit = split or "all", 0 if limit is None else limit
        if split != "all" or limit != 0:
            raise ValueError(
                f"A pair file is scored whole, since the split and the sampling would separate its pairs, so set "
                f"split to all and limit to 0, got split {split!r} and limit {limit}."
            )
        rows = read_rows(data)
        if len(rows) % 2:
            raise ValueError(f"{data} has an odd number of rows ({len(rows)}), so its rows are not pairs.")
    else:
        split, limit = split or "test", 4000 if limit is None else limit
        rows = sample_rows(partition_rows(read_rows(data), split, seed), limit, seed)
    predictor = predictor or DecisionPredictor(
        model_name_or_path, adapter_name_or_path, device=device, prompt_format=prompt_format, force_bos=force_bos,
        max_len=max_len, option_labels=option_labels, calibration=calibration, cuda_graphs=cuda_graphs,
    )
    calibrated = predictor.calibration is not None and not raw
    if ordinal_smoothing is None:
        ordinal_smoothing = (predictor.calibration or {}).get("ordinal_smoothing", DEFAULT_ORDINAL_SMOOTHING)

    log_probs = predictor.log_probs(rows)
    if pairs:  # a pair is kept only if both of its rows are
        whole = [a is not None and b is not None for a, b in zip(log_probs[0::2], log_probs[1::2])]
        kept = [(row, lp) for i, (row, lp) in enumerate(zip(rows, log_probs)) if whole[i // 2]]
    else:
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
    if pairs:
        report["pairs"] = {**pair_metrics(preds), "num_pairs_skipped": whole.count(False)}
        print(format_pairs(report["pairs"]))
    if calibrated and predictor.gate is not None:
        report["gate"] = evaluate_gate(predictor.gate, [row for row, _ in kept], [p.probs for p in preds])
        print(format_gate(predictor.gate, report["gate"]["by_kind"]))
    if out is not None:
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"Report written to {out}")
    return report


def format_pairs(metrics: Dict[str, Any]) -> str:
    """The pair metrics as one line of text."""
    if not metrics["pairs"]:
        return f"pairs  n=0  ({metrics['num_pairs_skipped']} left out over max_len)"
    return (
        f"pairs  n={metrics['pairs']:,}  same_answer={metrics['same_answer']:.3f}  "
        f"both_right={metrics['both_right']:.3f}  acc_first={metrics['acc_first']:.3f}  "
        f"acc_second={metrics['acc_second']:.3f}  ({metrics['num_pairs_skipped']} left out over max_len)"
    )


if __name__ == "__main__":
    import fire

    fire.Fire(evaluate)
