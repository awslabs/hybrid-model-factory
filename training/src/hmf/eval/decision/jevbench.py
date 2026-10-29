"""
A per-task comparison of two JevBench runs, with an exact McNemar test.

Two models' JevBench scores differ by the tasks only one of them answers correctly (the discordant
tasks). The exact McNemar test asks whether that split is more lopsided than chance. Under the
hypothesis that the models are equally good, each discordant task is a fair coin, and the p-value is
the two-sided binomial probability of a split at least as uneven. p >= 0.05 means no significant
difference.

    hmf decision-eval compare path/to/run_a/results.jsonl path/to/run_b/results.jsonl

A run is a JevBench results.jsonl (one line per task, with task_id and correct; a task without an
answer counts as wrong, as JevBench's summary counts it), or a CSV with the columns run, task_id, and
correct (the layout of Strands Decider's research/data/jevbench_results.csv), with the run named.
"""

import csv
import json
from math import comb
from typing import Dict, List, Optional


def load_results(path: str, run: Optional[str] = None) -> Dict[str, bool]:
    """
    Whether each task of a run was answered correctly.

    Args:
        path: A JevBench results.jsonl, or a CSV with the columns run, task_id, and correct.
        run: The run to read from a CSV. Needed for a CSV, unused for results.jsonl.
    """
    results: Dict[str, bool] = {}
    if path.endswith(".csv"):
        if run is None:
            raise ValueError(f"{path}: name the run to read from the CSV.")
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row["run"] == run:
                    results[row["task_id"]] = row["correct"] in ("1", "True", "true")
        if not results:
            raise ValueError(f"{path}: no rows for the run {run!r}.")
        return results

    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            if record["task_id"] in results:
                raise ValueError(f"{path}: the task {record['task_id']} appears twice.")
            results[record["task_id"]] = bool(record.get("correct"))
    return results


def mcnemar_exact(a_only: int, b_only: int) -> float:
    """The two-sided exact McNemar p-value of a_only and b_only discordant tasks."""
    n = a_only + b_only
    if n == 0:
        return 1.0
    tail = sum(comb(n, i) for i in range(min(a_only, b_only) + 1)) / 2**n
    return min(1.0, 2 * tail)


def compare(a: Dict[str, bool], b: Dict[str, bool]) -> Dict[str, object]:
    """
    The comparison of two runs on the tasks both answered.

    Returns:
        n (tasks compared), a_correct, b_correct, both_right, both_wrong, a_only, and b_only (the
        tasks only one run answers correctly, as lists of task ids), and p (the exact McNemar test).
    """
    tasks = sorted(set(a) & set(b))
    a_only: List[str] = [t for t in tasks if a[t] and not b[t]]
    b_only: List[str] = [t for t in tasks if b[t] and not a[t]]
    return {
        "n": len(tasks),
        "a_correct": sum(a[t] for t in tasks),
        "b_correct": sum(b[t] for t in tasks),
        "both_right": sum(a[t] and b[t] for t in tasks),
        "both_wrong": sum(not a[t] and not b[t] for t in tasks),
        "a_only": a_only,
        "b_only": b_only,
        "p": mcnemar_exact(len(a_only), len(b_only)),
    }


def main(a: str, b: str, a_run: Optional[str] = None, b_run: Optional[str] = None, out: Optional[str] = None) -> None:
    """
    Compares two JevBench runs task by task.

    Args:
        a: The first run's results.jsonl (or CSV).
        b: The second run's results.jsonl (or CSV).
        a_run: The run to read from a, if it is a CSV.
        b_run: The run to read from b, if it is a CSV.
        out: A JSON file to write the comparison to. Defaults to None.
    """
    result = compare(load_results(a, a_run), load_results(b, b_run))
    print(
        f"{result['n']} tasks: A {result['a_correct']} correct, B {result['b_correct']} correct "
        f"(net B - A {result['b_correct'] - result['a_correct']:+d}); both right {result['both_right']}, "
        f"both wrong {result['both_wrong']}, A only {len(result['a_only'])}, B only {len(result['b_only'])}; "
        f"exact McNemar p = {result['p']:.4f}"
    )
    for name in ("a_only", "b_only"):
        for task in result[name]:
            print(f"  {name}  {task}")
    if out is not None:
        with open(out, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    import fire

    fire.Fire(main)
