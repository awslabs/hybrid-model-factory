"""
Decision data files for evaluation, and the split of a file into a calibration half and a test half.

Temperatures must not be fitted on the rows they are then scored on, because that would report a
calibration error that the model does not reach on unseen rows. Calibration and evaluation therefore
split the same file in the same way, into two halves with no rows in common. The split and the
sampling are the same as in Strands Decider (training/evaluate.py), so the same file gives the same
rows in both programs.
"""

import hashlib
import json
import random
from typing import Any, Dict, List


SPLITS = ("calib", "test", "all")


def read_rows(path: str) -> List[Dict[str, Any]]:
    """The rows of a decision data file, which is JSON Lines (its non-empty lines) or a JSON list."""
    with open(path, encoding="utf-8") as f:
        if path.endswith(".json"):
            return json.load(f)
        return [json.loads(line) for line in f if line.strip()]


def partition_rows(rows: List[Dict[str, Any]], split: str, seed: int = 0) -> List[Dict[str, Any]]:
    """
    One half of a file's rows, which is "calib" for calibration, "test" for evaluation, or "all". A row's half
    depends on its position in the file only (a hash of the seed and the position), so a row that
    appears twice cannot land in both halves.

    Args:
        rows: All the rows of the file, in file order.
        split: "calib", "test", or "all".
        seed: Seed of the split. Defaults to 0.
    """
    if split == "all":
        return list(rows)
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
    want = 0 if split == "calib" else 1
    return [row for i, row in enumerate(rows) if (hashlib.sha256(f"{seed}:{i}".encode()).digest()[0] & 1) == want]


def sample_rows(rows: List[Dict[str, Any]], limit: int, seed: int = 0) -> List[Dict[str, Any]]:
    """
    At most limit rows, taken after a fixed shuffle rather than from the top of the file, which is
    often grouped by task. limit <= 0 keeps every row.
    """
    if limit <= 0 or limit >= len(rows):
        return list(rows)
    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    return shuffled[:limit]
