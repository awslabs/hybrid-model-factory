"""
DecisionPredictor gives a trained decision model's answers to decision rows, at temperature 1 or calibrated.

A trained model is a decision model (from hmf decision-init) and, when trained with LoRA, its adapter
(the training output_dir, the folder that holds adapter_config.json). Calibration (see calibrate.py)
saves its temperatures in decision_calibration.json in the adapter's folder, never in the decision
model's folder, which several adapters may be trained on. A model without an adapter (trained with
finetuning_type: full, or merged with hmf export) gets the file in its own folder.
"""

import json
import os
from typing import Any, Dict, List, Optional, Sequence

import torch

from ...data.decision_teachers import DecisionModelTeacher
from ...extras import logging
from .metrics import apply_temperature


logger = logging.get_logger(__name__)


CALIBRATION_FILE = "decision_calibration.json"


def calibration_path(model_name_or_path: str, adapter_name_or_path: Optional[str] = None) -> str:
    """Where a trained model's calibration is saved, which is its adapter's folder, else the model's folder."""
    return os.path.join(adapter_name_or_path or model_name_or_path, CALIBRATION_FILE)


def load_calibration(path: str) -> Optional[Dict[str, Any]]:
    """The calibration saved at path, or None if there is none."""
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


class DecisionPredictor:
    """
    A trained decision model's answers to decision rows (in the decision data format, see
    data/README.md), as its log-probabilities over each row's options at temperature 1, and its
    probabilities at the calibrated temperature of the row's kind. A kind without a temperature of
    its own gets the calibration's overall temperature, as a decision server applies it.

    Rows are rendered in their original option order, with the question's original phrasing; rows
    whose prompt is longer than max_len tokens get no answer (they are skipped, not truncated).

    Args:
        model_name_or_path: The decision model.
        adapter_name_or_path: Its LoRA adapter (the training output folder), merged before use.
            Defaults to None.
        device: Where to run the model. Defaults to the first GPU if there is one, else the CPU.
        prompt_format: The prompt format's name (decision_prompt_format in training). Defaults to
            "strands".
        force_bos: Whether prompts start with the BOS token (decision_force_bos in training).
            Defaults to False.
        max_len: The longest prompt answered, in tokens. Defaults to 4096.
        calibration: "auto" (the model's saved calibration, if any), the path of a calibration file,
            or None (temperature 1). Defaults to "auto".
    """

    def __init__(
        self,
        model_name_or_path: str,
        adapter_name_or_path: Optional[str] = None,
        device: Optional[str] = None,
        prompt_format: str = "strands",
        force_bos: bool = False,
        max_len: Optional[int] = 4096,
        calibration: Optional[str] = "auto",
    ):
        self.model_name_or_path = model_name_or_path
        self.adapter_name_or_path = adapter_name_or_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_len = max_len
        self.model = DecisionModelTeacher(
            model_name_or_path,
            self.device,
            adapter_name_or_path=adapter_name_or_path,
            prompt_format=prompt_format,
            force_bos=force_bos,
            max_len=max_len,
        )
        if calibration == "auto":
            calibration = calibration_path(model_name_or_path, adapter_name_or_path)
        self.calibration = load_calibration(calibration) if calibration is not None else None
        if self.calibration is not None:
            logger.info_rank0(
                f"Calibration {calibration}: temperature {self.calibration['temperature']:.4f}, by kind "
                f"{self.calibration['temperature_by_kind']}."
            )

    def temperature(self, kind: str) -> float:
        """The calibrated temperature of a kind, which is its own, else the overall one, else 1 (uncalibrated)."""
        if self.calibration is None:
            return 1.0
        return float(self.calibration["temperature_by_kind"].get(kind, self.calibration["temperature"]))

    def log_probs(self, rows: Sequence[Dict[str, Any]]) -> List[Optional[List[float]]]:
        """Each row's log-probabilities over its options at temperature 1, or None if its prompt is too long."""
        return self.model.log_probs(rows)

    def probabilities(self, rows: Sequence[Dict[str, Any]], calibrated: bool = True) -> List[Optional[List[float]]]:
        """Each row's probabilities over its options, calibrated or at temperature 1, or None if skipped."""
        results = []
        for row, log_probs in zip(rows, self.log_probs(rows)):
            if log_probs is None:
                results.append(None)
            else:
                temperature = self.temperature(row["kind"]) if calibrated else 1.0
                results.append(apply_temperature(log_probs, temperature))
        return results
