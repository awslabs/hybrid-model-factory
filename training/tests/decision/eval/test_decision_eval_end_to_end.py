"""
Tests for calibrating and evaluating a trained decision model end to end (hmf/eval/decision/), with a
tiny decision model and a LoRA adapter, on the CPU: where the calibration is saved, that the
predictor applies it, and the evaluation report. The model and the data are conftest.py's.
"""

import json
import math
import os

import pytest

from hmf.eval.decision.calibrate import calibrate
from hmf.eval.decision.evaluate import evaluate
from hmf.eval.decision.metrics import apply_temperature
from hmf.eval.decision.predictor import CALIBRATION_FILE, DecisionPredictor


def _predictor(trained, **kwargs) -> DecisionPredictor:
    return DecisionPredictor(trained["model_dir"], trained["adapter_dir"], device="cpu", **kwargs)


def test_calibration_is_saved_in_the_adapters_folder(trained):
    result = calibrate(
        trained["model_dir"], trained["data"], trained["adapter_dir"], split="all", limit=0, min_rows_per_kind=20,
        predictor=_predictor(trained, calibration=None),
    )

    path = os.path.join(trained["adapter_dir"], CALIBRATION_FILE)
    with open(path) as f:
        saved = json.load(f)
    assert saved == json.loads(json.dumps(result))
    assert not os.path.exists(os.path.join(trained["model_dir"], CALIBRATION_FILE))  # not in the shared model
    assert set(saved["temperature_by_kind"]) == {"noul", "choice", "score"}  # 30 rows each, at least 20
    assert all(0.25 <= t <= 6.0 for t in [saved["temperature"], *saved["temperature_by_kind"].values()])
    assert saved["num_rows"] == 91 and saved["num_skipped"] == 0
    # Temperatures never change the answers
    assert saved["before"]["accuracy"] == saved["after"]["accuracy"] == saved["after_overall_temperature"]["accuracy"]


def test_predictor_applies_the_calibration(trained):
    calibrate(trained["model_dir"], trained["data"], trained["adapter_dir"], split="all", limit=0,
              min_rows_per_kind=20, predictor=_predictor(trained, calibration=None))
    predictor = _predictor(trained)  # finds the calibration in the adapter's folder
    with open(os.path.join(trained["adapter_dir"], CALIBRATION_FILE)) as f:
        saved = json.load(f)
    rows = [trained["row"](i) for i in range(3)]

    log_probs = predictor.log_probs(rows)
    calibrated, raw = predictor.probabilities(rows), predictor.probabilities(rows, calibrated=False)

    for row, lp, probs, raw_probs in zip(rows, log_probs, calibrated, raw):
        temperature = saved["temperature_by_kind"][row["kind"]]
        assert predictor.temperature(row["kind"]) == temperature
        assert probs == pytest.approx(apply_temperature(lp, temperature))
        assert raw_probs == pytest.approx([math.exp(x) for x in lp])
    # Without a calibration, temperature 1
    assert _predictor(trained, calibration=None).temperature("choice") == 1.0


def test_a_kind_without_a_temperature_uses_the_overall_one(trained, tmp_path):
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps({"temperature": 1.7, "temperature_by_kind": {"choice": 0.8}}))
    predictor = _predictor(trained, calibration=str(path))
    assert predictor.temperature("choice") == 0.8 and predictor.temperature("noul") == 1.7


def test_evaluation_report(trained):
    calibrate(trained["model_dir"], trained["data"], trained["adapter_dir"], split="all", limit=0,
              min_rows_per_kind=20, predictor=_predictor(trained, calibration=None))
    out = trained["root"] / "report.json"

    report = evaluate(trained["model_dir"], trained["data"], trained["adapter_dir"], split="all", limit=0,
                      out=str(out), predictor=_predictor(trained, max_len=200))

    assert set(report) >= {"overall", "by_kind", "by_task", "by_confidence_band", "temperature_by_kind"}
    assert report["calibrated"] and report["num_rows"] == 91 and report["num_skipped"] == 1  # the long row
    assert report["overall"]["n"] == 90 and set(report["by_task"]) == {"a", "b"}
    assert sum(block.get("n", 0) for block in report["by_confidence_band"].values()) == 90
    with open(out) as f:
        assert json.load(f)["overall"] == report["overall"]

    # Raw, at temperature 1, with the same answers
    raw = evaluate(trained["model_dir"], trained["data"], trained["adapter_dir"], split="all", limit=0, raw=True,
                   predictor=_predictor(trained, max_len=200))
    assert not raw["calibrated"] and set(raw["temperature_by_kind"].values()) == {1.0}
    assert raw["overall"]["accuracy"] == report["overall"]["accuracy"]


def test_calibration_and_evaluation_use_different_halves(trained):
    calibration = calibrate(trained["model_dir"], trained["data"], trained["adapter_dir"], limit=0,
                            min_rows_per_kind=1, predictor=_predictor(trained, calibration=None))
    report = evaluate(trained["model_dir"], trained["data"], trained["adapter_dir"], limit=0,
                      predictor=_predictor(trained))
    assert calibration["split"] == "calib" and report["split"] == "test"
    assert calibration["num_rows"] + report["num_rows"] == 91
