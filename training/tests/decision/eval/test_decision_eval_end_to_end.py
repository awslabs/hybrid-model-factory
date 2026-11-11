"""
Tests for calibrating and evaluating a trained decision model end to end (hmf/eval/decision/), with a
tiny decision model and a LoRA adapter, on the CPU: where the calibration is saved, that the
predictor applies it, the evaluation report, and the evaluation of a pair file. The model and the data are
conftest.py's.
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
    assert saved["method"] == "nll"  # the default, saved with the temperatures
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


def test_predictor_warns_about_a_calibration_without_a_method(trained, monkeypatch, tmp_path):
    # A calibration saved before calibrate recorded its method was fitted the Strands Decider way
    from hmf.eval.decision import predictor as predictor_module

    warnings = []
    monkeypatch.setattr(predictor_module.logger, "warning_rank0", warnings.append)
    old = tmp_path / "old_calibration.json"
    old.write_text(json.dumps({"temperature": 1.0, "temperature_by_kind": {}}))
    _predictor(trained, calibration=str(old))
    assert len(warnings) == 1 and "strands_decider" in warnings[0]

    warnings.clear()
    new = tmp_path / "new_calibration.json"
    new.write_text(json.dumps({"method": "nll", "temperature": 1.0, "temperature_by_kind": {}}))
    _predictor(trained, calibration=str(new))
    assert warnings == []


def test_calibration_with_a_gate(trained):
    # The choice option the model answers most often, which temperatures do not change, is the one the gate takes
    rows = [row for row in (json.loads(line) for line in open(trained["data"])) if row["kind"] == "choice"]
    tops = [row["options"][max(range(len(lp)), key=lp.__getitem__)][0]
            for row, lp in zip(rows, _predictor(trained, calibration=None).log_probs(rows))]
    acted_on = max(set(tops), key=tops.count)
    other = next(name for name, _ in rows[0]["options"] if name != acted_on)
    settings = {"rule": "accuracy", "target": 0.0, "min_count": 1, "take_options": [acted_on, "true"]}
    calibration = calibrate(trained["model_dir"], trained["data"], trained["adapter_dir"], split="all", limit=0,
                            min_rows_per_kind=20, gate=settings, predictor=_predictor(trained, calibration=None))

    with open(os.path.join(trained["adapter_dir"], CALIBRATION_FILE)) as f:
        gate = json.load(f)["gate"]
    assert gate == json.loads(json.dumps(calibration["gate"]))
    assert gate["take_options"] == [acted_on, "true"] and gate["fitted"]["noul"]["rows"] == 31
    # At target 0, every candidate is taken, so each kind's threshold is its lowest candidate's probability
    predictor = _predictor(trained)
    candidates = []
    for row, probs in zip(rows, predictor.probabilities(rows)):
        top = max(range(len(probs)), key=probs.__getitem__)
        if row["options"][top][0] == acted_on:
            candidates.append(probs[top])
            assert predictor.takes("choice", acted_on, probs[top])
        assert not predictor.takes("choice", other, probs[top])
    assert gate["thresholds"]["choice"] == pytest.approx(min(candidates))
    assert gate["fitted"]["choice"]["taken"] == len(candidates)
    assert gate["thresholds"]["score"] is None  # no score answer is a candidate

    # Evaluation reports what the gate takes
    report = evaluate(trained["model_dir"], trained["data"], trained["adapter_dir"], split="all", limit=0,
                      predictor=predictor)
    assert report["gate"]["by_kind"]["choice"]["taken"] == len(candidates) > 0
    assert report["gate"]["by_kind"]["score"]["taken"] == 0
    raw = evaluate(trained["model_dir"], trained["data"], trained["adapter_dir"], split="all", limit=0, raw=True,
                   predictor=predictor)
    assert "gate" not in raw  # the gate holds only at the temperatures it was fitted at


def test_calibration_without_a_gate_saves_none(trained):
    calibration = calibrate(trained["model_dir"], trained["data"], trained["adapter_dir"], split="all", limit=0,
                            predictor=_predictor(trained, calibration=None))
    assert "gate" not in calibration and _predictor(trained).gate is None


def test_bad_gate_settings_fail_before_the_model_runs(trained):
    class Unused:
        def log_probs(self, rows):
            raise AssertionError("the model ran")

    with pytest.raises(ValueError, match="Unknown gate rule"):
        calibrate(trained["model_dir"], trained["data"], trained["adapter_dir"], gate={"rule": "x", "target": 1},
                  predictor=Unused())


def _pair_file(trained, path):
    """
    A pair file of 11 pairs, whose sixth pair has a second row too long for a max_len of 200. Leaving out
    single rows instead of whole pairs would pair the rows after it wrongly.
    """
    rows = [trained["row"](i) for i in range(22)]
    rows[11] = {**rows[11], "state": "state " * 300}
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return rows


def test_pair_file_leaves_out_a_pair_with_a_long_row_and_scores_the_others_as_pairs(trained, tmp_path):
    rows = _pair_file(trained, tmp_path / "pairs.jsonl")
    predictor = _predictor(trained, max_len=200)

    report = evaluate(trained["model_dir"], str(tmp_path / "pairs.jsonl"), trained["adapter_dir"], raw=True,
                      pairs=True, predictor=predictor)

    assert report["split"] == "all" and report["limit"] == 0  # the defaults for a pair file
    assert report["num_rows"] == 22 and report["num_skipped"] == 2 and report["overall"]["n"] == 20
    # The pair metrics, computed from the model's answers to the ten whole pairs
    kept = rows[:10] + rows[12:]
    answers = [max(range(len(p)), key=p.__getitem__) for p in predictor.probabilities(kept, calibrated=False)]
    first, second = list(zip(kept[0::2], answers[0::2])), list(zip(kept[1::2], answers[1::2]))
    assert report["pairs"] == pytest.approx({
        "pairs": 10,
        "acc_first": sum(a == row["label"] for row, a in first) / 10,
        "acc_second": sum(a == row["label"] for row, a in second) / 10,
        "same_answer": sum(a == b for (_, a), (_, b) in zip(first, second)) / 10,
        "both_right": sum(a == r["label"] and b == q["label"] for (r, a), (q, b) in zip(first, second)) / 10,
        "num_pairs_skipped": 1,
    })


def test_pair_file_refuses_a_split_a_limit_and_an_odd_number_of_rows(trained, tmp_path):
    _pair_file(trained, tmp_path / "pairs.jsonl")
    for options in ({"split": "test"}, {"limit": 100}):
        with pytest.raises(ValueError, match="scored whole"):
            evaluate(trained["model_dir"], str(tmp_path / "pairs.jsonl"), trained["adapter_dir"], pairs=True,
                     **options)
    odd = tmp_path / "odd.jsonl"
    odd.write_text("".join(json.dumps(trained["row"](i)) + "\n" for i in range(3)))
    with pytest.raises(ValueError, match="odd number of rows"):
        evaluate(trained["model_dir"], str(odd), trained["adapter_dir"], pairs=True)
