"""
Tests for fitting a gate (hmf/eval/decision/gate.py) on hand-made candidates whose threshold is known:
the rules, groups of equal probabilities, min_count, weights, take_options, and the settings check.
"""

import numpy as np
import pytest

from hmf.eval.decision.gate import (
    RULES,
    Candidates,
    check_gate_settings,
    evaluate_gate,
    fit_gate,
    fit_threshold,
    takes,
)


def _candidates(probability, correct, weight=None, total_weight=None) -> Candidates:
    weight = np.ones(len(probability)) if weight is None else np.asarray(weight, dtype=float)
    return Candidates(np.asarray(probability, dtype=float), np.asarray(correct, dtype=float), weight,
                      float(weight.sum() if total_weight is None else total_weight))


# Ten answers, most probable first, and whether each is right
PROBABILITY = [0.99, 0.98, 0.97, 0.95, 0.93, 0.90, 0.85, 0.80, 0.70, 0.60]
CORRECT = [1, 1, 1, 1, 0, 1, 1, 0, 0, 1]


def test_accuracy_takes_the_longest_prefix_that_reaches_the_target():
    # The top 7 are 86% right, and the top 8 are 75% right, so at 80% the threshold is the 7th answer's
    assert fit_threshold(_candidates(PROBABILITY, CORRECT), "accuracy", 0.8, {}, min_count=1) == 0.85
    # The top 5 are 80% right but the top 7 are better, so the longest prefix wins, not the first to fail
    assert fit_threshold(_candidates(PROBABILITY, CORRECT), "accuracy", 0.85, {}, min_count=1) == 0.85
    assert fit_threshold(_candidates(PROBABILITY, CORRECT), "accuracy", 1.0, {}, min_count=1) == 0.95


def test_a_threshold_never_splits_equal_probabilities():
    # The first three answers at 1.0 are right and the fourth is not: a threshold of 1.0 would take all four
    # (75% right), so at 80% there is none
    tied = _candidates([1.0, 1.0, 1.0, 1.0], [1, 1, 1, 0])
    assert fit_threshold(tied, "accuracy", 0.8, {}, min_count=1) is None
    # With six right answers below them, the whole set is 90% right
    longer = _candidates([1.0] * 4 + [0.9] * 6, [1, 1, 1, 0] + [1] * 6)
    assert fit_threshold(longer, "accuracy", 0.8, {}, min_count=1) == 0.9


def test_min_count():
    candidates = _candidates([0.9, 0.8, 0.7, 0.6, 0.5], [1, 1, 1, 0, 0])
    assert fit_threshold(candidates, "accuracy", 0.95, {}, min_count=3) == 0.7
    assert fit_threshold(candidates, "accuracy", 0.95, {}, min_count=4) is None


def test_weights():
    # The second answer is wrong and weighs 10, so taking it drops accuracy to 1/11
    candidates = _candidates([0.9, 0.8, 0.7], [1, 0, 1], weight=[1, 10, 1])
    assert fit_threshold(candidates, "accuracy", 0.6, {}, min_count=1) == 0.9
    assert fit_threshold(_candidates([0.9, 0.8, 0.7], [1, 0, 1]), "accuracy", 0.6, {}, min_count=1) == 0.7
    # min_count counts the Kish effective number: weights 1, 10, 1 are about 1.4 rows
    assert fit_threshold(candidates, "accuracy", 0.0, {}, min_count=2) is None


def test_accuracy_lower_bound():
    # 20 right answers: a Wilson lower bound of about 0.84; 100: about 0.96
    assert fit_threshold(_candidates(np.linspace(0.99, 0.9, 20), [1] * 20), "accuracy_lower_bound", 0.95, {},
                         min_count=1) is None
    probability = np.linspace(0.99, 0.5, 100)
    assert fit_threshold(_candidates(probability, [1] * 100), "accuracy_lower_bound", 0.95, {}, min_count=1) == (
        pytest.approx(probability[-1]))


def test_marginal_accuracy():
    # Cumulative accuracy stays at 4/6 or above, but the last two answers of the top 4, 5 or 6 are at most half right
    candidates = _candidates([0.9, 0.8, 0.7, 0.6, 0.5, 0.4], [1, 1, 1, 0, 0, 1])
    assert fit_threshold(candidates, "marginal_accuracy", 0.6, {"window": 2}, min_count=1) == 0.7
    assert fit_threshold(candidates, "accuracy", 0.6, {}, min_count=1) == 0.4


def test_coverage():
    # Ten candidates among twenty rows: taking at most a quarter of the rows takes the top five
    candidates = _candidates(PROBABILITY, CORRECT, total_weight=20)
    assert fit_threshold(candidates, "coverage", 0.25, {}, min_count=1) == 0.93


def test_cost_is_accuracy_against_the_fallback():
    candidates = _candidates(PROBABILITY, CORRECT)
    # Falling at most 0.15 short of a fallback that is always right is an accuracy of at least 0.85
    assert fit_threshold(candidates, "cost", 0.15, {}, min_count=1) == fit_threshold(
        candidates, "accuracy", 0.85, {}, min_count=1)
    # Against a fallback that is right 90% of the time, at most 0.05 short
    assert fit_threshold(candidates, "cost", 0.05, {"reference": 0.9}, min_count=1) == fit_threshold(
        candidates, "accuracy", 0.85, {}, min_count=1)


def test_net_saving():
    # Saving 1 per answer taken and losing 3 per wrong one: 1, 2, -1, 0, -3, -6
    candidates = _candidates([0.9, 0.8, 0.7, 0.6, 0.5, 0.4], [1, 1, 0, 1, 0, 0])
    params = {"saved_cost": 1, "error_cost": 3}
    assert fit_threshold(candidates, "net_saving", None, params, min_count=1) == 0.8
    # A saving that is never above 0 takes nothing
    assert fit_threshold(_candidates([0.9], [0]), "net_saving", None, params, min_count=1) is None
    with pytest.raises(ValueError, match="saved_cost"):
        fit_threshold(candidates, "net_saving", None, {"error_cost": 3}, min_count=1)


def test_every_rule_returns_one_value_per_prefix():
    candidates = _candidates(PROBABILITY, CORRECT)
    params = {"saved_cost": 1, "error_cost": 2}
    for name, rule in RULES.items():
        assert len(rule.function(candidates, params)) == len(PROBABILITY), name


def test_settings_are_checked():
    assert check_gate_settings({"target": 0.9}) == {
        "rule": "accuracy", "target": 0.9, "params": {}, "take_options": None, "min_count": 50, "use_weights": False,
    }
    with pytest.raises(ValueError, match="Unknown gate rule"):
        check_gate_settings({"rule": "precision", "target": 0.9})
    with pytest.raises(ValueError, match="Unknown gate settings"):
        check_gate_settings({"target": 0.9, "threshold": 0.5})
    with pytest.raises(ValueError, match="needs a target"):
        check_gate_settings({"rule": "accuracy"})
    check_gate_settings({"rule": "net_saving", "params": {"saved_cost": 1, "error_cost": 2}})  # no target needed


def test_settings_from_a_yaml_file(tmp_path):
    path = tmp_path / "gate.yaml"
    path.write_text("rule: coverage\ntarget: 0.5\ntake_options:\n  - act\n")
    assert check_gate_settings(str(path))["take_options"] == ["act"]


def test_settings_as_yaml_text():
    # As the command line passes them, where YAML reads true as a boolean and customer:a as a name
    settings = check_gate_settings("{rule: accuracy, target: 0.95, take_options: [customer:a, true, 2]}")
    assert settings["target"] == 0.95 and settings["take_options"] == ["customer:a", "true", "2"]
    with pytest.raises(ValueError, match="must be a mapping"):
        check_gate_settings("accuracy")


def _choice_row(label: int, weight=None) -> dict:
    row = {"kind": "choice", "options": [["act", None], ["wait", None]], "label": label}
    return row if weight is None else {**row, "weight": weight}


def test_fit_gate_and_take_options():
    # Choice rows whose top option is "act" (index 0) or "wait" (index 1), and noul rows without options
    rows = [_choice_row(0) for _ in range(4)] + [_choice_row(1)] + [_choice_row(1) for _ in range(3)]
    probabilities = [[0.9, 0.1], [0.8, 0.2], [0.7, 0.3], [0.6, 0.4], [0.95, 0.05], [0.1, 0.9], [0.2, 0.8], [0.3, 0.7]]
    rows += [{"kind": "noul", "label": 1}] * 2
    probabilities += [[0.2, 0.8], [0.4, 0.6]]

    gate = fit_gate(rows, probabilities, {"target": 0.75, "take_options": ["act", "true"], "min_count": 1})

    # Only the "act" answers are candidates for choice: four right, then one wrong at 0.95, which comes first
    assert gate["fitted"]["choice"]["candidates"] == 5 and gate["fitted"]["choice"]["rows"] == 8
    assert gate["thresholds"]["choice"] == 0.6  # all five: 4 of 5 right
    assert gate["fitted"]["choice"]["share"] == pytest.approx(5 / 8)
    assert gate["thresholds"]["noul"] == 0.6  # noul options are [false, true]

    assert takes(gate, "choice", "act", 0.65) and not takes(gate, "choice", "act", 0.55)
    assert not takes(gate, "choice", "wait", 0.99)  # not an option the caller acts on
    assert not takes(gate, "score", "1", 0.99)  # a kind without a threshold
    assert takes(None, "choice", "act", 0.99) is None


def test_fit_gate_with_weights():
    rows = [_choice_row(0, weight=1), _choice_row(1, weight=10), _choice_row(0, weight=1)]
    probabilities = [[0.9, 0.1], [0.8, 0.2], [0.7, 0.3]]
    settings = {"target": 0.6, "min_count": 1}
    assert fit_gate(rows, probabilities, settings)["thresholds"]["choice"] == 0.7
    assert fit_gate(rows, probabilities, {**settings, "use_weights": True})["thresholds"]["choice"] == 0.9


def test_evaluate_gate():
    gate = {"rule": "accuracy", "target": 0.9, "take_options": ["act"], "thresholds": {"choice": 0.75}}
    rows = [_choice_row(0), _choice_row(1), _choice_row(0), _choice_row(1)]
    probabilities = [[0.9, 0.1], [0.8, 0.2], [0.6, 0.4], [0.1, 0.9]]

    report = evaluate_gate(gate, rows, probabilities)

    # Taken: the first two (act, at 0.9 and 0.8), one of them right
    assert report["overall"] == {"rows": 4, "taken": 2, "share": 0.5, "accuracy": 0.5}
    assert report["by_kind"]["choice"]["taken"] == 2
