"""
A gate for a trained decision model, which decides whether a caller should act on an answer or
defer, for example to an LLM. An answer is taken when its option is one the caller can act on and
its probability, at the calibrated temperature of its kind, reaches the threshold of its kind.

The thresholds are fitted with the temperatures (see calibrate.py), on the same held-out rows, and
saved with them in decision_calibration.json, so that later evaluations and the decision server
apply both. A threshold fitted on calibrated probabilities only holds at those temperatures, and
keeping them in one file means they cannot get out of step.

For each kind, the fit sorts the candidate rows, those whose most probable option is in
take_options (every option if it is not set), by that option's probability, highest first, and gives
each a weight, which is the row's weight field with use_weights and 1 otherwise. The rule then gives
a value for each prefix of the candidates, that is, for the gate that takes the top 1, 2, ... of
them. A threshold can only end a prefix that does not split a group of equal probabilities, since a
threshold there would also take rows the value did not count, and that holds at least min_count
candidates (the Kish effective number with weights, (sum w)^2 / sum w^2). The threshold is the
probability of the last row of the longest such prefix whose value meets the target, at least it or
at most it depending on the rule, or of the prefix with the highest value for a rule that maximizes.
A kind where no prefix qualifies gets no threshold, so its answers are never taken.

Each rule in RULES is a function of the sorted candidates that returns one value per prefix:

    - accuracy, the weighted share of right answers among the taken ones, which must be at least
      target.
    - accuracy_lower_bound, the lower end of the Wilson interval of that share (params z, by default
      1.96, for a 95% interval), which must be at least target, so that a threshold resting on few
      answers must clear a higher share.
    - marginal_accuracy, the share of right answers among the last params window (by default 50)
      taken ones, which must be at least target, so that the answers near the threshold, and not
      only the ones above it, are right often enough.
    - coverage, the weighted share of all the kind's rows that are taken, which must be at most
      target.
    - cost, how far the accuracy of the taken answers falls short of the fallback's (params
      reference, by default 1, as when the labels are the fallback's own answers), which must be at
      most target, the cost of a fallback call divided by the cost of a wrong answer, so that the
      gate takes the decision model's answer when the saving pays for its extra errors.
    - net_saving, params saved_cost times the taken weight minus params error_cost times the weight
      of the wrong taken answers, which is maximized without a target, and nothing is taken unless
      the best saving is above 0.

    hmf decision-eval calibrate path/to/decision-model --adapter_name_or_path path/to/lora \\
        --data path/to/holdout.jsonl --gate '{rule: accuracy, target: 0.95}'   # or --gate path/to/gate.yaml
"""

import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import numpy as np

from ...data.decision_prompt import NOUL_OPTION_NAMES


@dataclass
class Candidates:
    """
    The rows a gate could take, sorted by probability, highest first.

    Args:
        probability: Each row's top probability.
        correct: Whether each row's top option is its label, as 0 or 1.
        weight: Each row's weight.
        total_weight: The weight of all the kind's rows, candidates or not.
    """

    probability: np.ndarray
    correct: np.ndarray
    weight: np.ndarray
    total_weight: float


@dataclass
class Rule:
    """A gate rule: its function, and whether its value must be at least the target, at most it, or the highest."""

    function: Callable[[Candidates, Dict[str, Any]], np.ndarray]
    goal: str  # "at_least", "at_most", or "maximize"


RULES: Dict[str, Rule] = {}
GOALS = ("at_least", "at_most", "maximize")


def gate_rule(name: str, goal: str):
    """Registers a gate rule under name (see the module's docstring)."""
    if goal not in GOALS:
        raise ValueError(f"A rule's goal is one of {GOALS}, got {goal!r}.")

    def register(function):
        RULES[name] = Rule(function, goal)
        return function

    return register


def _accuracy(c: Candidates) -> np.ndarray:
    return np.cumsum(c.weight * c.correct) / np.cumsum(c.weight)


def effective_count(weight: np.ndarray) -> np.ndarray:
    """The Kish effective number of rows of each prefix, (sum w)^2 / sum w^2, which is the count for equal weights."""
    return np.cumsum(weight) ** 2 / np.cumsum(weight**2)


@gate_rule("accuracy", "at_least")
def accuracy(c: Candidates, params: Dict[str, Any]) -> np.ndarray:
    return _accuracy(c)


@gate_rule("accuracy_lower_bound", "at_least")
def accuracy_lower_bound(c: Candidates, params: Dict[str, Any]) -> np.ndarray:
    z = float(params.get("z", 1.96))
    share, n = _accuracy(c), effective_count(c.weight)
    center = share + z * z / (2 * n)
    spread = z * np.sqrt(share * (1 - share) / n + z * z / (4 * n * n))
    return (center - spread) / (1 + z * z / n)


@gate_rule("marginal_accuracy", "at_least")
def marginal_accuracy(c: Candidates, params: Dict[str, Any]) -> np.ndarray:
    window = int(params.get("window", 50))
    right = np.concatenate([[0.0], np.cumsum(c.weight * c.correct)])
    total = np.concatenate([[0.0], np.cumsum(c.weight)])
    end = np.arange(1, len(c.weight) + 1)
    start = np.maximum(0, end - window)
    return (right[end] - right[start]) / (total[end] - total[start])


@gate_rule("coverage", "at_most")
def coverage(c: Candidates, params: Dict[str, Any]) -> np.ndarray:
    return np.cumsum(c.weight) / c.total_weight


@gate_rule("cost", "at_most")
def cost(c: Candidates, params: Dict[str, Any]) -> np.ndarray:
    return float(params.get("reference", 1.0)) - _accuracy(c)


@gate_rule("net_saving", "maximize")
def net_saving(c: Candidates, params: Dict[str, Any]) -> np.ndarray:
    for key in ("saved_cost", "error_cost"):
        if key not in params:
            raise ValueError(f"The net_saving rule needs params.{key}.")
    wrong = np.cumsum(c.weight * (1 - c.correct))
    return float(params["saved_cost"]) * np.cumsum(c.weight) - float(params["error_cost"]) * wrong


def _option_name(name: Any) -> str:
    # YAML reads the noul options true and false as booleans
    return str(name).lower() if isinstance(name, bool) else str(name)


def check_gate_settings(settings: Union[Dict[str, Any], str]) -> Dict[str, Any]:
    """
    The gate settings with their defaults, checked, so that a wrong setting fails before the model runs. The
    settings may also be YAML text, as the command line passes them, or the path of a YAML file.
    """
    if isinstance(settings, str):
        import yaml

        if os.path.isfile(settings):
            with open(settings, encoding="utf-8") as f:
                settings = f.read()
        settings = yaml.safe_load(settings)
    if not isinstance(settings, dict):
        raise ValueError(f"The gate settings must be a mapping, got {settings!r}.")
    known = {"rule", "target", "params", "take_options", "min_count", "use_weights"}
    unknown = set(settings) - known
    if unknown:
        raise ValueError(f"Unknown gate settings {sorted(unknown)}; the settings are {sorted(known)}.")
    rule = settings.get("rule", "accuracy")
    if rule not in RULES:
        raise ValueError(f"Unknown gate rule {rule!r}; use one of {', '.join(RULES)}.")
    target = settings.get("target")
    if RULES[rule].goal != "maximize" and target is None:
        raise ValueError(f"The gate rule {rule} needs a target.")
    take_options = settings.get("take_options")
    return {
        "rule": rule,
        "target": None if target is None else float(target),
        "params": dict(settings.get("params") or {}),
        "take_options": None if take_options is None else [_option_name(name) for name in take_options],
        "min_count": int(settings.get("min_count", 50)),
        "use_weights": bool(settings.get("use_weights", False)),
    }


def fit_threshold(
    candidates: Candidates, rule: str, target: Optional[float], params: Dict[str, Any], min_count: int
) -> Optional[float]:
    """
    The threshold of a kind's candidates (see the module's docstring), or None if no prefix qualifies.

    Args:
        candidates: The candidates, sorted by probability, highest first.
        rule: A name in RULES.
        target: The value the rule must meet (unused by rules that maximize).
        params: The rule's parameters.
        min_count: The fewest (effective) candidates a threshold rests on.
    """
    if len(candidates.probability) == 0:
        return None
    probability = candidates.probability
    if np.any(np.diff(probability) > 0):
        raise ValueError("The candidates must be sorted by probability, highest first.")
    values = np.asarray(RULES[rule].function(candidates, params), dtype=float)
    # A prefix may end where the next row has a lower probability, or at the last row
    ends_a_group = np.append(probability[1:] < probability[:-1], True)
    valid = ends_a_group & (effective_count(candidates.weight) >= min_count) & ~np.isnan(values)
    goal = RULES[rule].goal
    if goal == "at_least":
        meets = valid & (values >= target)
    elif goal == "at_most":
        meets = valid & (values <= target)
    else:
        meets = valid & (values > 0)
        if meets.any():
            best = np.max(values[meets])
            meets &= values == best
            return float(probability[np.flatnonzero(meets)[0]])  # the shortest prefix with the best value
    if not meets.any():
        return None
    return float(probability[np.flatnonzero(meets)[-1]])


def option_names(row: Dict[str, Any]) -> List[str]:
    """The names of a row's options, in their order: noul rows without options are [false, true]."""
    options = row.get("options")
    if options is None:
        return list(NOUL_OPTION_NAMES)
    return [str(option[0]) for option in options]


def fit_gate(
    rows: Sequence[Dict[str, Any]], probabilities: Sequence[Sequence[float]], settings: Union[Dict[str, Any], str]
) -> Dict[str, Any]:
    """
    Fits a gate on rows and their calibrated probabilities, one threshold per kind.

    Args:
        rows: The decision rows.
        probabilities: Each row's probabilities over its options, at its kind's calibrated temperature.
        settings: The gate settings (see check_gate_settings).

    Returns:
        The gate, with its settings, each kind's threshold (None if its answers are never taken),
        and what the gate takes of the rows it was fitted on.
    """
    gate = check_gate_settings(settings)
    take = None if gate["take_options"] is None else set(gate["take_options"])
    by_kind: Dict[str, List[tuple]] = {}
    totals: Dict[str, float] = {}  # the weight of each kind's rows
    counts: Dict[str, int] = {}
    for row, probs in zip(rows, probabilities):
        top = max(range(len(probs)), key=probs.__getitem__)
        weight = float(1.0 if row.get("weight") is None else row["weight"]) if gate["use_weights"] else 1.0
        totals[row["kind"]] = totals.get(row["kind"], 0.0) + weight
        counts[row["kind"]] = counts.get(row["kind"], 0) + 1
        if take is None or option_names(row)[top] in take:
            by_kind.setdefault(row["kind"], []).append((probs[top], float(top == int(row["label"])), weight))

    gate["thresholds"], gate["fitted"] = {}, {}
    for kind in sorted(totals):
        found = sorted(by_kind.get(kind, []), key=lambda c: -c[0])
        candidates = Candidates(
            np.array([c[0] for c in found], dtype=float), np.array([c[1] for c in found], dtype=float),
            np.array([c[2] for c in found], dtype=float), totals[kind],
        )
        threshold = fit_threshold(candidates, gate["rule"], gate["target"], gate["params"], gate["min_count"])
        gate["thresholds"][kind] = threshold
        gate["fitted"][kind] = {"rows": counts[kind], "candidates": len(found),
                                **taken_metrics(candidates, threshold)}
    return gate


def taken_metrics(candidates: Candidates, threshold: Optional[float]) -> Dict[str, Any]:
    """What a threshold takes of some candidates: how many, their (weighted) share of all the kind's rows, and their
    (weighted) accuracy."""
    taken = np.zeros(len(candidates.probability), dtype=bool) if threshold is None else (
        candidates.probability >= threshold
    )
    weight = candidates.weight[taken].sum()
    return {
        "taken": int(taken.sum()),
        "share": float(weight / candidates.total_weight) if candidates.total_weight else 0.0,
        "accuracy": float((candidates.weight * candidates.correct)[taken].sum() / weight) if weight else None,
    }


def takes(gate: Optional[Dict[str, Any]], kind: str, option: str, probability: float) -> Optional[bool]:
    """
    Whether a gate takes an answer: its option is one of take_options (if set), and its probability
    reaches its kind's threshold. None if there is no gate.
    """
    if gate is None:
        return None
    threshold = gate["thresholds"].get(kind)
    if threshold is None or (gate.get("take_options") is not None and option not in gate["take_options"]):
        return False
    return probability >= threshold


def evaluate_gate(
    gate: Dict[str, Any], rows: Sequence[Dict[str, Any]], probabilities: Sequence[Sequence[float]]
) -> Dict[str, Any]:
    """
    What a fitted gate takes of other rows, by kind and overall: the share of the rows taken, and the
    accuracy of the taken answers, to check that the threshold holds on rows it was not fitted on.
    """
    counts: Dict[str, List[int]] = {}
    for row, probs in zip(rows, probabilities):
        top = max(range(len(probs)), key=probs.__getitem__)
        taken = takes(gate, row["kind"], option_names(row)[top], probs[top])
        rows_taken_right = counts.setdefault(row["kind"], [0, 0, 0])
        rows_taken_right[0] += 1
        rows_taken_right[1] += bool(taken)
        rows_taken_right[2] += bool(taken) and top == int(row["label"])

    def block(n: int, taken: int, right: int) -> Dict[str, Any]:
        return {"rows": n, "taken": taken, "share": taken / n if n else 0.0,
                "accuracy": right / taken if taken else None}

    report = {"by_kind": {kind: block(*c) for kind, c in sorted(counts.items())}}
    report["overall"] = block(*(sum(c[i] for c in counts.values()) for i in range(3)))
    report.update({key: gate[key] for key in ("rule", "target")})
    return report


def format_gate(gate: Dict[str, Any], by_kind: Dict[str, Dict[str, Any]]) -> str:
    """What a gate takes, by kind (fit_gate's fitted, or evaluate_gate's by_kind), as text."""
    target = "" if gate.get("target") is None else f", target {gate['target']}"
    lines = [f"gate ({gate['rule']}{target})"]
    for kind, b in by_kind.items():
        threshold = gate["thresholds"].get(kind)
        threshold = "none" if threshold is None else f"{threshold:.4f}"
        accuracy = "-" if b["accuracy"] is None else f"{b['accuracy']:.3f}"
        lines.append(f"  {kind:<7} threshold {threshold}  taken {b['taken']:,} of {b['rows']:,} rows "
                     f"({b['share']:.1%})  accuracy {accuracy}")
    return "\n".join(lines)
