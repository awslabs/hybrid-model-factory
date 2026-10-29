"""
Asks a trained decision model questions and prints its decisions, which are the probability of each option,
the chosen option, and, for yes/no and rating questions, P(true) and the expected level.

    # A decision checkpoint plus a LoRA adapter trained with stage `decision`
    python scripts/decision_predict.py \
        --model_name_or_path path/to/decision-model \
        --adapter_name_or_path path/to/lora-output

    # Your own questions, in the decision format of data/README.md (a JSON list or JSONL file)
    python scripts/decision_predict.py --model_name_or_path ... --questions my_questions.jsonl

Without --questions, a few built-in examples are asked. The model answers at its saved calibration
(decision_calibration.json in the adapter's folder, written by hmf decision-eval calibrate) if it has
one, and at temperature 1 otherwise or with --raw. Questions longer than --max_len tokens are skipped.

Questions that give their correct label also get a correct/incorrect mark, and over all of them the
script reports accuracy, NLL, Brier score, and ECE, computed as in training. These are aggregate
measures, and on a handful of questions they say little.
"""

import json
import math
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import fire
import numpy as np

from hmf.data.decision_prompt import DECISION_KINDS, NOUL_OPTION_NAMES
from hmf.eval.decision.predictor import DecisionPredictor
from hmf.train.decision.metric import ComputeDecisionMetrics

# "label" denotes the correct/gold answer
EXAMPLES = [
    {
        "kind": "choice",
        "state": "Order 1182 was charged twice. The customer asked for the duplicate charge to be removed.",
        "instructions": "What should support do next?",
        "options": [
            ["refund", "return the duplicate payment"],
            ["ignore", "take no action"],
            ["escalate", "send the case to a human agent"],
        ],
        "label": 0,
    },
    {
        "kind": "noul",
        "state": "The team meeting was moved from Tuesday to Thursday at 3pm.",
        "instructions": "The meeting takes place on Tuesday.",
        "options": [["false", None], ["true", None]],
        "label": 0,
    },
    {
        "kind": "noul",
        "state": {"invoice": "INV-2041", "amount_due": 0, "status": "paid", "paid_on": "2026-10-02"},
        "instructions": "The invoice has been paid in full.",
        "options": [["false", None], ["true", None]],
        "label": 1,
    },
    {
        "kind": "choice",
        "state": "I've been charged for a subscription I canceled last month and nobody answers my emails!!",
        "instructions": "What is the customer's sentiment?",
        "options": [["positive", None], ["neutral", None], ["negative", None]],
        "label": 2,
    },
    {
        "kind": "score",
        "state": "Q: How do I reset my password? A: Go to Settings > Account > Reset password and follow the link "
        "we email you. The link expires after 24 hours.",
        "instructions": "How well does the answer address the question?",
        "options": [
            ["0", "does not address the question"],
            ["1", "partially addresses it"],
            ["2", "addresses it, with gaps"],
            ["3", "fully and clearly addresses it"],
        ],
        "label": 3,
    },
]


def _load_questions(path: Optional[str]) -> List[Dict[str, Any]]:
    if path is None:
        return EXAMPLES
    with open(path, encoding="utf-8") as f:
        if path.endswith(".jsonl"):
            questions = [json.loads(line) for line in f if line.strip()]
        else:
            questions = json.load(f)
    # A noul question may leave out its options, to use their default descriptions
    return [{**question, "options": question.get("options")} for question in questions]


def _option_names(question: Dict[str, Any]) -> List[str]:
    if question["kind"] == "noul":
        return list(NOUL_OPTION_NAMES)
    if question["kind"] == "score":
        return [f"{i} ({desc})" if desc else str(i) for i, (_, desc) in enumerate(question["options"])]
    return [name for name, _ in question["options"]]


def _print_decision(index: int, question: Dict[str, Any], probs: List[float]) -> None:
    names = _option_names(question)
    decision = max(range(len(probs)), key=probs.__getitem__)
    label = question.get("label")

    print(f"\n[{index}] {question['kind']}: {question['instructions']}")
    width = max(len(name) for name in names)
    for k, (name, p) in enumerate(zip(names, probs)):
        marks = ("  <- decision" if k == decision else "") + ("  (gold)" if k == label else "")
        print(f"    {name:<{width}}  {p:.3f}{marks}")

    if question["kind"] == "noul":
        print(f"    P(true) = {probs[1]:.3f}")
    if question["kind"] == "score":
        print(f"    expected level = {sum(k * p for k, p in enumerate(probs)):.2f}")
    if label is not None:
        print(f"    {'correct' if decision == label else 'incorrect'}")


def predict(
    model_name_or_path: str,
    adapter_name_or_path: Optional[str] = None,
    questions: Optional[str] = None,
    calibration: Optional[str] = "auto",
    raw: bool = False,
    prompt_format: str = "strands",
    force_bos: bool = False,
    max_len: int = 4096,
    device: Optional[str] = None,
):
    """
    Asks a trained decision model questions and prints its decisions.

    Args:
        model_name_or_path: The decision model (from hmf decision-init).
        adapter_name_or_path: Its LoRA adapter (the training output_dir). Defaults to None.
        questions: A JSON list or JSONL file of questions in the decision format. Defaults to None,
            which asks the built-in examples.
        calibration: "auto" (the model's saved calibration, if any) or the path of a calibration
            file. Defaults to "auto".
        raw: Whether to answer at temperature 1, ignoring any calibration. Defaults to False.
        prompt_format: The prompt format of training (decision_prompt_format). Defaults to "strands".
        force_bos: Whether prompts start with the BOS token (decision_force_bos in training).
            Defaults to False.
        max_len: The longest question answered, in tokens. Defaults to 4096.
        device: Where to run the model. Defaults to the first GPU if there is one, else the CPU.
    """
    predictor = DecisionPredictor(
        model_name_or_path,
        adapter_name_or_path,
        device=device,
        prompt_format=prompt_format,
        force_bos=force_bos,
        max_len=max_len,
        calibration=None if raw else calibration,
    )
    rows = _load_questions(questions)
    probabilities = predictor.probabilities(rows)
    calibrated = predictor.calibration is not None
    print(f"Answering at {'the saved calibration' if calibrated else 'temperature 1 (uncalibrated)'}.")

    for i, (question, probs) in enumerate(zip(rows, probabilities), start=1):
        if probs is None:
            print(f"\n[{i}] {question['kind']}: {question['instructions']}\n    skipped: longer than {max_len} tokens")
        else:
            _print_decision(i, question, probs)

    # Aggregate metrics over the answered questions that give their gold label, as in training's evaluation
    labeled = [(q, p) for q, p in zip(rows, probabilities) if p is not None and q.get("label") is not None]
    if labeled:
        stats = np.array(
            [
                [
                    max(range(len(p)), key=p.__getitem__),
                    q["label"],
                    DECISION_KINDS.index(q["kind"]),
                    math.log(max(p[q["label"]], 1e-12)),
                    max(p),
                    sum(x * x for x in p),
                ]
                for q, p in labeled
            ]
        )
        metrics = ComputeDecisionMetrics()(SimpleNamespace(predictions=stats))
        print(f"\nOver {len(labeled)} labeled questions (few questions make these noisy):")
        for name in ("accuracy", "nll", "brier", "ece"):
            print(f"    {name}: {metrics[name]:.3f}")


if __name__ == "__main__":
    fire.Fire(predict)
