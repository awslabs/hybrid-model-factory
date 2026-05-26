"""Eval format diagnostics for VL benchmarks.

Many VL benchmarks (ChartQA, DocVQA, MMMU, MMStar, AI2D) score predictions
by exact string match. Hybrid VL models trained on conversational data often
produce the right *answer* in the wrong *format* ("3." vs "3", "Answer: B"
vs "B", "There are three bars" vs "3"). This module reports:

  - the strict score (lmms-eval's relaxed_overall / anls / acc as computed)
  - a relaxed score that handles trailing punctuation, word-number conversion,
    5% numeric tolerance, first-number extraction from verbose text, MCQ
    letter extraction with several common phrasings

The gap between strict and relaxed quantifies how much of the
hybrid-vs-dense gap is format vs reasoning. Useful for diagnosing whether a
benchmark regression is genuine reasoning loss or a data-format artifact.

Usage from CLI:
    hmf eval-format chartqa <samples.jsonl>
    hmf eval-format mcq     <samples.jsonl>   # MMStar / MMMU / AI2D
    hmf eval-format docvqa  <samples.jsonl>
    hmf eval-format dump    <samples.jsonl>   # raw prediction inspection
"""
from __future__ import annotations

import json
import re
import sys
from typing import Any, Iterable


WORD_TO_NUM = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
}


def _load_samples(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]


def _get_pred(row: dict) -> str:
    p = row.get("filtered_resps", "")
    if isinstance(p, list):
        p = p[0] if p else ""
    return p


def _get_target(row: dict) -> str:
    t = row.get("target", "")
    if isinstance(t, list):
        t = t[0] if t else ""
    return t


# ---------------------------------------------------------------------------
# ChartQA-style: short factual answer with format tolerance
# ---------------------------------------------------------------------------


def _normalize_chart_answer(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"\.$|^the answer is\s*", "", s)
    s = re.sub(r"\s*%$", "", s)
    return s


def chartqa_relaxed_match(pred: str, target: str) -> bool:
    """Relaxed match: punctuation/word-number/numeric-tolerance/first-number."""
    p = _normalize_chart_answer(pred)
    t = _normalize_chart_answer(target)
    if p == t:
        return True

    for w, n in WORD_TO_NUM.items():
        if w in p and n == t:
            return True
        if n == p and w in t:
            return True

    # 5% numeric tolerance on whole-string number
    try:
        return abs(float(p) - float(t)) / max(abs(float(t)), 1e-6) < 0.05
    except Exception:
        pass

    # First numeric token in the prediction
    m = re.search(r"-?\d+\.?\d*", pred)
    if m:
        try:
            return abs(float(m.group()) - float(t)) / max(abs(float(t)), 1e-6) < 0.05
        except Exception:
            pass

    if t in ("yes", "no") and p.startswith(t):
        return True
    return False


def report_chartqa(path: str) -> None:
    rows = _load_samples(path)
    n = len(rows)
    strict_correct = sum(1 for r in rows if r.get("relaxed_overall", 0) > 0)
    relaxed_correct = 0
    verbose_count = 0
    extracts_recovered = 0
    for r in rows:
        pred, target = _get_pred(r), _get_target(r)
        if len(pred) > 20:
            verbose_count += 1
        if chartqa_relaxed_match(pred, target):
            relaxed_correct += 1
            if pred.strip() != target.strip() and len(pred) > len(target) + 2:
                extracts_recovered += 1

    print(f"path = {path}")
    print(f"n = {n}")
    print(f"strict (lmms-eval relaxed_overall>0) = "
          f"{strict_correct} ({strict_correct/n*100:.1f}%)")
    print(f"my-relaxed (numeric/word tolerance + first-number) = "
          f"{relaxed_correct} ({relaxed_correct/n*100:.1f}%)")
    print(f"verbose responses (>20 chars) = "
          f"{verbose_count} ({verbose_count/n*100:.1f}%)")
    print(f"recovered by extracting number from verbose pred = "
          f"{extracts_recovered} ({extracts_recovered/n*100:.1f}%)")

    print("\n=== sample failures (first 10) ===")
    shown = 0
    for r in rows:
        if r.get("relaxed_overall", 0) > 0:
            continue
        print(f"  target={_get_target(r)!r:20s} pred={_get_pred(r)!r}")
        shown += 1
        if shown >= 10:
            break


# ---------------------------------------------------------------------------
# DocVQA-style: refusal/verbose categorization
# ---------------------------------------------------------------------------


REFUSAL_PHRASES = (
    "i can't", "i cannot", "not visible", "don't see", "do not see",
    "cannot determine", "not provided", "unable to", "not legible",
    "not clear", "i'm sorry", "no information",
)


def report_docvqa(path: str) -> None:
    rows = _load_samples(path)
    n = len(rows)
    zero = sum(1 for r in rows if r.get("anls", 0) == 0)
    correct = n - zero
    verbose = 0
    refusals = 0
    single_token = 0
    for r in rows:
        pred = _get_pred(r)
        if len(pred) > 50:
            verbose += 1
        if len(pred.strip()) <= 2:
            single_token += 1
        if any(x in pred.lower() for x in REFUSAL_PHRASES):
            refusals += 1

    print(f"path             = {path}")
    print(f"n                = {n}")
    print(f"correct (anls>0) = {correct} ({correct/n*100:.1f}%)")
    print(f"zero ANLS        = {zero} ({zero/n*100:.1f}%)")
    print(f"verbose (>50 ch) = {verbose} ({verbose/n*100:.1f}%)")
    print(f"refusal phrases  = {refusals} ({refusals/n*100:.1f}%)")
    print(f"<=2 char preds   = {single_token} ({single_token/n*100:.1f}%)")

    print("\n=== sample failures (first 10) ===")
    shown = 0
    for r in rows:
        if r.get("anls", 0) > 0:
            continue
        print(f"  t={str(_get_target(r))[:50]!r}")
        print(f"  p={_get_pred(r)[:120]!r}")
        print()
        shown += 1
        if shown >= 10:
            break


# ---------------------------------------------------------------------------
# MCQ (MMStar / MMMU / AI2D): letter extraction
# ---------------------------------------------------------------------------


_LETTER_LEADING = re.compile(r"^\s*([A-D])(?:[\.:\)\s]|$)")
_LETTER_PHRASE = re.compile(
    r"(?:correct answer is|answer is|answer:|is)\s*([A-D])\b", re.I
)
_LETTER_FALLBACK = re.compile(r"\b([A-D])\b")


def extract_mcq_letter(s: str) -> str | None:
    """Best-effort A-D extraction from arbitrary model output."""
    s = s.strip()
    if not s:
        return None
    m = _LETTER_LEADING.match(s)
    if m:
        return m.group(1)
    m = _LETTER_PHRASE.search(s)
    if m:
        return m.group(1).upper()
    m = _LETTER_FALLBACK.search(s)
    if m:
        return m.group(1)
    return None


def report_mcq(path: str) -> None:
    rows = _load_samples(path)
    n = len(rows)
    strict = 0
    relaxed = 0
    for r in rows:
        pred = _get_pred(r)
        target = str(_get_target(r)).strip().upper()
        pred_letter = extract_mcq_letter(pred)
        if pred.strip().upper() == target:
            strict += 1
        if pred_letter and pred_letter == target:
            relaxed += 1

    print(f"path            = {path}")
    print(f"n               = {n}")
    print(f"strict-letter   = {strict} ({strict/n*100:.1f}%)")
    print(f"relaxed-extract = {relaxed} ({relaxed/n*100:.1f}%)")
    print(f"delta from extraction = +{(relaxed-strict)/n*100:.1f} pp")


# ---------------------------------------------------------------------------
# Raw prediction dump
# ---------------------------------------------------------------------------


def dump_predictions(path: str, n: int = 12) -> None:
    """Print full filtered_resps strings for the first ``n`` rows."""
    rows = _load_samples(path)
    print(f"path = {path}")
    for r in rows[:n]:
        print(f"  t={str(_get_target(r))[:60]!r}")
        print(f"  p={_get_pred(r)[:200]!r}")
        print()


__all__ = [
    "chartqa_relaxed_match",
    "extract_mcq_letter",
    "report_chartqa",
    "report_docvqa",
    "report_mcq",
    "dump_predictions",
]
