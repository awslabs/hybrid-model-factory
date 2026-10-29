"""
Checks a conversion between a decision model and a Strands Decider checkpoint, made with
scripts/hmf_decision_to_strands.py or scripts/strands_to_hmf_decision.py. Strands Decider's own
loader must give the probabilities the decision model gave. Run it with Strands Decider's Python environment, from its
repository; it imports nothing from HMF.

    cd path/to/strands-decider
    python path/to/<output_dir>.check/strands_parity_check.py path/to/<output_dir>.check

Both conversion scripts write the check folder (with --check_data) and a copy of this script into it:

    rows.jsonl        the rows checked, as in their data file
    hmf_probs.json    the decision model's probabilities over each row's options (null for a row not checked)
    meta.json         the Strands Decider checkpoint's folder

Both models run in bfloat16, the decision model with its LoRA adapter merged into the weights and Strands
Decider's without, so small differences are expected; a conversion error shows as much larger ones.
"""

import json
import os
import sys
from typing import List, Optional, Sequence


# The largest mean difference of the probabilities, and the smallest share of rows whose chosen option agrees
MAX_MEAN_DIFF = 0.01
MIN_AGREEMENT = 0.97


def compare(ours: Sequence[Optional[List[float]]], theirs: Sequence[List[float]]) -> dict:
    """The mean and largest differences of the probabilities, and how often the chosen option agrees."""
    pairs = [(a, b) for a, b in zip(ours, theirs) if a is not None]
    diffs = [abs(p - q) for a, b in pairs for p, q in zip(a, b)]
    agree = sum(max(range(len(a)), key=a.__getitem__) == max(range(len(b)), key=b.__getitem__) for a, b in pairs)
    return {
        "num_rows": len(pairs),
        "mean_diff": sum(diffs) / len(diffs),
        "max_diff": max(diffs),
        "agreement": agree / len(pairs),
    }


def passes(result: dict) -> bool:
    return result["mean_diff"] <= MAX_MEAN_DIFF and result["agreement"] >= MIN_AGREEMENT


def _strands_probs(checkpoint: str, rows: List[dict]) -> List[List[float]]:
    """Strands Decider's probabilities over each row's options, with its own loader and collator."""
    sys.path.insert(0, os.getcwd())  # Strands Decider's training package, from its repository
    import torch
    from strands_decider.modeling import StrandsDeciderModel, masked_log_softmax
    from training.data.format import Example
    from training.evaluate import collect_logits

    model = StrandsDeciderModel.load(checkpoint)
    examples = [Example.from_dict(row) for row in rows]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logits, _, n_slots, _ = collect_logits(
        model, examples, device=device, batch_size=8, max_length=model.config.max_length
    )
    probs = masked_log_softmax(logits, n_slots).exp()
    return [p[:n].tolist() for p, n in zip(probs, n_slots.tolist())]


def main(check_dir: str) -> None:
    with open(os.path.join(check_dir, "meta.json")) as f:
        checkpoint = json.load(f)["checkpoint"]
    with open(os.path.join(check_dir, "rows.jsonl"), encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    with open(os.path.join(check_dir, "hmf_probs.json")) as f:
        ours = json.load(f)

    result = compare(ours, _strands_probs(checkpoint, rows))
    print(
        f"Checked {result['num_rows']} rows: the probabilities differ by {result['mean_diff']:.5f} on average and "
        f"{result['max_diff']:.5f} at most; the chosen option agrees on {result['agreement']:.1%} of rows."
    )
    if not passes(result):
        sys.exit(f"FAILED: expected a mean difference <= {MAX_MEAN_DIFF} and agreement >= {MIN_AGREEMENT:.0%}.")
    print("PASSED")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python strands_parity_check.py <check_dir>")
    main(sys.argv[1])
