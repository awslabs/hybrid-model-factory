"""
Quickstart: Prune a MoE model with HOPE in three steps.

Usage:
    python examples/quickstart.py \
        --model-path /path/to/moe-model \
        --prompts calibration_prompts.txt \
        --prune-frac 0.25 \
        --out-dir /path/to/out-dir

If --prompts is not provided, a handful of built-in demo prompts are used.
In practice, calibration should use hundreds to thousands of domain-specific
prompts for best results (see the paper for details).
"""

import argparse
import json
import os


DEMO_PROMPTS = [
    "Write a Python function that computes the Fibonacci sequence.",
    "Explain the theory of general relativity in simple terms.",
    "What are the main differences between TCP and UDP?",
    "Translate the following English text to French: The weather is nice today.",
    "Solve the equation 2x + 5 = 17.",
]


def load_prompts(path):
    """
    Load prompts from a .txt (one string per line) or .json (list of strings,
    or list of lists of ints (pre-tokenized prompts).
    """
    if path.endswith(".json"):
        with open(path) as f:
            return json.load(f)
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def main():
    parser = argparse.ArgumentParser(
        description="Prune a MoE model with HOPE",
    )
    parser.add_argument(
        "--model-path", required=True,
        help="Path to a HuggingFace MoE model directory.",
    )
    parser.add_argument(
        "--prompts", default=None,
        help="Path to calibration prompts (.txt or .json). "
             "If omitted, uses a few built-in demo prompts.",
    )
    parser.add_argument(
        "--prune-frac", type=float, default=0.25,
        help="Fraction of experts to prune per layer (default: 0.25).",
    )
    parser.add_argument(
        "--out-dir", required=True,
        help="Directory to save observations, pruning set, and pruned model.",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    obs_path = os.path.join(args.out_dir, "observations.h5")
    pruneset_path = os.path.join(args.out_dir, "pruneset.json")
    pruned_model_path = os.path.join(args.out_dir, "pruned_model")

    # Load prompts
    if args.prompts:
        prompts = load_prompts(args.prompts)
        print("Loaded %d prompts from %s" % (len(prompts), args.prompts))
    else:
        prompts = DEMO_PROMPTS
        print("Using %d built-in demo prompts (pass --prompts for real use)" %
              len(prompts))

    # Step 1: Calibrate — collect expert interaction statistics
    print("\n[1/3] Calibrating F-matrix...")
    from hope.calibrate import calibrate
    calibrate(args.model_path, prompts, obs_path)

    # Step 2: Solve — find the optimal pruning set via QP
    print("\n[2/3] Solving HOPE QP (pruning %.0f%% of experts)..." %
          (args.prune_frac * 100))
    from hope.solve import solve
    solve(obs_path, args.prune_frac, pruneset_path)

    # Step 3: Prune — apply the pruning set to the model
    print("\n[3/3] Pruning model...")
    from hope.prune import prune_model
    prune_model(args.model_path, pruneset_path, pruned_model_path)

    # Summary
    with open(pruneset_path) as f:
        ps = json.load(f)
    num_layers = len(ps)
    num_pruned = len(next(iter(ps.values())))

    print("\n" + "=" * 60)
    print("Done!")
    print("  Observations:  %s" % obs_path)
    print("  Pruning set:   %s (%d experts pruned per layer)" %
          (pruneset_path, num_pruned))
    print("  Pruned model:  %s" % pruned_model_path)
    print()
    print("Load the pruned model:")
    print("  from transformers import AutoModelForCausalLM")
    print("  model = AutoModelForCausalLM.from_pretrained(\"%s\")" %
          pruned_model_path)
    print("=" * 60)


if __name__ == "__main__":
    main()
