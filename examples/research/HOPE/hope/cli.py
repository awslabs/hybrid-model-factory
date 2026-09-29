"""
CLI entry point for HOPE expert pruning.

Usage:
    hope calibrate --model-path <path> --prompts <file> --out-path <path>
    hope solve --obs-path <path> --prune-frac 0.5 --out-path <path>
    hope baselines --obs-path <path> --prune-frac 0.5 --method reap --out-path <path>
    hope prune --model-path <path> --pruneset-path <path> --out-path <path>
"""

import click


@click.group()
def cli():
    """HOPE: Higher-Order Pruning of Experts for MoE LLMs."""
    pass


@cli.command()
@click.option("--model-path", required=True,
              type=click.Path(exists=True, file_okay=False),
              help="Path to the HuggingFace MoE model.")
@click.option("--prompts", required=True,
              type=click.Path(exists=True, dir_okay=False),
              help="Path to prompts file: .json (list of strings, or list of "
              "token ID lists) or .txt (one prompt per line).")
@click.option("--out-path", required=True,
              help="Output HDF5 path for observations.")
@click.option("--limit", type=float, default=None,
              help="Subsample: fraction (<1) or count (>=1) of prompts.")
@click.option("--seed", type=int, default=20260423,
              help="Random seed for subsampling.")
@click.option("--stats-on-cpu", is_flag=True,
              help="Keep accumulators on CPU instead of GPU.")
@click.option("--max-prompt-length", type=int, default=None,
              help="Truncate prompts longer than this (tokens).")
def calibrate(model_path, prompts, out_path, limit, seed, stats_on_cpu,
              max_prompt_length):
    """Collect expert interaction statistics (F-matrix) from a model."""
    import json

    if prompts.endswith(".json"):
        with open(prompts) as f:
            prompt_data = json.load(f)
    else:
        with open(prompts) as f:
            prompt_data = [line.strip() for line in f if line.strip()]

    from hope.calibrate import calibrate as _calibrate
    _calibrate(
        model_path, prompt_data, out_path,
        limit=limit, seed=seed, stats_on_cpu=stats_on_cpu,
        max_prompt_length=max_prompt_length,
    )


@cli.command()
@click.option("--obs-path", required=True,
              type=click.Path(exists=True, dir_okay=False),
              help="Path to HDF5 observations from calibration.")
@click.option("--prune-frac", required=True, type=float,
              help="Fraction of experts to prune per layer, or integer number to prune per layer.")
@click.option("--out-path", required=True,
              help="Output JSON path for the pruning set.")
@click.option("--task-id", default=None,
              help="Task ID in the HDF5 (default: first available).")
def solve(obs_path, prune_frac, out_path, task_id):
    """Solve the HOPE QP to find the optimal pruning set."""
    from hope.solve import solve as _solve
    _solve(
        obs_path, prune_frac, out_path, task_id=task_id,
    )


@cli.command()
@click.option("--obs-path", required=True,
              type=click.Path(exists=True, dir_okay=False),
              help="Path to HDF5 observations from calibration.")
@click.option("--prune-frac", required=True, type=float,
              help="Fraction of experts to prune per layer, or integer number to prune per layer.")
@click.option("--out-path", required=True,
              help="Output JSON path for the pruning set.")
@click.option("--method", required=True,
              type=click.Choice(["reap", "ean", "man", "freq"]),
              help="First-order scoring method.")
@click.option("--task-id", default=None,
              help="Task ID in the HDF5 (default: first available).")
def baselines(obs_path, prune_frac, out_path, method, task_id):
    """Compute first-order baseline pruning sets (REAP, EAN, MAN, FREQ)."""
    from hope.baselines import solve_baselines
    solve_baselines(obs_path, prune_frac, out_path, method, task_id=task_id)


@cli.command()
@click.option("--model-path", required=True,
              type=click.Path(exists=True, file_okay=False),
              help="Path to the original HuggingFace model.")
@click.option("--pruneset-path", required=True,
              type=click.Path(exists=True, dir_okay=False),
              help="Path to JSON pruning set.")
@click.option("--out-path", required=True,
              help="Path to save the pruned model.")
def prune(model_path, pruneset_path, out_path):
    """Apply a pruning set to a model checkpoint and save the pruned model."""
    from hope.prune import prune_model
    prune_model(model_path, pruneset_path, out_path)
