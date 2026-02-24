#!/usr/bin/env python3
"""Select optimal layers for sliding window conversion from a CSV.

Input CSV format: must have `layer_num` column. All other columns are treated as
task scores. A row with layer_num=-1 is treated as baseline and excluded from
selection.

Example:
    python layer_selection.py results.csv --M 14
    python layer_selection.py results.csv --M 14 --task-filter helmet
"""
import csv
import argparse

META_COLS = {"layer_num", "layer_type", "n_runs", "codename", "window_size", "model"}


def load_scores(path, task_filter):
    with open(path) as f:
        rows = list(csv.DictReader(f))

    task_cols = [c for c in rows[0] if c not in META_COLS]
    if task_filter:
        task_cols = [c for c in task_cols if task_filter in c]

    layers, baseline = {}, None
    for row in rows:
        scores = {c: float(row[c]) for c in task_cols if row[c]}
        layer_num = int(row["layer_num"])
        if layer_num == -1:
            baseline = scores
        else:
            layers[layer_num] = scores
    return layers, baseline, task_cols


def rank_layers(layers, task_names):
    """Rank layers by mean score across tasks (higher = more performance retained)."""
    scored = {}
    for layer, scores in layers.items():
        vals = [scores[t] for t in task_names if t in scores]
        if vals:
            scored[layer] = sum(vals) / len(vals)
    return sorted(scored.items(), key=lambda x: x[1], reverse=True)


def main():
    parser = argparse.ArgumentParser(description="Select top-M layers for sliding window conversion")
    parser.add_argument("input", help="CSV with layer_num + task score columns")
    parser.add_argument("--M", type=int, default=14, help="Number of layers to select")
    parser.add_argument("--task-filter", default=None, help="Only use tasks containing this substring")
    parser.add_argument("--exclude-layer-0", action="store_true")
    args = parser.parse_args()

    layers, baseline, task_names = load_scores(args.input, args.task_filter)
    if args.exclude_layer_0:
        layers.pop(0, None)

    print(f"Tasks ({len(task_names)}): {task_names}")
    print(f"Layers: {len(layers)}")
    bl_mean = None
    if baseline:
        bl_mean = sum(baseline.values()) / len(baseline)
        print(f"Baseline mean: {bl_mean:.4f}")

    ranked = rank_layers(layers, task_names)
    selected = sorted([l for l, _ in ranked[:args.M]])

    print("\nFull ranking:")
    for i, (layer, score) in enumerate(ranked):
        drop = f" | drop: {score - bl_mean:+.4f}" if bl_mean else ""
        print(f"  {i+1:2d}. layer {layer:2d}  score: {score:.4f}{drop}")

    print(f"\nSelected (M={args.M}): {selected}")


if __name__ == "__main__":
    main()
