# Priming Layer Selection

The `vllm-inference/selection/` directory contains tools for choosing which layers to convert when creating a new Hybrid model config
and the raw importance scores used for selecting layers for priming in Qwen3-8B/32B models.

### `create_swa_variants.py`

Creates symlinked model directories where each variant has a single layer replaced with Sliding Window Attention. Supports Qwen2, Qwen3, and Qwen3-MoE architectures. This is used to evaluate per-layer importance — measuring performance drops when a layer is individually replaced gives layer importance scores which can be used for selection.

```bash
python vllm-inference/selection/create_swa_variants.py \
  --src /path/to/public-qwen-model \
  --dst /path/to/qwen-swa-variants \
  --window-size 2048
```

This creates one directory per layer (e.g., `swa_layer_0/`, `swa_layer_1/`, ...), each with a modified `config.json` that sets that single layer to SWA. All model weight files are symlinked to avoid large model weight duplication.

### `layer_selection.py`

After evaluating each SWA variant on your target benchmarks, use this script to rank layers and select the top-M for hybridization. For example:

```bash
cd vllm-inference/selection/
python layer_selection.py importance_scores_qwen3-8b.csv --M 18
```

Custom importance score csv files should have the following structure:

```csv
layer_num,task_1,task_2,...,task_K
-1,<baseline_score>,<baseline_score>,...,<baseline_score>
0,<score>,<score>,...,<score>
1,<score>,<score>,...,<score>
...
N,<score>,<score>,...,<score>
```

- `layer_num`: integer layer index. The row with `layer_num=-1` is the baseline (full-attention model, no SWA replacement).
- All other columns are treated as task scores (float). Column names are free-form, but a consistent prefix is useful with `--task-filter` (e.g. `helmet_cite__32768::suite`).

Each non-baseline row contains the score when *only that layer* is replaced with SWA. The script ranks layers by mean score (higher = more performance retained when that layer is replaced) and outputs the top-M layers to convert.

Pre-computed importance scores for Qwen3-8B and Qwen3-32B using 5 HELMET tasks at 32k context are included in `vllm-inference/selection/`.
