#!/usr/bin/env bash
# Fine-tunes Strands Decider v19 (SD-v19, strands-decider-2B-hobson-v19 on the Hugging Face Hub, built on Qwen3.5-2B)
# on a Mac (Apple silicon), as in Part 2 of examples/decision/README.md, with training in MLX and evaluation in
# PyTorch on MPS. From training/, in HMF's environment with the mac extra (pip install -e ".[mac]"):
#
#   bash examples/decision/finetune_mac.sh                  # about 45 minutes on an M3 Pro, after downloading SD-v19
#   MAX_SAMPLES=64 bash examples/decision/finetune_mac.sh   # 64 rows for each set, to try it out in a few minutes
#
# This is an illustrative run, which shows how fine-tuning works on a Mac and how long it takes, rather than a recipe
# for a better model. It fine-tunes SD-v19 on rows of massive_intent (choosing the intent of a request to a voice
# assistant, among up to 24 options), one of the tasks that Strands Decider held out of SD-v19's training, and
# measures SD-v19 on other rows of the same task before and after. A model fine-tuned on one task this way gets better
# at that task, but its results on other kinds of questions are not checked here.
#
# It runs these steps, in order:
#   1. Clone Strands Decider (unless STRANDS_DIR points to a clone), build its held-out tasks, and link its data/
#   2. Download SD-v19 from the Hugging Face Hub and convert it to an HMF decision model, with its adapter merged
#   3. Split massive_intent's rows into training, calibration, and test rows that do not overlap
#   4. Evaluate SD-v19 on the test rows, with the calibration it was published with
#   5. Tokenize the training rows
#   6. Fine-tune a new LoRA adapter on SD-v19 with MLX
#   7. Calibrate the fine-tuned model on the calibration rows, as Strands Decider calibrates (strands_decider)
#   8. Evaluate the fine-tuned model on the test rows, and compare it with SD-v19
# Steps whose outputs exist are skipped, so after a failure, running the script again continues where it stopped.

set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/mac_helpers.sh"
SMOKE="${MAX_SAMPLES:+-smoke}"
TRAIN_ROWS="${MAX_SAMPLES:-3000}"
CALIB_ROWS="${MAX_SAMPLES:-500}"
TEST_ROWS="${MAX_SAMPLES:-1000}"

SD_V19=StrandsAgents/strands-decider-2B-hobson-v19
M=./models/strands-v19-merged                              # SD-v19 as an HMF decision model, from step 2
A=$M/LoRA-massive-intent-mac$SMOKE                         # the fine-tuned adapter, from step 6
DATA=./data/decision/massive_intent$SMOKE                  # the split rows, from step 3
TOKENIZED=./data/tokenized/decision/strands_v19_finetune/massive_intent$SMOKE
REPORTS=reports/strands_v19_massive_intent_mac$SMOKE

require_mlx

step "1. Strands Decider's held-out tasks"
strands_data holdout_v5_norule.jsonl
strands_checksums holdout_v5_norule.jsonl

step "2. Download and convert SD-v19"
if [ -f "$M/config.json" ]; then
    echo "Already converted: $M"
else
    "$PY" scripts/strands_to_hmf_decision.py --checkpoint "$SD_V19" --output_dir "$M" --merge
fi

step "3. Split massive_intent into $TRAIN_ROWS training, $CALIB_ROWS calibration, and $TEST_ROWS test rows"
if [ -f "$DATA/dataset_info.json" ]; then
    echo "Already split: $DATA"
else
    "$PY" - "$DATA" "$TRAIN_ROWS" "$CALIB_ROWS" "$TEST_ROWS" <<'EOF'
import json
import os
import random
import sys

folder, sizes = sys.argv[1], [int(n) for n in sys.argv[2:]]
with open("data/strands-decider/holdout_v5_norule.jsonl") as f:
    rows = [line for line in f if line.strip() and json.loads(line)["task"] == "massive_intent"]
random.Random(0).shuffle(rows)
if sum(sizes) > len(rows):
    sys.exit(f"massive_intent has {len(rows)} rows, fewer than the {sum(sizes)} asked for.")
os.makedirs(folder, exist_ok=True)
start, info = 0, {}
for name, size in zip(("train", "calib", "test"), sizes):
    with open(os.path.join(folder, f"{name}.jsonl"), "w") as f:
        f.writelines(rows[start : start + size])
    start += size
    info[f"massive_intent_{name}"] = {"file_name": f"{name}.jsonl", "formatting": "decision"}
with open(os.path.join(folder, "dataset_info.json"), "w") as f:
    json.dump(info, f, indent=2)
print(f"Wrote {', '.join(f'{n} {name}' for name, n in zip(('train', 'calib', 'test'), sizes))} rows to {folder}.")
EOF
fi

mkdir -p "$REPORTS"
EVALUATE=(--data "$DATA/test.jsonl" --split all --limit 0)

step "4. Evaluate SD-v19 on the test rows"
if [ -f "$REPORTS/before.json" ]; then
    echo "Already evaluated: $REPORTS/before.json"
else
    hmf decision-eval evaluate "$M" "${EVALUATE[@]}" --out "$REPORTS/before.json"
fi

step "5. Tokenize the training rows"
TOKENIZE=examples/decision/tokenization/strands_v19_finetune/massive_intent.yaml
if [ -n "${MAX_SAMPLES:-}" ]; then
    TOKENIZE=$(derive "$TOKENIZE" tokenize.yaml "dataset_dir=$DATA" "tokenized_path=$TOKENIZED")
fi
hmf preprocess "$TOKENIZE" --shard_size 20000 --parallel_jobs 8

step "6. Fine-tune SD-v19 with MLX"
if [ -f "$A/adapter_model.safetensors" ]; then
    echo "Already trained: $A"
else
    TRAIN=()
    if [ -n "${MAX_SAMPLES:-}" ]; then
        TRAIN=("dataset_dir=$DATA" "tokenized_path=$TOKENIZED" "output_dir=$A" "logging_steps=1")
    fi
    hmf train examples/decision/train/strands_v19_finetune/lora_massive_intent_mac.yaml ${TRAIN[@]+"${TRAIN[@]}"}
fi

step "7. Calibrate the fine-tuned model on the calibration rows"
hmf decision-eval calibrate "$M" --adapter_name_or_path "$A" --data "$DATA/calib.jsonl" --split all --limit 0 \
    --method strands_decider

step "8. Evaluate the fine-tuned model on the test rows"
hmf decision-eval evaluate "$M" --adapter_name_or_path "$A" "${EVALUATE[@]}" --out "$REPORTS/after.json"

step "Before and after fine-tuning, on $TEST_ROWS test rows of massive_intent"
"$PY" - "$REPORTS" <<'EOF'
import json
import sys

reports = {name: json.load(open(f"{sys.argv[1]}/{name}.json"))["overall"] for name in ("before", "after")}
print(f"{'':10s} {'accuracy':>9s} {'NLL':>7s} {'Brier':>7s} {'ECE':>7s}")
for name, metrics in reports.items():
    print(f"{name:10s} {metrics['accuracy']:9.3f} {metrics['nll']:7.3f} {metrics['brier']:7.3f} {metrics['ece']:7.3f}")
EOF
echo "Fine-tuned model: $M with the adapter $A. Reports: $REPORTS"
