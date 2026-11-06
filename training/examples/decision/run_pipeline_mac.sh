#!/usr/bin/env bash
# Decision model pipeline on a Mac (Apple silicon), with Qwen3-0.6B: Strands Decider's v19 recipe, as in Part 1 of
# examples/decision/README.md, with training in MLX and labeling and evaluation in PyTorch on MPS. From training/,
# in HMF's environment with the mac extra (pip install -e ".[mac]"):
#
#   bash examples/decision/run_pipeline_mac.sh                  # the full recipe
#   MAX_SAMPLES=32 bash examples/decision/run_pipeline_mac.sh   # the same steps on the first 32 rows of each dataset
#
# With MAX_SAMPLES, every step runs on the first MAX_SAMPLES rows of each dataset, without validation splits, and
# writes to folders ending in -smoke, so that the whole pipeline runs in about 15 minutes after the data is built.
#
# It runs these steps, in order:
#   1. Clone Strands Decider (unless STRANDS_DIR points to a clone), build its data, and link it into data/
#   2. Convert Qwen3.5-4B's answers to the multi-step rows, which Strands Decider ships, to an HMF teacher file
#   3. Convert the base model into a decision model
#   4. Tokenize and label model 1's data (the anchor, and the 4B's answers as the teacher)
#   5. Train model 1
#   6. Tokenize and label model 2's data (the anchor, and model 1's answers as the teacher)
#   7. Train model 2
#   8. Calibrate and evaluate model 2
# Steps whose outputs exist are skipped, so after a failure, running the script again continues where it stopped.

set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/mac_helpers.sh"
SMOKE="${MAX_SAMPLES:+-smoke}"

M=./models/Qwen3-0.6B-Base-Decision                  # the decision model, from step 3
A1=$M/LoRA-strands-v14-mac$SMOKE                      # model 1's adapter, from step 5
A2=$M/LoRA-strands-v19-mac$SMOKE                      # model 2's adapter, from step 7
T1=./data/tokenized/decision/qwen3_0_6b/strands_v14$SMOKE   # model 1's tokenized data and labels
T2=./data/tokenized/decision/qwen3_0_6b/strands_v19$SMOKE   # model 2's tokenized data and labels
TEACHER=./data/decision/teachers/teacher_multistep_v14.hmf.jsonl
REPORTS=reports/qwen3_0_6b_v19_mac$SMOKE

require_mlx

step "1. Strands Decider's data"
strands_data train_v5.jsonl holdout_v5_norule.jsonl multistep_v14.jsonl multistep_v14_eval.jsonl \
    generated_v16.jsonl generated_v16_eval.jsonl generated_v18.jsonl generated_v18_eval.jsonl \
    adequacy_hs2.jsonl adequacy_hs2_eval.jsonl adequacy_gen.jsonl adequacy_gen_eval.jsonl
# The teacher's answers attach to the multi-step rows by their position, so both files must be the exact ones
# Strands Decider recorded
strands_checksums multistep_v14.jsonl synthetic/teacher_multistep_v14.jsonl

step "2. Qwen3.5-4B's answers to the multi-step rows, as an HMF teacher file"
if [ -f "$TEACHER" ]; then
    echo "Already converted: $TEACHER"
else
    mkdir -p "$(dirname "$TEACHER")"
    "$PY" scripts/strands_teacher_to_hmf.py --teacher data/strands-decider/synthetic/teacher_multistep_v14.jsonl \
        --datasets strands_multistep_v14 --output "$TEACHER"
fi

step "3. Convert Qwen3-0.6B-Base into a decision model"
if [ -f "$M/config.json" ]; then
    echo "Already converted: $M"
else
    hmf decision-init examples/decision/init/qwen3_0_6b/pointer.yaml
fi

# With MAX_SAMPLES, the configs get the smaller data, no validation split, and the -smoke folders
TOKENIZE_V14=examples/decision/tokenization/qwen3_0_6b/strands_v14.yaml
TOKENIZE_V19=examples/decision/tokenization/qwen3_0_6b/strands_v19_replay.yaml
EVAL=examples/decision/eval/strands_v19_mac.yaml
TRAIN_V14=(); TRAIN_V19=()
if [ -n "${MAX_SAMPLES:-}" ]; then
    SMALL=("max_samples=$MAX_SAMPLES" "val_size=0")
    TOKENIZE_V14=$(derive "$TOKENIZE_V14" tokenize_v14.yaml "${SMALL[@]}" "tokenized_path=$T1")
    TOKENIZE_V19=$(derive "$TOKENIZE_V19" tokenize_v19.yaml "${SMALL[@]}" "tokenized_path=$T2" \
        "decision_teacher_adapter=$A1")
    EVAL=$(derive "$EVAL" eval.yaml "calibration.limit=200" "benchmarks.*.limit=50")
    NO_EVAL=("do_eval=false" "eval_strategy='no'" "logging_steps=1")  # quoted, as YAML reads a bare no as false
    TRAIN_V14=("tokenized_path=$T1" "output_dir=$A1" "${NO_EVAL[@]}")
    TRAIN_V19=("tokenized_path=$T2" "output_dir=$A2" "${NO_EVAL[@]}")
fi

step "4. Tokenize and label model 1's data (anchor: Qwen3-0.6B-Base; teacher: Qwen3.5-4B's shipped answers)"
hmf preprocess "$TOKENIZE_V14" --shard_size 20000 --parallel_jobs 8

step "5. Train model 1 with MLX"
if [ -f "$A1/adapter_model.safetensors" ]; then
    echo "Already trained: $A1"
else
    hmf train examples/decision/train/qwen3_0_6b/lora_v14_mac.yaml ${TRAIN_V14[@]+"${TRAIN_V14[@]}"}
fi

step "6. Tokenize and label model 2's data (anchor: Qwen3-0.6B-Base; teacher: model 1)"
hmf preprocess "$TOKENIZE_V19" --shard_size 20000 --parallel_jobs 8

step "7. Train model 2 with MLX"
if [ -f "$A2/adapter_model.safetensors" ]; then
    echo "Already trained: $A2"
else
    hmf train examples/decision/train/qwen3_0_6b/lora_v19_mac.yaml ${TRAIN_V19[@]+"${TRAIN_V19[@]}"}
fi

step "8. Calibrate and evaluate model 2"
hmf decision-eval run "$EVAL" model_name_or_path=$M adapter_name_or_path=$A2 output_dir=$REPORTS

step "Done"
echo "Model 2: $M with the adapter $A2. Reports: $REPORTS"
