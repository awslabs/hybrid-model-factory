#!/usr/bin/env bash
# Decision model pipeline: steps 1 to 7 of Part 1 of examples/decision/README.md, which explains each
# step. Before running it, build Strands Decider's data and link it into data/strands-decider, as
# "Before You Start" in the README describes. Then, from training/:
#
#   bash examples/decision/run_pipeline.sh
#
# To continue after a failure, comment out the steps that have already finished.

set -euo pipefail

M=./models/Qwen3.5-2B-Base-Decision     # the decision model, from step 1
A=$M/LoRA-strands-v19                   # model 2's adapter, from step 5
J=jevbench                              # a JevBench checkout, cloned below if it does not exist

# 1. Convert the base model into a decision model
hmf decision-init examples/decision/init/qwen3_5_2b/pointer.yaml

# 2. Tokenize and label model 1's data
hmf preprocess examples/decision/tokenization/qwen3_5_2b/strands_v14.yaml --shard_size 20000 --parallel_jobs 32

# 3. Train model 1
hmf train examples/decision/train/qwen3_5_2b/lora_v14.yaml

# 4. Tokenize and label model 2's data, with model 1 as the teacher
hmf preprocess examples/decision/tokenization/qwen3_5_2b/strands_v19_replay.yaml --shard_size 20000 --parallel_jobs 32

# 5. Train model 2
hmf train examples/decision/train/qwen3_5_2b/lora_v19.yaml

# 6 and 7. Calibrate and evaluate model 2, and run JevBench, at the version Strands Decider's results used
if [ ! -d $J ]; then
    git clone https://github.com/fstandhartinger/jevbench $J
    git -C $J checkout 1bcc55eb6c8cffde2306b3db03ede39b61c6152a
fi
hmf decision-eval run examples/decision/eval/strands_v19.yaml model_name_or_path=$M adapter_name_or_path=$A \
    output_dir=reports/qwen3_5_2b_v19 jevbench.dir=$J jevbench.compare=null
