# Training Decision Models

A **decision model** reads a situation, a question about it, and a list of possible answers (the options), and returns how likely each option is to be the right one. HMF builds it from a language model, by removing the part that writes text and adding a small part that scores the options (the "head"). Training then teaches the model to answer well and to give probabilities you can trust.

This guide has two parts:

- [Part 1: Converting an LLM to a Decision Model](#part-1-converting-an-llm-to-a-decision-model) turns a language model into a decision model and trains it from the start.
- [Part 2: Fine-tuning an Existing Decision Model](#part-2-fine-tuning-an-existing-decision-model) adapts a decision model that is already trained to your own questions, which takes much less data and time.

Run every command from the `training/` folder. The steps use example configs, which you copy and change to point at your own paths (the paths to change are marked in each file).

## Part 1: Converting an LLM to a Decision Model

This part converts Qwen3.5-2B-Base into a decision model and trains it. As a worked example, it follows the recipe of Strands Decider's published v19 model ([strands-decider-2B-hobson-v19](https://huggingface.co/StrandsAgents/strands-decider-2B-hobson-v19)), so that the results can be compared with v19's (see [What to Expect](#what-to-expect)). Other recipes use the same steps with other settings. For example, training on the correct answers alone needs no labeling in steps 2 and 4 and a single training run, with [train/qwen3_5_2b/lora_ce.yaml](train/qwen3_5_2b/lora_ce.yaml).

### Overview of the Steps

These steps train two models, one after the other:

1. **Model 1** learns from the data, and from a larger model's answers (Qwen3.5-4B) on the hardest rows, which are long, multi-step documents.
2. **Model 2** is the final model. It starts again from the base model and learns from more data, and from model 1's answers on those same multi-step rows. That keeps what model 1 learned about them, which more data alone would make it forget.

Both also learn to stay close to the base model's own answers (the "anchor"). [recipes.md](recipes.md) explains these choices.

### Before You Start

- **GPUs.** The times below are for 8 H200s.
- **The data.** The training and evaluation files used here are built by [Strands Decider](https://github.com/strands-labs/strands-decider). Clone it, build its data, and link its `data/` folder into HMF:

  ```bash
  git clone https://github.com/strands-labs/strands-decider path/to/strands-decider
  (cd path/to/strands-decider && training/recipe.sh build fetch multistep generated adequacy)
  ln -s path/to/strands-decider/data data/strands-decider
  ```

  The files are registered in `data/dataset_info.json` under names starting with `strands_`.
- **JevBench**, which [step 7](#7-run-jevbench) sets up.

### 1. Convert the Base Model

Turn the language model into a decision model:

```bash
hmf decision-init examples/decision/init/qwen3_5_2b/pointer.yaml
```

Like Stage 0 of priming (`hmf prime-init`), this step builds a new architecture from the base model's weights without training it. The language model's head, which turns its last hidden states into next-token probabilities, is removed, and a small decision head that scores the options (about 2 million parameters) takes its place. The decision head starts from random weights, and training teaches it. The result is saved as a new model folder (`output_dir` in the config). Every later step starts from this folder, and training never changes it.

### 2. Prepare Model 1's Data

This step and step 4 use Strands Decider's data as an example. [Preparing the Data](#preparing-the-data) in Part 2 describes the format for your own data.

```bash
hmf preprocess examples/decision/tokenization/qwen3_5_2b/strands_v14.yaml --shard_size 20000 --parallel_jobs 32
```

This does two things:

1. **Tokenizes** the rows (`train_v5` and `multistep_v14`), turning text into the model's input, and keeps 3% of them aside to check training.
2. **Labels** them with the extra targets model 1 learns from, using the GPUs:
   - the **anchor**, which is the base model's own answer to each row;
   - the **teacher**, which is Qwen3.5-4B's answer to each multi-step row.

The tokens go to `tokenized_path`, and the labels to a folder inside it, `labels/<decision_label_folder>`. The last line of the output gives the counts; for this data, about 110,000 rows, of which about 89,000 have an anchor (rows with up to 9 options) and about 12,500 have a teacher (the multi-step rows). Running the command again skips the work that is already done.

### 3. Train Model 1

```bash
hmf train examples/decision/train/qwen3_5_2b/lora_v14.yaml
```

Model 1 is trained with LoRA. The weights of the model from step 1 stay frozen, and training learns small low-rank adapters (rank 16) on its attention, MLP, and Gated DeltaNet layers, together with all of the decision head's weights. Training takes about 25 minutes on 8 H200s.

At the start, the log says how many rows have an anchor and a teacher (`Anchor term: weight 0.3, on … rows`). During training, it shows the three parts of the loss, which are `ce_loss` (learning the correct answers), `anchor_kl`, and `teacher_kl`. At the end, it reports accuracy and calibration on the 3% kept aside.

The trained model is saved in the config's `output_dir`. It holds only the LoRA adapters and the trained decision head, which are applied on top of the model from step 1, so it is small.

### 4. Prepare Model 2's Data

```bash
hmf preprocess examples/decision/tokenization/qwen3_5_2b/strands_v19_replay.yaml --shard_size 20000 --parallel_jobs 32
```

The same as step 2, for model 2's data (six files). The teacher is now model 1, so the config points at the model from step 1 (`decision_teacher_model`) and at model 1's adapter from step 3 (`decision_teacher_adapter`). About 12,500 rows should get a teacher.

### 5. Train Model 2

```bash
hmf train examples/decision/train/qwen3_5_2b/lora_v19.yaml
```

Model 2 is trained with LoRA in the same way, in about 20 minutes on 8 H200s. It starts from the model of step 1, not from model 1, because model 1 only provides answers to learn from.

### 6. Calibrate and Evaluate

A model's probabilities are usually a little over- or under-confident. **Calibration** corrects that with one number per question type, fitted on held-out rows, and saves it with the adapter. It does not change which answer the model picks.

```bash
M=path/to/decision-model           # step 1
A=path/to/model-2-adapter          # step 5's output_dir
D=data/strands-decider

hmf decision-eval calibrate $M --adapter_name_or_path $A --data $D/holdout_v5_norule.jsonl
hmf decision-eval evaluate $M --adapter_name_or_path $A --data $D/holdout_v5_norule.jsonl --limit 6000
hmf decision-eval evaluate $M --adapter_name_or_path $A --data $D/multistep_v14_eval.jsonl --split all --limit 0
hmf decision-eval evaluate $M --adapter_name_or_path $A --data $D/adequacy_hs2_eval.jsonl --split all --limit 0
```

The first evaluation uses tasks the model never trained on. Calibration and evaluation use different halves of that file, so the model is never scored on the rows it was calibrated on. [The evaluation guide](../../src/hmf/eval/decision/README.md) explains every number in the reports.

### 7. Run JevBench

[JevBench](https://github.com/fstandhartinger/jevbench) is an external benchmark of 231 decision tasks. It sends each task to a server that runs your model, on the same machine. JevBench needs nothing installed, because it runs from its folder.

```bash
J=path/to/jevbench
git clone https://github.com/fstandhartinger/jevbench $J
git -C $J checkout 1bcc55eb6c8cffde2306b3db03ede39b61c6152a   # the version Strands Decider's results used

hmf decision-serve $M --adapter_name_or_path $A &   # the server, on 127.0.0.1:8099
mkdir -p reports/jevbench
cat $J/datasets/public/{original,easy,hard}.jsonl > reports/jevbench/tasks.jsonl
PYTHONPATH=$J python -m jevbench.cli run --tasks reports/jevbench/tasks.jsonl --adapter typesafe \
    --endpoint http://127.0.0.1:8099 --key-env '' --model my-model \
    --cost-basis no_billable_account_public_endpoint --reserve-usd 0 \
    --results reports/jevbench/results.jsonl --raw-dir reports/jevbench/raw --ledger reports/jevbench/ledger.jsonl
PYTHONPATH=$J python -m jevbench.cli summarize --tasks reports/jevbench/tasks.jsonl \
    --results reports/jevbench/results.jsonl --ledger reports/jevbench/ledger.jsonl \
    --public-export reports/jevbench/summary.json
kill %1   # stop the server
```

`summary.json` gives the number of tasks answered correctly (`n_correct`). To compare with Strands Decider's v19 task by task:

```bash
hmf decision-eval compare path/to/strands-decider/research/data/jevbench_results.csv \
    reports/jevbench/results.jsonl --a_run v19
```

The last number, `p`, tells whether the two differ by more than chance. A value of 0.05 or more means they do not.

#### Steps 6 and 7 in One Command

An evaluation suite runs both steps, and writes every result to one folder with a summary table:

```bash
hmf decision-eval run examples/decision/eval/strands_v19.yaml model_name_or_path=$M adapter_name_or_path=$A \
    output_dir=reports/my-model jevbench.dir=$J
```

The suite file lists what to run. Before running it, set the path of the reference run in its `jevbench.compare` section (Strands Decider's `research/data/jevbench_results.csv`), or delete that section. [The evaluation guide](../../src/hmf/eval/decision/README.md#all-at-once-evaluation-suites) explains the file.

### 8. Ask the Model Questions

The trained model can be asked questions directly, from a script or from Python. Both answer at the calibration saved in step 6. The script prints, for each question, the probability of each option and the chosen option, and for yes/no and rating questions also P(true) and the expected level:

```bash
python scripts/decision_predict.py --model_name_or_path $M --adapter_name_or_path $A
```

Without `--questions`, it asks a few built-in examples. With `--questions my_questions.jsonl`, it asks questions in the decision format ([Preparing the Data](#preparing-the-data)), and marks those that include a `label` as correct or incorrect. `--raw` answers at temperature 1, without the calibration.

In Python, `DecisionPredictor` gives the same answers:

```python
from hmf.eval.decision.predictor import DecisionPredictor

predictor = DecisionPredictor("path/to/decision-model", "path/to/model-2-adapter")
question = {
    "kind": "choice",
    "state": "Order 1182 was charged twice. The customer asked for the duplicate charge to be removed.",
    "instructions": "What should support do next?",
    "options": [["refund", "return the duplicate payment"], ["ignore", None], ["escalate", "send the case to a human agent"]],
}
[probabilities] = predictor.probabilities([question])   # one probability per option, in the order given
```

A question longer than 4,096 tokens gets `None` instead of probabilities. To answer requests over HTTP, `hmf decision-serve` (step 7) runs the model behind JevBench's protocol, on this machine only.

### What to Expect

A model trained with these steps matches v19 when it scores 157 to 177 on JevBench, its `p` against v19 is at least 0.05, and its HelpSteer2 accuracy (the last evaluation of step 6) is at least 0.689. One run of these steps in HMF gave the results below:

| | v19 (Strands Decider) | trained in HMF with these steps |
| --- | --- | --- |
| JevBench, of 231 | 167 | 168 (p = 1.00) |
| HelpSteer2 | 0.739 | 0.726 |
| held-out tasks: accuracy | 0.647 | 0.643 |
| HotpotQA (a source never trained on) | 0.726 | 0.734 |

Two runs with the same settings differ by about 3 JevBench tasks, and sometimes more. Seven training runs of these steps in HMF ranged from 164 to 176.

The same steps on a larger hybrid model score higher (see [Example: GKA-primed-HQwen3-8B-Instruct](#example-gka-primed-hqwen3-8b-instruct)).

### Other Models and Data

The steps stay the same for other models and other data, with the changes listed below.

- **Another base model.** Convert it in step 1 (`base_model_name_or_path`), and point every later config at the new model folder. If its tokenizer differs from Qwen3.5's (for example any Qwen3 or Llama model), redo steps 2 and 4, because the data must be tokenized for it. Set the anchor (`decision_anchor_model`) to the new base model.
- **HMF's hybrid models** (with GKA, Gated DeltaNet, or Mamba layers) work as they are. Set `lora_target: all`, so that the adapters also cover their layers.
- **Larger models** need more GPU memory. The first two settings below reduce the memory that training uses, and the script after them shows where training time goes.
  - `decision_checkpoint_min_tokens: 4096` saves memory on long batches only, at little cost in speed (it recomputes some values instead of storing them).
  - DeepSpeed ZeRO-3 splits the model across the GPUs, for models that do not fit on one. [train/qwen3_5_2b/lora_v19_z3.yaml](train/qwen3_5_2b/lora_v19_z3.yaml) trains model 2 with it.
  - `scripts/decision_benchmark.py` times the forward and backward passes of a model on its tokenized data, on one GPU, to show where training time goes. It is useful when a new model trains more slowly than expected.
- **Your own data.** Any data in the decision format works (see [Preparing the Data](#preparing-the-data)). The anchor and teacher are optional, and without them training learns the correct answers only.

#### Example: GKA-primed-HQwen3-8B-Instruct

The configs in the `hqwen3_gka_8b/` folders follow the same steps on [GKA-primed-HQwen3-8B-Instruct](https://huggingface.co/amazon/GKA-primed-HQwen3-8B-Instruct), an HMF hybrid with Gated KalmaNet layers. They replace the 2B configs of steps 1 to 5, and steps 6 and 7 stay the same.

| step | config |
| --- | --- |
| 1. Convert the base model | [init/hqwen3_gka_8b/pointer.yaml](init/hqwen3_gka_8b/pointer.yaml) |
| 2. Prepare model 1's data | [tokenization/hqwen3_gka_8b/strands_v14.yaml](tokenization/hqwen3_gka_8b/strands_v14.yaml) |
| 3. Train model 1 | [train/hqwen3_gka_8b/lora_v14.yaml](train/hqwen3_gka_8b/lora_v14.yaml) |
| 4. Prepare model 2's data | [tokenization/hqwen3_gka_8b/strands_v19_replay.yaml](tokenization/hqwen3_gka_8b/strands_v19_replay.yaml) |
| 5. Train model 2 | [train/hqwen3_gka_8b/lora_v19.yaml](train/hqwen3_gka_8b/lora_v19.yaml) |

These configs apply the changes listed above. Because the model's tokenizer is Qwen3's, steps 2 and 4 tokenize the data again, and the anchor is the GKA model itself. Training uses `lora_target: all`, and `decision_checkpoint_min_tokens: 4096` for the long multi-step batches. This model's language model head has its own weights, so step 1 makes the model about 620 million parameters smaller.

On 8 H200s, training takes about 70 minutes for model 1 and about 90 minutes for model 2. One run of these configs gave the results below, next to v19 and to the 2B model trained with these steps. The GKA model's JevBench window is 3,072 tokens (`jevbench.max_len=3072` in step 7).

| | v19 (Strands Decider) | Qwen3.5-2B-Base | GKA-primed-HQwen3-8B-Instruct |
| --- | --- | --- | --- |
| JevBench, of 231 | 167 | 168 | 191 (p = 0.0002 against v19) |
| HelpSteer2 | 0.739 | 0.726 | 0.756 |
| held-out tasks: accuracy | 0.647 | 0.643 | 0.672 |
| HotpotQA (a source never trained on) | 0.726 | 0.734 | 0.837 |

## Part 2: Fine-tuning an Existing Decision Model

Fine-tuning teaches a trained decision model your own questions. The model already knows how to read options and give probabilities, so it needs far less data and time than training from a language model.

### The Starting Model

Any trained decision model can be fine-tuned:

- a model trained with Part 1, as the decision model folder from step 1 and its adapter;
- the same model merged into one folder with `hmf export` (`stage: decision`, with the decision model, its adapter, and an `export_dir`);
- a model trained with another tool and converted to HMF's format, such as Strands Decider's (next section).

### Example: Starting From Strands Decider's v19

Strands Decider publishes its trained models on the Hugging Face Hub. A script converts one into an HMF decision model. The conversion needs nothing from Strands Decider, and downloads the model and its base model when they are not local folders.

```bash
# A decision model and its adapter, as training in Part 1 writes them
python scripts/strands_to_hmf_decision.py --checkpoint StrandsAgents/strands-decider-2B-hobson-v19 \
    --output_dir path/to/strands-v19

# One decision model, with the adapter merged into its weights
python scripts/strands_to_hmf_decision.py --checkpoint StrandsAgents/strands-decider-2B-hobson-v19 \
    --output_dir path/to/strands-v19-merged --merge
```

The first command writes the decision model to `path/to/strands-v19/model` and its adapter, which also holds the trained head, to `path/to/strands-v19/adapter`. The second writes one decision model to `path/to/strands-v19-merged`. Either way, the converted model gives the same answers as the original, and keeps its calibration.

At the end, the conversion prints the values of three settings that the configs of Part 2 must use for this model, which are the longest prompt (`cutoff_len`), whether prompts start with the BOS token (`decision_force_bos`), and the smoothing of rating questions (`decision_ordinal_smoothing`). With `--check_data path/to/rows.jsonl`, it also prepares a check that Strands Decider's own code gives the converted model's probabilities on those rows, which runs in Strands Decider's environment. Models built on Qwen3.5 can be converted, but models built on Gemma 4 cannot yet.

### Three Ways to Start Training

A training config can start from a model with an adapter in three ways. In all three, the head continues from the starting model's trained head, and only the adapter and the head are trained.

| | what to set | what training learns | what you get |
| --- | --- | --- | --- |
| **Continue the adapter** | `model_name_or_path` (the decision model) and `adapter_name_or_path` (its adapter) | more of the same adapter, with its own rank and layers (`lora_rank`, `lora_alpha`, and `lora_target` have no effect) | one adapter on the decision model |
| **Add a new adapter on top** | the same two settings, and `create_new_adapter: true` | a new adapter, with the config's LoRA settings, on top of the starting adapter, which is merged into the model when training starts | the new adapter, which works only on top of the merged starting model |
| **Add a new adapter to the merged model** | `model_name_or_path` (the merged folder), without `adapter_name_or_path` | a new adapter, with the config's LoRA settings | one adapter on the merged model |

The third way is the simplest to use afterwards, because the result is an ordinary adapter on an ordinary model, and it is the one the rest of this part uses. The second way gives the same model without writing a merged copy first, but its result needs both adapters. To evaluate or serve it, first merge both into one model with `hmf export`, listing them in order in `adapter_name_or_path` (`path/to/adapter,path/to/new-adapter`).

### Preparing the Data

Each row of the data is one question about a situation (the "state"), with its options and the correct answer. There are three kinds of question:

- `noul` is a yes or no question, whose options are `false` and `true`.
- `choice` asks for one answer out of several.
- `score` asks for a rating on an ordered scale, whose options are the levels, from lowest to highest.

These three rows ask about the same customer message, one row per line of a JSON Lines file:

```json
{"kind": "noul", "state": "My payouts have failed for three days and I have rent due tomorrow. Please fix this today.", "instructions": "Does the customer need an answer today?", "options": [["false", "the request can wait"], ["true", "the request is urgent"]], "label": 1, "task": "urgency"}
{"kind": "choice", "state": "My payouts have failed for three days and I have rent due tomorrow. Please fix this today.", "instructions": "Which team should handle this message?", "options": [["billing", "payments, payouts, and refunds"], ["sales", "plans and pricing"], ["technical", "errors in the product"]], "label": 0, "task": "routing", "instruction_variants": ["Who should answer this message?"]}
{"kind": "score", "state": "My payouts have failed for three days and I have rent due tomorrow. Please fix this today.", "instructions": "How frustrated is the customer?", "options": [["calm", "no sign of frustration"], ["annoyed", "some frustration"], ["very frustrated", "strong frustration or distress"]], "label": 2, "task": "frustration"}
```

`label` is the position of the correct option, counting from 0. `task` names the kind of question, so that results can be reported per task. `instruction_variants` lists other ways to ask the question, which training picks from at random so that the model does not depend on one wording. `weight` (not shown) makes a row count more or less in training. [data/README.md](../../data/README.md) ("Decision Format") describes every field.

Register your files in a `dataset_info.json` in your own data folder, and set `dataset_dir` to that folder in the configs:

```json
{
  "my_train": {"file_name": "train.jsonl", "formatting": "decision"},
  "my_dev": {"file_name": "dev.jsonl", "formatting": "decision"}
}
```

Keep three sets of rows out of training, one to choose the checkpoint (dev), one to calibrate, and one to measure the final results (test). When several rows come from the same document, put all of them in the same set, because a question about a document the model trained on is easier than one about a new document. Training's own validation split (`val_size` in the tokenization config) picks rows at random within each task, so it does not keep documents together. A separate dev file is therefore scored after training (below), because `hmf preprocess` tokenizes only the datasets in `dataset`.

### Training

Tokenizing and training use the same commands as Part 1, with configs copied from Part 1's and a few settings changed. In the tokenization config:

```yaml
model_name_or_path: path/to/strands-v19-merged   # the starting model, whose tokenizer is used
dataset_dir: path/to/my-data
dataset: my_train
tokenized_path: path/to/my-data/tokenized
decision_num_augmentations: 3                    # one new shuffling of the options per epoch
val_size: 0.03                                   # optional, keeps 3% of the rows of each task to check training
# Optional: label each row with the starting model's own answers, to keep the new model close to them
decision_label_folder: starting-model
decision_teacher_model: path/to/strands-v19-merged
```

Then, in the training config:

```yaml
model_name_or_path: path/to/strands-v19-merged
dataset_dir: path/to/my-data
dataset: my_train
tokenized_path: path/to/my-data/tokenized
decision_label_folder: starting-model            # leave out, with the next line, without the labels above
decision_teacher_kl_weight: 0.3
num_train_epochs: 3.0
learning_rate: 1.0e-4
decision_head_lr: 1.0e-4                         # the head is already trained, so no larger step than the adapter
lr_scheduler_type: constant_with_warmup
eval_strategy: "no"                              # "epoch" with val_size above; "no" without a validation split
save_strategy: epoch                             # one checkpoint per epoch, to choose from
```

The list below explains how these settings differ from Part 1's.

- **The starting model** replaces the converted base model, in both configs. The data must be tokenized with its tokenizer.
- **The head's learning rate** equals the adapter's. In Part 1 the head starts from random weights and needs a step 10 times larger; here it is already trained.
- **The KL term** keeps the new model's answers on your rows close to the starting model's, so that training changes the model more gradually. It replaces Part 1's anchor toward the base model. It does not preserve the model's results on other kinds of questions; for that, add rows of those questions to the training data. A weight of 0 (or leaving both settings out) trains on the correct answers alone.
- **Several epochs, a constant learning rate, and a checkpoint per epoch**, so that each checkpoint is the model after that many epochs and the best one can be chosen afterwards. A constant learning rate keeps the checkpoints comparable, which a learning rate that decays to 0 at the end would not.

For the first or second way to start training, add `adapter_name_or_path` (and `create_new_adapter: true` for the second) to the training config, and `decision_teacher_adapter` to the tokenization config.

### Choosing the Checkpoint, Calibrating, and Evaluating

Score every checkpoint on the dev set at temperature 1, and keep the one with the best results:

```bash
M=path/to/strands-v19-merged
for C in path/to/output_dir/checkpoint-*; do
  hmf decision-eval evaluate $M --adapter_name_or_path $C --data path/to/my-data/dev.jsonl \
      --split all --limit 0 --raw --out reports/dev/$(basename $C).json
done
```

Then calibrate the chosen checkpoint on its own held-out set, and measure it on the test set, as in [step 6](#6-calibrate-and-evaluate):

```bash
C=path/to/output_dir/checkpoint-NNN   # the chosen checkpoint
hmf decision-eval calibrate $M --adapter_name_or_path $C --data path/to/my-data/calib.jsonl --split all --limit 0
hmf decision-eval evaluate  $M --adapter_name_or_path $C --data path/to/my-data/test.jsonl  --split all --limit 0
```

`--split all --limit 0` uses every row of each file, since these files are already separate sets. The fine-tuned model is then asked questions as in [step 8](#8-ask-the-model-questions). To compare with the starting model, run the same commands without `--adapter_name_or_path`. Calibrating it replaces the calibration it came with.

## More Details

- [recipes.md](recipes.md) describes the recipes of Part 1, and the reasons behind their settings.
- [data/README.md](../../data/README.md) describes the data format, tokenizing, and labeling.
- [src/hmf/eval/decision/README.md](../../src/hmf/eval/decision/README.md) describes calibration, the evaluation sets, and JevBench.
