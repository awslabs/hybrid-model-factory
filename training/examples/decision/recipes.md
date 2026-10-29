# Decision Model Recipes

This document describes the recipes behind the decision models of Part 1 of [README.md](README.md),
which settings they share, how they differ, and why. Part 1 of README.md runs the v19 recipe step
by step.

Both recipes train Qwen3.5-2B-Base on the Strands Decider v19 data. The cross-entropy baseline
learns from the correct answers only. The v19 recipe is Strands Decider's, and also learns from the
base model's and a teacher's answers. Its published results give a reference to compare against
(see [What to Expect](README.md#what-to-expect)).

The v19 recipe is also given for GKA-primed-HQwen3-8B-Instruct, an HMF hybrid, with the same
settings except for those its size and architecture need. [README.md](README.md#example-gka-primed-hqwen3-8b-instruct)
lists its configs and explains what changes.

## What Both Share

**Student.** The student is Qwen3.5-2B-Base, converted with `hmf decision-init`. The conversion
removes the language model's head and puts a pointer head (256-dimensional, randomly initialized)
in its place. Training is done with LoRA, with adapters (rank 16, alpha 32, dropout 0.05) on the
attention, MLP, and Gated DeltaNet projections, and it trains all of the pointer head's weights.
Every training run starts from this converted model, with new adapters.

**Training.** Each run trains for one epoch with a global batch of 32, in bf16, with AdamW (beta2
0.95). The learning rate is 1e-4 for the adapters and 1e-3 for the head, with a cosine schedule and
3% warmup, and gradients are clipped at 1.0. Weight decay of 0.01 applies to the head's matrices
only. Batches group rows of similar length, and are padded to a multiple of 256 tokens.

**Data preparation.** Each row is rendered in the strands prompt format. Training rows are
augmented, which means that their options are shuffled (`noul`, `choice`) or their rubric is
reversed (`score`), and that one phrasing of the question is sampled. 3% of the rows of each task
are held out for validation.

**Loss terms.** Both recipes use cross-entropy on the correct option, with ordinal smoothing 0.1 for
`score` rows. Two optional terms can be added, each a KL divergence over the options.

- **Anchor**, KL(base model ‖ student). The target is the frozen Qwen3.5-2B-Base's own answer,
  given by its next-token probabilities for the option numbers "1", "2", ... after `<answer>`. It
  keeps the student close to where it started. Only rows with at most 9 options have an anchor.
- **Teacher**, KL(teacher ‖ student). The target is another model's probabilities over the
  options. Only some rows have a teacher.

## The Recipes

| | Cross-entropy baseline | v19 recipe |
| --- | --- | --- |
| Results | validation only | matches v19 (JevBench 168, see [README.md](README.md#what-to-expect)) |
| Training runs | 1 | 2 (model 1, then model 2) |
| Data | the six v19 files | model 1: `train_v5` + `multistep_v14`; model 2: the six v19 files |
| Anchor (weight 0.3) | none | Qwen3.5-2B-Base, for both models |
| Teacher (weight 1.0) | none | model 1: Qwen3.5-4B; model 2: model 1's probabilities |
| Rows with a teacher | none | the 12,909 multi-step rows |

### Cross-Entropy Baseline

The baseline trains with cross-entropy only, on the six v19 files. After one epoch, it reaches
0.857 accuracy and 0.017 expected calibration error (ECE) on the rows held out for validation (3% of each task, see "Data
preparation" above).

### The v19 Recipe

This is Strands Decider's v19 recipe, with every label generated in HMF. Model 1 (which Strands Decider
calls the parent, because it follows its v14 recipe) is trained first, and is used only to provide
teacher labels for model 2.

1. **Teacher for model 1.** Qwen3.5-4B, the instruct model, answers each multi-step question
   without any training. It reads the question through its chat template, with the options shown
   as letters and thinking off, and its probabilities for the letters are the targets.
2. **Model 1.** Train on `train_v5` + `multistep_v14` with cross-entropy + 0.3 × anchor + 1.0 ×
   the Qwen3.5-4B teacher, at a 3,072-token window.
3. **Replay.** Model 1 answers the multi-step questions, and its probabilities are the targets
   for model 2.
4. **Model 2.** A new student, which starts again from the base model rather than from model 1,
   trains on the six v19 files with cross-entropy + 0.3 × anchor + 1.0 × model 1's probabilities.

Strands Decider reports that adding the new files without the replay made the model worse on the
multi-step questions, and the replay is meant to prevent that. We have not trained this recipe
without the replay.

## The Data

| file | rows | contents | Baseline, and the v19 recipe's model 2 | The v19 recipe's model 1 |
| --- | --- | --- | --- | --- |
| `train_v5` | about 100,000 | classification from about 21 public datasets, covering topic, intent, sentiment, entailment, rating scales, yes/no, and rule chaining | yes | yes |
| `multistep_v14` | 12,909 | multi-step documents from ContractNLI, MuSiQue, and BoardgameQA | yes | yes |
| `generated_v16` + `generated_v18` | 3,815 | generated workplace documents with skill-specific questions | yes | |
| `adequacy_hs2` + `adequacy_gen` | 6,166 | whether a response adequately answers a request (HelpSteer2, and generated) | yes | |

## Running Them

| step | Cross-entropy baseline | v19 recipe |
| --- | --- | --- |
| Tokenize and label | `hmf preprocess` with `tokenization/qwen3_5_2b/strands_v19.yaml` | model 1: `hmf preprocess` with `tokenization/qwen3_5_2b/strands_v14.yaml` |
| Train | `hmf train train/qwen3_5_2b/lora_ce.yaml` | model 1: `hmf train train/qwen3_5_2b/lora_v14.yaml` |
| Replay | | `hmf preprocess` with `tokenization/qwen3_5_2b/strands_v19_replay.yaml` (labels the six-file data with model 1) |
| Train model 2 | | `hmf train train/qwen3_5_2b/lora_v19.yaml` |

Part 1 of [README.md](README.md) runs these commands step by step.

## Evaluation

Both recipes are evaluated on the same sets as Strands Decider's v19, after one temperature per
question kind is fitted on `holdout_v5_norule` to minimize the expected calibration error (ECE).

- **JevBench public** (231 tasks) is the main measure for ranking models. v19 scores 167 of 231
  (0.723).
- `holdout_v5_norule` holds short tasks never trained on.
- `multistep_v14_eval` holds MuSiQue, HotpotQA (never trained on), and the other multi-step sets.
- `generated_v16_eval` and `generated_v18_eval` hold document domains never trained on.
- `adequacy_hs2_eval` and `adequacy_gen_eval` measure answer adequacy.

Strands Decider considers a retrain to match v19 when it scores 157 to 177 on JevBench public, is
not significantly different from v19 task by task (McNemar test, p ≥ 0.05), and reaches at least
0.689 on HelpSteer2 adequacy. Differences of fewer than about 10 JevBench tasks are within
retraining noise.
