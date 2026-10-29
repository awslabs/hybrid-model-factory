The [dataset_info.json](dataset_info.json) contains all available datasets. If you are using a custom dataset, please **make sure** to add a *dataset description* in `dataset_info.json` and specify `dataset: dataset_name` before training to use it.

The `dataset_info.json` file should be put in the `dataset_dir` directory. You can change `dataset_dir` to use another directory. The default value is `./data`.

Currently we support datasets in **alpaca** and **sharegpt** format, and in **decision** format for decision models. Allowed file types include json, jsonl, csv, parquet, arrow.

> **Note:** Example dataset links below point to the [LlamaFactory](https://github.com/hiyouga/LLaMA-Factory) repository, from which Hybrid Model Factory's training code is derived.

```json
"dataset_name": {
  "hf_hub_url": "the name of the dataset repository on the Hugging Face hub. (if specified, ignore script_url, file_name and cloud_file_name)",
  "ms_hub_url": "the name of the dataset repository on the Model Scope hub. (if specified, ignore script_url, file_name and cloud_file_name)",
  "script_url": "the name of the directory containing a dataset loading script. (if specified, ignore file_name and cloud_file_name)",
  "cloud_file_name": "the name of the dataset file in s3/gcs cloud storage. (if specified, ignore file_name)",
  "file_name": "the name of the dataset folder or dataset file in this directory. (required if above are not specified)",
  "formatting": "the format of the dataset. (optional, default: alpaca, can be chosen from {alpaca, sharegpt, decision})",
  "ranking": "whether the dataset is a preference dataset or not. (default: False)",
  "subset": "the name of the subset. (optional, default: None)",
  "split": "the name of dataset split to be used. (optional, default: train)",
  "folder": "the name of the folder of the dataset repository on the Hugging Face hub. (optional, default: None)",
  "num_samples": "the number of samples in the dataset to be used. (optional, default: None)",
  "columns (optional)": {
    "prompt": "the column name in the dataset containing the prompts. (default: instruction)",
    "query": "the column name in the dataset containing the queries. (default: input)",
    "response": "the column name in the dataset containing the responses. (default: output)",
    "history": "the column name in the dataset containing the histories. (default: None)",
    "messages": "the column name in the dataset containing the messages. (default: conversations)",
    "system": "the column name in the dataset containing the system prompts. (default: None)",
    "tools": "the column name in the dataset containing the tool description. (default: None)",
    "images": "the column name in the dataset containing the image inputs. (default: None)",
    "videos": "the column name in the dataset containing the videos inputs. (default: None)",
    "audios": "the column name in the dataset containing the audios inputs. (default: None)",
    "chosen": "the column name in the dataset containing the chosen answers. (default: None)",
    "rejected": "the column name in the dataset containing the rejected answers. (default: None)",
    "kto_tag": "the column name in the dataset containing the kto tags. (default: None)"
  },
  "tags (optional, used for the sharegpt format)": {
    "role_tag": "the key in the message represents the identity. (default: from)",
    "content_tag": "the key in the message represents the content. (default: value)",
    "user_tag": "the value of the role_tag represents the user. (default: human)",
    "assistant_tag": "the value of the role_tag represents the assistant. (default: gpt)",
    "observation_tag": "the value of the role_tag represents the tool results. (default: observation)",
    "function_tag": "the value of the role_tag represents the function call. (default: function_call)",
    "system_tag": "the value of the role_tag represents the system prompt. (default: system, can override system column)"
  }
}
```

## Alpaca Format

### Supervised Fine-Tuning Dataset

* [Example dataset](https://github.com/hiyouga/LLaMA-Factory/blob/762b480131908d37736ad9aa3f12e87f8f7e6313/data/alpaca_en_demo.json)

In supervised fine-tuning, the `instruction` column will be concatenated with the `input` column and used as the user prompt, then the user prompt would be `instruction\ninput`. The `output` column represents the model response.

For reasoning models, if the dataset contains chain-of-thought (CoT), the CoT needs to be placed in the model responses, such as `<think>cot</think>output`.

The `system` column will be used as the system prompt if specified.

The `history` column is a list consisting of string tuples representing prompt-response pairs in the history messages. Note that the responses in the history **will also be learned by the model** in supervised fine-tuning.

```json
[
  {
    "instruction": "user instruction (required)",
    "input": "user input (optional)",
    "output": "model response (required)",
    "system": "system prompt (optional)",
    "history": [
      ["user instruction in the first round (optional)", "model response in the first round (optional)"],
      ["user instruction in the second round (optional)", "model response in the second round (optional)"]
    ]
  }
]
```

Regarding the above dataset, the *dataset description* in `dataset_info.json` should be:

```json
"dataset_name": {
  "file_name": "data.json",
  "columns": {
    "prompt": "instruction",
    "query": "input",
    "response": "output",
    "system": "system",
    "history": "history"
  }
}
```

> [!TIP]  
> If the model has reasoning capabilities (e.g. Qwen3) but the dataset does not contain chain-of-thought (CoT), LLaMA-Factory will automatically add empty CoT to the data. When `enable_thinking` is `True` (slow thinking, by default), the empty CoT will be added to the model responses and loss computation will be considered; otherwise (fast thinking), it will be added to the user prompts and loss computation will be ignored. Please keep the `enable_thinking` parameter consistent during training and inference.
>
> If you want to train data containing CoT with slow thinking and data without CoT with fast thinking, you can set `enable_thinking` to `None`. However, this feature is relatively complicated and should be used with caution.

### Pre-training Dataset

- [Example dataset](https://github.com/hiyouga/LLaMA-Factory/blob/762b480131908d37736ad9aa3f12e87f8f7e6313/data/c4_demo.jsonl)

In pre-training, only the `text` column will be used for model learning.

```json
[
  {"text": "document"},
  {"text": "document"}
]
```

Regarding the above dataset, the *dataset description* in `dataset_info.json` should be:

```json
"dataset_name": {
  "file_name": "data.json",
  "columns": {
    "prompt": "text"
  }
}
```

### Preference Dataset

> **Note:** Preference-based training regimes (DPO, ORPO, SimPO, reward modeling) are inherited from LlamaFactory and have not been tested with Hybrid Model Factory. See [Training.md](../../docs/Training.md) for details.

Preference datasets are used for reward modeling, DPO training, ORPO and SimPO training.

It requires a better response in `chosen` column and a worse response in `rejected` column.

```json
[
  {
    "instruction": "user instruction (required)",
    "input": "user input (optional)",
    "chosen": "chosen answer (required)",
    "rejected": "rejected answer (required)"
  }
]
```

Regarding the above dataset, the *dataset description* in `dataset_info.json` should be:

```json
"dataset_name": {
  "file_name": "data.json",
  "ranking": true,
  "columns": {
    "prompt": "instruction",
    "query": "input",
    "chosen": "chosen",
    "rejected": "rejected"
  }
}
```

### KTO Dataset

> **Note:** KTO training is inherited from LlamaFactory and has not been tested with Hybrid Model Factory. See [Training.md](../../docs/Training.md) for details.

An additional column `kto_tag` is required. Please refer to the [sharegpt](#sharegpt-format) format for details.

### Multimodal Image Dataset

An additional column `images` is required. Please refer to the [sharegpt](#sharegpt-format) format for details.

### Multimodal Video Dataset

An additional column `videos` is required. Please refer to the [sharegpt](#sharegpt-format) format for details.

### Multimodal Audio Dataset

An additional column `audios` is required. Please refer to the [sharegpt](#sharegpt-format) format for details.

## Sharegpt Format

### Supervised Fine-Tuning Dataset

- [Example dataset](https://github.com/hiyouga/LLaMA-Factory/blob/762b480131908d37736ad9aa3f12e87f8f7e6313/data/glaive_toolcall_en_demo.json)

Compared to the alpaca format, the sharegpt format allows the datasets have **more roles**, such as human, gpt, observation and function. They are presented in a list of objects in the `conversations` column.

Note that the human and observation should appear in odd positions, while gpt and function should appear in even positions. The gpt and function will be learned by the model.

```json
[
  {
    "conversations": [
      {
        "from": "human",
        "value": "user instruction"
      },
      {
        "from": "function_call",
        "value": "tool arguments"
      },
      {
        "from": "observation",
        "value": "tool result"
      },
      {
        "from": "gpt",
        "value": "model response"
      }
    ],
    "system": "system prompt (optional)",
    "tools": "tool description (optional)"
  }
]
```

Regarding the above dataset, the *dataset description* in `dataset_info.json` should be:

```json
"dataset_name": {
  "file_name": "data.json",
  "formatting": "sharegpt",
  "columns": {
    "messages": "conversations",
    "system": "system",
    "tools": "tools"
  }
}
```

### Pre-training Dataset

Not yet supported, please use the [alpaca](#alpaca-format) format.

### Preference Dataset

> **Note:** Preference-based training regimes (DPO, ORPO, SimPO, reward modeling) are inherited from LlamaFactory and have not been tested with Hybrid Model Factory. See [Training.md](../../docs/Training.md) for details.

- [Example dataset](https://github.com/hiyouga/LLaMA-Factory/blob/762b480131908d37736ad9aa3f12e87f8f7e6313/data/dpo_en_demo.json)

Preference datasets in sharegpt format also require a better message in `chosen` column and a worse message in `rejected` column.

```json
[
  {
    "conversations": [
      {
        "from": "human",
        "value": "user instruction"
      },
      {
        "from": "gpt",
        "value": "model response"
      },
      {
        "from": "human",
        "value": "user instruction"
      }
    ],
    "chosen": {
      "from": "gpt",
      "value": "chosen answer (required)"
    },
    "rejected": {
      "from": "gpt",
      "value": "rejected answer (required)"
    }
  }
]
```

Regarding the above dataset, the *dataset description* in `dataset_info.json` should be:

```json
"dataset_name": {
  "file_name": "data.json",
  "formatting": "sharegpt",
  "ranking": true,
  "columns": {
    "messages": "conversations",
    "chosen": "chosen",
    "rejected": "rejected"
  }
}
```

### KTO Dataset

> **Note:** KTO training is inherited from LlamaFactory and has not been tested with Hybrid Model Factory. See [Training.md](../../docs/Training.md) for details.

- [Example dataset](https://github.com/hiyouga/LLaMA-Factory/blob/762b480131908d37736ad9aa3f12e87f8f7e6313/data/kto_en_demo.json)

KTO datasets require a extra `kto_tag` column containing the boolean human feedback.

```json
[
  {
    "conversations": [
      {
        "from": "human",
        "value": "user instruction"
      },
      {
        "from": "gpt",
        "value": "model response"
      }
    ],
    "kto_tag": "human feedback [true/false] (required)"
  }
]
```

Regarding the above dataset, the *dataset description* in `dataset_info.json` should be:

```json
"dataset_name": {
  "file_name": "data.json",
  "formatting": "sharegpt",
  "columns": {
    "messages": "conversations",
    "kto_tag": "kto_tag"
  }
}
```

### Multimodal Image Dataset

- [Example dataset](https://github.com/hiyouga/LLaMA-Factory/blob/762b480131908d37736ad9aa3f12e87f8f7e6313/data/mllm_demo.json)

Multimodal image datasets require an `images` column containing the paths to the input images.

The number of images should be identical to the `<image>` tokens in the conversations.

```json
[
  {
    "conversations": [
      {
        "from": "human",
        "value": "<image>user instruction"
      },
      {
        "from": "gpt",
        "value": "model response"
      }
    ],
    "images": [
      "image path (required)"
    ]
  }
]
```

Regarding the above dataset, the *dataset description* in `dataset_info.json` should be:

```json
"dataset_name": {
  "file_name": "data.json",
  "formatting": "sharegpt",
  "columns": {
    "messages": "conversations",
    "images": "images"
  }
}
```

### Multimodal Video Dataset

- [Example dataset](https://github.com/hiyouga/LLaMA-Factory/blob/762b480131908d37736ad9aa3f12e87f8f7e6313/data/mllm_video_demo.json)

Multimodal video datasets require a `videos` column containing the paths to the input videos.

The number of videos should be identical to the `<video>` tokens in the conversations.

```json
[
  {
    "conversations": [
      {
        "from": "human",
        "value": "<video>user instruction"
      },
      {
        "from": "gpt",
        "value": "model response"
      }
    ],
    "videos": [
      "video path (required)"
    ]
  }
]
```

Regarding the above dataset, the *dataset description* in `dataset_info.json` should be:

```json
"dataset_name": {
  "file_name": "data.json",
  "formatting": "sharegpt",
  "columns": {
    "messages": "conversations",
    "videos": "videos"
  }
}
```

### Multimodal Audio Dataset

- [Example dataset](https://github.com/hiyouga/LLaMA-Factory/blob/762b480131908d37736ad9aa3f12e87f8f7e6313/data/mllm_audio_demo.json)

Multimodal audio datasets require an `audios` column containing the paths to the input audios.

The number of audios should be identical to the `<audio>` tokens in the conversations.

```json
[
  {
    "conversations": [
      {
        "from": "human",
        "value": "<audio>user instruction"
      },
      {
        "from": "gpt",
        "value": "model response"
      }
    ],
    "audios": [
      "audio path (required)"
    ]
  }
]
```

Regarding the above dataset, the *dataset description* in `dataset_info.json` should be:

```json
"dataset_name": {
  "file_name": "data.json",
  "formatting": "sharegpt",
  "columns": {
    "messages": "conversations",
    "audios": "audios"
  }
}
```

### OpenAI Format

The openai format is simply a special case of the sharegpt format, where the first message may be a system prompt.

```json
[
  {
    "messages": [
      {
        "role": "system",
        "content": "system prompt (optional)"
      },
      {
        "role": "user",
        "content": "user instruction"
      },
      {
        "role": "assistant",
        "content": "model response"
      }
    ]
  }
]
```

Regarding the above dataset, the *dataset description* in `dataset_info.json` should be:

```json
"dataset_name": {
  "file_name": "data.json",
  "formatting": "sharegpt",
  "columns": {
    "messages": "messages"
  },
  "tags": {
    "role_tag": "role",
    "content_tag": "content",
    "user_tag": "user",
    "assistant_tag": "assistant",
    "system_tag": "system"
  }
}
```

## Decision Format

Decision models (`stage: decision`) read a prompt that lists options and return a probability for each one. Each row is one question about a state, with its options and the correct answer:

```json
[
  {
    "kind": "choice",
    "state": "Order 1182 was charged twice. The customer asked for the duplicate charge to be removed.",
    "instructions": "What should support do next?",
    "options": [["refund", "return the duplicate payment"], ["ignore", null], ["escalate", "send the case to a human agent"]],
    "label": 0,
    "task": "support_routing",
    "weight": 1.0,
    "instruction_variants": ["How should support respond to this ticket?"]
  }
]
```

| field | description |
| --- | --- |
| `kind` | `noul` (yes/no), `choice` (one of N options), or `score` (an ordered rubric). |
| `state` | What is being judged: text, or a JSON object or list (rendered as indented JSON). |
| `instructions` | The question. |
| `options` | `[name, description]` pairs; a description may be `null` or a JSON object or list. For `noul`, the names are `false` and `true` (either may be left out, to use a default description). For `score`, the options are the rubric levels, lowest first; they are shown as levels `0`, `1`, ... |
| `label` | The index of the correct option in `options`. For `noul`, 0 is `false` and 1 is `true`. |
| `task` | Optional, default `unknown`. The validation split holds out the same fraction of each task. |
| `weight` | Optional, default 1.0. The row's weight in the loss. |
| `instruction_variants` | Optional, default none. Other phrasings of the question, sampled during training. |

Regarding the above dataset, the *dataset description* in `dataset_info.json` should be:

```json
"dataset_name": {
  "file_name": "data.jsonl",
  "formatting": "decision"
}
```

If the dataset uses other column names, map them under `columns`, e.g. `"columns": {"state": "context", "instructions": "question"}`. Unlike the other formats, columns that are not listed keep their default names. Decision datasets must be local JSON or JSONL files (or folders of them), and cannot be streamed.

Tokenize decision data with `hmf preprocess` and `stage: decision` (see [examples/decision/tokenization/qwen3_5_2b/strands_v19.yaml](../examples/decision/tokenization/qwen3_5_2b/strands_v19.yaml)). No chat template is used: each row is rendered as plain text by the prompt format `decision_prompt_format` (default `strands`). Training rows are augmented, so the model reads the options from the prompt rather than from their position:

| argument | default | description |
| --- | --- | --- |
| `decision_prompt_format` | `strands` | The prompt format rows are rendered with. |
| `decision_num_augmentations` | 1 | Augmentations of each training row. An augmentation is one augmented copy of a row, rendered with its options in one order (shuffled for `noul` and `choice`, possibly reversed for `score`) and one phrasing of its question. Epoch e trains on augmentation e (modulo the number of augmentations). |
| `decision_shuffle_options` | true | Shuffle the options of `noul` and `choice` rows. |
| `decision_reverse_score_prob` | 0.5 | Probability of reversing a `score` row's rubric (and its label). `score` options are otherwise never shuffled. |
| `decision_vary_instructions` | true | Sample among a row's phrasings of the question. |
| `decision_ordinal_smoothing` | 0.1 | Target mass moved to the neighboring levels of `score` rows. |
| `decision_force_bos` | false | Start prompts with the BOS token if the tokenizer does not add it. |

With `val_size`, the validation split is held out per task and rendered once, without augmentation. Prompts longer than `cutoff_len` are dropped. `seed` sets both the validation split and the augmentations.

The Strands Decider training data is registered as `strands_*` datasets, read from `strands-decider/` in this directory, which should be a link to the `data/` folder of a [strands-decider](https://github.com/strands-labs/strands-decider) checkout in which the data has been built.

### Decision Labels

Besides the gold label, a decision model can learn from other models' distributions over a row's options. These extra targets are added to tokenized decision data by a labeling pass. When the tokenization config has labeling keys (`decision_label_folder`, and an anchor or a teacher), `hmf preprocess` labels the data right after tokenizing it, on all visible GPUs:

```bash
hmf preprocess examples/decision/tokenization/qwen3_5_2b/strands_v14.yaml   # tokens, then labels
```

The labeling keys are checked before tokenizing starts. Running the command again skips what is done. Tokens that exist are not tokenized again, and a complete label set is not labeled again (an interrupted one is). Without GPUs, labeling runs on the CPU, which is slow for large models.

Data tokenized earlier is labeled with `hmf decision-label`, from the same config:

```bash
hmf decision-label examples/decision/tokenization/qwen3_5_2b/strands_v14.yaml
```

Two kinds of target can be added, each a distribution over the row's options:

- **Anchor**, a frozen causal LM's own answer to the decision prompt (usually the student's base model), given by its next-token probabilities for the options' labels `1`, `2`, ... right after `<answer>`. It is computed for every training row whose labels are all single tokens (up to 9 options for most tokenizers). The model reads the tokenized prompts as they are, so it must use the tokenizer the data was tokenized with.
- **Teacher**, another model's distribution, for the rows of the datasets listed in `decision_teacher_datasets`. It comes from a teacher model or from a teacher file (both below).

Both are stored in each row's augmented option order, so they line up with the model's output. Only the training split is labeled.

| argument | description |
| --- | --- |
| `decision_label_folder` | The label set's folder, as a folder name rather than a path. The label set is stored in `<tokenized_path>/labels/<decision_label_folder>/`, next to its tokens. Labeling writes it, and training with `stage: decision` reads it (see below). |
| `decision_anchor_model` | Causal LM (with its LM head) whose answers are the anchor targets. Unset: no anchor. |
| `decision_teacher_model` | Model whose answers are the teacher targets: a causal LM or a decision model. Unset: no teacher model. |
| `decision_teacher_adapter` | LoRA adapter of a decision teacher model, merged before labeling. |
| `decision_teacher_prompt` | How a causal LM teacher is asked each row, either `decision` (default) or `strands`. |
| `decision_teacher_max_len` | Rows whose teacher prompt is longer than this many tokens get no teacher. Unset: no limit. |
| `decision_teacher_file` | Teacher file whose probabilities are the teacher targets. Unset: no teacher file. |
| `decision_teacher_datasets` | Datasets whose rows get teacher targets, separated by commas. Unset: all datasets. |

Set at most one teacher, either `decision_teacher_model` or `decision_teacher_file`. Any key can be overridden on the command line, e.g. to label the same tokens with another teacher under another name:

```bash
hmf decision-label examples/decision/tokenization/qwen3_5_2b/strands_v19_replay.yaml decision_label_folder=replay_other \
    decision_teacher_model=path/to/other-model decision_teacher_adapter=path/to/other-lora
```

The label set is stored next to the tokens it was made for, without copying them:

```
<tokenized_path>/
├── dataset_dict.json, train/, validation/     # tokens, from hmf preprocess
└── labels/<decision_label_folder>/
    ├── train/                                 # one row per training row, in the same order
    └── labels_info.json                       # the models and files used, and how many rows got each target
```

Each label row stores its training row's key (`source`, `row_index`, `augmentation`, `option_order`) and a fingerprint of its tokens, and the label columns `ref_probs`, `has_ref` (anchor) and `teacher_probs`, `has_teacher` (teacher). A tokenized dataset can have several label sets. To train with one, set `tokenized_path` and `decision_label_folder: <folder>` in the training config. The label set is then joined to the training rows after checking that every row's key matches and that a sample of rows has the same tokens, so labels made for other tokens (e.g. after re-tokenizing with another seed) are rejected. To relabel, delete `labels/<folder>/` or choose another folder. The folder's name is up to you; the models and settings used are recorded in its `labels_info.json`.

Training uses the targets through two weights, which default to 0 (cross-entropy only):

| argument | the term it weights |
| --- | --- |
| `decision_anchor_kl_weight` | KL(anchor ‖ model), averaged over the rows of each batch that have an anchor |
| `decision_teacher_kl_weight` | KL(teacher ‖ model), averaged over the rows of each batch that have a teacher |

The loss is the cross-entropy plus each weighted term. Row weights apply to the cross-entropy only, and a batch without rows that have a target adds nothing for that term. Training stops with an error if a weight is above 0 without `decision_label_folder`, or if the label set has no rows with that target. The training logs show `ce_loss`, `anchor_kl` and `teacher_kl` (unweighted) beside the loss. Validation rows have no label set, so `eval_loss` is the cross-entropy alone.

#### Teacher Models

A teacher model reads each original row once, from the dataset files: its options in original order (the order they are listed in the data file; for `noul` rows, `[false, true]`) and its question in its original phrasing. Its answer is then reordered like each augmentation of the row. What it reads depends on the model:

- **A causal LM** answers a teacher prompt with one of the options' labels, and its next-token probabilities for the labels are the teacher's distribution. Rows whose labels are not all single tokens get no teacher. `decision_teacher_prompt` selects the prompt:

  | prompt | what the teacher reads | labels | suits |
  | --- | --- | --- | --- |
  | `decision` | The decision prompt the student trains on (`decision_prompt_format`). | `1`, `2`, ... | any causal LM, including base models |
  | `strands` | Strands Decider's teacher prompt, through the model's chat template with thinking disabled: a system message, then the row as JSON (`evidence`, `criterion`, and lettered `options`). | `A` to `P` (at most 16 options) | instruction-tuned models |

- **A decision model** (a checkpoint of `hmf decision-init` or of training, with `model_type: hmf_decision`, detected automatically) reads the row as in training and gives its own distribution, at temperature 1. With `decision_teacher_adapter`, its LoRA adapter is merged first. Training toward an earlier model's answers keeps a new model close to it (replay).

On several GPUs, each GPU reads a slice of the original rows, so each row is read once.

#### Teacher Files

A teacher file holds the teacher's probabilities for some rows, computed elsewhere. It is a JSON Lines file with one line per row:

```json
{"source": "strands-decider/multistep_v14.jsonl", "row_index": 12, "probs": [0.82, 0.11, 0.07]}
```

| field | description |
| --- | --- |
| `source` | The dataset's `file_name` in `dataset_info.json` (for a dataset that is a folder, `<file_name>/<file>`). |
| `row_index` | The row's position among the non-empty lines of that file, counting from 0. |
| `probs` | One probability per option, summing to 1, in the row's original option order, which is the order its options are listed in the data file, except for `noul` rows, which are always `[false, true]`. |

Rows that are not in the file get no teacher. Labeling stops with an error if a line misses a field, if `probs` does not sum to 1, if a row appears twice, or if a row's `probs` does not have one probability per option.

Strands Decider's own teacher files (`{"i": ..., "probs": [...]}`, where `i` counts the rows of several training files in a row) can be converted with `scripts/strands_teacher_to_hmf.py`, given the HMF datasets of those files in order:

```bash
python scripts/strands_teacher_to_hmf.py --teacher data/strands-decider/synthetic/replay_v14_multistep.jsonl \
    --datasets strands_train_v5,strands_multistep_v14 --output ./data/decision/teachers/replay_v14_multistep.hmf.jsonl
```
