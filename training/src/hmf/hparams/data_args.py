# Copyright 2025 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/examples/pytorch/language-modeling/run_clm.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


@dataclass
class DataArguments:
    r"""Arguments pertaining to what data we are going to input our model for training and evaluation."""

    template: str | None = field(
        default=None,
        metadata={"help": "Which template to use for constructing prompts in training and inference."},
    )
    dataset: str | None = field(
        default=None,
        metadata={"help": "The name of dataset(s) to use for training. Use commas to separate multiple datasets."},
    )
    eval_dataset: str | None = field(
        default=None,
        metadata={"help": "The name of dataset(s) to use for evaluation. Use commas to separate multiple datasets."},
    )
    dataset_dir: str = field(
        default="data",
        metadata={"help": "Path to the folder containing the datasets."},
    )
    media_dir: str | None = field(
        default=None,
        metadata={"help": "Path to the folder containing the images, videos or audios. Defaults to `dataset_dir`."},
    )
    cutoff_len: int = field(
        default=2048,
        metadata={"help": "The cutoff length of the tokenized inputs in the dataset."},
    )
    filter: str | None = field(
        default=None,
        metadata={"help": "Filter spec to apply after alignment, e.g., 'repetition_ngram10_repeat3_window1024'"},
    )
    train_on_prompt: bool = field(
        default=False,
        metadata={"help": "Whether or not to disable the mask on the prompt."},
    )
    mask_history: bool = field(
        default=False,
        metadata={"help": "Whether or not to mask the history and train on the last turn only."},
    )
    streaming: bool = field(
        default=False,
        metadata={"help": "Enable dataset streaming."},
    )
    buffer_size: int = field(
        default=16384,
        metadata={"help": "Size of the buffer to randomly sample examples from in dataset streaming."},
    )
    mix_strategy: Literal["concat", "interleave_under", "interleave_over", "interleave_once"] = field(
        default="concat",
        metadata={
            "help": "Strategy to use in dataset mixing (concat/interleave) (undersampling/oversampling/sampling w.o. replacement)."
        },
    )
    interleave_probs: str | None = field(
        default=None,
        metadata={"help": "Probabilities to sample data from datasets. Use commas to separate multiple datasets."},
    )
    max_merged_samples: int | None = field(
        default=None,
        metadata={"help": "Maximum number of samples after merging datasets. Only used with mix_strategy='interleave_fast'."}
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets."},
    )
    preprocessing_batch_size: int = field(
        default=1000,
        metadata={"help": "The number of examples in one group in pre-processing."},
    )
    preprocessing_num_workers: int | None = field(
        default=None,
        metadata={"help": "The number of processes to use for the pre-processing."},
    )
    max_samples: int | None = field(
        default=None,
        metadata={"help": "For debugging purposes, truncate the number of examples for each dataset."},
    )
    eval_num_beams: int | None = field(
        default=None,
        metadata={"help": "Number of beams to use for evaluation. This argument will be passed to `model.generate`"},
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={"help": "Whether or not to ignore the tokens corresponding to the pad label in loss computation."},
    )
    val_size: float = field(
        default=0.0,
        metadata={"help": "Size of the validation set, should be an integer or a float in range `[0,1)`."},
    )
    eval_on_each_dataset: bool = field(
        default=False,
        metadata={"help": "Whether or not to evaluate on each dataset separately."},
    )
    packing: bool | None = field(
        default=None,
        metadata={"help": "Enable sequences packing in training. Will automatically enable in pre-training."},
    )
    add_eos_token: bool | None = field(
        default=True,
        metadata={
            "help": (
                "Append EOS token to pretraining data. "
                "Set to False if raw data contains EOS token delimiters. "
                "Defaults to True."
            )
        },
    )
    neat_packing: bool = field(
        default=False,
        metadata={"help": "Enable sequence packing without cross-attention."},
    )
    packing_method: Literal["greedy", "random"] = field(
        default="greedy",
        metadata={"help": "Strategy for packing"},
    )
    shuffle_for_sequence_parallel: bool = field(
        default=True,
        metadata={
            "help": "Shuffle dataset before sequence parallel preprocessing (should shuffle before pad & split)."
        },
    )
    tool_format: str | None = field(
        default=None,
        metadata={"help": "Tool format to use for constructing function calling examples."},
    )
    default_system: str | None = field(
        default=None,
        metadata={"help": "Override the default system message in the template."},
    )
    enable_thinking: bool | None = field(
        default=True,
        metadata={"help": "Whether or not to enable thinking mode for reasoning models."},
    )
    tokenized_path: str | None = field(
        default=None,
        metadata={
            "help": (
                "Path to save or load the tokenized datasets. "
                "If tokenized_path not exists, it will save the tokenized datasets. "
                "If tokenized_path exists, it will load the tokenized datasets."
            )
        },
    )
    data_shared_file_system: bool = field(
        default=False,
        metadata={"help": "Whether or not to use a shared file system for the datasets."},
    )
    decision_prompt_format: str = field(
        default="strands",
        metadata={"help": "Prompt format used to render decision rows (stage `decision`)."},
    )
    decision_num_augmentations: int = field(
        default=1,
        metadata={
            "help": (
                "Number of augmentations of each decision training row. An augmentation is one augmented copy of "
                "a row, rendered with its options in one order (shuffled for noul and choice, possibly "
                "reversed for score) and one phrasing of its question. Epoch e trains on augmentation e (modulo "
                "the number of augmentations)."
            )
        },
    )
    decision_shuffle_options: bool = field(
        default=True,
        metadata={"help": "Whether or not to shuffle the options of noul and choice training rows."},
    )
    decision_reverse_score_prob: float = field(
        default=0.5,
        metadata={"help": "Probability of reversing the rubric (and the label) of a score training row."},
    )
    decision_vary_instructions: bool = field(
        default=True,
        metadata={"help": "Whether or not to sample among the phrasings of a decision training row's question."},
    )
    decision_ordinal_smoothing: float = field(
        default=0.1,
        metadata={"help": "Probability mass moved to the neighboring levels in the targets of score rows."},
    )
    decision_force_bos: bool = field(
        default=False,
        metadata={
            "help": "Whether or not to start decision prompts with the BOS token if the tokenizer does not add it."
        },
    )
    decision_label_folder: str | None = field(
        default=None,
        metadata={
            "help": (
                "Folder of a label set of tokenized decision data, as a folder name rather than a path. The label "
                "set is stored in <tokenized_path>/labels/<decision_label_folder>/, next to the tokens it was made "
                "for. In a tokenization config, labeling (hmf preprocess or hmf decision-label) writes it; in a "
                "training config with stage `decision`, training reads it and joins it to the training rows."
            )
        },
    )
    decision_anchor_model: str | None = field(
        default=None,
        metadata={
            "help": (
                "Causal LM whose own answers are the anchor targets (hmf decision-label), given by its probabilities "
                "for the options' labels (1, 2, ...) after the answer marker. Must use the tokenizer the data was "
                "tokenized with."
            )
        },
    )
    decision_teacher_model: str | None = field(
        default=None,
        metadata={
            "help": (
                "Model whose answers are the teacher targets (hmf decision-label), either a causal LM, which reads "
                "each row through decision_teacher_prompt, or a decision model (an hmf_decision checkpoint), which "
                "reads it as in training. Exclusive with decision_teacher_file."
            )
        },
    )
    decision_teacher_adapter: str | None = field(
        default=None,
        metadata={"help": "LoRA adapter of a decision teacher model, merged before labeling."},
    )
    decision_teacher_prompt: str = field(
        default="decision",
        metadata={
            "help": (
                "How a causal LM teacher model is asked each row, either `decision` (the decision prompt the student "
                "trains on) or `strands` (Strands Decider's chat prompt, for instruction-tuned models)."
            )
        },
    )
    decision_teacher_max_len: int | None = field(
        default=None,
        metadata={
            "help": "Rows whose teacher prompt is longer than this many tokens get no teacher. Defaults to no limit."
        },
    )
    decision_teacher_file: str | None = field(
        default=None,
        metadata={
            "help": (
                "Teacher file whose probabilities are the teacher targets (hmf decision-label), in JSON Lines of "
                '{"source": ..., "row_index": ..., "probs": [...]}, one line per row; see data/README.md.'
            )
        },
    )
    decision_teacher_datasets: str | None = field(
        default=None,
        metadata={
            "help": (
                "Datasets whose rows get teacher targets (hmf decision-label). Use commas to separate multiple "
                "datasets. Defaults to all datasets."
            )
        },
    )

    def __post_init__(self):
        def split_arg(arg):
            if isinstance(arg, str):
                return [item.strip() for item in arg.split(",")]
            return arg

        self.dataset = split_arg(self.dataset)
        self.eval_dataset = split_arg(self.eval_dataset)
        self.decision_teacher_datasets = split_arg(self.decision_teacher_datasets)

        if self.media_dir is None:
            self.media_dir = self.dataset_dir

        if self.dataset is None and self.val_size > 1e-6:
            raise ValueError("Cannot specify `val_size` if `dataset` is None.")

        if self.eval_dataset is not None and self.val_size > 1e-6:
            raise ValueError("Cannot specify `val_size` if `eval_dataset` is not None.")

        if self.interleave_probs is not None:
            if self.mix_strategy == "concat":
                raise ValueError("`interleave_probs` is only valid for interleaved mixing.")

            self.interleave_probs = list(map(float, split_arg(self.interleave_probs)))
            if self.dataset is not None and len(self.dataset) != len(self.interleave_probs):
                raise ValueError("The length of dataset and interleave probs should be identical.")

            if self.eval_dataset is not None and len(self.eval_dataset) != len(self.interleave_probs):
                raise ValueError("The length of eval dataset and interleave probs should be identical.")

        if self.streaming and self.val_size > 1e-6 and self.val_size < 1:
            raise ValueError("Streaming mode should have an integer val size.")

        if self.streaming and self.max_samples is not None:
            raise ValueError("`max_samples` is incompatible with `streaming`.")

        if self.mask_history and self.train_on_prompt:
            raise ValueError("`mask_history` is incompatible with `train_on_prompt`.")

        if self.neat_packing:
            self.packing = True

        if self.decision_num_augmentations < 1:
            raise ValueError("`decision_num_augmentations` must be at least 1.")

        if not 0.0 <= self.decision_reverse_score_prob <= 1.0:
            raise ValueError("`decision_reverse_score_prob` must be in [0, 1].")

        if not 0.0 <= self.decision_ordinal_smoothing < 1.0:
            raise ValueError("`decision_ordinal_smoothing` must be in [0, 1).")


    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
