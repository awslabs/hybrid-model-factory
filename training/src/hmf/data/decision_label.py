"""
The labeling pass for decision data, which adds training targets beyond the correct label to a
tokenized decision dataset, without copying its tokens. It runs after tokenization, from the same
config. hmf preprocess labels the data after tokenizing it when the config has labeling keys
(decision_label_folder, and an anchor or a teacher). Data tokenized earlier is labeled with

    hmf decision-label config.yaml [key=value ...]

where key=value overrides the config, e.g. to label the same tokens with another teacher under
another name.

The labels of the training split are written next to the tokens, as a named label set:

    <tokenized_path>/
    ├── dataset_dict.json, train/, validation/     # tokens, from hmf preprocess
    └── labels/<decision_label_folder>/
        ├── train/                                 # one row per training row, in the same order
        └── labels_info.json                       # how the labels were made

Each label row has the training row's key (source, row_index, augmentation, option_order) and a
fingerprint of its tokens, and the label columns:

    - ref_probs, has_ref: the anchor, from decision_anchor_model, which reads each tokenized prompt
      (see CausalLMScorer).
    - teacher_probs, has_teacher: the teacher, for rows of the datasets in decision_teacher_datasets
      (all datasets if unset). It comes from decision_teacher_model (a causal LM or a decision
      model, see load_teacher_model) or from decision_teacher_file (see TeacherFile). A teacher
      model reads each original row once, from the dataset files.

Both are in the row's augmented option order, so they line up with the model's output. Training
reads a label set with decision_label_folder: <folder>, and load_decision_labels joins it to the tokens
after checking that every row matches. On several GPUs, each GPU labels the training rows of a
slice of the original rows.
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from datasets import Dataset, Features, Sequence as SequenceFeature, Value, concatenate_datasets, load_from_disk

from ..extras import logging
from .data_utils import decision_dataset_files, decision_sources, read_decision_rows
from .decision_prompt import get_prompt_format
from .decision_teachers import CausalLMScorer, ScoringQuery, TeacherFile, load_teacher_model, to_augmented_order
from .parser import get_dataset_list


logger = logging.get_logger(__name__)


LABELS_DIR = "labels"
KEY_COLUMNS = ("source", "row_index", "augmentation", "option_order")
LABEL_COLUMNS = ("ref_probs", "has_ref", "teacher_probs", "has_teacher")
# Rows whose token fingerprints are compared when joining labels to tokens
NUM_FINGERPRINTS_CHECKED = 2000
# The columns of a labeled slice. Explicit, so that a slice without any anchor or teacher (all [])
# has the same types as the others
SHARD_FEATURES = Features(
    {
        "index": Value("int64"),
        "fingerprint": Value("string"),
        "ref_probs": SequenceFeature(Value("float64")),
        "has_ref": Value("bool"),
        "teacher_probs": SequenceFeature(Value("float64")),
        "has_teacher": Value("bool"),
    }
)


def labels_path(tokenized_path: str, name: str) -> str:
    """The folder of the label set called name."""
    return os.path.join(tokenized_path, LABELS_DIR, name)


def tokens_fingerprint(input_ids: Sequence[int]) -> str:
    """A short hash of a row's tokens, to detect labels made for other tokens."""
    return hashlib.sha1(np.asarray(input_ids, dtype=np.int64).tobytes()).hexdigest()[:16]


def _load_args(config_path: str, overrides: Sequence[str] = ()):
    """
    The arguments of a tokenization config, with key=value overrides. Distributed training settings
    are dropped.
    """
    from omegaconf import OmegaConf

    from ..hparams.parser import _parse_train_args

    config = OmegaConf.load(os.path.abspath(config_path))
    config = OmegaConf.to_container(OmegaConf.merge(config, OmegaConf.from_dotlist(list(overrides))))
    for key in ("deepspeed", "fsdp", "fsdp_config", "ddp_timeout", "ddp_find_unused_parameters", "local_rank"):
        config.pop(key, None)
    model_args, data_args, training_args, _, _ = _parse_train_args(config)
    return model_args, data_args, training_args


def _check_args(data_args) -> None:
    if data_args.tokenized_path is None:
        raise ValueError("Labeling needs tokenized_path: tokenize the data with hmf preprocess first.")
    if not os.path.isfile(os.path.join(data_args.tokenized_path, "dataset_dict.json")):
        raise ValueError(f"No tokenized dataset at {data_args.tokenized_path}: run hmf preprocess first.")
    _check_labeling_keys(data_args)


def _check_labeling_keys(data_args) -> None:
    """Checks the labeling keys of a config, without loading any model or data."""
    if data_args.decision_label_folder is None:
        raise ValueError("Set decision_label_folder: the folder in <tokenized_path>/labels/ to write the label set to.")
    if data_args.decision_teacher_model is not None and data_args.decision_teacher_file is not None:
        raise ValueError("Set one teacher: decision_teacher_model or decision_teacher_file, not both.")
    if data_args.decision_anchor_model is None and not _has_teacher(data_args):
        raise ValueError(
            "Nothing to label: set decision_anchor_model and/or a teacher (decision_teacher_model or "
            "decision_teacher_file)."
        )


def _has_teacher(data_args) -> bool:
    return data_args.decision_teacher_model is not None or data_args.decision_teacher_file is not None


# The keys that ask hmf preprocess to label the data
LABELING_KEYS = ("decision_label_folder", "decision_anchor_model", "decision_teacher_model", "decision_teacher_file")


def check_labeling_config(config_path: str) -> None:
    """
    Checks a config's labeling keys before tokenizing, so that a mistake is found in seconds rather
    than after tokenization. It checks the label set's name, that there is at least one target and at
    most one teacher, the teacher datasets, and the teacher file.
    """
    _, data_args, _ = _load_args(config_path)
    _check_labeling_keys(data_args)
    if data_args.decision_teacher_datasets is not None:
        get_dataset_list(data_args.decision_teacher_datasets, data_args.dataset_dir)  # unknown names raise
    if data_args.decision_teacher_file is not None and not os.path.isfile(data_args.decision_teacher_file):
        raise ValueError(f"decision_teacher_file {data_args.decision_teacher_file} does not exist.")


def is_labeled(tokenized_path: str, name: str) -> bool:
    """Whether the label set called name is complete (its labels_info.json is written last)."""
    return os.path.isfile(os.path.join(labels_path(tokenized_path, name), "labels_info.json"))


def _teacher_sources(data_args) -> set:
    """
    The sources (see decision_sources) whose rows get a teacher, which are those of
    decision_teacher_datasets, or of all datasets if it is unset.
    """
    names = data_args.decision_teacher_datasets or data_args.dataset
    sources = set()
    for attr in get_dataset_list(names, data_args.dataset_dir):
        files = decision_dataset_files(attr, data_args.dataset_dir)
        sources.update(source for _, source in decision_sources(files, attr))
    return sources


def _original_rows(data_args, keys: set) -> Dict[Tuple[str, int], dict]:
    """The original rows with the given (source, row_index) keys, read from the dataset files."""
    rows = {}
    wanted_sources = {source for source, _ in keys}
    for attr in get_dataset_list(data_args.dataset, data_args.dataset_dir):
        files = decision_dataset_files(attr, data_args.dataset_dir)
        if not wanted_sources & {source for _, source in decision_sources(files, attr)}:
            continue
        for source, row_index, row in read_decision_rows(files, attr):
            if (source, row_index) in keys:
                rows[(source, row_index)] = row
    if len(rows) != len(keys):
        missing = sorted(keys - set(rows))[:3]
        raise ValueError(f"Rows {missing} of the tokenized data are not in the dataset files: did the files change?")
    return rows


def _shard_indices(dataset: Dataset, shard_index: int, num_shards: int) -> List[int]:
    """
    The training rows of a slice of the original rows, so that a teacher model reads each original
    row on one GPU only (its augmentations are all in the same slice).
    """
    keys = list(zip(dataset["source"], dataset["row_index"]))
    rank = {key: r for r, key in enumerate(sorted(set(keys)))}
    return [i for i, key in enumerate(keys) if rank[key] % num_shards == shard_index]


def label_shard(
    config_path: str,
    shard_index: int = 0,
    num_shards: int = 1,
    device: Optional[str] = None,
    overrides: Sequence[str] = (),
) -> Dataset:
    """
    Labels the training rows of one slice of the original rows of a tokenized dataset.

    Args:
        config_path: The tokenization config, with the labeling keys.
        shard_index: Which slice to label. Defaults to 0.
        num_shards: The number of slices. Defaults to 1 (all rows).
        device: Where to run the models. Defaults to the first GPU if there is one, else the CPU.
        overrides: key=value overrides of the config. Defaults to none.

    Returns:
        A dataset with each labeled row's index, its token fingerprint, and the label columns.
    """
    model_args, data_args, _ = _load_args(config_path, overrides)
    _check_args(data_args)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    dataset = load_from_disk(data_args.tokenized_path)["train"]
    indices = _shard_indices(dataset, shard_index, num_shards)
    columns = ["input_ids", "answer_idx", "opt_idx", "option_order", "source", "row_index"]
    rows = dataset.select(indices).select_columns(columns).to_dict()
    num_options = [len(opt_idx) for opt_idx in rows["opt_idx"]]
    keys = list(zip(rows["source"], rows["row_index"]))

    refs: List[Optional[List[float]]] = [None] * len(indices)
    if data_args.decision_anchor_model is not None:
        anchor = CausalLMScorer(data_args.decision_anchor_model, device)
        from transformers import AutoTokenizer

        data_tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
        if anchor.tokenizer.get_vocab() != data_tokenizer.get_vocab():
            raise ValueError(
                f"The anchor model {data_args.decision_anchor_model} reads the tokenized prompts, so it must use the "
                f"tokenizer the data was tokenized with ({model_args.model_name_or_path})."
            )
        # The options' labels in the prompt format (1, 2, ...), in the order the prompt shows them
        prompt_format = get_prompt_format(data_args.decision_prompt_format)
        refs = anchor(
            [
                ScoringQuery(ids, answer_idx, [prompt_format.option_label(k) for k in range(n)])
                for ids, answer_idx, n in zip(rows["input_ids"], rows["answer_idx"], num_options)
            ]
        )
        del anchor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # The teacher of each original row of the slice, in original order
    teacher_probs: Dict[Tuple[str, int], Optional[List[float]]] = {}
    teacher_name = data_args.decision_teacher_file or data_args.decision_teacher_model
    if _has_teacher(data_args):
        sources = _teacher_sources(data_args)
        wanted = {key for key in keys if key[0] in sources}
        if data_args.decision_teacher_file is not None:
            teacher_file = TeacherFile(data_args.decision_teacher_file)
            teacher_probs = {key: teacher_file.get(*key) for key in wanted}
        else:
            originals = _original_rows(data_args, wanted)
            wanted = sorted(wanted)
            teacher = load_teacher_model(data_args, device)
            found = teacher([originals[key] for key in wanted])
            teacher_probs = dict(zip(wanted, found))
            num_found = sum(probs is not None for probs in found)
            logger.info_rank0(f"Teacher: {num_found:,} of {len(wanted):,} original rows labeled.")

    teachers: List[Optional[List[float]]] = [None] * len(indices)
    for j, key in enumerate(keys):
        probs = teacher_probs.get(key)
        if probs is None:
            continue
        if len(probs) != num_options[j]:
            raise ValueError(
                f"{teacher_name}: {key[0]} row {key[1]} has {num_options[j]} options, "
                f"but its teacher has {len(probs)} probabilities."
            )
        teachers[j] = to_augmented_order(probs, rows["option_order"][j])

    return Dataset.from_dict(
        {
            "index": indices,
            "fingerprint": [tokens_fingerprint(ids) for ids in rows["input_ids"]],
            "ref_probs": [r if r is not None else [] for r in refs],
            "has_ref": [r is not None for r in refs],
            "teacher_probs": [t if t is not None else [] for t in teachers],
            "has_teacher": [t is not None for t in teachers],
        },
        features=SHARD_FEATURES,
    )


def merge_shards(config_path: str, shards: Sequence[Dataset], overrides: Sequence[str] = ()) -> Dataset:
    """Puts the labeled slices back in the order of the training rows, with each row's key."""
    _, data_args, _ = _load_args(config_path, overrides)
    dataset = load_from_disk(data_args.tokenized_path)["train"]
    merged = concatenate_datasets(list(shards)).sort("index").flatten_indices()
    if merged["index"] != list(range(len(dataset))):
        raise RuntimeError("The labeled slices do not cover every training row exactly once.")
    keys = dataset.select_columns(list(KEY_COLUMNS))
    return concatenate_datasets([keys, merged.remove_columns("index")], axis=1)


def _label_on_gpus(config_path: str, num_gpus: int, parts_dir: str, overrides: Sequence[str]) -> List[Dataset]:
    """Labels one slice of the rows per GPU, in parallel processes, and returns the slices."""
    processes = []
    for shard_index in range(num_gpus):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(shard_index))
        command = [
            sys.executable, "-m", "hmf.data.decision_label", config_path, *overrides,
            "--shard_index", str(shard_index), "--num_shards", str(num_gpus), "--parts_dir", parts_dir,
        ]
        processes.append(subprocess.Popen(command, env=env))
    failed = [i for i, process in enumerate(processes) if process.wait() != 0]
    if failed:
        raise RuntimeError(f"Labeling failed on GPU(s) {failed}; see their output above.")
    return [load_from_disk(os.path.join(parts_dir, f"shard_{i}")) for i in range(num_gpus)]


def run_decision_label(config_path: str, num_gpus: Optional[int] = None, overrides: Sequence[str] = ()) -> str:
    """
    Labels the training split of a tokenized decision dataset and writes the label set.

    Args:
        config_path: The tokenization config, with the labeling keys.
        num_gpus: The number of GPUs to label on. Defaults to all visible GPUs (or the CPU).
        overrides: key=value overrides of the config. Defaults to none.

    Returns:
        The folder of the label set.
    """
    _, data_args, _ = _load_args(config_path, overrides)
    _check_args(data_args)
    output_path = labels_path(data_args.tokenized_path, data_args.decision_label_folder)
    if is_labeled(data_args.tokenized_path, data_args.decision_label_folder):
        raise ValueError(
            f"The label set {output_path} already exists. Delete it, or choose another decision_label_folder."
        )
    if os.path.exists(output_path):  # an interrupted run
        logger.warning_rank0(f"Removing the incomplete label set {output_path}.")
        shutil.rmtree(output_path)

    num_gpus = num_gpus if num_gpus is not None else max(1, torch.cuda.device_count())
    if not torch.cuda.is_available():
        logger.warning_rank0(
            "No GPUs visible: labeling on the CPU, which is slow for large models. To use GPUs, check "
            "CUDA_VISIBLE_DEVICES."
        )
    if num_gpus > 1:
        parts_dir = output_path + ".parts"
        shutil.rmtree(parts_dir, ignore_errors=True)  # left by an interrupted run
        shards = _label_on_gpus(config_path, num_gpus, parts_dir, overrides)
    else:
        parts_dir, shards = None, [label_shard(config_path, overrides=overrides)]
    labels = merge_shards(config_path, shards, overrides)

    labels.save_to_disk(os.path.join(output_path, "train"))
    info = {
        "tokenized_path": os.path.abspath(data_args.tokenized_path),
        "anchor_model": data_args.decision_anchor_model,
        "teacher_model": data_args.decision_teacher_model,
        "teacher_adapter": data_args.decision_teacher_adapter,
        "teacher_prompt": data_args.decision_teacher_prompt if data_args.decision_teacher_model else None,
        "teacher_max_len": data_args.decision_teacher_max_len,
        "teacher_file": data_args.decision_teacher_file,
        "teacher_datasets": data_args.decision_teacher_datasets,
        "num_rows": len(labels),
        "num_with_anchor": int(sum(labels["has_ref"])),
        "num_with_teacher": int(sum(labels["has_teacher"])),
    }
    with open(os.path.join(output_path, "labels_info.json"), "w") as f:
        json.dump(info, f, indent=2)
    if parts_dir is not None:
        shutil.rmtree(parts_dir, ignore_errors=True)

    print(
        f"Label set {data_args.decision_label_folder}: {info['num_rows']:,} training rows, "
        f"{info['num_with_anchor']:,} with an anchor, {info['num_with_teacher']:,} with a teacher. "
        f"Written to {output_path}"
    )
    return output_path


def load_decision_labels(train_dataset: Dataset, tokenized_path: str, name: str) -> Dataset:
    """
    Joins the label set called name to the training rows it was made for.

    Every row's key (source, row_index, augmentation, option_order) must match, and so must the
    token fingerprints of an even sample of rows, so that labels made for other tokens (e.g. after
    re-tokenizing with another seed) are rejected rather than silently joined.

    Args:
        train_dataset: The tokenized training split.
        tokenized_path: Where the tokenized dataset, and its label sets, are stored.
        name: The label set.

    Returns:
        The training split with the label columns added.
    """
    path = labels_path(tokenized_path, name)
    if not is_labeled(tokenized_path, name):
        all_labels = os.path.join(tokenized_path, LABELS_DIR)
        names = os.listdir(all_labels) if os.path.isdir(all_labels) else []
        available = sorted(n for n in names if is_labeled(tokenized_path, n))
        raise ValueError(f"No label set {name!r} in {tokenized_path}. Available: {available or 'none'}.")
    labels = load_from_disk(os.path.join(path, "train"))

    mismatch = f"The label set {name!r} was made for other tokens than {tokenized_path}: label the data again."
    if len(labels) != len(train_dataset):
        raise ValueError(f"{mismatch} ({len(labels):,} label rows, {len(train_dataset):,} training rows)")
    for column in KEY_COLUMNS:
        if labels[column] != train_dataset[column]:
            raise ValueError(f"{mismatch} (the {column} column differs)")
    num_checked = min(NUM_FINGERPRINTS_CHECKED, len(labels))
    sample = sorted(set(np.linspace(0, len(labels) - 1, num_checked, dtype=int).tolist()))
    fingerprints = labels.select(sample)["fingerprint"]
    if fingerprints != [tokens_fingerprint(ids) for ids in train_dataset.select(sample)["input_ids"]]:
        raise ValueError(f"{mismatch} (the tokens differ)")

    logger.info_rank0(f"Joined label set {name!r} ({path}).")
    return concatenate_datasets([train_dataset, labels.select_columns(list(LABEL_COLUMNS))], axis=1)


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Label a tokenized decision dataset (see hmf.data.decision_label).")
    parser.add_argument("config", help="The tokenization config, with the labeling keys")
    parser.add_argument("overrides", nargs="*", help="key=value overrides of the config")
    parser.add_argument("--num_gpus", type=int, default=None, help="GPUs to label on (default: all visible)")
    # Used internally, to label one GPU's slice when labeling on several GPUs
    parser.add_argument("--shard_index", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--num_shards", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--parts_dir", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    if args.shard_index is not None:
        shard = label_shard(args.config, args.shard_index, args.num_shards, overrides=args.overrides)
        shard.save_to_disk(os.path.join(args.parts_dir, f"shard_{args.shard_index}"))
    else:
        run_decision_label(args.config, args.num_gpus, args.overrides)


if __name__ == "__main__":
    main()
