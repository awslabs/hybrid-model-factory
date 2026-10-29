# Copyright 2025 the LlamaFactory team.
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

import json
import os
import random
from enum import Enum, unique
from typing import TYPE_CHECKING, Any, Iterator, Optional, TypedDict, Union
import numpy as np

import fsspec
from datasets import DatasetDict, concatenate_datasets, interleave_datasets
from datasets import Dataset, Features, Sequence, Value

from ..extras import logging
from .decision_prompt import render_content


if TYPE_CHECKING:
    from datasets import Dataset, IterableDataset

    from ..hparams import DataArguments


logger = logging.get_logger(__name__)


SLOTS = list[Union[str, set[str], dict[str, str]]]


@unique
class Role(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    FUNCTION = "function"
    OBSERVATION = "observation"
    PT_SEPARATOR = "pt_separator"
    PT_CONTENT = "pt_content"


class DatasetModule(TypedDict):
    train_dataset: Optional[Union["Dataset", "IterableDataset"]]
    eval_dataset: Optional[Union["Dataset", "IterableDataset", dict[str, "Dataset"]]]


def merge_dataset(
    all_datasets: list[Union["Dataset", "IterableDataset"]], 
    data_args: "DataArguments", 
    seed: int, 
    max_merged_samples: Optional[int] = None,
) -> Union["Dataset", "IterableDataset"]:
    r"""Merge multiple datasets to a unified dataset."""
    if len(all_datasets) == 1:
        return all_datasets[0]

    strategy = data_args.mix_strategy

    # Warn if max_merged_samples is provided but won't be used
    if max_merged_samples is not None and strategy != "interleave_fast":
        logger.warning_rank0_once(
            f"max_merged_samples={max_merged_samples:,} is specified but will be ignored with mix_strategy='{strategy}'. "
            f"Use mix_strategy='interleave_fast' to apply max_merged_samples during merging."
        )

    # Concat strategy
    if strategy == "concat":
        if data_args.streaming:
            logger.warning_rank0_once("The samples between different datasets will not be mixed in streaming mode.")
        return concatenate_datasets(all_datasets)

    # Interleave strategies
    if strategy in ("interleave_under", "interleave_over", "interleave_once", "interleave_fast"):
        if not data_args.streaming:
            logger.warning_rank0_once("We recommend using `mix_strategy=concat` in non-streaming mode.")

        all_datasets = [dataset.shuffle(seed=42) for dataset in all_datasets]
        logger.info_rank0(f"Interleave Probs: {data_args.interleave_probs}")

        if strategy == "interleave_under":
            logger.info_rank0("Using interleave_datasets (stopping_strategy=first_exhausted)")
            return interleave_datasets(
                datasets=all_datasets,
                probabilities=data_args.interleave_probs,
                seed=seed,
                stopping_strategy="first_exhausted",
            )

        if strategy == "interleave_over":
            logger.info_rank0("Using interleave_datasets (stopping_strategy=all_exhausted)")
            return interleave_datasets(
                datasets=all_datasets,
                probabilities=data_args.interleave_probs,
                seed=seed,
                stopping_strategy="all_exhausted",
            )

        if strategy == "interleave_once":
            logger.info_rank0("Using interleave_datasets (stopping_strategy=all_exhausted_without_replacement)")
            return interleave_datasets(
                datasets=all_datasets,
                probabilities=data_args.interleave_probs,
                seed=seed,
                stopping_strategy="all_exhausted_without_replacement",
            )

        if strategy == "interleave_fast":
            if max_merged_samples is None:
                raise ValueError("mix_strategy='interleave_fast' requires max_merged_samples to be specified.")
            if data_args.interleave_probs is None:
                raise ValueError("mix_strategy='interleave_fast' requires interleave_probs to be specified.")
            
            logger.info_rank0(f"Using fast_interleave_with_limit (max_merged_samples={max_merged_samples:,})")
            return fast_interleave_with_limit(
                all_datasets=all_datasets,
                probabilities=data_args.interleave_probs,
                max_samples=max_merged_samples,
                seed=seed,
            )

    raise ValueError(
        f"Unknown mixing strategy: {strategy}. "
        f"Choose from: 'concat', 'interleave_under', 'interleave_over', 'interleave_once', 'interleave_fast'."
    )

def stratified_split(dataset: "Dataset", val_size: float, stratify_by: str, seed: int) -> tuple["Dataset", "Dataset"]:
    """
    Splits a dataset into train and validation sets with the same proportion of each value of a
    column, so that small groups (e.g. small tasks) are represented in the validation set.

    Args:
        dataset: The dataset to split.
        val_size: The validation fraction in [0, 1), or the number of validation rows if > 1.
        stratify_by: The column whose values define the groups.
        seed: Seed of the split.

    Returns:
        The train and validation datasets. A group with a single row goes to the train set.
    """
    fraction = val_size if val_size < 1 else val_size / len(dataset)
    groups: dict[Any, list[int]] = {}
    for index, value in enumerate(dataset[stratify_by]):
        groups.setdefault(value, []).append(index)

    rng = random.Random(seed)
    train_indices, val_indices = [], []
    for value in sorted(groups, key=str):
        indices = groups[value]
        rng.shuffle(indices)
        num_train = max(1, int(len(indices) * (1 - fraction))) if len(indices) > 1 else 1
        train_indices += indices[:num_train]
        val_indices += indices[num_train:]

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    return dataset.select(train_indices), dataset.select(val_indices)


def split_dataset(
    dataset: Optional[Union["Dataset", "IterableDataset"]],
    eval_dataset: Optional[Union["Dataset", "IterableDataset", dict[str, "Dataset"]]],
    data_args: "DataArguments",
    seed: int,
    stratify_by: Optional[str] = None,
) -> tuple[dict, dict]:
    r"""Split the dataset and returns two dicts containing train set and validation set.

    Support both map dataset and iterable dataset. If stratify_by names a column, a map dataset is
    split with the same validation fraction in each group of that column (see stratified_split).

    Returns:
        train_dict: Dictionary containing training data with key "train"
        eval_dict: Dictionary containing evaluation data with keys "validation" or "validation_{name}"
    """
    if eval_dataset is not None and data_args.val_size > 1e-6:
        raise ValueError("Cannot specify `val_size` if `eval_dataset` is not None.")

    # the train and eval better to in dict dtype and separately return for cpode clearly and good handle outside
    train_dict, eval_dict = {}, {}

    if dataset is not None:
        if data_args.streaming:
            dataset = dataset.shuffle(buffer_size=data_args.buffer_size, seed=seed)

        if data_args.val_size > 1e-6:
            if data_args.streaming:
                eval_dict["validation"] = dataset.take(int(data_args.val_size))
                train_dict["train"] = dataset.skip(int(data_args.val_size))
            elif stratify_by is not None:
                train_dict["train"], eval_dict["validation"] = stratified_split(
                    dataset, data_args.val_size, stratify_by, seed
                )
            else:
                val_size = int(data_args.val_size) if data_args.val_size > 1 else data_args.val_size
                split_result = dataset.train_test_split(test_size=val_size, seed=seed)
                train_dict["train"] = split_result["train"]
                eval_dict["validation"] = split_result["test"]
        else:
            train_dict["train"] = dataset

    if eval_dataset is not None:
        if isinstance(eval_dataset, dict):
            for name, data in eval_dataset.items():
                eval_dict[f"validation_{name}"] = data
        else:
            if data_args.streaming:
                eval_dataset = eval_dataset.shuffle(buffer_size=data_args.buffer_size, seed=seed)

            eval_dict["validation"] = eval_dataset

    return train_dict, eval_dict


def get_dataset_module(dataset: Union["Dataset", "DatasetDict"]) -> "DatasetModule":
    r"""Convert dataset or dataset dict to dataset module."""
    dataset_module: DatasetModule = {}
    if isinstance(dataset, DatasetDict):  # dataset dict
        if "train" in dataset:
            dataset_module["train_dataset"] = dataset["train"]

        if "validation" in dataset:
            dataset_module["eval_dataset"] = dataset["validation"]
        else:
            eval_dataset = {}
            for key in dataset.keys():
                if key.startswith("validation_"):
                    eval_dataset[key[len("validation_") :]] = dataset[key]

            if len(eval_dataset):
                dataset_module["eval_dataset"] = eval_dataset

    else:  # single dataset
        dataset_module["train_dataset"] = dataset

    return dataset_module


def setup_fs(path: str, anon: bool = False) -> "fsspec.AbstractFileSystem":
    r"""Set up a filesystem object based on the path protocol."""
    storage_options = {"anon": anon} if anon else {}
    if path.startswith("s3://"):
        fs = fsspec.filesystem("s3", **storage_options)
    elif path.startswith(("gs://", "gcs://")):
        fs = fsspec.filesystem("gcs", **storage_options)
    else:
        raise ValueError(f"Unsupported protocol in path: {path}. Use 's3://' or 'gs://'.")

    if not fs.exists(path):
        raise ValueError(f"Path does not exist: {path}.")

    return fs


def _read_json_with_fs(fs: "fsspec.AbstractFileSystem", path: str) -> list[Any]:
    r"""Helper function to read JSON/JSONL files using fsspec."""
    with fs.open(path, "r") as f:
        if path.endswith(".jsonl"):
            return [json.loads(line) for line in f if line.strip()]
        else:
            return json.load(f)


def read_cloud_json(cloud_path: str) -> list[Any]:
    r"""Read a JSON/JSONL file from cloud storage (S3 or GCS).

    Args:
        cloud_path: str
            Cloud path in the format:
            - 's3://bucket-name/file.json' for AWS S3
            - 'gs://bucket-name/file.jsonl' or 'gcs://bucket-name/file.jsonl' for Google Cloud Storage
    """
    try:
        fs = setup_fs(cloud_path, anon=True)  # try with anonymous access first
    except Exception:
        fs = setup_fs(cloud_path)  # try again with credentials

    # filter out non-JSON files
    files = [x["Key"] for x in fs.listdir(cloud_path)] if fs.isdir(cloud_path) else [cloud_path]
    files = filter(lambda file: file.endswith(".json") or file.endswith(".jsonl"), files)
    if not files:
        raise ValueError(f"No JSON/JSONL files found in the specified path: {cloud_path}.")

    return sum([_read_json_with_fs(fs, file) for file in files], [])

# modified from https://github.com/jzhang38/EasyContext/
def preprocess_sp_dataset(seq_ids, world_size, sequence_parallel_mode):
    if sequence_parallel_mode == "zigzag-ring":
        step = len(seq_ids) // (2 * world_size)
        value_chunks = [seq_ids[s : s + step] for s in range(0, len(seq_ids), step)]
        local_values = list()
        for rank in range(world_size):
            local_values.append(value_chunks[rank] + value_chunks[2 * world_size - rank - 1])
        return local_values
    elif sequence_parallel_mode == "ulysses":
        step = len(seq_ids) // world_size
        local_values = [seq_ids[s : s + step] for s in range(0, len(seq_ids), step)]
        return local_values
    else:
        raise NotImplementedError("Other sequence parallel modes are to be implemented.")
    

COMPLEX_COLUMNS = {'messages', 'tools', 'images'}

def jsonl_generator(filepath, column_mapping):
    """Yield rows with needed columns, stringified appropriately."""
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            
            output = {}
            for canonical_name, source_col in column_mapping.items():
                value = row.get(source_col)
                if value is not None and canonical_name in COMPLEX_COLUMNS:
                    output[canonical_name] = json.dumps(value)
                else:
                    output[canonical_name] = value
            yield output

def deserialize_transform(batch):
    result = {}
    for key, values in batch.items():
        if key in COMPLEX_COLUMNS:
            result[key] = [json.loads(v) if v is not None else None for v in values]
        else:
            result[key] = values
    return result

def load_jsonl(filepath, dataset_attr, cache_dir=None):
    """Load JSONL using column names from dataset_attr."""
    
    # Map canonical name -> source column name (from dataset_attr)
    column_mapping = {}
    
    if dataset_attr.messages:
        column_mapping["messages"] = dataset_attr.messages
    if dataset_attr.tools:
        column_mapping["tools"] = dataset_attr.tools
    if dataset_attr.system:
        column_mapping["system"] = dataset_attr.system
    if dataset_attr.images:
        column_mapping["images"] = dataset_attr.images
    
    # Build features
    features = Features({name: Value("string") for name in column_mapping})
    
    logger.info_rank0(f"Column mapping: {column_mapping}")
    
    dataset = Dataset.from_generator(
        jsonl_generator,
        gen_kwargs={"filepath": filepath, "column_mapping": column_mapping},
        split=dataset_attr.split,
        cache_dir=cache_dir,
        features=features,
    )
    dataset.set_transform(deserialize_transform)
    return dataset


def fast_interleave_with_limit(
    all_datasets: list[Union["Dataset", "IterableDataset"]],
    probabilities: list[float],
    max_samples: int,
    seed: int,
) -> "Dataset":
    """
    Fast alternative to interleave_datasets when you know the max samples needed.
    
    Instead of iteratively sampling one-at-a-time (slow), this:
    1. Calculates exactly how many samples needed from each dataset
    2. Samples directly from each (with replacement if needed)
    3. Concatenates and shuffles
    
    ~10-100x faster than interleave_datasets for large datasets.
    """
    from tqdm import tqdm
    
    rng = np.random.default_rng(seed)
    
    # Normalize probabilities
    probs = np.array(probabilities)
    probs = probs / probs.sum()
    
    sampled_datasets = []
    
    for i, (ds, prob) in tqdm(
        enumerate(zip(all_datasets, probs)), 
        total=len(all_datasets),
        desc="Sampling datasets"
    ):
        n_needed = int(max_samples * prob)
        n_available = len(ds)
        
        if n_needed == 0:
            continue
        
        if n_needed <= n_available:
            # Sample without replacement
            indices = rng.choice(n_available, size=n_needed, replace=False)
        else:
            # Need to repeat samples (like all_exhausted does)
            indices = rng.choice(n_available, size=n_needed, replace=True)
        
        # Sort indices for faster disk access (sequential reads)
        indices = np.sort(indices)
        
        logger.info_rank0(f"  Dataset {i}: sampling {n_needed:,} from {n_available:,} "
                         f"({'with' if n_needed > n_available else 'without'} replacement)")
        
        sampled_datasets.append(ds.select(indices.tolist()))
    
    logger.info_rank0(f"Concatenating {len(sampled_datasets)} sampled datasets...")
    combined = concatenate_datasets(sampled_datasets)
    
    logger.info_rank0(f"Shuffling {len(combined):,} samples...")
    return combined.shuffle(seed=seed)

# Columns of a decision row, as named in DatasetAttr (which maps them to the source's column names)
DECISION_COLUMNS = ("state", "instructions", "instruction_variants", "kind", "options", "label", "task", "weight")

DECISION_FEATURES = Features(
    {
        "state": Value("string"),
        "instructions": Value("string"),
        "instruction_variants": Sequence(Value("string")),
        "kind": Value("string"),
        "options": Sequence(Sequence(Value("string"))),
        "label": Value("int64"),
        "task": Value("string"),
        "weight": Value("float64"),
        "source": Value("string"),
        "row_index": Value("int64"),
    }
)

# The same columns after DecisionDatasetConverter, which prefixes them with an underscore
DECISION_ALIGNED_FEATURES = Features({f"_{name}": feature for name, feature in DECISION_FEATURES.items()})


def flatten_decision_row(values: dict[str, Any]) -> dict[str, Any]:
    """
    Flattens a decision row's structured content (state, instructions, option descriptions) to text.
    Optional columns may be missing, e.g. in a row read straight from its data file.
    """
    variants, options = values.get("instruction_variants"), values.get("options")
    return {
        **values,
        "state": render_content(values["state"]),
        "instructions": render_content(values["instructions"]),
        "instruction_variants": None if variants is None else [render_content(v) for v in variants],
        "options": None
        if options is None
        else [[str(name), None if desc is None else render_content(desc)] for name, desc in options],
    }


def decision_dataset_files(dataset_attr, dataset_dir: str) -> list[str]:
    """The files of a local decision dataset, which are its file or the files of its folder."""
    local_path = os.path.join(dataset_dir, dataset_attr.dataset_name)
    if os.path.isdir(local_path):
        return [os.path.join(local_path, name) for name in os.listdir(local_path)]
    if os.path.isfile(local_path):
        return [local_path]
    raise ValueError(f"File {local_path} not found.")


def decision_sources(filepaths: list[str], dataset_attr) -> list[tuple[str, str]]:
    """
    The (file, source) pairs of a decision dataset's files, in the order they are read. A row's
    source is the dataset's file name, plus the file's name for a dataset that is a folder.
    """
    filepaths = sorted(filepaths)
    if len(filepaths) == 1:
        return [(filepaths[0], dataset_attr.dataset_name)]
    return [(path, f"{dataset_attr.dataset_name}/{os.path.basename(path)}") for path in filepaths]


def read_decision_rows(filepaths: list[str], dataset_attr) -> Iterator[tuple[str, int, dict[str, Any]]]:
    """
    Reads the rows of a decision dataset's files as they are stored, without flattening structured
    content. Columns are renamed with the column names in dataset_attr.

    Args:
        filepaths: The files of the dataset.
        dataset_attr: The dataset's attributes.

    Yields:
        The (source, row_index, row) of each row, where row_index is the row's index among the file's
        rows (its non-empty lines, for JSON Lines).
    """
    column_mapping = {name: getattr(dataset_attr, name) for name in DECISION_COLUMNS}
    for filepath, source in decision_sources(filepaths, dataset_attr):
        with open(filepath, "r", encoding="utf-8") as f:
            if filepath.endswith(".json"):
                rows = json.load(f)
            else:
                rows = (json.loads(line) for line in f if line.strip())

            for row_index, row in enumerate(rows):
                yield source, row_index, {name: row.get(column) for name, column in column_mapping.items()}


def _decision_generator(filepaths: list[str], dataset_attr, file_stats):
    """
    Yields the decision rows of each file, flattened, with their source and their index in that file.
    file_stats is unused, and only makes the datasets cache key change when a file changes.
    """
    for source, row_index, row in read_decision_rows(filepaths, dataset_attr):
        yield {**flatten_decision_row(row), "source": source, "row_index": row_index}


def load_decision_jsonl(filepaths: list[str], dataset_attr, cache_dir: Optional[str] = None) -> "Dataset":
    """
    Loads decision rows from JSONL (or JSON) files, using the column names in dataset_attr.

    Structured content (a dict or list state, instructions, or option description) is rendered to
    text here, as datasets cannot hold a column whose type varies between rows. Each row records
    where it came from, as its source (the dataset's file name, plus the file for a directory) and its
    index among the file's rows. This identity is stable across machines and shuffles, so
    per-row data computed elsewhere (e.g. teacher distributions) can be joined on it.

    Args:
        filepaths: The files of the dataset.
        dataset_attr: The dataset's attributes.
        cache_dir: Where datasets caches the loaded rows.

    Returns:
        A dataset with the columns of DECISION_FEATURES. Missing optional columns are None.
    """
    filepaths = sorted(filepaths)
    # Size and modification time are part of the cache key, so an edited file is loaded again
    file_stats = [(os.path.getsize(path), os.path.getmtime(path)) for path in filepaths]

    logger.info_rank0(f"Decision column mapping: { {name: getattr(dataset_attr, name) for name in DECISION_COLUMNS} }")
    return Dataset.from_generator(
        _decision_generator,
        gen_kwargs={"filepaths": filepaths, "dataset_attr": dataset_attr, "file_stats": file_stats},
        split=dataset_attr.split,
        cache_dir=cache_dir,
        features=DECISION_FEATURES,
    )
