"""
Checks decision data tokenized by `hmf preprocess` against the raw rows it was built from. Unlike
the other decision data tests, which use toy data, these run on a real tokenization output:

    hmf preprocess examples/decision/tokenization/qwen3_5_2b/strands_v19.yaml
    HMF_DECISION_TOKENIZE_CONFIG=examples/decision/tokenization/qwen3_5_2b/strands_v19.yaml \
        pytest tests/decision/data/test_decision_tokenized.py -v -rs -s

The tokenization config names everything the checks need, which is tokenized_path, the tokenizer,
the datasets, and the decision options. The raw rows are loaded as the tokenizer saw them. All tests are
skipped when HMF_DECISION_TOKENIZE_CONFIG is not set.
"""

import os
import random
from collections import Counter

import pyarrow.compute as pc
import pytest
from datasets import load_from_disk

from hmf.data import get_template_and_fix_tokenizer
from hmf.data.decision_prompt import NOUL_OPTION_NAMES, encode_decision_prompt, get_prompt_format
from hmf.data.loader import _load_single_dataset
from hmf.data.parser import get_dataset_list
from hmf.data.sharded_preprocess_and_tokenize import get_args_from_config
from hmf.model import load_tokenizer


TOKENIZE_CONFIG = os.getenv("HMF_DECISION_TOKENIZE_CONFIG")
NUM_ROUND_TRIP_TRAIN_ROWS = 3000

pytestmark = pytest.mark.skipif(
    TOKENIZE_CONFIG is None, reason="set HMF_DECISION_TOKENIZE_CONFIG to a decision tokenization config"
)

COLUMNS = {
    "input_ids", "attention_mask", "opt_idx", "answer_idx", "label", "label_dist", "option_order",
    "kind", "task", "weight", "source", "row_index", "augmentation",
}
# Small columns, read whole; input_ids and attention_mask are only read for sampled rows
SMALL_COLUMNS = sorted(COLUMNS - {"input_ids", "attention_mask"})


@pytest.fixture(scope="module")
def args():
    model_args, data_args, training_args, _, _ = get_args_from_config(TOKENIZE_CONFIG)
    return model_args, data_args, training_args


@pytest.fixture(scope="module")
def tokenized(args):
    return load_from_disk(args[1].tokenized_path)


@pytest.fixture(scope="module")
def columns(tokenized) -> dict:
    """The small columns of each split, plus each row's length, as Python lists."""
    out = {}
    for split, dataset in tokenized.items():
        out[split] = dataset.select_columns(SMALL_COLUMNS).to_dict()
        out[split]["length"] = pc.list_value_length(dataset.data.table.column("input_ids")).to_pylist()
    return out


@pytest.fixture(scope="module")
def raw(args) -> dict:
    """The aligned raw rows, keyed by (source, row_index), as the tokenizer saw them."""
    model_args, data_args, training_args = args
    rows = {}
    for dataset_attr in get_dataset_list(data_args.dataset, data_args.dataset_dir):
        dataset = _load_single_dataset(dataset_attr, model_args, data_args, training_args)
        for row in dataset:
            rows[(row["_source"], row["_row_index"])] = row
    return rows


@pytest.fixture(scope="module")
def tokenizer(args):
    model_args, data_args, _ = args
    tokenizer = load_tokenizer(model_args)["tokenizer"]
    get_template_and_fix_tokenizer(tokenizer, data_args)
    return tokenizer


def _rows(split_columns: dict):
    """Iterates over the rows of a split's columns."""
    names = list(split_columns)
    for values in zip(*(split_columns[name] for name in names)):
        yield dict(zip(names, values))


def _num_options(row: dict) -> int:
    return len(NOUL_OPTION_NAMES) if row["_kind"] == "noul" else len(row["_options"])


def test_schema(args, tokenized):
    data_args = args[1]
    expected_splits = {"train", "validation"} if data_args.val_size > 1e-6 else {"train"}

    assert set(tokenized) == expected_splits
    for dataset in tokenized.values():
        assert set(dataset.column_names) == COLUMNS


def test_summary(args, columns, raw):
    data_args = args[1]
    print(f"\nraw rows: {len(raw):,}")
    for source, n in sorted(Counter(source for source, _ in raw).items()):
        print(f"  {source}: {n:,}")
    for split, split_columns in columns.items():
        lengths = sorted(split_columns["length"])
        print(
            f"{split}: {len(lengths):,} rows | prompt length mean {sum(lengths) / len(lengths):.0f}, "
            f"p50 {lengths[len(lengths) // 2]}, p99 {lengths[int(len(lengths) * 0.99)]}, max {lengths[-1]} "
            f"(cutoff_len {data_args.cutoff_len})"
        )
        print(f"  kinds: {dict(Counter(split_columns['kind']))}")
        print(f"  tasks: {len(set(split_columns['task']))}, largest: {Counter(split_columns['task']).most_common(5)}")


def test_every_row_tokenized_once_per_augmentation(args, columns, raw):
    num_augmentations = args[1].decision_num_augmentations
    train = Counter(zip(columns["train"]["source"], columns["train"]["row_index"]))
    train_columns = columns["train"]
    augmentations = Counter(zip(train_columns["source"], train_columns["row_index"], train_columns["augmentation"]))
    validation = Counter()
    if "validation" in columns:
        validation = Counter(zip(columns["validation"]["source"], columns["validation"]["row_index"]))

    missing = set(raw) - set(train) - set(validation)
    assert not missing, (
        f"{len(missing):,} raw rows are not in the tokenized data (dropped?), e.g. {sorted(missing)[:5]}"
    )
    assert set(train) | set(validation) == set(raw)
    assert not set(train) & set(validation), "rows in both splits"
    assert set(train.values()) == {num_augmentations}, "training rows with a missing augmentation (dropped?)"
    assert set(augmentations.values()) == {1}
    assert set(columns["train"]["augmentation"]) == set(range(num_augmentations))
    assert set(validation.values()) <= {1}


def test_validation_split(args, columns, raw):
    data_args = args[1]
    if "validation" not in columns:
        pytest.skip("no validation split (val_size is 0)")

    # Per task, the number of rows stratified_split holds out
    fraction = data_args.val_size if data_args.val_size < 1 else data_args.val_size / len(raw)
    task_sizes = Counter(row["_task"] for row in raw.values())
    expected = {}
    for task, n in task_sizes.items():
        num_train = max(1, int(n * (1 - fraction))) if n > 1 else 1
        if n - num_train:
            expected[task] = n - num_train

    validation = columns["validation"]
    print(f"\nvalidation: {len(validation['task']):,} of {len(raw):,} rows ({len(validation['task']) / len(raw):.2%})")
    assert dict(Counter(validation["task"])) == expected
    assert set(validation["augmentation"]) == {0}
    assert all(order == list(range(len(order))) for order in validation["option_order"]), "augmented validation rows"


def test_labels_and_targets(args, columns, raw):
    smoothing = args[1].decision_ordinal_smoothing
    for split, split_columns in columns.items():
        for row in _rows(split_columns):
            source_row = raw[(row["source"], row["row_index"])]
            n = _num_options(source_row)
            where = f"{split} {row['source']} row {row['row_index']} augmentation {row['augmentation']}"

            assert sorted(row["option_order"]) == list(range(n)), where
            assert len(row["opt_idx"]) == len(row["label_dist"]) == n, where
            assert row["option_order"][row["label"]] == source_row["_label"], where
            assert (row["kind"], row["task"], row["weight"]) == (
                source_row["_kind"], source_row["_task"], source_row["_weight"]
            ), where

            dist = row["label_dist"]
            assert abs(sum(dist) - 1.0) < 1e-6, where
            assert max(range(n), key=dist.__getitem__) == row["label"], where
            if row["kind"] != "score" or smoothing == 0:
                assert dist[row["label"]] == 1.0, where
            else:
                assert abs(dist[row["label"]] - (1.0 - smoothing)) < 1e-6, where


def test_positions_and_lengths(args, columns):
    cutoff_len = args[1].cutoff_len
    for split, split_columns in columns.items():
        for row in _rows(split_columns):
            where = f"{split} {row['source']} row {row['row_index']} augmentation {row['augmentation']}"
            positions = row["opt_idx"] + [row["answer_idx"]]
            assert row["length"] <= cutoff_len, where
            assert all(a < b for a, b in zip(positions, positions[1:])), where
            assert row["answer_idx"] < row["length"], where


def test_round_trip(args, tokenized, columns, raw, tokenizer):
    # Rendering the raw row with the recorded option order, and one of its phrasings, must give
    # exactly the stored tokens and positions
    data_args = args[1]
    prompt_format = get_prompt_format(data_args.decision_prompt_format)
    rng = random.Random(0)
    num_train_rows = len(tokenized["train"])
    samples = {"train": rng.sample(range(num_train_rows), min(NUM_ROUND_TRIP_TRAIN_ROWS, num_train_rows))}
    if "validation" in tokenized:
        samples["validation"] = range(len(tokenized["validation"]))

    num_alternative = 0
    for split, indices in samples.items():
        for row in tokenized[split].select(indices):
            source_row = raw[(row["source"], row["row_index"])]
            phrasings = [source_row["_instructions"]]
            if split == "train":
                phrasings += source_row["_instruction_variants"] or []

            matches = []
            for phrasing in phrasings:
                rendered = prompt_format.render(
                    source_row["_state"], phrasing, source_row["_kind"],
                    [tuple(option) for option in source_row["_options"]], option_order=row["option_order"],
                )
                encoded = encode_decision_prompt(tokenizer, rendered, force_bos=data_args.decision_force_bos)
                if (encoded.input_ids, encoded.opt_idx, encoded.answer_idx) == (
                    row["input_ids"], row["opt_idx"], row["answer_idx"]
                ):
                    matches.append(phrasing)

            assert matches, f"{split} {row['source']} row {row['row_index']}: stored tokens match no rendered prompt"
            if split == "train" and source_row["_instructions"] not in matches:
                num_alternative += 1

    num_train = len(samples["train"])
    num_validation = len(samples.get("validation", []))
    print(f"\nround trip: {num_train:,} training rows and {num_validation:,} validation rows match")
    print(f"  {num_alternative:,} of the {num_train:,} training rows used an alternative phrasing")
