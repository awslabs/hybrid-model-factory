"""
End-to-end tests for decision data, covering loading (load_decision_jsonl), conversion
(DecisionDatasetConverter), the stratified validation split, and tokenization through both
get_dataset and the sharded `hmf preprocess` path (run_preprocess).

Everything runs offline. The datasets are tiny strands-format JSONL files written to a temporary
folder, and the tokenizer is a tiny word-level tokenizer saved next to them.
"""

import json
from collections import Counter

import pytest
import yaml
from datasets import Dataset, load_from_disk
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from hmf.data import get_dataset, get_template_and_fix_tokenizer
from hmf.data.converter import DecisionDatasetConverter
from hmf.data.data_utils import load_decision_jsonl, stratified_split
from hmf.data.decision_prompt import render_content
from hmf.data.parser import get_dataset_list
from hmf.data.sharded_preprocess_and_tokenize import run_preprocess
from hmf.hparams.parser import _parse_train_args, _tokenized_eval_split_exists
from hmf.model import load_tokenizer


VAL_SIZE = 0.25
NUM_AUGMENTATIONS = 2

# Rows per task in toy_a, and the validation rows stratified_split takes from each at VAL_SIZE:
# max(1, int(n * 0.75)) rows stay in train, a single row always does.
TASK_SIZES = {"t1": 8, "t2": 4, "t3": 1}
EXPECTED_VAL = {"t1": 2, "t2": 1, "t3": 0, "unknown": 1}


def _row(i: int, task=None) -> dict:
    """A decision row; the kind cycles through noul, choice, and score."""
    kind = ("noul", "choice", "score")[i % 3]
    if kind == "noul":
        options, label = [["false", None], ["true", "it holds"]], i % 2
    elif kind == "choice":
        options, label = [["a", "first"], ["b", {"detail": i}], ["c", None]], i % 3
    else:
        options, label = [["0", "low"], ["1", "mid"], ["2", "high"], ["3", "top"]], i % 4
    row = {
        "kind": kind,
        "state": {"id": i, "notes": ["x", "y"]} if i % 4 == 0 else f"State number {i}.",
        "instructions": f"Question {i}?",
        "options": options,
        "label": label,
    }
    if task is not None:
        row["task"] = task
        row["weight"] = 0.5 if i % 5 == 0 else 1.0
        row["instruction_variants"] = [f"Asked differently {i}?"] if i % 2 == 0 else []
    return row


def _toy_a_rows() -> list:
    rows, i = [], 0
    for task, size in TASK_SIZES.items():
        for _ in range(size):
            rows.append(_row(i, task))
            i += 1
    return rows


def _toy_b_rows() -> list:
    """Rows with other column names (mapped in dataset_info.json) and no optional columns."""
    rows = []
    for i in range(4):
        row = _row(100 + i)
        rows.append({"context": row["state"], "question": row["instructions"], "answer": row["label"],
                     "kind": row["kind"], "options": row["options"]})
    return rows


def _write_jsonl(path, rows, blank_lines: bool = False) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
            if blank_lines:
                f.write("\n")


def _save_tiny_tokenizer(path) -> None:
    tokenizer = Tokenizer(models.WordLevel({"[UNK]": 0, "[PAD]": 1}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="[UNK]", pad_token="[PAD]")
    # A chat template HMF cannot parse into a template, as with Qwen3.5, which decision data must not need
    tokenizer.chat_template = "{% for message in messages %}{{ message['content'] }}{% endfor %}"
    tokenizer.save_pretrained(path)


@pytest.fixture
def workspace(tmp_path) -> dict:
    """A dataset folder with toy_a (strands columns, with a blank line between rows) and toy_b
    (mapped columns, defaults only), and a tiny tokenizer."""
    data_dir, tokenizer_dir = tmp_path / "data", tmp_path / "tokenizer"
    data_dir.mkdir()
    _write_jsonl(data_dir / "toy_a.jsonl", _toy_a_rows(), blank_lines=True)
    _write_jsonl(data_dir / "toy_b.jsonl", _toy_b_rows())
    dataset_info = {
        "toy_a": {"file_name": "toy_a.jsonl", "formatting": "decision"},
        "toy_b": {
            "file_name": "toy_b.jsonl",
            "formatting": "decision",
            "columns": {"state": "context", "instructions": "question", "label": "answer"},
        },
        "toy_hub": {"hf_hub_url": "some/dataset", "formatting": "decision"},
    }
    (data_dir / "dataset_info.json").write_text(json.dumps(dataset_info))
    _save_tiny_tokenizer(tokenizer_dir)
    return {"root": tmp_path, "data_dir": str(data_dir), "tokenizer_dir": str(tokenizer_dir)}


def _config(workspace: dict, **overrides) -> dict:
    config = {
        "model_name_or_path": workspace["tokenizer_dir"],
        "stage": "decision",
        "do_train": True,
        "finetuning_type": "full",
        "dataset": "toy_a,toy_b",
        "dataset_dir": workspace["data_dir"],
        "cutoff_len": 4096,
        "val_size": VAL_SIZE,
        "seed": 0,
        "overwrite_cache": True,
        "preprocessing_batch_size": 2,
        "decision_num_augmentations": NUM_AUGMENTATIONS,
        "output_dir": str(workspace["root"] / "output"),
        "report_to": "none",
    }
    config.update(overrides)
    return config


def _get_dataset(config: dict) -> dict:
    model_args, data_args, training_args, _, _ = _parse_train_args(config)
    tokenizer_module = load_tokenizer(model_args)
    template = get_template_and_fix_tokenizer(tokenizer_module["tokenizer"], data_args)
    return get_dataset(template, model_args, data_args, training_args, stage="decision", **tokenizer_module)


def _check_splits(train: Dataset, validation: Dataset) -> None:
    """Checks the train and validation splits of the toy datasets."""
    all_rows = {("toy_a.jsonl", i) for i in range(sum(TASK_SIZES.values()))}
    all_rows |= {("toy_b.jsonl", i) for i in range(4)}

    # Every row is in exactly one split; training rows once per augmentation, validation rows once
    train_rows = Counter(zip(train["source"], train["row_index"]))
    val_rows = Counter(zip(validation["source"], validation["row_index"]))
    assert set(train_rows) | set(val_rows) == all_rows
    assert not set(train_rows) & set(val_rows)
    assert set(train_rows.values()) == {NUM_AUGMENTATIONS}
    assert set(val_rows.values()) == {1}
    assert sorted(set(train["augmentation"])) == list(range(NUM_AUGMENTATIONS))

    # Stratified by task
    assert dict(Counter(validation["task"])) == {task: n for task, n in EXPECTED_VAL.items() if n}

    # Validation rows are canonical
    assert set(validation["augmentation"]) == {0}
    assert all(order == list(range(len(order))) for order in validation["option_order"])

    # Defaults for toy_b's missing columns
    toy_b = [i for i, source in enumerate(train["source"]) if source == "toy_b.jsonl"]
    assert {train["task"][i] for i in toy_b} == {"unknown"}
    assert {train["weight"][i] for i in toy_b} == {1.0}


# Loading and conversion


def test_load_decision_jsonl(workspace):
    data_dir = workspace["data_dir"]
    dataset_attr = get_dataset_list(["toy_a"], data_dir)[0]
    dataset = load_decision_jsonl([f"{data_dir}/toy_a.jsonl"], dataset_attr)
    rows = _toy_a_rows()

    assert len(dataset) == len(rows)  # blank lines are skipped
    assert dataset["row_index"] == list(range(len(rows)))
    assert set(dataset["source"]) == {"toy_a.jsonl"}
    for loaded, row in zip(dataset, rows):
        # Structured content is rendered to text, exactly as the prompt formats render it
        assert loaded["state"] == render_content(row["state"])
        assert loaded["options"] == [[name, None if d is None else render_content(d)] for name, d in row["options"]]


def test_load_decision_folder(tmp_path):
    folder = tmp_path / "data" / "parts"
    folder.mkdir(parents=True)
    _write_jsonl(folder / "b.jsonl", [_row(0, "t"), _row(1, "t")])
    _write_jsonl(folder / "a.jsonl", [_row(2, "t")])
    (tmp_path / "data" / "dataset_info.json").write_text(
        json.dumps({"parts": {"file_name": "parts", "formatting": "decision"}})
    )
    dataset_attr = get_dataset_list(["parts"], str(tmp_path / "data"))[0]

    dataset = load_decision_jsonl([str(folder / "b.jsonl"), str(folder / "a.jsonl")], dataset_attr)

    # Files are read in sorted order, and each row is indexed within its own file
    assert list(zip(dataset["source"], dataset["row_index"])) == [("parts/a.jsonl", 0), ("parts/b.jsonl", 0),
                                                                  ("parts/b.jsonl", 1)]


def test_column_mapping(workspace):
    dataset_attr = get_dataset_list(["toy_b"], workspace["data_dir"])[0]

    assert (dataset_attr.state, dataset_attr.instructions, dataset_attr.label) == ("context", "question", "answer")
    # Unmapped decision columns keep their default names
    assert (dataset_attr.kind, dataset_attr.options, dataset_attr.task) == ("kind", "options", "task")


def _loaded_example(**overrides) -> dict:
    example = {
        "state": "s", "instructions": "q", "instruction_variants": None, "kind": "choice",
        "options": [["a", None], ["b", None]], "label": 1, "task": None, "weight": None,
        "source": "toy.jsonl", "row_index": 3,
    }
    example.update(overrides)
    return example


def test_converter_defaults():
    converted = DecisionDatasetConverter(dataset_attr=None, data_args=None)(_loaded_example())

    assert converted["_task"] == "unknown"
    assert converted["_weight"] == 1.0
    assert converted["_instruction_variants"] == []
    assert (converted["_source"], converted["_row_index"]) == ("toy.jsonl", 3)


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"kind": "ranking"}, "unknown kind"),
        ({"label": None}, "missing label"),
        ({"options": None}, "needs options"),
        ({"kind": "score", "options": []}, "needs options"),
    ],
)
def test_converter_errors(overrides, message):
    converter = DecisionDatasetConverter(dataset_attr=None, data_args=None)
    with pytest.raises(ValueError, match=message) as error:
        converter(_loaded_example(**overrides))
    assert "toy.jsonl row 3" in str(error.value)


def test_noul_without_options_is_allowed():
    converted = DecisionDatasetConverter(dataset_attr=None, data_args=None)(
        _loaded_example(kind="noul", options=None, label=0)
    )
    assert converted["_options"] == []


# Stratified split


def test_stratified_split():
    tasks = ["a"] * 10 + ["b"] * 5 + ["c"]
    dataset = Dataset.from_dict({"_task": tasks, "i": list(range(len(tasks)))})

    train, validation = stratified_split(dataset, 0.2, stratify_by="_task", seed=0)

    assert sorted(train["i"] + validation["i"]) == list(range(len(tasks)))
    assert dict(Counter(validation["_task"])) == {"a": 2, "b": 1}  # c has a single row, which goes to train
    again = stratified_split(dataset, 0.2, stratify_by="_task", seed=0)
    assert (train["i"], validation["i"]) == (again[0]["i"], again[1]["i"])
    other = stratified_split(dataset, 0.2, stratify_by="_task", seed=1)
    assert validation["i"] != other[1]["i"]

    # An integer val_size is a number of rows, as a fraction of the dataset (4 / 16) in each group.
    # Train sizes are rounded down, so validation can get slightly more (a 10 -> 7 + 3, b 5 -> 3 + 2)
    _, validation = stratified_split(dataset, 4, stratify_by="_task", seed=0)
    assert dict(Counter(validation["_task"])) == {"a": 3, "b": 2}


# Tokenization


def test_get_dataset(workspace):
    dataset_module = _get_dataset(_config(workspace))

    _check_splits(dataset_module["train_dataset"], dataset_module["eval_dataset"])


def test_hub_decision_dataset_rejected(workspace):
    with pytest.raises(ValueError, match="local JSON/JSONL"):
        _get_dataset(_config(workspace, dataset="toy_hub", val_size=0.0))


def test_run_preprocess(workspace):
    tokenized_path = workspace["root"] / "tokenized" / "toy"
    config_path = workspace["root"] / "tokenize.yaml"
    config_path.write_text(yaml.safe_dump(_config(workspace, tokenized_path=str(tokenized_path))))

    assert run_preprocess(str(config_path), shard_size=4, parallel_jobs=2) == 0

    tokenized = load_from_disk(str(tokenized_path))
    assert set(tokenized) == {"train", "validation"}
    _check_splits(tokenized["train"], tokenized["validation"])

    # Training can evaluate on the tokenized validation split without val_size or eval_dataset
    assert _tokenized_eval_split_exists(str(tokenized_path))
    assert not _tokenized_eval_split_exists(str(workspace["root"] / "missing"))

    # Training then loads both splits from tokenized_path
    dataset_module = _get_dataset(_config(workspace, tokenized_path=str(tokenized_path)))
    assert len(dataset_module["train_dataset"]) == len(tokenized["train"])
    assert len(dataset_module["eval_dataset"]) == len(tokenized["validation"])
