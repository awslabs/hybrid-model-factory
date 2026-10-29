"""
Tests for the labeling pass of decision data (data/decision_label.py, data/decision_teachers.py):
the anchor, teacher models (causal LMs with each teacher prompt, and decision models), teacher
files, the label set's layout, labeling in slices, joining a label set to the training rows, and
training with a label set. The strands teacher prompt's exact messages, and converting Strands teacher
files, are in tests/decision/test_strands_decider_reproduction.py.

Everything runs offline, with tiny decision datasets, a tiny word-level tokenizer whose option numbers
and letters are single tokens, and a tiny causal LM as the anchor and teacher model. One test needs
two GPUs (labeling on both), and is skipped otherwise.
"""

import json
import math
import os
import shutil

import pytest
import torch
import transformers
import yaml
from datasets import load_from_disk
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerFast

import hmf.model.hybrid_zoo.models.model_register  # noqa: F401  # Registers decision models
from hmf.data import get_dataset, get_template_and_fix_tokenizer
from hmf.data.decision_label import (
    KEY_COLUMNS,
    LABEL_COLUMNS,
    _shard_indices,
    label_shard,
    labels_path,
    merge_shards,
    run_decision_label,
)
from hmf.data.decision_prompt import StrandsPromptFormat, encode_decision_prompt
from hmf.data.decision_teachers import (
    StrandsTeacherPrompt,
    TeacherFile,
    to_augmented_order,
)
from hmf.hparams.parser import _parse_train_args
from hmf.model import load_tokenizer
from hmf.priming.decision_init import DecisionInitConfig, init_decision_model
from hmf.train.tuner import run_exp


NUM_AUGMENTATIONS = 2
WORDS = ["state", "question", "options", "answer", "true", "false", "a", "b", "c"]
LETTERS = list("ABCDEFGHIJKLMNOP")
# A minimal chat template, for the strands teacher prompt
CHAT_TEMPLATE = (
    "{% for m in messages %}{{ m['role'] }} : {{ m['content'] }} \n{% endfor %}"
    "{% if add_generation_prompt %}assistant :{% endif %}"
)


def _save_tiny_model(path, extra_words=()) -> None:
    """
    A tiny causal LM and a word-level tokenizer with a chat template, in which 1 to 9 and A to P
    are single tokens (10 is not).
    """
    vocab = {"[UNK]": 0, "[PAD]": 1}
    for word in [str(n) for n in range(1, 10)] + LETTERS + WORDS + list(extra_words):
        vocab[word] = len(vocab)
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="[UNK]", pad_token="[PAD]")
    tokenizer.chat_template = CHAT_TEMPLATE
    tokenizer.save_pretrained(path)

    torch.manual_seed(0)
    config = transformers.Qwen3Config(
        vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=2,
        num_key_value_heads=1, head_dim=16, tie_word_embeddings=True,
    )
    transformers.Qwen3ForCausalLM(config).save_pretrained(path)


def _row(i: int, kind: str, num_options: int = 3) -> dict:
    if kind == "noul":
        options, label = [["false", None], ["true", None]], i % 2
    elif kind == "choice":
        options, label = [[f"option{k}", f"description {k}"] for k in range(num_options)], i % num_options
    else:
        options, label = [[str(k), f"level {k}"] for k in range(4)], i % 4
    return {"kind": kind, "state": f"state {i}", "instructions": f"question {i}?", "options": options,
            "label": label, "task": "t"}


def _write_jsonl(path, rows) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


@pytest.fixture
def workspace(tmp_path) -> dict:
    """Two tiny decision datasets (toy_a has a 10-option row), a tiny model, and their config."""
    data_dir, model_dir = tmp_path / "data", tmp_path / "model"
    data_dir.mkdir()
    toy_a = [_row(i, ("noul", "choice", "score")[i % 3]) for i in range(9)] + [_row(9, "choice", num_options=10)]
    toy_b = [_row(i, "choice") for i in range(4)]
    _write_jsonl(data_dir / "toy_a.jsonl", toy_a)
    _write_jsonl(data_dir / "toy_b.jsonl", toy_b)
    (data_dir / "dataset_info.json").write_text(json.dumps({
        "toy_a": {"file_name": "toy_a.jsonl", "formatting": "decision"},
        "toy_b": {"file_name": "toy_b.jsonl", "formatting": "decision"},
    }))
    _save_tiny_model(model_dir)
    return {"root": tmp_path, "data_dir": str(data_dir), "model_dir": str(model_dir), "toy_a": toy_a, "toy_b": toy_b}


def _config(workspace: dict, **overrides) -> dict:
    config = {
        "model_name_or_path": workspace["model_dir"],
        "stage": "decision",
        "do_train": True,
        "finetuning_type": "full",
        "dataset": "toy_a,toy_b",
        "dataset_dir": workspace["data_dir"],
        "tokenized_path": str(workspace["root"] / "tokenized"),
        "cutoff_len": 4096,
        "val_size": 0.0,
        "seed": 0,
        "overwrite_cache": True,
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


def _tokenize(workspace: dict, **overrides) -> None:
    """Tokenizes the toy datasets to the config's tokenized_path (without labeling keys)."""
    _get_dataset(_config(workspace, **overrides))


def _label_config(workspace: dict, name: str = "labels_a", **overrides) -> str:
    """Writes a labeling config (the tokenization config plus labeling keys) and returns its path."""
    config = _config(workspace, decision_label_folder=name, **overrides)
    path = workspace["root"] / f"label_{name}.yaml"
    path.write_text(yaml.safe_dump(config))
    return str(path)


def _write_teacher_file(workspace: dict, sources_and_rows, path=None) -> tuple:
    """A teacher file with made-up probabilities for the given (source, row) pairs."""
    path = path or workspace["root"] / "teacher.jsonl"
    entries = {}
    with open(path, "w") as f:
        for source, row_index, row in sources_and_rows:
            num_options = 2 if row["kind"] == "noul" else len(row["options"])
            weights = [k + 1 + row_index for k in range(num_options)]
            probs = [w / sum(weights) for w in weights]
            entries[(source, row_index)] = probs
            f.write(json.dumps({"source": source, "row_index": row_index, "probs": probs}) + "\n")
    return str(path), entries


# Anchor


def test_anchor_matches_the_model(workspace):
    _tokenize(workspace)
    config_path = _label_config(workspace, decision_anchor_model=workspace["model_dir"])

    # On the CPU the anchor runs in float32, as the reference below does (on a GPU it runs in bf16)
    labels = merge_shards(config_path, [label_shard(config_path, device="cpu")])
    tokens = load_from_disk(str(workspace["root"] / "tokenized"))["train"]

    tokenizer = AutoTokenizer.from_pretrained(workspace["model_dir"])
    number_ids = [tokenizer.encode(str(n), add_special_tokens=False)[0] for n in range(1, 10)]
    model = AutoModelForCausalLM.from_pretrained(workspace["model_dir"], dtype=torch.float32).eval()
    for row, label in zip(tokens, labels):
        num_options = len(row["opt_idx"])
        if num_options > 9:  # the 10-option row, whose "10" is not a single token
            assert not label["has_ref"] and label["ref_probs"] == []
            continue
        with torch.no_grad():
            logits = model(input_ids=torch.tensor([row["input_ids"]])).logits[0, row["answer_idx"]]
        # The option shown at position k is numbered k + 1, so the anchor is in augmented order
        expected = torch.softmax(logits[number_ids[:num_options]], dim=-1).tolist()
        assert label["has_ref"]
        assert label["ref_probs"] == pytest.approx(expected, abs=1e-5)


def test_anchor_needs_the_data_tokenizer(workspace):
    _tokenize(workspace)
    other_model = workspace["root"] / "other_model"
    _save_tiny_model(other_model, extra_words=["extra"])  # a different vocabulary
    config_path = _label_config(workspace, decision_anchor_model=str(other_model))

    with pytest.raises(ValueError, match="tokenizer the data was tokenized with"):
        run_decision_label(config_path, num_gpus=1)


# Teacher files


def test_teacher_file(workspace):
    _tokenize(workspace)
    teacher_path, entries = _write_teacher_file(
        workspace,
        [("toy_a.jsonl", i, row) for i, row in enumerate(workspace["toy_a"]) if i < 6]
        + [("toy_b.jsonl", 0, workspace["toy_b"][0])],  # toy_b is not a teacher dataset below
    )
    config_path = _label_config(workspace, decision_teacher_file=teacher_path, decision_teacher_datasets="toy_a")

    labels = load_from_disk(os.path.join(run_decision_label(config_path, num_gpus=1), "train"))

    for label in labels:
        key = (label["source"], label["row_index"])
        if label["source"] == "toy_a.jsonl" and key in entries:
            assert label["has_teacher"]
            # Original order, reordered like the row's options
            assert label["teacher_probs"] == pytest.approx(to_augmented_order(entries[key], label["option_order"]))
        else:
            assert not label["has_teacher"] and label["teacher_probs"] == []
    assert sum(labels["has_teacher"]) == 6 * NUM_AUGMENTATIONS


def test_teacher_file_with_wrong_number_of_probs(workspace):
    _tokenize(workspace)
    path = workspace["root"] / "teacher.jsonl"
    path.write_text(json.dumps({"source": "toy_a.jsonl", "row_index": 1, "probs": [0.5, 0.5]}) + "\n")  # 3 options
    config_path = _label_config(workspace, decision_teacher_file=str(path))

    with pytest.raises(ValueError, match="has 3 options, but its teacher has 2 probabilities"):
        run_decision_label(config_path, num_gpus=1)


@pytest.mark.parametrize(
    "lines, message",
    [
        ([{"source": "s", "row_index": 0}], "missing"),
        ([{"source": "s", "row_index": 0, "probs": [0.5, 0.4]}], "sum to"),
        ([{"source": "s", "row_index": 0, "probs": [0.5, 0.5]}] * 2, "a second entry"),
    ],
)
def test_teacher_file_errors(tmp_path, lines, message):
    path = tmp_path / "teacher.jsonl"
    path.write_text("".join(json.dumps(line) + "\n" for line in lines))
    with pytest.raises(ValueError, match=message):
        TeacherFile(str(path))


def test_to_augmented_order():
    assert to_augmented_order([0.7, 0.2, 0.1], [2, 0, 1]) == [0.1, 0.7, 0.2]


# Teacher models


def _expected_teacher(model_dir, input_ids, read_idx, labels) -> list:
    """A causal LM's distribution over the labels at a prompt's read position, computed directly."""
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float32).eval()
    label_ids = [tokenizer.convert_tokens_to_ids(label) for label in labels]
    with torch.no_grad():
        logits = model(input_ids=torch.tensor([input_ids])).logits[0, read_idx]
    return torch.softmax(logits[label_ids], dim=-1).tolist()


def _rows_by_key(workspace) -> dict:
    return {
        **{("toy_a.jsonl", i): row for i, row in enumerate(workspace["toy_a"])},
        **{("toy_b.jsonl", i): row for i, row in enumerate(workspace["toy_b"])},
    }


def test_teacher_model_with_decision_prompt(workspace):
    _tokenize(workspace)
    config_path = _label_config(
        workspace, decision_teacher_model=workspace["model_dir"], decision_teacher_datasets="toy_a"
    )

    # On the CPU the teacher runs in float32, as the reference does
    labels = merge_shards(config_path, [label_shard(config_path, device="cpu")])

    tokenizer = AutoTokenizer.from_pretrained(workspace["model_dir"])
    originals = _rows_by_key(workspace)
    for label in labels:
        row = originals[(label["source"], label["row_index"])]
        num_options = len(label["option_order"])
        if label["source"] != "toy_a.jsonl" or num_options > 9:  # not a teacher dataset, or "10" is not a token
            assert not label["has_teacher"]
            continue
        # The original row, in original order and with its original phrasing
        options = [tuple(option) for option in row["options"]]
        encoded = encode_decision_prompt(
            tokenizer, StrandsPromptFormat.render(row["state"], row["instructions"], row["kind"], options)
        )
        expected = _expected_teacher(
            workspace["model_dir"], encoded.input_ids, encoded.answer_idx, [str(k + 1) for k in range(num_options)]
        )
        assert label["has_teacher"]
        assert label["teacher_probs"] == pytest.approx(to_augmented_order(expected, label["option_order"]), abs=1e-5)


def test_teacher_model_with_strands_prompt(workspace):
    _tokenize(workspace)
    config_path = _label_config(
        workspace, decision_teacher_model=workspace["model_dir"], decision_teacher_prompt="strands"
    )

    labels = merge_shards(config_path, [label_shard(config_path, device="cpu")])

    tokenizer = AutoTokenizer.from_pretrained(workspace["model_dir"])
    originals = _rows_by_key(workspace)
    for label in labels:
        built = StrandsTeacherPrompt().build(tokenizer, originals[(label["source"], label["row_index"])])
        shown = _expected_teacher(
            workspace["model_dir"], built.query.input_ids, built.query.read_idx, built.query.labels
        )
        original = [0.0] * len(shown)
        for position, index in enumerate(built.original_order):
            original[index] = shown[position]
        assert label["has_teacher"]  # all rows: 10 options is within the 16 letters
        assert label["teacher_probs"] == pytest.approx(to_augmented_order(original, label["option_order"]), abs=1e-5)


def test_teacher_max_len(workspace):
    _tokenize(workspace)
    config_path = _label_config(
        workspace, decision_teacher_model=workspace["model_dir"], decision_teacher_max_len=5  # shorter than all
    )
    labels = merge_shards(config_path, [label_shard(config_path, device="cpu")])
    assert not any(labels["has_teacher"])


@pytest.fixture
def decision_teacher(workspace) -> dict:
    """A tiny decision model converted from the tiny causal LM, and a LoRA adapter trained on it (random)."""
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModel

    decision_dir, adapter_dir = workspace["root"] / "decision", workspace["root"] / "adapter"
    init_decision_model(
        DecisionInitConfig(
            base_model_name_or_path=workspace["model_dir"], output_dir=str(decision_dir), dtype=torch.float32,
            decision_head_dim=16, seed=0,
        )
    )
    torch.manual_seed(1)
    model = AutoModel.from_pretrained(str(decision_dir), dtype=torch.float32)
    lora = LoraConfig(target_modules=["q_proj", "v_proj"], modules_to_save=["decision_head"], init_lora_weights=False)
    get_peft_model(model, lora).save_pretrained(str(adapter_dir))
    return {"model_dir": str(decision_dir), "adapter_dir": str(adapter_dir)}


def _expected_decision_teacher(model_dir, adapter_dir, row) -> list:
    """A decision model's distribution over an original row's options, computed directly."""
    from peft import PeftModel
    from transformers import AutoModel

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModel.from_pretrained(model_dir, dtype=torch.float32)
    if adapter_dir is not None:
        model = PeftModel.from_pretrained(model, adapter_dir).merge_and_unload()
    model.eval()
    options = [tuple(option) for option in row["options"]]
    encoded = encode_decision_prompt(
        tokenizer, StrandsPromptFormat.render(row["state"], row["instructions"], row["kind"], options)
    )
    with torch.no_grad():
        out = model(
            input_ids=torch.tensor([encoded.input_ids]),
            attention_mask=torch.ones(1, len(encoded.input_ids), dtype=torch.long),
            opt_idx=torch.tensor([encoded.opt_idx]),
            answer_idx=torch.tensor([encoded.answer_idx]),
        )
    return out.log_probs[0].exp().tolist()


@pytest.mark.parametrize("with_adapter", [False, True])
def test_decision_model_teacher(workspace, decision_teacher, with_adapter):
    _tokenize(workspace)
    adapter_dir = decision_teacher["adapter_dir"] if with_adapter else None
    overrides = {"decision_teacher_adapter": adapter_dir} if with_adapter else {}
    config_path = _label_config(workspace, decision_teacher_model=decision_teacher["model_dir"], **overrides)

    labels = merge_shards(config_path, [label_shard(config_path, device="cpu")])

    originals = _rows_by_key(workspace)
    for label in labels:
        expected = _expected_decision_teacher(
            decision_teacher["model_dir"], adapter_dir, originals[(label["source"], label["row_index"])]
        )
        assert label["has_teacher"]  # a pointer head reads any number of options
        assert label["teacher_probs"] == pytest.approx(to_augmented_order(expected, label["option_order"]), abs=1e-5)

    if with_adapter:  # the adapter changes the answers
        row = workspace["toy_a"][1]
        without = _expected_decision_teacher(decision_teacher["model_dir"], None, row)
        assert _expected_decision_teacher(decision_teacher["model_dir"], adapter_dir, row) != pytest.approx(without)


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"decision_teacher_model": "x", "decision_teacher_file": "y"}, "Set one teacher"),
        ({"decision_teacher_adapter": "x"}, "decision_teacher_adapter is for decision models"),
        ({"decision_teacher_prompt": "unknown"}, "Unknown decision_teacher_prompt"),
    ],
)
def test_teacher_config_errors(workspace, overrides, message):
    _tokenize(workspace)
    overrides = {"decision_teacher_model": workspace["model_dir"], **overrides}
    with pytest.raises(ValueError, match=message):
        label_shard(_label_config(workspace, **overrides), device="cpu")


# Label sets


def test_label_set_layout(workspace):
    _tokenize(workspace)
    teacher_path, _ = _write_teacher_file(workspace, [("toy_b.jsonl", 1, workspace["toy_b"][1])])
    config_path = _label_config(
        workspace, decision_anchor_model=workspace["model_dir"], decision_teacher_file=teacher_path
    )

    output = run_decision_label(config_path, num_gpus=1)

    tokenized_path = str(workspace["root"] / "tokenized")
    assert output == labels_path(tokenized_path, "labels_a") == os.path.join(tokenized_path, "labels", "labels_a")
    labels = load_from_disk(os.path.join(output, "train"))
    tokens = load_from_disk(tokenized_path)["train"]
    assert set(labels.column_names) == set(KEY_COLUMNS) | set(LABEL_COLUMNS) | {"fingerprint"}
    for column in KEY_COLUMNS:
        assert labels[column] == tokens[column]  # one label row per training row, in the same order

    with open(os.path.join(output, "labels_info.json")) as f:
        info = json.load(f)
    assert info["num_rows"] == len(tokens)
    assert info["num_with_anchor"] == len(tokens) - NUM_AUGMENTATIONS  # all but the 10-option row
    assert info["num_with_teacher"] == NUM_AUGMENTATIONS

    # The tokens are untouched, and an existing label set is not overwritten
    assert sorted(os.listdir(tokenized_path)) == ["dataset_dict.json", "labels", "train"]
    with pytest.raises(ValueError, match="already exists"):
        run_decision_label(config_path, num_gpus=1)


def test_labeling_in_slices_matches_all_at_once(workspace):
    _tokenize(workspace)
    teacher_path, _ = _write_teacher_file(workspace, [("toy_a.jsonl", 2, workspace["toy_a"][2])])
    config_path = _label_config(
        workspace, decision_anchor_model=workspace["model_dir"], decision_teacher_file=teacher_path
    )

    whole = merge_shards(config_path, [label_shard(config_path)])
    sliced = merge_shards(config_path, [label_shard(config_path, 1, 2), label_shard(config_path, 0, 2)])

    assert sliced.column_names == whole.column_names
    for column in whole.column_names:
        if column == "ref_probs":
            for a, b in zip(sliced[column], whole[column]):
                assert a == pytest.approx(b, abs=1e-6)
        else:
            assert sliced[column] == whole[column], column


def test_slices_split_by_original_row(workspace):
    _tokenize(workspace)
    dataset = load_from_disk(str(workspace["root"] / "tokenized"))["train"]
    keys = list(zip(dataset["source"], dataset["row_index"]))

    slices = [_shard_indices(dataset, i, 3) for i in range(3)]
    assert sorted(i for indices in slices for i in indices) == list(range(len(dataset)))  # every row, once
    # A teacher model reads each original row on one GPU only, so all its augmentations are in one slice
    owners = {}
    for shard, indices in enumerate(slices):
        for i in indices:
            assert owners.setdefault(keys[i], shard) == shard


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs two GPUs")
def test_labeling_on_two_gpus(workspace):
    _tokenize(workspace)
    teacher_path, _ = _write_teacher_file(workspace, [("toy_a.jsonl", 2, workspace["toy_a"][2])])
    config_path = _label_config(workspace)  # the labeling keys are given as command-line overrides
    overrides = ["decision_label_folder=on_gpus", f"decision_anchor_model={workspace['model_dir']}",
                 f"decision_teacher_file={teacher_path}"]

    output = run_decision_label(config_path, num_gpus=2, overrides=overrides)

    assert output == labels_path(str(workspace["root"] / "tokenized"), "on_gpus")
    on_gpus = load_from_disk(os.path.join(output, "train"))
    on_cpu = merge_shards(config_path, [label_shard(config_path, device="cpu", overrides=overrides)], overrides)
    for column in on_cpu.column_names:
        if column == "ref_probs":  # bf16 on the GPUs
            for a, b in zip(on_gpus[column], on_cpu[column]):
                assert a == pytest.approx(b, abs=2e-2)
        else:
            assert on_gpus[column] == on_cpu[column], column


def test_nothing_to_label(workspace):
    _tokenize(workspace)
    with pytest.raises(ValueError, match="Nothing to label"):
        run_decision_label(_label_config(workspace), num_gpus=1)


# Joining label sets


def test_join(workspace):
    _tokenize(workspace)
    teacher_path, _ = _write_teacher_file(workspace, [("toy_a.jsonl", 0, workspace["toy_a"][0])])
    run_decision_label(_label_config(workspace, decision_anchor_model=workspace["model_dir"]), num_gpus=1)
    run_decision_label(_label_config(workspace, name="labels_b", decision_teacher_file=teacher_path), num_gpus=1)

    for name in ("labels_a", "labels_b"):  # two label sets of the same tokens
        train = _get_dataset(_config(workspace, decision_label_folder=name))["train_dataset"]
        labels = load_from_disk(os.path.join(labels_path(str(workspace["root"] / "tokenized"), name), "train"))
        assert set(LABEL_COLUMNS) <= set(train.column_names)
        for column in LABEL_COLUMNS:
            assert train[column] == labels[column], (name, column)

    # Without decision_label_folder, the tokens load as before
    assert not set(LABEL_COLUMNS) & set(_get_dataset(_config(workspace))["train_dataset"].column_names)


def test_join_rejects_labels_of_other_tokens(workspace):
    _tokenize(workspace)
    run_decision_label(_label_config(workspace, decision_anchor_model=workspace["model_dir"]), num_gpus=1)
    # Re-tokenize with another seed (other option orders) and bring the old labels along
    other_path = workspace["root"] / "tokenized_seed1"
    _tokenize(workspace, tokenized_path=str(other_path), seed=1)
    shutil.copytree(workspace["root"] / "tokenized" / "labels", other_path / "labels")

    with pytest.raises(ValueError, match="made for other tokens"):
        _get_dataset(_config(workspace, tokenized_path=str(other_path), decision_label_folder="labels_a"))


def test_join_unknown_label_set(workspace):
    _tokenize(workspace)
    run_decision_label(_label_config(workspace, decision_anchor_model=workspace["model_dir"]), num_gpus=1)

    with pytest.raises(ValueError, match=r"No label set 'missing'.*Available: \['labels_a'\]"):
        _get_dataset(_config(workspace, decision_label_folder="missing"))


def test_labels_need_tokenized_data(workspace):
    # decision_label_folder is set, but the data is not tokenized yet, so it must be tokenized and labeled first
    with pytest.raises(ValueError, match="decision_label_folder needs labeled tokenized data"):
        _get_dataset(_config(workspace, decision_label_folder="labels_a"))


# Training with a label set


def test_training_with_a_label_set(workspace, decision_teacher):
    _tokenize(workspace)
    rows = [("toy_a.jsonl", i, row) for i, row in enumerate(workspace["toy_a"])]
    teacher_path, _ = _write_teacher_file(workspace, rows)
    run_decision_label(
        _label_config(workspace, decision_anchor_model=workspace["model_dir"], decision_teacher_file=teacher_path),
        num_gpus=1,
    )
    output_dir = workspace["root"] / "train_output"

    run_exp(
        {
            "model_name_or_path": decision_teacher["model_dir"],
            "stage": "decision",
            "do_train": True,
            "finetuning_type": "lora",
            "lora_target": "q_proj,v_proj",
            "lora_rank": 4,
            "dataset": "toy_a,toy_b",
            "dataset_dir": workspace["data_dir"],
            "tokenized_path": str(workspace["root"] / "tokenized"),
            "decision_label_folder": "labels_a",
            "decision_anchor_kl_weight": 0.3,
            "decision_teacher_kl_weight": 1.0,
            "cutoff_len": 4096,
            "max_steps": 3,
            "per_device_train_batch_size": 8,
            "learning_rate": 1e-3,
            "logging_steps": 1,
            "save_strategy": "no",
            "output_dir": str(output_dir),
            "overwrite_output_dir": True,
            "report_to": "none",
            "bf16": torch.cuda.is_available(),
            "seed": 0,
        }
    )

    # Every step logs the cross-entropy, and the KL terms of the batches that have their targets
    with open(output_dir / "trainer_state.json") as f:
        logs = [entry for entry in json.load(f)["log_history"] if "loss" in entry]
    assert len(logs) == 3
    for entry in logs:
        assert "ce_loss" in entry
        assert not any(isinstance(value, float) and math.isnan(value) for value in entry.values())
    assert any("anchor_kl" in entry for entry in logs) and any("teacher_kl" in entry for entry in logs)


# Tokenizing and labeling in one command (hmf preprocess)


def _preprocess(config_path) -> int:
    # Imported here, because importing it hides the GPUs from this process (see its CUDA_VISIBLE_DEVICES)
    from hmf.data.sharded_preprocess_and_tokenize import run_preprocess

    return run_preprocess(config_path, shard_size=1000, parallel_jobs=2)


def test_preprocess_labels_after_tokenizing(workspace):
    teacher_path, _ = _write_teacher_file(workspace, [("toy_a.jsonl", 2, workspace["toy_a"][2])])
    config_path = _label_config(
        workspace, val_size=0.2, decision_anchor_model=workspace["model_dir"], decision_teacher_file=teacher_path
    )

    assert _preprocess(config_path) == 0

    tokenized_path = str(workspace["root"] / "tokenized")
    assert set(load_from_disk(tokenized_path)) == {"train", "validation"}
    labels = load_from_disk(os.path.join(labels_path(tokenized_path, "labels_a"), "train"))
    # The same labels as labeling directly (on the GPUs, if there are any, the anchor runs in bf16)
    expected = merge_shards(config_path, [label_shard(config_path, device="cpu")])
    for column in expected.column_names:
        if column == "ref_probs":
            for a, b in zip(labels[column], expected[column]):
                assert a == pytest.approx(b, abs=2e-2)
        else:
            assert labels[column] == expected[column], column

    # Training joins them
    train = _get_dataset(_config(workspace, decision_label_folder="labels_a"))["train_dataset"]
    assert train["has_teacher"] == labels["has_teacher"]


def test_preprocess_without_labeling_keys(workspace):
    config_path = workspace["root"] / "tokenize.yaml"
    config_path.write_text(yaml.safe_dump(_config(workspace, val_size=0.2)))

    assert _preprocess(str(config_path)) == 0

    assert not os.path.exists(workspace["root"] / "tokenized" / "labels")


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({}, "Nothing to label"),
        ({"decision_anchor_model": "x", "decision_teacher_datasets": "not_a_dataset"}, "not_a_dataset"),
        ({"decision_teacher_file": "missing.jsonl"}, "does not exist"),
    ],
)
def test_preprocess_checks_labeling_keys_first(workspace, overrides, message):
    config_path = _label_config(workspace, val_size=0.2, **overrides)

    with pytest.raises(ValueError, match=message):
        _preprocess(config_path)

    assert not os.path.exists(workspace["root"] / "tokenized")  # before tokenizing


def test_preprocess_again_skips_what_is_done(workspace, capsys):
    config_path = _label_config(workspace, val_size=0.2, decision_anchor_model=workspace["model_dir"])
    assert _preprocess(config_path) == 0
    info_path = os.path.join(labels_path(str(workspace["root"] / "tokenized"), "labels_a"), "labels_info.json")
    written = os.path.getmtime(info_path)
    capsys.readouterr()

    assert _preprocess(config_path) == 0

    output = capsys.readouterr().out
    assert "Already complete" in output and "Already labeled" in output
    assert os.path.getmtime(info_path) == written


def test_interrupted_label_set(workspace):
    _tokenize(workspace)
    tokenized_path = str(workspace["root"] / "tokenized")
    # What an interrupted run leaves, which is the labels without labels_info.json, written last
    os.makedirs(os.path.join(labels_path(tokenized_path, "labels_a"), "train"))

    with pytest.raises(ValueError, match=r"No label set 'labels_a'.*Available: none"):
        _get_dataset(_config(workspace, decision_label_folder="labels_a"))  # not joined

    # Labeling again replaces it
    run_decision_label(_label_config(workspace, decision_anchor_model=workspace["model_dir"]), num_gpus=1)
    assert "has_ref" in _get_dataset(_config(workspace, decision_label_folder="labels_a"))["train_dataset"].column_names
