"""
Tests that HMF can reproduce Strands Decider's results (https://github.com/strands-labs/strands-decider,
at commit b81d443, the version of its v19 recipe). HMF's decision models are not tied to Strands
Decider, but one of the setups they support matches it exactly, with the strands prompt format, the strands
teacher prompt, its teacher files, its split of a file into calibration and test halves, and its
checkpoint format. These tests check that setup against that version; they do not follow later changes
to Strands Decider.

The expected values were produced by Strands Decider's own code, which cannot be imported here (it
needs a newer transformers than HMF). Everything runs offline, on the CPU, with tiny models and data,
except the opt-in test at the end (see there).
"""

import importlib.util
import json
import os
from pathlib import Path

import pytest
import torch
import transformers
from peft import LoraConfig, PeftModel, get_peft_model
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizerFast

import hmf.model.hybrid_zoo.models.model_register  # noqa: F401  # Registers decision models
from hmf.data.decision_prompt import StrandsPromptFormat, encode_decision_prompt
from hmf.data.decision_teachers import CausalLMScorer, ModelTeacher, StrandsTeacherPrompt
from hmf.eval.decision.data import partition_rows
from hmf.model.hybrid_zoo.models.decision.modeling_decision import PointerHead
from hmf.priming.decision_init import DecisionInitConfig, init_decision_model


TRAINING_DIR = Path(__file__).resolve().parents[2]


def _script(name: str):
    """A script of training/scripts, imported."""
    path = TRAINING_DIR / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_jsonl(path, rows) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


# The prompt


def test_strands_noul():
    rendered = StrandsPromptFormat.render(
        "  The meeting moved. ", "Is it Tuesday?", "noul", [("false", None), ("true", "it is")], option_order=[1, 0]
    )

    assert rendered.text == (
        "<state>\nThe meeting moved.\n</state>\n"
        '<question type="noul">\nDecide whether the statement is true of the state.\nIs it Tuesday?\n<options>\n'
        "1. true — it is\n2. false\n</options>\n</question>\n<answer>"
    )


def test_strands_noul_default_descriptions():
    # Options that are not given show their default description; both options are always shown
    rendered = StrandsPromptFormat.render("s", "q", "noul", [("true", "custom")])

    assert "1. false — the statement does not hold for this state\n2. true — custom\n" in rendered.text


def test_strands_choice():
    rendered = StrandsPromptFormat.render(
        {"a": 1}, "Pick", "choice", [("x", "first\n option"), ("y", None), ("z", {"k": "v"})]
    )

    assert rendered.text == (
        '<state>\n{\n  "a": 1\n}\n</state>\n'
        '<question type="choice">\nSelect exactly one option.\nPick\n<options>\n'
        '1. x — first option\n2. y\n3. z — { "k": "v" }\n</options>\n</question>\n<answer>'
    )


def test_strands_score_reversed():
    rendered = StrandsPromptFormat.render(
        "s", "Rate", "score", [("0", "bad"), ("1", "ok"), ("2", "good")], option_order=[2, 1, 0]
    )

    assert rendered.text == (
        "<state>\ns\n</state>\n"
        '<question type="score">\nRate the state against the ordered levels below (lowest first).\nRate\n'
        "<options>\n1. 2 — good\n2. 1 — ok\n3. 0 — bad\n</options>\n</question>\n<answer>"
    )


# The teacher prompt


class _StubChatTokenizer:
    """Records the messages it is given; the prompt's text is not needed to check them."""

    chat_template = "stub"

    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return "the prompt"

    def encode(self, text, add_special_tokens=True):
        assert text == "the prompt" and not add_special_tokens
        return [5, 6, 7]


@pytest.mark.parametrize(
    "row, user_message, original_order",
    [
        (  # noul, shown true first, where a missing description gets Strands' default
            {"kind": "noul", "state": "The meeting moved to Thursday.", "instructions": "The meeting is on Tuesday.",
             "options": [["false", None], ["true", "the meeting is on Tuesday"]]},
            '{"evidence": "The meeting moved to Thursday.", "criterion": "The meeting is on Tuesday.", "options": '
            '[{"letter": "A", "description": "true: the meeting is on Tuesday"}, '
            '{"letter": "B", "description": "false: The proposition is false."}]}',
            [1, 0],
        ),
        (  # choice, where an option without a description is described by its name
            {"kind": "choice", "state": "Charged twice.", "instructions": "What next?",
             "options": [["refund", "return the payment"], ["ignore", None]]},
            '{"evidence": "Charged twice.", "criterion": "What next?", "options": '
            '[{"letter": "A", "description": "refund: return the payment"}, '
            '{"letter": "B", "description": "ignore: ignore"}]}',
            [0, 1],
        ),
        (  # score, with levels named by their index
            {"kind": "score", "state": "An answer.", "instructions": "How good is it?",
             "options": [["0", "poor"], ["1", "good"]]},
            '{"evidence": "An answer.", "criterion": "How good is it?", "options": '
            '[{"letter": "A", "description": "0: poor"}, {"letter": "B", "description": "1: good"}]}',
            [0, 1],
        ),
        (  # a JSON-record state stays nested JSON, and non-ASCII text is kept
            {"kind": "noul", "state": {"note": "café", "amount_due": 0}, "instructions": "It is paid.",
             "options": None},
            '{"evidence": {"note": "café", "amount_due": 0}, "criterion": "It is paid.", "options": '
            '[{"letter": "A", "description": "true: The proposition is true."}, '
            '{"letter": "B", "description": "false: The proposition is false."}]}',
            [1, 0],
        ),
    ],
)
def test_strands_teacher_prompt(row, user_message, original_order):
    tokenizer = _StubChatTokenizer()
    built = StrandsTeacherPrompt().build(tokenizer, row)

    [(messages, kwargs)] = tokenizer.calls
    assert messages == [
        {"role": "system", "content": StrandsTeacherPrompt.SYSTEM},
        {"role": "user", "content": user_message},
    ]
    assert kwargs == {"tokenize": False, "add_generation_prompt": True, "enable_thinking": False}
    assert built.query.input_ids == [5, 6, 7] and built.query.read_idx == 2  # read at the last token
    assert built.query.labels == ["A", "B"]
    assert built.original_order == original_order


def test_strands_teacher_prompt_limits():
    row = {"kind": "choice", "state": "s", "instructions": "q", "options": [[f"o{k}", None] for k in range(17)]}
    assert StrandsTeacherPrompt().build(_StubChatTokenizer(), row) is None  # 17 options, 16 letters

    no_template = _StubChatTokenizer()
    no_template.chat_template = None
    with pytest.raises(ValueError, match="needs a teacher with a chat template"):
        StrandsTeacherPrompt().build(no_template, {**row, "options": [["a", None], ["b", None]]})


# Teacher files (scripts/strands_teacher_to_hmf.py)


@pytest.fixture
def workspace(tmp_path) -> dict:
    """Two registered decision datasets, toy_a (10 rows) and toy_b (4 rows)."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    row = {"kind": "noul", "state": "s", "instructions": "q", "options": [["false", None], ["true", None]], "label": 0}
    _write_jsonl(data_dir / "toy_a.jsonl", [row] * 10)
    _write_jsonl(data_dir / "toy_b.jsonl", [row] * 4)
    (data_dir / "dataset_info.json").write_text(json.dumps({
        "toy_a": {"file_name": "toy_a.jsonl", "formatting": "decision"},
        "toy_b": {"file_name": "toy_b.jsonl", "formatting": "decision"},
    }))
    return {"root": tmp_path, "data_dir": str(data_dir)}


def test_convert_strands_teacher_file(workspace):
    # Strands' i counts the rows of toy_a (10 rows), then toy_b
    strands_path = workspace["root"] / "strands_teacher.jsonl"
    strands_path.write_text(
        json.dumps({"i": 3, "probs": [0.6, 0.4]}) + "\n" + json.dumps({"i": 11, "probs": [0.2, 0.3, 0.5]}) + "\n"
    )
    output = workspace["root"] / "teacher.hmf.jsonl"

    converter = _script("strands_teacher_to_hmf")
    converter.convert(str(strands_path), "toy_a,toy_b", str(output), dataset_dir=workspace["data_dir"])

    lines = [json.loads(line) for line in output.read_text().splitlines()]
    assert lines == [
        {"source": "toy_a.jsonl", "row_index": 3, "probs": [0.6, 0.4]},
        {"source": "toy_b.jsonl", "row_index": 1, "probs": [0.2, 0.3, 0.5]},
    ]

    strands_path.write_text(json.dumps({"i": 14, "probs": [1.0]}) + "\n")  # beyond the 14 rows
    with pytest.raises(ValueError, match="beyond the 14 rows"):
        converter.convert(str(strands_path), "toy_a,toy_b", str(output), dataset_dir=workspace["data_dir"])


# The calibration and test halves


ROWS = [{"i": i} for i in range(20)]
# Strands Decider's halves of a 20-row file (seed 0). A row is in the calibration half when the first
# byte of sha256("0:<position>") is even
CALIB = [0, 3, 4, 5, 9, 10, 11, 12, 13, 15]


def test_the_halves_are_strands_halves():
    assert [row["i"] for row in partition_rows(ROWS, "calib")] == CALIB
    assert [row["i"] for row in partition_rows(ROWS, "test")] == [i for i in range(20) if i not in CALIB]
    assert partition_rows(ROWS, "all") == ROWS


# The checkpoint (scripts/hmf_decision_to_strands.py)


HEAD_DIM = 16
WORDS = ["state", "question", "options", "answer", "true", "false", "refund", "ignore", "1", "2"]


@pytest.fixture
def trained(tmp_path) -> dict:
    """A tiny causal LM, the decision model converted from it, and a LoRA adapter with a trained head."""
    source_dir, decision_dir, adapter_dir = tmp_path / "source", tmp_path / "decision", tmp_path / "adapter"
    vocab = {"[UNK]": 0, "[PAD]": 1, **{word: i + 2 for i, word in enumerate(WORDS)}}
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="[UNK]", pad_token="[PAD]").save_pretrained(
        source_dir
    )
    torch.manual_seed(0)
    config = transformers.Qwen3Config(
        vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=2,
        num_key_value_heads=1, head_dim=16, tie_word_embeddings=True,
    )
    transformers.Qwen3ForCausalLM(config).save_pretrained(source_dir)
    init_decision_model(
        DecisionInitConfig(
            base_model_name_or_path=str(source_dir), output_dir=str(decision_dir), dtype=torch.float32,
            decision_head_dim=HEAD_DIM, seed=0,
        )
    )

    # A LoRA adapter as training saves it, with random LoRA weights and a trained copy of the head
    torch.manual_seed(1)
    model = AutoModel.from_pretrained(str(decision_dir), dtype=torch.float32)
    lora = LoraConfig(target_modules=["q_proj", "v_proj"], modules_to_save=["decision_head"], init_lora_weights=False)
    peft_model = get_peft_model(model, lora)
    trained_head = peft_model.base_model.model.decision_head.modules_to_save["default"]
    with torch.no_grad():
        for param in trained_head.parameters():
            param.add_(torch.randn_like(param))
    peft_model.save_pretrained(str(adapter_dir))
    return {
        "source_dir": str(source_dir),
        "decision_dir": str(decision_dir),
        "adapter_dir": str(adapter_dir),
        "head": {name: value.detach().clone() for name, value in trained_head.state_dict().items()},
    }


def _inputs(tokenizer) -> tuple:
    row = {"kind": "choice", "state": "state refund", "instructions": "question ?",
           "options": [("refund", "refund"), ("ignore", None), ("true", "false")]}
    encoded = encode_decision_prompt(
        tokenizer, StrandsPromptFormat.render(row["state"], row["instructions"], row["kind"], row["options"])
    )
    return (
        torch.tensor([encoded.input_ids]),
        torch.tensor([encoded.opt_idx]),
        torch.tensor([encoded.answer_idx]),
    )


def test_converted_checkpoint_gives_the_same_logits(trained, tmp_path):
    output_dir = tmp_path / "strands"

    _script("hmf_decision_to_strands").convert(trained["decision_dir"], trained["adapter_dir"], str(output_dir))

    assert sorted(os.listdir(output_dir / "lora")) == ["adapter_config.json", "adapter_model.safetensors"]
    with open(output_dir / "strands_decider_config.json") as f:
        config = json.load(f)
    assert config["head_type"] == "pointer" and config["pointer_dim"] == HEAD_DIM and config["use_lora"]
    assert config["base_model"] == trained["source_dir"]  # recorded by hmf decision-init
    assert config["lora_targets"] == ["q_proj", "v_proj"] and config["temperature_by_kind"] == {}

    # The head is the trained copy, in float32, in safetensors (no pickle)
    head_state = load_file(output_dir / "head.safetensors")
    assert set(head_state) == set(trained["head"])
    for name, value in trained["head"].items():
        assert head_state[name].dtype == torch.float32
        torch.testing.assert_close(head_state[name], value)

    # Loaded as Strands Decider loads it, with the bare decoder of the base model, then lora/, then the head
    decoder = PeftModel.from_pretrained(
        AutoModel.from_pretrained(trained["source_dir"], dtype=torch.float32), str(output_dir / "lora")
    ).eval()
    head = PointerHead(decoder.config.hidden_size, dim=HEAD_DIM)
    head.load_state_dict(head_state)
    head.eval()

    # Our model, the decision model with the adapter
    ours = PeftModel.from_pretrained(
        AutoModel.from_pretrained(trained["decision_dir"], dtype=torch.float32), trained["adapter_dir"]
    ).eval()

    input_ids, opt_idx, answer_idx = _inputs(AutoTokenizer.from_pretrained(output_dir))
    with torch.no_grad():
        hidden = decoder(input_ids=input_ids).last_hidden_state
        option_mask = torch.ones_like(opt_idx, dtype=torch.bool)
        theirs = head(hidden[0, answer_idx], hidden[0, opt_idx[0]].unsqueeze(0), option_mask)
        expected = ours(
            input_ids=input_ids, attention_mask=torch.ones_like(input_ids), opt_idx=opt_idx, answer_idx=answer_idx
        ).logits
    torch.testing.assert_close(theirs, expected, rtol=1e-5, atol=1e-5)


def test_convert_refuses(trained, tmp_path):
    converter = _script("hmf_decision_to_strands")
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(ValueError, match="already exists"):
        converter.convert(trained["decision_dir"], trained["adapter_dir"], str(existing))

    # A causal LM, not a decision model
    with pytest.raises(ValueError, match="is not a decision model"):
        converter.convert(trained["source_dir"], trained["adapter_dir"], str(tmp_path / "out"))

    # A check on a data file that does not exist, refused before converting
    with pytest.raises(ValueError, match="does not exist"):
        converter.convert(
            trained["decision_dir"], trained["adapter_dir"], str(tmp_path / "out2"), check_data="missing.jsonl"
        )
    assert not os.path.exists(tmp_path / "out2")


def test_check_folder(trained, tmp_path):
    rows = [
        {"kind": "choice", "state": "state refund", "instructions": "question ?", "label": 0,
         "options": [["refund", "refund"], ["ignore", None], ["true", "false"]]},
        {"kind": "noul", "state": "state", "instructions": "question ?", "label": 1,
         "options": [["false", None], ["true", None]]},
    ]
    data_path = tmp_path / "rows.jsonl"
    data_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    output_dir = tmp_path / "strands"

    _script("hmf_decision_to_strands").convert(
        trained["decision_dir"], trained["adapter_dir"], str(output_dir), check_data=str(data_path)
    )

    check_dir = tmp_path / "strands.check"
    assert sorted(os.listdir(check_dir)) == ["hmf_probs.json", "meta.json", "rows.jsonl", "strands_parity_check.py"]
    with open(check_dir / "meta.json") as f:
        assert json.load(f)["checkpoint"] == str(output_dir)
    assert [json.loads(line) for line in (check_dir / "rows.jsonl").read_text().splitlines()] == rows
    with open(check_dir / "hmf_probs.json") as f:
        ours = json.load(f)
    assert [len(probs) for probs in ours] == [3, 2] and all(sum(p) == pytest.approx(1.0) for p in ours)


def test_parity_comparison():
    check = _script("strands_parity_check")
    ours = [[0.7, 0.2, 0.1], [0.4, 0.6], None]  # None for a row the decision model did not check
    close = [[0.705, 0.195, 0.1], [0.41, 0.59], [0.5, 0.5]]

    result = check.compare(ours, close)
    assert result["num_rows"] == 2 and result["agreement"] == 1.0
    assert result["max_diff"] == pytest.approx(0.01)
    assert check.passes(result)

    # Another chosen option on one of two rows
    assert not check.passes(check.compare(ours, [[0.7, 0.2, 0.1], [0.6, 0.4], [0.5, 0.5]]))


# Converting back (scripts/strands_to_hmf_decision.py)


def _strands_checkpoint(trained, tmp_path, **settings) -> Path:
    """The trained model converted to a Strands Decider checkpoint, with some of its settings changed."""
    path = tmp_path / "strands"
    _script("hmf_decision_to_strands").convert(trained["decision_dir"], trained["adapter_dir"], str(path))
    config_path = path / "strands_decider_config.json"
    config = json.loads(config_path.read_text())
    config.update({"torch_dtype": "float32", **settings})  # The tiny models are in float32
    config_path.write_text(json.dumps(config))
    return path


def _decision_logits(model_dir, adapter_dir=None) -> torch.Tensor:
    """A decision model's logits on the row of _inputs, with its adapter if it has one."""
    model = AutoModel.from_pretrained(str(model_dir), dtype=torch.float32)
    if adapter_dir is not None:
        model = PeftModel.from_pretrained(model, str(adapter_dir))
    model.eval()
    input_ids, opt_idx, answer_idx = _inputs(AutoTokenizer.from_pretrained(str(model_dir)))
    with torch.no_grad():
        return model(
            input_ids=input_ids, attention_mask=torch.ones_like(input_ids), opt_idx=opt_idx, answer_idx=answer_idx
        ).logits


def _safetensors(path) -> dict:
    with safe_open(str(path), framework="pt") as f:
        return {key: f.get_tensor(key) for key in f.keys()}


def test_strands_checkpoint_converts_to_an_adapter(trained, tmp_path):
    output_dir = tmp_path / "converted"

    _script("strands_to_hmf_decision").convert(str(_strands_checkpoint(trained, tmp_path)), str(output_dir))

    assert sorted(os.listdir(output_dir)) == ["adapter", "model"]
    assert not (output_dir / "adapter" / "decision_calibration.json").exists()  # The checkpoint is uncalibrated
    with open(output_dir / "adapter" / "adapter_config.json") as f:
        adapter_config = json.load(f)
    assert adapter_config["base_model_name_or_path"] == str(output_dir / "model")
    assert adapter_config["modules_to_save"] == ["decision_head"] and adapter_config["task_type"] is None
    assert sorted(adapter_config["target_modules"]) == ["q_proj", "v_proj"]

    # The adapter is the one HMF trained, weight for weight, and so is the model
    converted = _safetensors(output_dir / "adapter" / "adapter_model.safetensors")
    original = _safetensors(Path(trained["adapter_dir"]) / "adapter_model.safetensors")
    assert set(converted) == set(original)
    for key, value in original.items():
        torch.testing.assert_close(converted[key], value, rtol=0, atol=0)
    torch.testing.assert_close(
        _decision_logits(output_dir / "model", output_dir / "adapter"),
        _decision_logits(trained["decision_dir"], trained["adapter_dir"]),
    )


def test_strands_checkpoint_converts_to_a_merged_model(trained, tmp_path):
    output_dir = tmp_path / "merged"

    checkpoint = _strands_checkpoint(trained, tmp_path)
    _script("strands_to_hmf_decision").convert(str(checkpoint), str(output_dir), merge=True)

    weights = {}
    for path in output_dir.glob("*.safetensors"):
        with safe_open(str(path), framework="pt") as f:
            weights.update({key: f.get_slice(key).get_dtype() for key in f.keys()})
    assert not any("lora_" in key or "modules_to_save" in key for key in weights)
    assert {dtype for key, dtype in weights.items() if key.startswith("decision_head.")} == {"F32"}
    assert not (output_dir / "adapter_config.json").exists()
    assert not [name for name in os.listdir(tmp_path) if name.startswith(".strands_to_hmf_decision-")]  # Steps deleted
    torch.testing.assert_close(
        _decision_logits(output_dir), _decision_logits(trained["decision_dir"], trained["adapter_dir"]),
        rtol=1e-5, atol=1e-5,
    )


def test_strands_checkpoint_onto_an_existing_decision_model(trained, tmp_path):
    output_dir = tmp_path / "converted"

    _script("strands_to_hmf_decision").convert(
        str(_strands_checkpoint(trained, tmp_path)), str(output_dir), decision_model=trained["decision_dir"]
    )

    assert os.listdir(output_dir) == ["adapter"]
    with open(output_dir / "adapter" / "adapter_config.json") as f:
        assert json.load(f)["base_model_name_or_path"] == trained["decision_dir"]
    torch.testing.assert_close(
        _decision_logits(trained["decision_dir"], output_dir / "adapter"),
        _decision_logits(trained["decision_dir"], trained["adapter_dir"]),
    )


def test_strands_calibration_is_kept(trained, tmp_path):
    checkpoint = _strands_checkpoint(trained, tmp_path, temperature=1.3, temperature_by_kind={"noul": 0.8})
    converter = _script("strands_to_hmf_decision")

    converter.convert(str(checkpoint), str(tmp_path / "converted"))
    converter.convert(str(checkpoint), str(tmp_path / "merged"), merge=True)

    for path in (tmp_path / "converted" / "adapter", tmp_path / "merged"):
        with open(path / "decision_calibration.json") as f:
            calibration = json.load(f)
        assert calibration["temperature"] == 1.3 and calibration["temperature_by_kind"] == {"noul": 0.8}
        assert calibration["ordinal_smoothing"] == 0.1


def test_older_strands_checkpoints(trained, tmp_path):
    # Checkpoints saved before Strands Decider's rename, with the head pickled as slot_head.pt
    checkpoint = _strands_checkpoint(trained, tmp_path)
    (checkpoint / "strands_decider_config.json").rename(checkpoint / "hobson_config.json")
    torch.save(load_file(checkpoint / "head.safetensors"), checkpoint / "slot_head.pt")
    (checkpoint / "head.safetensors").unlink()
    output_dir = tmp_path / "converted"

    _script("strands_to_hmf_decision").convert(str(checkpoint), str(output_dir))

    torch.testing.assert_close(
        _decision_logits(output_dir / "model", output_dir / "adapter"),
        _decision_logits(trained["decision_dir"], trained["adapter_dir"]),
    )


def test_strands_conversion_refuses(trained, tmp_path):
    converter = _script("strands_to_hmf_decision")
    checkpoint = _strands_checkpoint(trained, tmp_path)
    config_path = checkpoint / "strands_decider_config.json"
    config = json.loads(config_path.read_text())

    def convert_with(match, **settings):
        config_path.write_text(json.dumps({**config, **settings}))
        with pytest.raises(ValueError, match=match):
            converter.convert(str(checkpoint), str(tmp_path / "out"))
        config_path.write_text(json.dumps(config))
        assert not (tmp_path / "out").exists()

    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(ValueError, match="already exists"):
        converter.convert(str(checkpoint), str(existing))
    with pytest.raises(ValueError, match="not a Strands Decider checkpoint"):
        converter.convert(trained["decision_dir"], str(tmp_path / "out"))

    convert_with("only pointer heads", head_type="slot")
    convert_with("only LoRA checkpoints", use_lora=False)
    convert_with("only LoRA checkpoints", full_weight_targets=["q_proj"])
    convert_with("the head's width is 16", pointer_dim=32)

    # A Gemma 4 base model, refused from its config.json alone
    gemma = tmp_path / "gemma"
    gemma.mkdir()
    (gemma / "config.json").write_text(json.dumps({"model_type": "gemma4"}))
    convert_with("cannot convert yet", base_model=str(gemma))

    # An existing decision model whose head has another width
    other_dir = tmp_path / "other_decision"
    init_decision_model(
        DecisionInitConfig(
            base_model_name_or_path=trained["source_dir"], output_dir=str(other_dir), dtype=torch.float32,
            decision_head_dim=2 * HEAD_DIM,
        )
    )
    with pytest.raises(ValueError, match="differs from the checkpoint's"):
        converter.convert(str(checkpoint), str(tmp_path / "out"), decision_model=str(other_dir))


def test_strands_conversion_refuses_unknown_lora_weights(trained, tmp_path):
    # A LoRA weight on a module the decision model does not have
    checkpoint = _strands_checkpoint(trained, tmp_path)
    lora_path = checkpoint / "lora" / "adapter_model.safetensors"
    weights = load_file(lora_path)
    key = next(iter(weights))
    weights[key.replace("layers.0", "layers.7")] = weights[key].clone()
    save_file(weights, lora_path)

    with pytest.raises(ValueError, match="do not match the modules"):
        _script("strands_to_hmf_decision").convert(str(checkpoint), str(tmp_path / "out"))


def test_strands_conversion_check_folder(trained, tmp_path):
    rows = [{"kind": "choice", "state": "state refund", "instructions": "question ?", "label": 0,
             "options": [["refund", "refund"], ["ignore", None], ["true", "false"]]}]
    data_path = tmp_path / "rows.jsonl"
    data_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    checkpoint = _strands_checkpoint(trained, tmp_path)

    _script("strands_to_hmf_decision").convert(str(checkpoint), str(tmp_path / "converted"), check_data=str(data_path))

    check_dir = tmp_path / "converted.check"
    assert sorted(os.listdir(check_dir)) == ["hmf_probs.json", "meta.json", "rows.jsonl", "strands_parity_check.py"]
    with open(check_dir / "meta.json") as f:
        assert json.load(f)["checkpoint"] == str(checkpoint)


# Opt-in test with a real teacher model
#
# This test needs a GPU and HMF_STRANDS_DATA_DIR, Strands Decider's data folder, and is skipped otherwise:
#
#     HMF_STRANDS_DATA_DIR=path/to/strands-decider/data \
#         pytest tests/decision/test_strands_decider_reproduction.py -v -rs


@pytest.mark.skipif(
    not os.environ.get("HMF_STRANDS_DATA_DIR") or not torch.cuda.is_available(),
    reason="needs a GPU and HMF_STRANDS_DATA_DIR (Strands Decider's data folder)",
)
def test_reproduces_strands_teacher():
    """The strands prompt with Qwen3.5-4B gives Strands Decider's own teacher file (up to bf16 noise)."""
    data_dir = Path(os.environ["HMF_STRANDS_DATA_DIR"])
    with open(data_dir / "multistep_v14.jsonl", encoding="utf-8") as f:
        rows = [json.loads(line) for line, _ in zip((line for line in f if line.strip()), range(256))]
    with open(data_dir / "synthetic" / "teacher_multistep_v14.jsonl", encoding="utf-8") as f:
        reference = {entry["i"]: entry["probs"] for entry in map(json.loads, f) if entry["i"] < len(rows)}

    scorer = CausalLMScorer("Qwen/Qwen3.5-4B", "cuda")
    found = ModelTeacher(scorer, StrandsTeacherPrompt(), max_len=4096)(rows)

    # Strands labels the same rows, with at most 16 options and prompts of at most 4,096 tokens
    assert {i for i, probs in enumerate(found) if probs is not None} == set(reference)
    diffs = [abs(p - q) for i, probs in reference.items() for p, q in zip(found[i], probs)]
    agree = sum(
        max(range(len(probs)), key=probs.__getitem__) == max(range(len(probs)), key=found[i].__getitem__)
        for i, probs in reference.items()
    ) / len(reference)
    assert sum(diffs) / len(diffs) < 0.01 and agree > 0.98, (sum(diffs) / len(diffs), max(diffs), agree)
