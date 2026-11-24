"""
Tests for the label heads of decision models (LabelHead and LabelPointerHead in modeling_decision.py), the letter
labels they read (data/decision_prompt.py), and converting a causal LM into a decision model with one (hmf
decision-init with decision_head_type label, label_trained, or label_pointer).

The conversion tests build tiny random causal LMs and a tiny word-level tokenizer that has the letters A to Z and
two two-letter codes, so they need no network access. They run in float32 on the GPU if there is one (Qwen3.5's
Gated DeltaNet kernels need it when flash-linear-attention is installed), else on the CPU. An untrained label head
must answer exactly as its source causal LM does with its next-token logits of the options' labels (CausalLMScorer).
"""

import pytest
import torch
import transformers
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizerFast

import hmf.model.hybrid_zoo.models.model_register  # noqa: F401  # Registers hybrid and decision models
from hmf.data.decision_prompt import OptionLabels, StrandsPromptFormat, letter_codes
from hmf.data.decision_teachers import CausalLMScorer, DecisionModelTeacher, DecisionTeacherPrompt
from hmf.eval.decision.predictor import DecisionPredictor
from hmf.model.hybrid_zoo.models.decision.configuration_decision import DecisionConfig
from hmf.model.hybrid_zoo.models.decision.modeling_decision import (
    MASK_VALUE,
    DecisionModel,
    LabelHead,
    LabelPointerHead,
    PointerHead,
)
from hmf.priming.decision_init import DecisionInitConfig, init_decision_model


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# The untrained head against its source LM: exact up to the order of float32 sums, which differs between kernels
TOLERANCE = {"rtol": 1e-4, "atol": 1e-5}
HIDDEN_SIZE = 64
VOCAB_SIZE = 128
HEAD_DIM = 16
SEED = 0
LETTERS = [chr(ord("A") + i) for i in range(26)]
# The tiny tokenizer's two-letter codes, as every other two-letter code is unknown to it
TWO_LETTER_CODES = ["AA", "AB"]


def _tiny_tokenizer() -> PreTrainedTokenizerFast:
    words = ["[UNK]", "[PAD]", *[f"tok{i}" for i in range(30)], *LETTERS, *TWO_LETTER_CODES]
    tokenizer = Tokenizer(models.WordLevel({word: i for i, word in enumerate(words)}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    return PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="[UNK]", pad_token="[PAD]")


# Letter labels


def test_letter_codes_are_the_letters_then_the_single_token_two_letter_codes():
    assert letter_codes(_tiny_tokenizer()) == LETTERS + TWO_LETTER_CODES


def test_letter_codes_need_every_letter():
    vocab = {"[UNK]": 0, **{letter: i + 1 for i, letter in enumerate(LETTERS[:-1])}}
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    with pytest.raises(ValueError, match="A to Z"):
        letter_codes(PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="[UNK]"))


def test_option_labels():
    assert OptionLabels("numbers")(3) == ["1", "2", "3"]
    labels = OptionLabels("letters", _tiny_tokenizer())
    assert labels(3) == ["A", "B", "C"]
    assert labels(28) == LETTERS + TWO_LETTER_CODES
    with pytest.raises(ValueError, match="more than the 28 letter codes"):
        labels(29)
    with pytest.raises(ValueError, match="needs one"):
        OptionLabels("letters")
    with pytest.raises(ValueError, match="Unknown decision_option_labels"):
        OptionLabels("roman")


def test_render_with_letter_labels():
    options = [("refund", "return the payment"), ("ignore", None), ("escalate", "send it to a human")]
    numbered = StrandsPromptFormat.render("the state", "what now?", "choice", options, option_order=[2, 0, 1])
    lettered = StrandsPromptFormat.render(
        "the state", "what now?", "choice", options, option_order=[2, 0, 1], labels=["A", "B", "C"]
    )
    lines = [lettered.text[start:end] for start, end in lettered.option_spans]
    assert lines == ["A. escalate — send it to a human", "B. refund — return the payment", "C. ignore"]
    assert lettered.text == numbered.text.replace("1. ", "A. ").replace("2. ", "B. ").replace("3. ", "C. ")
    # Without labels, the options are numbered as before
    assert [numbered.text[s:e] for s, e in numbered.option_spans][0] == "1. escalate — send it to a human"
    with pytest.raises(ValueError, match="2 labels for 3 options"):
        StrandsPromptFormat.render("s", "q", "choice", options, labels=["A", "B"])


# LabelHead and LabelPointerHead


def _head_inputs(batch: int = 3, num_options: int = 4):
    torch.manual_seed(SEED)
    answer_states = torch.randn(batch, HIDDEN_SIZE)
    option_states = torch.randn(batch, num_options, HIDDEN_SIZE)
    option_mask = torch.ones(batch, num_options, dtype=torch.bool)
    option_mask[1, 2:] = False
    return answer_states, option_states, option_mask


def _label_head(trainable: bool = False, softcap=None, num_labels: int = 6) -> LabelHead:
    head = LabelHead(HIDDEN_SIZE, num_labels, trainable=trainable, softcap=softcap)
    head.copy_rows(torch.randn(num_labels, HIDDEN_SIZE))
    return head


@pytest.mark.parametrize("softcap", [None, 3.0])
def test_label_head_matches_formula(softcap):
    head = _label_head(softcap=softcap)
    answer_states, option_states, option_mask = _head_inputs()
    logits = head(answer_states, option_states, option_mask)
    expected = answer_states @ head.rows[:4].T
    if softcap:
        expected = softcap * torch.tanh(expected / softcap)
    torch.testing.assert_close(logits[option_mask], expected[option_mask])
    assert (logits[~option_mask] == MASK_VALUE).all()


def test_label_head_rows_are_a_buffer_unless_trained():
    frozen, trained = _label_head(), _label_head(trainable=True)
    assert list(frozen.parameters()) == [] and "rows" in dict(frozen.named_buffers())
    assert [name for name, _ in trained.named_parameters()] == ["rows"]
    assert "rows" in frozen.state_dict() and "rows" in trained.state_dict()


def test_label_head_computes_in_fp32():
    head = _label_head()
    answer_states, option_states, option_mask = _head_inputs()
    logits = head(answer_states.bfloat16(), option_states.bfloat16(), option_mask)
    assert logits.dtype == torch.float32


def test_label_head_refuses_more_options_than_rows():
    head = _label_head(num_labels=3)
    with pytest.raises(ValueError, match="at most 3 options"):
        head(*_head_inputs(num_options=4))


def test_label_pointer_head_starts_as_the_label_head_and_trains_its_pointer():
    torch.manual_seed(SEED)
    head = LabelPointerHead(HIDDEN_SIZE, 6, dim=HEAD_DIM, dropout=0.0)
    head.reset_parameters()
    rows = torch.randn(6, HIDDEN_SIZE)
    head.copy_rows(rows)
    label = LabelHead(HIDDEN_SIZE, 6)
    label.copy_rows(rows)
    inputs = _head_inputs()
    logits = head(*inputs)
    torch.testing.assert_close(logits, label(*inputs))
    # The query projection starts at zero, and the gradient reaches it
    logits[inputs[2]].sum().backward()
    assert head.pointer.q.weight.grad.abs().sum() > 0
    assert [name for name, _ in head.named_parameters() if name.startswith("label")] == []


# Converted models


def _tiny_qwen3(tie_word_embeddings: bool):
    return transformers.Qwen3ForCausalLM(
        transformers.Qwen3Config(
            vocab_size=VOCAB_SIZE, hidden_size=HIDDEN_SIZE, intermediate_size=128, num_hidden_layers=2,
            num_attention_heads=4, num_key_value_heads=2, head_dim=16, tie_word_embeddings=tie_word_embeddings,
        )
    )


def _tiny_qwen3_5_text():
    if not hasattr(transformers, "Qwen3_5TextConfig"):
        pytest.skip(f"transformers {transformers.__version__} has no Qwen3.5")
    return transformers.Qwen3_5ForCausalLM(
        transformers.Qwen3_5TextConfig(
            vocab_size=VOCAB_SIZE, hidden_size=HIDDEN_SIZE, intermediate_size=128, num_hidden_layers=2,
            num_attention_heads=2, num_key_value_heads=1, layer_types=["linear_attention", "full_attention"],
            tie_word_embeddings=True,
        )
    )


SOURCES = {
    "qwen3_tied": lambda: _tiny_qwen3(True),
    "qwen3_untied": lambda: _tiny_qwen3(False),
    "qwen3_5_text": _tiny_qwen3_5_text,
}
HEAD_TYPES = ("label", "label_trained", "label_pointer")


def _convert(root, source_name: str, head_type: str) -> dict:
    source_dir, output_dir = str(root / "source"), str(root / head_type)
    if not (root / "source").exists():
        torch.manual_seed(SEED)
        SOURCES[source_name]().save_pretrained(source_dir)
        _tiny_tokenizer().save_pretrained(source_dir)
    init_decision_model(
        DecisionInitConfig(
            base_model_name_or_path=source_dir, output_dir=output_dir, dtype=torch.float32,
            decision_head_type=head_type, decision_head_dim=HEAD_DIM, seed=SEED,
        )
    )
    return {"source_dir": source_dir, "output_dir": output_dir}


@pytest.fixture(scope="module", params=list(SOURCES))
def source_root(request, tmp_path_factory):
    return request.param, tmp_path_factory.mktemp(request.param)


@pytest.fixture(scope="module", params=HEAD_TYPES)
def converted(request, source_root) -> dict:
    source_name, root = source_root
    return {"head_type": request.param, **_convert(root, source_name, request.param)}


def _inputs(encoded) -> dict:
    """The model inputs of one encoded prompt, on DEVICE."""
    return {
        "input_ids": torch.tensor([encoded.input_ids], device=DEVICE),
        "attention_mask": torch.ones(1, len(encoded.input_ids), dtype=torch.long, device=DEVICE),
        "opt_idx": torch.tensor([encoded.opt_idx], device=DEVICE),
        "answer_idx": torch.tensor([encoded.answer_idx], device=DEVICE),
    }


ROWS = [
    {"kind": "choice", "state": "tok1 tok2 tok3", "instructions": "tok4 tok5",
     "options": [["tok6", "tok7"], ["tok8", None], ["tok9", "tok10"]]},
    {"kind": "noul", "state": "tok11 tok12", "instructions": "tok13", "options": [["false", None], ["true", None]]},
    {"kind": "score", "state": "tok14", "instructions": "tok15 tok16",
     "options": [["0", "tok17"], ["1", "tok18"], ["2", "tok19"], ["3", "tok20"]]},
]


def test_config(converted):
    config = DecisionConfig.from_pretrained(converted["output_dir"])
    assert config.decision_head_type == converted["head_type"]
    assert config.decision_num_labels == len(LETTERS + TWO_LETTER_CODES)
    assert config.decision_option_labels == "letters"
    assert config.decision_head_init == "lm_head"
    assert config.decision_label_softcap is None


def test_rows_are_the_source_lm_rows_of_the_letter_tokens(converted):
    model = AutoModel.from_pretrained(converted["output_dir"], dtype=torch.float32)
    source = transformers.AutoModelForCausalLM.from_pretrained(converted["source_dir"], dtype=torch.float32)
    tokenizer = AutoTokenizer.from_pretrained(converted["source_dir"])
    token_ids = [tokenizer.convert_tokens_to_ids(code) for code in LETTERS + TWO_LETTER_CODES]
    head = model.decision_head
    rows = head.label.rows if isinstance(head, LabelPointerHead) else head.rows
    torch.testing.assert_close(rows, source.get_output_embeddings().weight[token_ids], rtol=0, atol=0)
    assert rows.dtype == torch.float32
    if converted["head_type"] == "label_pointer":
        assert isinstance(head.pointer, PointerHead) and (head.pointer.q.weight == 0).all()


def test_only_the_trained_label_head_trains_its_rows(converted):
    model = AutoModel.from_pretrained(converted["output_dir"], dtype=torch.float32)
    names = [name for name, _ in model.decision_head.named_parameters()]
    expected = {"label": [], "label_trained": ["rows"]}.get(converted["head_type"])
    if expected is None:
        assert names and all(name.startswith("pointer.") for name in names)
    else:
        assert names == expected


def test_untrained_label_head_answers_as_its_source_lm(converted):
    tokenizer = AutoTokenizer.from_pretrained(converted["output_dir"])
    prompt = DecisionTeacherPrompt("strands", option_labels="letters")
    queries = [prompt.build(tokenizer, row).query for row in ROWS]
    expected = CausalLMScorer(converted["source_dir"], DEVICE, dtype=torch.float32)(queries)

    model = AutoModel.from_pretrained(converted["output_dir"], dtype=torch.float32).to(DEVICE).eval()
    for query, row, probs in zip(queries, ROWS, expected):
        encoded = prompt.encode(tokenizer, row)
        assert encoded.input_ids == query.input_ids and query.labels == LETTERS[: len(encoded.opt_idx)]
        with torch.no_grad():
            out = model(**_inputs(encoded))
        torch.testing.assert_close(out.log_probs[0].exp().cpu(), torch.tensor(probs), **TOLERANCE)


def test_backward_reaches_the_backbone(converted):
    model = AutoModel.from_pretrained(converted["output_dir"], dtype=torch.float32).to(DEVICE).train()
    tokenizer = AutoTokenizer.from_pretrained(converted["output_dir"])
    encoded = DecisionTeacherPrompt("strands", option_labels="letters").encode(tokenizer, ROWS[0])
    model(**_inputs(encoded)).log_probs[0, 0].backward()
    assert model.model.layers[-1].mlp.down_proj.weight.grad.abs().sum() > 0


def test_the_predictor_asks_a_label_head_with_letters(converted):
    predictor = DecisionPredictor(converted["output_dir"], device=DEVICE, dtype=torch.float32, calibration=None)
    assert predictor.prompt_settings["option_labels"] == "letters"
    tokenizer = AutoTokenizer.from_pretrained(converted["output_dir"])
    prompt = DecisionTeacherPrompt("strands", option_labels="letters")
    expected = CausalLMScorer(converted["source_dir"], DEVICE, dtype=torch.float32)(
        [prompt.build(tokenizer, row).query for row in ROWS]
    )
    for got, want in zip(predictor.probabilities(ROWS, calibrated=False), expected):
        torch.testing.assert_close(torch.tensor(got), torch.tensor(want), **TOLERANCE)


def test_a_label_head_refuses_numbered_options(converted):
    with pytest.raises(ValueError, match="reads options labeled with letters"):
        DecisionModelTeacher(converted["output_dir"], DEVICE, option_labels="numbers")


def test_the_server_answers_with_a_label_head(converted):
    from hmf.eval.decision.server import DecisionRequest, DecisionServer

    server = DecisionServer(DecisionPredictor(converted["output_dir"], device=DEVICE, calibration=None))
    request = DecisionRequest(
        state="tok1 tok2", questions={"pick": {"type": "choice", "instructions": "tok3",
                                               "criteria": {"tok4": "tok5", "tok6": None, "tok7": "tok8"}}}
    )
    probabilities = server.answer(request)["answers"]["pick"]["probabilities"]
    assert list(probabilities) == ["tok4", "tok6", "tok7"]
    assert abs(sum(probabilities.values()) - 1) < 1e-3  # the server rounds each probability


def test_decision_init_checks_the_head_init(tmp_path):
    config = DecisionInitConfig(
        base_model_name_or_path=str(tmp_path / "missing"), output_dir=str(tmp_path / "out"),
        decision_head_type="label", decision_head_init="random",
    )
    with pytest.raises(ValueError, match="initialized with lm_head, not random"):
        init_decision_model(config)
    config.decision_head_type, config.decision_head_init = "pointer", "lm_head"
    with pytest.raises(ValueError, match="initialized with random, not lm_head"):
        init_decision_model(config)


def test_config_needs_labels_for_a_label_head():
    with pytest.raises(ValueError, match="decision_num_labels"):
        DecisionConfig(decision_head_type="label")
    with pytest.raises(ValueError, match="decision_head_type"):
        DecisionConfig(decision_head_type="slot")
    assert DecisionConfig().decision_option_labels is None


def test_a_decision_model_builds_the_head_of_its_config():
    text_config = _tiny_qwen3(True).config
    for head_type, cls in [("pointer", PointerHead), ("label", LabelHead), ("label_trained", LabelHead),
                           ("label_pointer", LabelPointerHead)]:
        config = DecisionConfig(text_config=text_config, decision_head_type=head_type, decision_num_labels=5,
                                decision_head_dim=HEAD_DIM)
        assert type(DecisionModel(config).decision_head) is cls


# LoRA training


def _batch(tokenizer) -> dict:
    """ROWS as one right-padded batch, labeled with letters, with the first option of each as the label."""
    prompt = DecisionTeacherPrompt("strands", option_labels="letters")
    encoded = [prompt.encode(tokenizer, row) for row in ROWS]
    length, width = max(len(e.input_ids) for e in encoded), max(len(e.opt_idx) for e in encoded)
    input_ids = torch.full((len(encoded), length), tokenizer.pad_token_id)
    attention_mask = torch.zeros_like(input_ids)
    opt_idx = torch.full((len(encoded), width), -1)
    for j, e in enumerate(encoded):
        input_ids[j, : len(e.input_ids)] = torch.tensor(e.input_ids)
        attention_mask[j, : len(e.input_ids)] = 1
        opt_idx[j, : len(e.opt_idx)] = torch.tensor(e.opt_idx)
    answer_idx = torch.tensor([e.answer_idx for e in encoded])
    return {"input_ids": input_ids, "attention_mask": attention_mask, "opt_idx": opt_idx, "answer_idx": answer_idx,
            "label": torch.zeros(len(encoded), dtype=torch.long)}


def test_lora_trains_and_reloads_a_label_head(converted, tmp_path):
    import torch.nn.functional as F
    from peft import PeftModel

    from hmf.hparams.parser import _parse_train_args
    from hmf.model import load_model, load_tokenizer

    model_args, _, _, finetuning_args, _ = _parse_train_args({
        "model_name_or_path": converted["output_dir"], "stage": "decision", "do_train": True,
        "finetuning_type": "lora", "lora_rank": 4, "lora_target": "q_proj,v_proj",
        "disable_gradient_checkpointing": True, "output_dir": "unused", "report_to": "none",
    })
    tokenizer = load_tokenizer(model_args)["tokenizer"]
    model = load_model(tokenizer, model_args, finetuning_args, is_trainable=True).to(DEVICE)
    batch = {key: value.to(DEVICE) for key, value in _batch(tokenizer).items()}
    inputs = {key: value for key, value in batch.items() if key != "label"}
    trainable = {name for name, param in model.named_parameters() if param.requires_grad}
    rows_trainable = any(name.endswith("rows") for name in trainable)
    assert rows_trainable == (converted["head_type"] == "label_trained")

    model.train()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-2)
    for _ in range(3):
        F.nll_loss(model(**inputs).log_probs, batch["label"]).backward()
        optimizer.step()
        optimizer.zero_grad()
    model.eval()
    with torch.no_grad():
        trained = model(**inputs).log_probs
    adapter_dir = str(tmp_path / "adapter")
    model.save_pretrained(adapter_dir)

    # The saved adapter, with its copy of the head, reloads exactly
    base = AutoModel.from_pretrained(converted["output_dir"], dtype=torch.float32)
    reloaded = PeftModel.from_pretrained(base, adapter_dir).to(DEVICE).eval()
    with torch.no_grad():
        torch.testing.assert_close(reloaded(**inputs).log_probs, trained)
    # Merged into the decision model, it answers as the trained model did, up to the rounding of the merged
    # weights, which a GPU's TF32 matmuls make about 1e-3
    base = AutoModel.from_pretrained(converted["output_dir"], dtype=torch.float32)
    merged = PeftModel.from_pretrained(base, adapter_dir).merge_and_unload().to(DEVICE).eval()
    with torch.no_grad():
        torch.testing.assert_close(merged(**inputs).log_probs, trained, rtol=1e-2, atol=5e-3)
    untrained = AutoModel.from_pretrained(converted["output_dir"], dtype=torch.float32).to(DEVICE).eval()
    with torch.no_grad():
        assert not torch.allclose(untrained(**inputs).log_probs, trained)
