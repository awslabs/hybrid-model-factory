"""
Tests for decision prompt formats and encoding (data/decision_prompt.py).

The encoding tests use a tiny offline tokenizer, plus real Qwen3.5 and Llama 3 tokenizers (set
QWEN_TOKENIZER / TINY_LLAMA3 to use local copies). Strands Decider's exact prompt strings are tested
in tests/decision/test_strands_decider_reproduction.py.
"""

import os

import pytest
from tokenizers import Tokenizer, models, pre_tokenizers, processors
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from hmf.data.decision_prompt import (
    DECISION_PROMPT_FORMATS,
    StrandsPromptFormat,
    adds_bos_token,
    encode_decision_prompt,
    get_prompt_format,
    render_content,
)


QWEN_TOKENIZER = os.getenv("QWEN_TOKENIZER", "Qwen/Qwen3.5-2B-Base")
TINY_LLAMA3 = os.getenv("TINY_LLAMA3", "llamafactory/tiny-random-Llama-3")


def _tiny_tokenizer(bos: bool = False, add_bos: bool = False) -> PreTrainedTokenizerFast:
    """A word-level tokenizer, in which every word or punctuation run is one token (all unknown)."""
    tokenizer = Tokenizer(models.WordLevel({"[UNK]": 0, "[PAD]": 1, "<s>": 2}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    if add_bos:
        tokenizer.post_processor = processors.TemplateProcessing(single="<s> $A", special_tokens=[("<s>", 2)])
    return PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, unk_token="[UNK]", pad_token="[PAD]", bos_token="<s>" if bos else None
    )


# render_content


def test_render_content():
    assert render_content(None) == ""
    assert render_content("  text \n") == "text"
    assert render_content({"a": [1, "é"]}) == '{\n  "a": [\n    1,\n    "é"\n  ]\n}'
    assert render_content(["x", 2]) == '[\n  "x",\n  2\n]'


# StrandsPromptFormat


@pytest.mark.parametrize(
    "kind, options, order",
    [
        ("noul", [("false", "no"), ("true", "yes")], None),
        ("choice", [("a", "first"), ("b", None), ("c", "third")], [2, 0, 1]),
        ("score", [("0", "low"), ("1", "mid"), ("2", "high")], [2, 1, 0]),
    ],
)
def test_strands_spans(kind, options, order):
    rendered = StrandsPromptFormat.render("state", "question", kind, options, option_order=order)
    lines = rendered.text.split("\n")
    option_lines = lines[lines.index("<options>") + 1 : lines.index("</options>")]

    assert [rendered.text[start:end] for start, end in rendered.option_spans] == option_lines
    assert rendered.text[slice(*rendered.answer_span)] == StrandsPromptFormat.ANSWER_MARKER
    assert rendered.answer_span[1] == len(rendered.text)


@pytest.mark.parametrize(
    "kind, options, order",
    [
        ("noul", [("false", None), ("maybe", None)], None),
        ("noul", [("true", None), ("true", None)], None),
        ("choice", [("a", None), ("a", None)], None),
        ("choice", [("a", None), ("b", None)], [0, 0]),
        ("choice", [("a", None), ("b", None)], [0, 1, 2]),
        ("ranking", [("a", None), ("b", None)], None),
    ],
)
def test_strands_rejects_invalid_rows(kind, options, order):
    with pytest.raises(ValueError):
        StrandsPromptFormat.render("s", "q", kind, options, option_order=order)


def test_get_prompt_format():
    assert get_prompt_format("strands") is StrandsPromptFormat
    assert set(DECISION_PROMPT_FORMATS) == {"strands"}
    with pytest.raises(ValueError, match="decision_prompt_format"):
        get_prompt_format("unknown")


# encode_decision_prompt


def _assert_positions(tokenizer, rendered, encoded):
    """Each read position is the last token lying entirely within its span."""
    offsets = tokenizer(rendered.text, return_offsets_mapping=True)["offset_mapping"]
    shift = len(encoded.input_ids) - len(offsets)  # 1 if a BOS token was prepended
    spans = list(rendered.option_spans) + [rendered.answer_span]
    positions = list(encoded.opt_idx) + [encoded.answer_idx]

    for (start, end), position in zip(spans, positions):
        lo, hi = offsets[position - shift]
        assert hi > lo and start <= lo and hi <= end
        later = offsets[position - shift + 1 :]
        assert not any(b > a and start <= a and b <= end for a, b in later)


def _rendered_examples():
    return [
        StrandsPromptFormat.render("The order was charged twice.", "What next?", "choice",
                                   [("refund", "return the payment"), ("ignore", None), ("escalate", "to a human")],
                                   option_order=[1, 2, 0]),
        StrandsPromptFormat.render("The meeting moved to Thursday.", "It is on Tuesday.", "noul", []),
        StrandsPromptFormat.render({"ticket": 7}, "Rate it.", "score", [("0", "bad"), ("1", "fine"), ("2", "great")]),
    ]


def test_encode_tiny_tokenizer():
    tokenizer = _tiny_tokenizer()
    for rendered in _rendered_examples():
        encoded = encode_decision_prompt(tokenizer, rendered)

        assert encoded.input_ids == tokenizer(rendered.text)["input_ids"]
        assert len(encoded.opt_idx) == len(rendered.option_spans)
        assert encoded.answer_idx == len(encoded.input_ids) - 1
        _assert_positions(tokenizer, rendered, encoded)


@pytest.mark.parametrize("name", [QWEN_TOKENIZER, TINY_LLAMA3])
def test_encode_real_tokenizer(name):
    tokenizer = AutoTokenizer.from_pretrained(name)
    for rendered in _rendered_examples():
        encoded = encode_decision_prompt(tokenizer, rendered)

        assert encoded.input_ids == tokenizer(rendered.text)["input_ids"]
        assert encoded.answer_idx == len(encoded.input_ids) - 1
        assert encoded.opt_idx == sorted(encoded.opt_idx)  # options are read in rendered order
        _assert_positions(tokenizer, rendered, encoded)


def test_adds_bos_token():
    assert not adds_bos_token(_tiny_tokenizer(bos=False))
    assert not adds_bos_token(_tiny_tokenizer(bos=True, add_bos=False))
    assert adds_bos_token(_tiny_tokenizer(bos=True, add_bos=True))


def test_force_bos():
    rendered = _rendered_examples()[0]
    plain = encode_decision_prompt(_tiny_tokenizer(bos=True), rendered)

    # A BOS token the tokenizer does not add is prepended, and the read positions shift with it
    tokenizer = _tiny_tokenizer(bos=True)
    forced = encode_decision_prompt(tokenizer, rendered, force_bos=True)
    assert forced.input_ids == [tokenizer.bos_token_id] + plain.input_ids
    assert forced.opt_idx == [p + 1 for p in plain.opt_idx]
    assert forced.answer_idx == plain.answer_idx + 1
    _assert_positions(tokenizer, rendered, forced)

    # A tokenizer that already adds BOS gets no second one
    tokenizer = _tiny_tokenizer(bos=True, add_bos=True)
    assert encode_decision_prompt(tokenizer, rendered, force_bos=True) == encode_decision_prompt(tokenizer, rendered)

    # Without a BOS token, force_bos changes nothing
    tokenizer = _tiny_tokenizer(bos=False)
    assert encode_decision_prompt(tokenizer, rendered, force_bos=True) == encode_decision_prompt(tokenizer, rendered)


def test_encode_rejects_slow_tokenizer():
    class SlowTokenizer:
        is_fast = False

    with pytest.raises(ValueError, match="fast tokenizer"):
        encode_decision_prompt(SlowTokenizer(), _rendered_examples()[0])
