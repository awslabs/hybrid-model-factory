"""
Tests for the decision dataset processor (data/processor/decision.py), covering augmentation, labels
and targets, reproducibility, evaluation rows, and dropping long prompts. All tests run
offline with a tiny word-level tokenizer.
"""

from types import SimpleNamespace

import pytest
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from hmf.data.decision_prompt import StrandsPromptFormat, encode_decision_prompt
from hmf.data.processor.decision import DecisionDatasetProcessor


def _tiny_tokenizer(bos: bool = False, words: tuple = ()) -> PreTrainedTokenizerFast:
    """A word-level tokenizer, in which every word or punctuation run is one token, unknown unless in words."""
    vocab = {"[UNK]": 0, "[PAD]": 1, "<s>": 2, **{word: i + 3 for i, word in enumerate(words)}}
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    return PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, unk_token="[UNK]", pad_token="[PAD]", bos_token="<s>" if bos else None
    )


def _data_args(**overrides) -> SimpleNamespace:
    args = dict(
        decision_prompt_format="strands",
        decision_num_augmentations=1,
        decision_shuffle_options=True,
        decision_reverse_score_prob=0.5,
        decision_vary_instructions=True,
        decision_ordinal_smoothing=0.1,
        decision_force_bos=False,
        cutoff_len=4096,
    )
    args.update(overrides)
    return SimpleNamespace(**args)


def _processor(is_eval: bool = False, seed: int = 0, tokenizer=None, **data_args) -> DecisionDatasetProcessor:
    return DecisionDatasetProcessor(
        template=None,
        tokenizer=tokenizer or _tiny_tokenizer(),
        processor=None,
        data_args=_data_args(**data_args),
        is_eval=is_eval,
        seed=seed,
    )


ROWS = [
    {
        "_kind": "choice",
        "_state": "Order 7 was charged twice.",
        "_instructions": "What should support do?",
        "_instruction_variants": ["How should support respond?", "Pick the next step."],
        "_options": [["refund", "return the payment"], ["ignore", None], ["escalate", "to a human"], ["wait", None]],
        "_label": 2,
        "_task": "support",
        "_weight": 1.0,
    },
    {
        "_kind": "noul",
        "_state": "The meeting moved to Thursday.",
        "_instructions": "The meeting is on Tuesday.",
        "_instruction_variants": [],
        "_options": [["false", None], ["true", None]],
        "_label": 0,
        "_task": "calendar",
        "_weight": 0.5,
    },
    {
        "_kind": "score",
        "_state": "The reply was polite but missed the question.",
        "_instructions": "Rate the reply.",
        "_instruction_variants": None,
        "_options": [["0", "useless"], ["1", "weak"], ["2", "good"], ["3", "excellent"]],
        "_label": 1,
        "_task": "adequacy",
        "_weight": 1.0,
    },
]


def _batch(rows=ROWS, source="toy", start=0) -> dict:
    """Columnar batch of rows, as datasets.map passes it, with row identities."""
    batch = {key: [row[key] for row in rows] for key in rows[0]}
    batch["_source"] = [source] * len(rows)
    batch["_row_index"] = list(range(start, start + len(rows)))
    return batch


def _rows(output: dict) -> list:
    """The processor's columnar output as a list of rows."""
    return [dict(zip(output, values)) for values in zip(*output.values())]


def _canonical_label_dist(row: dict, smoothing: float) -> list:
    num_options = len(row["_options"])
    dist = [0.0] * num_options
    label = row["_label"]
    neighbors = [level for level in (label - 1, label + 1) if 0 <= level < num_options]
    if row["_kind"] == "score" and smoothing > 0 and neighbors:
        dist[label] = 1.0 - smoothing
        for level in neighbors:
            dist[level] = smoothing / len(neighbors)
    else:
        dist[label] = 1.0
    return dist


def test_output_columns():
    out = _rows(_processor().preprocess_dataset(_batch()))

    assert len(out) == len(ROWS)
    assert set(out[0]) == {
        "input_ids", "attention_mask", "opt_idx", "answer_idx", "label", "label_dist", "option_order",
        "kind", "task", "weight", "source", "row_index", "augmentation",
    }
    for row, source_row in zip(out, ROWS):
        assert row["attention_mask"] == [1] * len(row["input_ids"])
        assert len(row["opt_idx"]) == len(row["option_order"]) == len(row["label_dist"])
        assert row["kind"] == source_row["_kind"]
        assert row["task"] == source_row["_task"]
        assert row["weight"] == source_row["_weight"]


def test_label_follows_its_option():
    processor = _processor(decision_num_augmentations=50)
    for row in _rows(processor.preprocess_dataset(_batch())):
        source_row = ROWS[row["row_index"]]
        assert row["option_order"][row["label"]] == source_row["_label"]


def test_matches_direct_encoding():
    # The output is exactly the prompt format's rendered prompt, encoded, under the recorded option order
    tokenizer = _tiny_tokenizer()
    processor = _processor(tokenizer=tokenizer, decision_num_augmentations=5, decision_vary_instructions=False)
    for row in _rows(processor.preprocess_dataset(_batch())):
        source_row = ROWS[row["row_index"]]
        rendered = StrandsPromptFormat.render(
            source_row["_state"], source_row["_instructions"], source_row["_kind"],
            source_row["_options"], option_order=row["option_order"],
        )
        encoded = encode_decision_prompt(tokenizer, rendered)
        assert (row["input_ids"], row["opt_idx"], row["answer_idx"]) == (
            encoded.input_ids, encoded.opt_idx, encoded.answer_idx
        )


def test_option_orders():
    out = _rows(_processor(decision_num_augmentations=200).preprocess_dataset(_batch()))
    orders = {
        kind: {tuple(r["option_order"]) for r in out if r["kind"] == kind} for kind in ("choice", "noul", "score")
    }

    # noul and choice options are shuffled; score rubrics are only ever kept or reversed
    assert len(orders["choice"]) > 10
    assert orders["noul"] == {(0, 1), (1, 0)}
    assert orders["score"] == {(0, 1, 2, 3), (3, 2, 1, 0)}


@pytest.mark.parametrize("prob, expected", [(0.0, {(0, 1, 2, 3)}), (1.0, {(3, 2, 1, 0)})])
def test_reverse_score_prob(prob, expected):
    processor = _processor(decision_num_augmentations=20, decision_reverse_score_prob=prob)
    out = _rows(processor.preprocess_dataset(_batch([ROWS[2]])))

    assert {tuple(r["option_order"]) for r in out} == expected


def test_no_shuffle():
    out = _rows(_processor(decision_num_augmentations=10, decision_shuffle_options=False).preprocess_dataset(_batch()))

    assert all(r["option_order"] == list(range(len(r["option_order"]))) for r in out)


@pytest.mark.parametrize("smoothing", [0.1, 0.0])
def test_label_dist(smoothing):
    processor = _processor(decision_num_augmentations=20, decision_ordinal_smoothing=smoothing)
    for row in _rows(processor.preprocess_dataset(_batch())):
        canonical = _canonical_label_dist(ROWS[row["row_index"]], smoothing)
        expected = [canonical[i] for i in row["option_order"]]
        assert row["label_dist"] == pytest.approx(expected)
        assert sum(row["label_dist"]) == pytest.approx(1.0)


def test_label_dist_at_rubric_end():
    # The lowest level has a single neighbor, which gets all of the smoothing mass
    row = dict(ROWS[2], _label=0)
    out = _rows(_processor(decision_shuffle_options=False).preprocess_dataset(_batch([row])))

    assert out[0]["label_dist"] == pytest.approx([0.9, 0.1, 0.0, 0.0])


def test_instruction_variants_sampled():
    # Each phrasing of ROWS[0] starts with a different word, which the tokenizer knows
    first_words = ("What", "How", "Pick")
    tokenizer = _tiny_tokenizer(words=first_words)
    processor = _processor(tokenizer=tokenizer, decision_num_augmentations=60)
    texts = [tokenizer.decode(r["input_ids"]) for r in _rows(processor.preprocess_dataset(_batch([ROWS[0]])))]

    for word in first_words:
        assert any(word in text.split() for text in texts), f"the phrasing starting with {word!r} was never sampled"


def test_augmentations():
    out = _rows(_processor(decision_num_augmentations=3).preprocess_dataset(_batch()))

    assert len(out) == 3 * len(ROWS)
    for row_index in range(len(ROWS)):
        assert sorted(r["augmentation"] for r in out if r["row_index"] == row_index) == [0, 1, 2]


def test_reproducible_and_independent_of_batching():
    processor = _processor(decision_num_augmentations=4)
    together = _rows(processor.preprocess_dataset(_batch()))
    again = _rows(processor.preprocess_dataset(_batch()))
    separately = [r for i in range(len(ROWS)) for r in _rows(processor.preprocess_dataset(_batch([ROWS[i]], start=i)))]

    assert together == again
    key = lambda r: (r["row_index"], r["augmentation"])  # noqa: E731
    assert sorted(together, key=key) == sorted(separately, key=key)


def test_more_augmentations_keep_the_first():
    one = _rows(_processor(decision_num_augmentations=1).preprocess_dataset(_batch()))
    three = _rows(_processor(decision_num_augmentations=3).preprocess_dataset(_batch()))

    assert one == [r for r in three if r["augmentation"] == 0]


def test_seed_and_row_identity_change_augmentations():
    def orders(**kwargs):
        processor = _processor(decision_num_augmentations=10, seed=kwargs.pop("seed", 0))
        return [r["option_order"] for r in _rows(processor.preprocess_dataset(_batch([ROWS[0]], **kwargs)))]

    assert orders() != orders(seed=1)
    assert orders() != orders(source="other")
    assert orders() != orders(start=5)


def test_eval_rows_are_canonical():
    tokenizer = _tiny_tokenizer()
    processor = _processor(is_eval=True, tokenizer=tokenizer, decision_num_augmentations=5)
    out = _rows(processor.preprocess_dataset(_batch()))

    assert len(out) == len(ROWS)
    for row in out:
        source_row = ROWS[row["row_index"]]
        assert row["augmentation"] == 0
        assert row["option_order"] == list(range(len(row["option_order"])))
        rendered = StrandsPromptFormat.render(
            source_row["_state"], source_row["_instructions"], source_row["_kind"], source_row["_options"]
        )
        assert row["input_ids"] == encode_decision_prompt(tokenizer, rendered).input_ids


def test_drops_prompts_longer_than_cutoff_len():
    lengths = [len(r["input_ids"]) for r in _rows(_processor().preprocess_dataset(_batch()))]
    cutoff_len = max(lengths) - 1

    out = _rows(_processor(cutoff_len=cutoff_len).preprocess_dataset(_batch()))

    assert len(out) == sum(length <= cutoff_len for length in lengths) < len(ROWS)
    assert all(len(r["input_ids"]) <= cutoff_len for r in out)


def test_label_out_of_range_raises():
    with pytest.raises(ValueError, match="out of range"):
        _processor().preprocess_dataset(_batch([dict(ROWS[0], _label=4)]))


def test_force_bos():
    tokenizer = _tiny_tokenizer(bos=True)
    plain = _rows(_processor(tokenizer=tokenizer).preprocess_dataset(_batch()))
    forced = _rows(_processor(tokenizer=tokenizer, decision_force_bos=True).preprocess_dataset(_batch()))

    for p, f in zip(plain, forced):
        assert f["input_ids"] == [tokenizer.bos_token_id] + p["input_ids"]
        assert f["opt_idx"] == [i + 1 for i in p["opt_idx"]]
        assert f["answer_idx"] == p["answer_idx"] + 1
