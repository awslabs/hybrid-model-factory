"""
Prompt formats for decision models, and the step that turns a rendered prompt into model inputs.

A decision row is a state, a question (instructions), a list of options, and the question's kind:

    - noul: a yes/no question. The options are named "false" and "true", in that order.
    - choice: pick one of N named options.
    - score: rate against an ordered rubric. Options are the rubric levels, lowest first.

A prompt format renders a row into text, and records where each option's line and the answer
marker are. Formats are registered in DECISION_PROMPT_FORMATS and selected with
decision_prompt_format. The decision model reads the hidden state at the last token of each
option's line and at the answer marker (see model/hybrid_zoo/models/decision/), so
encode_decision_prompt maps those character spans to token positions.

Training and inference must use the same format, so both go through this module.
"""

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple, Type


if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer


DECISION_KINDS = ("noul", "choice", "score")

# A noul question always renders both options, in this order, so canonical index 0 is "false"
NOUL_OPTION_NAMES = ("false", "true")


@dataclass(frozen=True)
class RenderedPrompt:
    """
    A decision row rendered to text.

    Args:
        text: The prompt.
        option_spans: The (start, end) character span of each option's line in text, in rendered
            order.
        answer_span: The (start, end) character span of the answer marker in text.
    """

    text: str
    option_spans: Tuple[Tuple[int, int], ...]
    answer_span: Tuple[int, int]


@dataclass(frozen=True)
class EncodedPrompt:
    """
    A rendered prompt tokenized, with the token positions the decision model reads.

    Args:
        input_ids: Token ids of the prompt.
        opt_idx: Position of the last token of each option's line, in rendered order.
        answer_idx: Position of the last token of the answer marker.
    """

    input_ids: List[int]
    opt_idx: List[int]
    answer_idx: int


def render_content(content: Any) -> str:
    """
    Flattens a state, instruction, or option description into text. Strings are stripped; dicts and
    lists are rendered as indented JSON, so the same content always gives the same text.

    Args:
        content: A string, a JSON-serializable dict or list, or None.

    Returns:
        The content as text. None gives an empty string.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    return json.dumps(content, indent=2, ensure_ascii=False, sort_keys=False)


class DecisionPromptFormat(ABC):
    """
    A prompt format, which renders a decision row into text and records where each option's line
    and the answer marker are. Each option's line starts with its label (option_label), which a causal
    LM can answer with right after the answer marker (see data/decision_teachers.py).
    """

    @classmethod
    def option_label(cls, position: int) -> str:
        """
        The label of the option shown at a position, counting from 0. Defaults to its number,
        counting from 1.
        """
        return str(position + 1)

    @classmethod
    @abstractmethod
    def render(
        cls,
        state: Any,
        instructions: Any,
        kind: str,
        options: Sequence[Tuple[str, Optional[str]]],
        option_order: Optional[Sequence[int]] = None,
    ) -> RenderedPrompt:
        """
        Renders a decision row.

        Args:
            state: The state being judged.
            instructions: The question.
            kind: One of DECISION_KINDS.
            options: The (name, description) of each option, in canonical order. Descriptions may
                be None.
            option_order: A permutation of the option indices: option_order[k] is the canonical
                index of the option shown at position k. If None, options are shown in canonical
                order.

        Returns:
            The RenderedPrompt.
        """
        ...


class StrandsPromptFormat(DecisionPromptFormat):
    """
    The layout used by Strands Decider:

        <state>
        {state}
        </state>
        <question type="{kind}">
        {header}
        {instructions}
        <options>
        1. {name} — {description}
        2. {name} — {description}
        </options>
        </question>
        <answer>

    The header depends on the kind (HEADERS). Options are numbered from 1, and an option without a
    description shows only its name. A noul question always shows both options; an option that is
    not given shows its default description in NOUL_DESCRIPTIONS.
    """

    HEADERS = {
        "noul": "Decide whether the statement is true of the state.",
        "choice": "Select exactly one option.",
        "score": "Rate the state against the ordered levels below (lowest first).",
    }
    NOUL_DESCRIPTIONS = {
        "false": "the statement does not hold for this state",
        "true": "the statement holds for this state",
    }
    ANSWER_MARKER = "<answer>"

    @classmethod
    def _option_pairs(cls, kind: str, options: Sequence[Tuple[str, Optional[str]]]) -> List[Tuple[str, str]]:
        """The (name, description) pair of each option, in canonical order, as this format shows it."""
        if kind == "noul":
            names = [name for name, _ in options]
            if len(set(names)) != len(names) or not set(names) <= set(NOUL_OPTION_NAMES):
                raise ValueError(f"noul options must be named from {NOUL_OPTION_NAMES} without repeats, got {names}")
            descriptions = {**cls.NOUL_DESCRIPTIONS, **{name: desc for name, desc in options}}
            return [(name, render_content(descriptions[name])) for name in NOUL_OPTION_NAMES]

        if kind == "choice":
            names = [name for name, _ in options]
            if len(set(names)) != len(names):
                raise ValueError(f"choice option names must be unique, got {names}")
            return [(name, render_content(desc)) for name, desc in options]

        if kind == "score":
            # Levels are named by their index; the description is the rubric text
            return [(str(i), desc or "") for i, (_, desc) in enumerate(options)]

        raise ValueError(f"Unknown decision kind '{kind}'. Supported kinds: {DECISION_KINDS}")

    @classmethod
    def render(
        cls,
        state: Any,
        instructions: Any,
        kind: str,
        options: Sequence[Tuple[str, Optional[str]]],
        option_order: Optional[Sequence[int]] = None,
    ) -> RenderedPrompt:
        pairs = cls._option_pairs(kind, options)
        if option_order is not None:
            if sorted(option_order) != list(range(len(pairs))):
                raise ValueError(f"option_order must be a permutation of range({len(pairs)}), got {option_order}")
            pairs = [pairs[i] for i in option_order]

        text = (
            f"<state>\n{render_content(state)}\n</state>\n"
            f'<question type="{kind}">\n{cls.HEADERS[kind]}\n{render_content(instructions)}\n<options>\n'
        )

        # One line per option, numbered from 1. Descriptions are collapsed to a single line so each
        # option's span is exactly one line.
        option_spans = []
        for i, (name, desc) in enumerate(pairs):
            desc = " ".join(desc.split())
            line = f"{cls.option_label(i)}. {name}" + (f" — {desc}" if desc else "")
            if i > 0:
                text += "\n"
            option_spans.append((len(text), len(text) + len(line)))
            text += line

        text += "\n</options>\n</question>\n" + cls.ANSWER_MARKER
        return RenderedPrompt(
            text=text,
            option_spans=tuple(option_spans),
            answer_span=(len(text) - len(cls.ANSWER_MARKER), len(text)),
        )


# Prompt formats, selected by decision_prompt_format
DECISION_PROMPT_FORMATS: Dict[str, Type[DecisionPromptFormat]] = {
    "strands": StrandsPromptFormat,
}


def get_prompt_format(name: str) -> Type[DecisionPromptFormat]:
    """Returns the prompt format called name."""
    if name not in DECISION_PROMPT_FORMATS:
        raise ValueError(
            f"Unknown decision_prompt_format '{name}'. Supported formats: {list(DECISION_PROMPT_FORMATS)}"
        )
    return DECISION_PROMPT_FORMATS[name]


def _last_token_in_span(offsets: Sequence[Tuple[int, int]], start: int, end: int) -> int:
    """Index of the last token lying entirely within characters [start, end), or -1 if there is none."""
    last = -1
    for j, (lo, hi) in enumerate(offsets):
        if hi > lo and lo >= start and hi <= end:  # special tokens have an empty span
            last = j
    return last


def adds_bos_token(tokenizer: "PreTrainedTokenizer") -> bool:
    """Whether the tokenizer has a BOS token and adds it when encoding text with special tokens."""
    if tokenizer.bos_token_id is None:
        return False
    input_ids = tokenizer("x")["input_ids"]
    return len(input_ids) > 0 and input_ids[0] == tokenizer.bos_token_id


def encode_decision_prompt(
    tokenizer: "PreTrainedTokenizer",
    rendered: RenderedPrompt,
    force_bos: bool = False,
) -> EncodedPrompt:
    """
    Tokenizes a rendered prompt and finds the token positions the decision model reads, which are the
    last token of each option's line and the last token of the answer marker. The last token of a span
    is the one that has read all of it under causal attention.

    Args:
        tokenizer: A fast tokenizer (offset mappings are required). Special tokens are added as the
            tokenizer does by default.
        rendered: The output of a prompt format.
        force_bos: Whether to start the prompt with the tokenizer's BOS token when the tokenizer
            does not add it, e.g. for instruction-tuned tokenizers that leave BOS to the chat
            template. Defaults to False.

    Returns:
        The EncodedPrompt.
    """
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError("encode_decision_prompt needs a fast tokenizer for offset mappings.")

    encoding = tokenizer(rendered.text, return_offsets_mapping=True)
    input_ids, offsets = list(encoding["input_ids"]), encoding["offset_mapping"]

    opt_idx = [_last_token_in_span(offsets, start, end) for start, end in rendered.option_spans]
    answer_idx = _last_token_in_span(offsets, *rendered.answer_span)
    if min(opt_idx + [answer_idx]) < 0:
        raise ValueError("A token span of the prompt is empty; the prompt cannot be read by the decision model.")

    bos_token_id = tokenizer.bos_token_id
    if force_bos and bos_token_id is not None and (not input_ids or input_ids[0] != bos_token_id):
        input_ids = [bos_token_id] + input_ids
        opt_idx = [position + 1 for position in opt_idx]
        answer_idx += 1

    return EncodedPrompt(input_ids=input_ids, opt_idx=opt_idx, answer_idx=answer_idx)
