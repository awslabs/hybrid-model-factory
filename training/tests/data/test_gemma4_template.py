"""
Tests that the gemma4 template formats conversations as Gemma 4's own chat template does. The expected strings were
rendered with gemma-4-E4B-it's chat_template.jinja (with enable_thinking where a model turn has thinking): no system
turn without a system message, thinking kept in the last model turn only, no empty thought blocks, and <|think|> at
the start of the system turn when the last turn's thinking is kept. A stand-in tokenizer encodes each character as
its own id, so the encoded ids read back as the formatted text. The tests run on the CPU.
"""

import pytest

from hmf.data.template import TEMPLATES

BOS = 0x110000  # past the last Unicode character, so it can't be confused with one
THOUGHT = "<|channel>thought\n{}\n<channel|>{}"


class CharTokenizer:
    """Encodes each character as its own id."""

    bos_token_id = BOS
    eos_token_id = BOS + 1

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]

    def convert_tokens_to_ids(self, token):
        return BOS + 2


def _format(messages, system=None, enable_thinking=True):
    template = TEMPLATES["gemma4"]
    template.enable_thinking = enable_thinking
    pairs = template.encode_multiturn(CharTokenizer(), messages, system)
    return "".join("<bos>" if i == BOS else chr(i) for prompt, response in pairs for i in prompt + response)


def _turns(*contents):
    return [{"role": "user" if i % 2 == 0 else "assistant", "content": c} for i, c in enumerate(contents)]


def test_conversation_without_a_system_message_has_no_system_turn():
    assert _format(_turns("Hi", "Hello!")) == "<bos><|turn>user\nHi<turn|>\n<|turn>model\nHello!<turn|>\n"


def test_system_message_and_two_turns():
    assert _format(_turns("Hi", "Hello!", "Bye", "Ok"), system="You are helpful.") == (
        "<bos><|turn>system\nYou are helpful.<turn|>\n<|turn>user\nHi<turn|>\n<|turn>model\nHello!<turn|>\n"
        "<|turn>user\nBye<turn|>\n<|turn>model\nOk<turn|>\n"
    )


def test_thinking_is_kept_in_the_last_model_turn_only():
    messages = _turns("2+2?", THOUGHT.format("Add them.", "4"), "3+3?", THOUGHT.format("Add again.", "6"))
    assert _format(messages) == (
        "<bos><|turn>system\n<|think|>\n<turn|>\n<|turn>user\n2+2?<turn|>\n<|turn>model\n4<turn|>\n"
        "<|turn>user\n3+3?<turn|>\n<|turn>model\n<|channel>thought\nAdd again.\n<channel|>6<turn|>\n"
    )


def test_thinking_with_a_system_message():
    assert _format(_turns("2+2?", THOUGHT.format("Add them.", "4")), system="Be brief.") == (
        "<bos><|turn>system\n<|think|>\nBe brief.<turn|>\n<|turn>user\n2+2?<turn|>\n<|turn>model\n"
        "<|channel>thought\nAdd them.\n<channel|>4<turn|>\n"
    )


def test_thinking_is_removed_when_disabled():
    messages = _turns("2+2?", THOUGHT.format("Add them.", "4"))
    assert _format(messages, enable_thinking=False) == "<bos><|turn>user\n2+2?<turn|>\n<|turn>model\n4<turn|>\n"


def test_tool_messages_raise():
    messages = [{"role": "user", "content": "go"}, {"role": "function", "content": "{}"},
                {"role": "observation", "content": "ok"}, {"role": "assistant", "content": "done"}]
    with pytest.raises(NotImplementedError, match="tool calls"):
        _format(messages)
