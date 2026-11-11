"""
Tests that the minicpm5 template formats conversations as MiniCPM5's own chat template does. The expected strings
were rendered with MiniCPM5-2B's chat_template.jinja: an empty thought block in every assistant turn without
thinking, the thinking of every turn kept, tool declarations in the system turn (after the system message, or on
their own without one), tool calls as XML with CDATA for values that hold a newline, and tool responses in a user
turn, where the responses to several calls share one turn (the data converters join consecutive tool messages into
one observation). A stand-in tokenizer encodes each character as its own id, so the encoded ids read back as the
formatted text. The tests run on the CPU.
"""

import json

from hmf.data.template import TEMPLATES

BOS = 0x110000  # past the last Unicode character, so it can't be confused with one
TOOL = {
    "name": "get_weather",
    "description": "Today's forecast",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "days": {"type": "integer"}}},
}
CALL = {"name": "get_weather", "arguments": {"city": "Paris\nFrance", "days": 3}}
CALL2 = {"name": "get_weather", "arguments": {"city": "Rome", "days": 1}}

TOOLS_WITHOUT_SYSTEM = (
    '<s><|im_start|>system\n# Tools\n\nYou are provided with function signatures within <tools></tools> '
    'XML tags:\n<tools>\n{"type": "function", "function": {"name": "get_weather", "description": '
    '"Today\'s forecast", "parameters": {"type": "object", "properties": {"city": {"type": "string"}, '
    '"days": {"type": "integer"}}}}}\n</tools>\n\nTool usage guidelines:\n'
    '- You may call zero or more functions. If no function calls are needed, just answer normally '
    'and do not include any <function ... </function>.\n'
    '- When calling a function, return an XML object within <function ... </function> using:\n'
    '<function name="function-name"><param name="param-name">param-value</param></function>\n'
    '- param-value may be multi-line. If it contains <, & or newline characters, wrap it in a '
    'CDATA block: <param name="param-name"><![CDATA[...multi-line value...]]></param><|im_end|>\n'
    '<|im_start|>user\nWeather?<|im_end|>\n<|im_start|>assistant\n'
    '<think>\n\n</think>\n\n<function name="get_weather"><param name="city"><![CDATA[Paris\n'
    'France]]></param><param name="days">3</param></function><|im_end|>\n'
    '<|im_start|>user\n<tool_response>\nSunny\n</tool_response><|im_end|>\n'
    '<|im_start|>assistant\n<think>\n\n</think>\n\nIt is sunny.<|im_end|>\n'
)

TOOLS_WITH_SYSTEM = (
    '<s><|im_start|>system\nBe brief.\n\n# Tools\n\nYou are provided with function signatures '
    'within <tools></tools> XML tags:\n<tools>\n{"type": "function", "function": {"name": "get_weather", '
    '"description": "Today\'s forecast", "parameters": {"type": "object", "properties": {"city": '
    '{"type": "string"}, "days": {"type": "integer"}}}}}\n'
    '</tools>\n\nTool usage guidelines:\n- You may call zero or more functions. If no function '
    'calls are needed, just answer normally and do not include any <function ... </function>.\n'
    '- When calling a function, return an XML object within <function ... </function> using:\n'
    '<function name="function-name"><param name="param-name">param-value</param></function>\n'
    '- param-value may be multi-line. If it contains <, & or newline characters, wrap it in a '
    'CDATA block: <param name="param-name"><![CDATA[...multi-line value...]]></param><|im_end|>\n'
    '<|im_start|>user\nWeather?<|im_end|>\n<|im_start|>assistant\n'
    '<think>\n\n</think>\n\n<function name="get_weather"><param name="city"><![CDATA[Paris\n'
    'France]]></param><param name="days">3</param></function><|im_end|>\n'
    '<|im_start|>user\n<tool_response>\nSunny\n</tool_response><|im_end|>\n'
    '<|im_start|>assistant\n<think>\n\n</think>\n\nIt is sunny.<|im_end|>\n'
)

TWO_TOOL_CALLS = (
    '<s><|im_start|>system\n# Tools\n\nYou are provided with function signatures within <tools></tools> '
    'XML tags:\n<tools>\n{"type": "function", "function": {"name": "get_weather", "description": '
    '"Today\'s forecast", "parameters": {"type": "object", "properties": {"city": {"type": "string"}, '
    '"days": {"type": "integer"}}}}}\n</tools>\n\nTool usage guidelines:\n'
    '- You may call zero or more functions. If no function calls are needed, just answer normally '
    'and do not include any <function ... </function>.\n'
    '- When calling a function, return an XML object within <function ... </function> using:\n'
    '<function name="function-name"><param name="param-name">param-value</param></function>\n'
    '- param-value may be multi-line. If it contains <, & or newline characters, wrap it in a '
    'CDATA block: <param name="param-name"><![CDATA[...multi-line value...]]></param><|im_end|>\n'
    '<|im_start|>user\nWeather?<|im_end|>\n<|im_start|>assistant\n'
    '<think>\n\n</think>\n\n<function name="get_weather"><param name="city"><![CDATA[Paris\n'
    'France]]></param><param name="days">3</param></function>\n'
    '<function name="get_weather"><param name="city">Rome</param><param name="days">1</param></function><|im_end'
    '|>\n<|im_start|>user\n<tool_response>\nSunny\n</tool_response><|im_end|>\n'
    '<|im_start|>assistant\n<think>\n\n</think>\n\nIt is sunny.<|im_end|>\n'
)

PARALLEL_TOOL_RESPONSES = (
    '<s><|im_start|>system\n# Tools\n\nYou are provided with function signatures within <tools></tools> '
    'XML tags:\n<tools>\n{"type": "function", "function": {"name": "get_weather", "description": '
    '"Today\'s forecast", "parameters": {"type": "object", "properties": {"city": {"type": "string"}, '
    '"days": {"type": "integer"}}}}}\n</tools>\n\nTool usage guidelines:\n'
    '- You may call zero or more functions. If no function calls are needed, just answer normally '
    'and do not include any <function ... </function>.\n'
    '- When calling a function, return an XML object within <function ... </function> using:\n'
    '<function name="function-name"><param name="param-name">param-value</param></function>\n'
    '- param-value may be multi-line. If it contains <, & or newline characters, wrap it in a '
    'CDATA block: <param name="param-name"><![CDATA[...multi-line value...]]></param><|im_end|>\n'
    '<|im_start|>user\nWeather?<|im_end|>\n<|im_start|>assistant\n'
    '<think>\n\n</think>\n\n<function name="get_weather"><param name="city"><![CDATA[Paris\n'
    'France]]></param><param name="days">3</param></function>\n'
    '<function name="get_weather"><param name="city">Rome</param><param name="days">1</param></function>'
    '<|im_end|>\n<|im_start|>user\n<tool_response>\nSunny\n</tool_response>\n'
    '<tool_response>\nRain\n</tool_response><|im_end|>\n'
    '<|im_start|>assistant\n<think>\n\n</think>\n\nIt is mixed.<|im_end|>\n'
)


class CharTokenizer:
    """Encodes each character as its own id."""

    bos_token_id = BOS
    eos_token_id = BOS + 1

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]

    def convert_tokens_to_ids(self, token):
        return BOS + 2


def _format(messages, system=None, tools=None):
    template = TEMPLATES["minicpm5"]
    template.enable_thinking = True
    pairs = template.encode_multiturn(CharTokenizer(), messages, system, tools)
    return "".join("<s>" if i == BOS else chr(i) for prompt, response in pairs for i in prompt + response)


def _turns(*contents):
    return [{"role": "user" if i % 2 == 0 else "assistant", "content": c} for i, c in enumerate(contents)]


def _tool_conversation(calls, responses=("Sunny",), answer="It is sunny."):
    return [
        {"role": "user", "content": "Weather?"},
        {"role": "function", "content": json.dumps(calls if len(calls) > 1 else calls[0])},
        # One observation for all the responses, joined as the data converters join consecutive tool messages
        {"role": "observation", "content": "\n</tool_response>\n<tool_response>\n".join(responses)},
        {"role": "assistant", "content": answer},
    ]


def test_turns_without_thinking_get_an_empty_thought_block():
    assert _format(_turns("Hi", "Hello!")) == (
        "<s><|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nHello!<|im_end|>\n"
    )


def test_system_message_and_two_turns():
    assert _format(_turns("Hi", "Hello!", "Bye", "Ok"), system="Be brief.") == (
        "<s><|im_start|>system\nBe brief.<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n"
        "<think>\n\n</think>\n\nHello!<|im_end|>\n<|im_start|>user\nBye<|im_end|>\n<|im_start|>assistant\n"
        "<think>\n\n</think>\n\nOk<|im_end|>\n"
    )


def test_thinking_is_kept_in_every_turn():
    messages = _turns("2+2?", "<think>\nAdd.\n</think>\n\n4", "3+3?", "<think>\nAdd again.\n</think>\n\n6")
    assert _format(messages) == (
        "<s><|im_start|>user\n2+2?<|im_end|>\n<|im_start|>assistant\n<think>\nAdd.\n</think>\n\n4<|im_end|>\n"
        "<|im_start|>user\n3+3?<|im_end|>\n<|im_start|>assistant\n<think>\nAdd again.\n</think>\n\n6<|im_end|>\n"
    )


def test_tool_declarations_without_a_system_message_start_the_system_turn():
    assert _format(_tool_conversation([CALL]), tools=json.dumps([TOOL])) == TOOLS_WITHOUT_SYSTEM


def test_tool_declarations_follow_the_system_message():
    assert _format(_tool_conversation([CALL]), system="Be brief.", tools=json.dumps([TOOL])) == TOOLS_WITH_SYSTEM


def test_two_tool_calls_in_one_message_are_written_on_separate_lines():
    assert _format(_tool_conversation([CALL, CALL2]), tools=json.dumps([TOOL])) == TWO_TOOL_CALLS


def test_responses_to_two_tool_calls_share_one_user_turn():
    conversation = _tool_conversation([CALL, CALL2], responses=("Sunny", "Rain"), answer="It is mixed.")
    assert _format(conversation, tools=json.dumps([TOOL])) == PARALLEL_TOOL_RESPONSES
