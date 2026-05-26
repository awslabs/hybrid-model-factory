"""Qwen-native tool-calling smoke test.

Loads a (hybrid) Qwen-VL model and prompts it with three short scenarios that
should each elicit a single Qwen-format
``<tool_call>{"name": ..., "arguments": ...}</tool_call>`` emission. Reports
pass/fail per case and an aggregate.

Useful as a regression check after tool-calling SFT — if a hybrid VLM trained
on tool-call data drops below 3/3 on these cases, the effective LR or data
mix is likely off.
"""
from __future__ import annotations

import json
from typing import Any

# Side-effect import: registers HybridQwen2_5_VLConfig with the Auto* classes.
from ..model.hybrid_zoo.models import model_register  # noqa: F401


SYSTEM_PROMPT_TMPL = (
    "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.\n\n"
    "# Tools\n\nYou may call one or more functions to assist with the user query.\n\n"
    "You are provided with function signatures within <tools></tools> XML tags:\n"
    "<tools>\n{tools_str}\n</tools>\n\n"
    "For each function call, return a json object with function name and arguments within "
    "<tool_call></tool_call> XML tags:\n"
    '<tool_call>\n{{"name": <function-name>, "arguments": <args-json-object>}}\n</tool_call>'
)


def _build_prompt(tools: list[dict], user: str) -> str:
    tools_str = "\n".join(json.dumps(t.get("function", t)) for t in tools)
    sys_str = SYSTEM_PROMPT_TMPL.format(tools_str=tools_str)
    return (
        f"<|im_start|>system\n{sys_str}<|im_end|>\n"
        f"<|im_start|>user\n{user}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


# Three canonical smoke scenarios. Each: (label, tools, user prompt, expected fn).
DEFAULT_TESTS: list[tuple[str, list[dict], str, str]] = [
    (
        "weather",
        [{
            "name": "get_current_weather", "description": "Get weather",
            "parameters": {
                "type": "object",
                "properties": {"location": {"type": "string"},
                               "unit": {"type": "string"}},
                "required": ["location"],
            },
        }],
        "Weather in San Francisco fahrenheit?",
        "get_current_weather",
    ),
    (
        "math",
        [{
            "name": "calculate", "description": "Compute math",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
            },
        }],
        "What is 47 * 38?",
        "calculate",
    ),
    (
        "translate",
        [
            {"name": "send_email", "description": "Send email",
             "parameters": {"type": "object",
                            "properties": {"to": {"type": "string"}},
                            "required": ["to"]}},
            {"name": "translate_text", "description": "Translate",
             "parameters": {"type": "object",
                            "properties": {"text": {"type": "string"},
                                           "target_lang": {"type": "string"}},
                            "required": ["text", "target_lang"]}},
        ],
        "Translate 'hello' to French.",
        "translate_text",
    ),
]


def run_toolcall_smoke(model_path: str) -> int:
    """Load ``model_path``, run the smoke tests. Returns 0 if all pass, else 1."""
    import torch
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    print(f"Loading {model_path} ...")
    model = AutoModelForImageTextToText.from_pretrained(
        model_path, dtype=torch.bfloat16, trust_remote_code=True
    ).cuda().eval()
    tok = AutoTokenizer.from_pretrained(model_path)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params: {n_params:,}")

    passed = 0
    for name, tools, user, expect in DEFAULT_TESTS:
        prompt = _build_prompt(tools, user)
        ids = tok(prompt, return_tensors="pt").input_ids.cuda()
        with torch.no_grad():
            out = model.generate(
                input_ids=ids, max_new_tokens=200, do_sample=False,
                pad_token_id=tok.eos_token_id,
            )
        gen = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=False)
        print(f"\n=== {name} ===")
        print(f"OUT: {gen[:200]}")

        if "<tool_call>" not in gen:
            print("  ❌ FAIL — no <tool_call>")
            continue
        try:
            payload = gen.split("<tool_call>")[1].split("</tool_call>")[0].strip()
            fn = json.loads(payload).get("name")
        except Exception as e:
            print(f"  ❌ FAIL — parse error: {e}")
            continue

        print(f"  parsed function: {fn}")
        if fn == expect:
            passed += 1
            print("  ✅ PASS")
        else:
            print(f"  ❌ FAIL — expected {expect}")

    total = len(DEFAULT_TESTS)
    print(f"\nTool calling: {passed}/{total} pass")
    return 0 if passed == total else 1


__all__ = ["run_toolcall_smoke"]
