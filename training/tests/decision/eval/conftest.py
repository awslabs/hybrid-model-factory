"""
Fixtures for the tests of hmf/eval/decision/: a tiny decision model, a LoRA adapter trained on it
(random) and a data file of decision rows. Everything runs on the CPU, offline.
"""

import json

import pytest
import torch
import transformers
from peft import LoraConfig, get_peft_model
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import AutoModel, PreTrainedTokenizerFast

import hmf.model.hybrid_zoo.models.model_register  # noqa: F401  # Registers decision models
from hmf.priming.decision_init import DecisionInitConfig, init_decision_model


WORDS = ["state", "question", "options", "answer", "true", "false", "refund", "ignore", "escalate", "low", "high"]


def decision_row(i: int) -> dict:
    kind = ("noul", "choice", "score")[i % 3]
    if kind == "noul":
        options, label = [["false", None], ["true", None]], i % 2
    elif kind == "choice":
        options, label = [["refund", None], ["ignore", None], ["escalate", None]], i % 3
    else:
        options, label = [["0", "low"], ["1", "low"], ["2", "high"]], i % 3
    return {"kind": kind, "state": "state " * (1 + i % 4), "instructions": "question ?", "options": options,
            "label": label, "task": ("a", "b")[i % 2]}


@pytest.fixture(scope="module")
def trained(tmp_path_factory) -> dict:
    """A tiny decision model, a LoRA adapter trained on it (random), a data file, and the row maker."""
    root = tmp_path_factory.mktemp("eval")
    source_dir, decision_dir, adapter_dir = root / "source", root / "decision", root / "adapter"
    vocab = {"[UNK]": 0, "[PAD]": 1}
    for word in [str(n) for n in range(10)] + WORDS:
        vocab[word] = len(vocab)
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="[UNK]", pad_token="[PAD]").save_pretrained(
        source_dir
    )
    torch.manual_seed(0)
    transformers.Qwen3ForCausalLM(transformers.Qwen3Config(
        vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=2,
        num_key_value_heads=1, head_dim=16, tie_word_embeddings=True,
    )).save_pretrained(source_dir)
    init_decision_model(DecisionInitConfig(
        base_model_name_or_path=str(source_dir), output_dir=str(decision_dir), dtype=torch.float32,
        decision_head_dim=16, decision_head_dropout=0.0, seed=0,
    ))
    model = AutoModel.from_pretrained(str(decision_dir), dtype=torch.float32)
    lora = LoraConfig(target_modules=["q_proj", "v_proj"], modules_to_save=["decision_head"], init_lora_weights=False)
    get_peft_model(model, lora).save_pretrained(str(adapter_dir))

    data = root / "rows.jsonl"
    rows = [decision_row(i) for i in range(90)]
    rows.append({**decision_row(0), "state": "state " * 300})  # too long for a small max_len
    data.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return {"root": root, "model_dir": str(decision_dir), "adapter_dir": str(adapter_dir), "data": str(data),
            "row": decision_row}
