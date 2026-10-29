"""
Tests for decision models on HMF's hybrid backbones (model/hybrid_zoo/models/decision/), covering which
backbones get the padding mask (uses_padding_mask), that leaving it out of right-padded prompts is
exact, and, on a GPU, a tiny GKA hybrid through decision-init, LoRA training with evaluation, and
labeling with it as the anchor and as a decision-model teacher. GKA refuses a padding mask in eval
mode, so evaluation and labeling fail if the mask reaches it.

The GPU tests need CUDA (GKA's kernels) and are skipped otherwise.
"""

import json

import pytest
import torch
import transformers
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizerFast

import hmf.model.hybrid_zoo.models.model_register  # noqa: F401  # Registers hybrid and decision models
from hmf.data.decision_prompt import StrandsPromptFormat, encode_decision_prompt
from hmf.model.hybrid_zoo.models.decision.configuration_decision import DecisionConfig
from hmf.model.hybrid_zoo.models.decision.modeling_decision import DecisionModel, uses_padding_mask


WORDS = ["state", "question", "options", "answer", "true", "false", "refund", "ignore", "escalate", "level"]
needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="GKA's kernels need CUDA")


def _save_tokenizer(path) -> None:
    """A word-level tokenizer in which the option numbers 1 to 9 are single tokens."""
    vocab = {"[UNK]": 0, "[PAD]": 1}
    for word in [str(n) for n in range(1, 10)] + WORDS:
        vocab[word] = len(vocab)
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="[UNK]", pad_token="[PAD]").save_pretrained(path)


def _row(i: int) -> dict:
    kind = ("noul", "choice", "score")[i % 3]
    if kind == "noul":
        options, label = [["false", None], ["true", None]], i % 2
    elif kind == "choice":
        options, label = [["refund", "refund"], ["ignore", None], ["escalate", "escalate"]], i % 3
    else:
        options, label = [["0", "level"], ["1", "level"], ["2", "level"]], i % 3
    return {"kind": kind, "state": "state " * (1 + i % 5), "instructions": "question ?", "options": options,
            "label": label, "task": ("a", "b")[i % 2]}


def _encode(tokenizer, rows) -> list:
    encoded = []
    for row in rows:
        options = [tuple(option) for option in row["options"]]
        rendered = StrandsPromptFormat.render(row["state"], row["instructions"], row["kind"], options)
        encoded.append(encode_decision_prompt(tokenizer, rendered))
    return encoded


def _batch(encoded, pad_token_id: int, device: str) -> dict:
    """Right-padded inputs of DecisionModel.forward."""
    seq_len = max(len(e.input_ids) for e in encoded)
    num_options = max(len(e.opt_idx) for e in encoded)
    input_ids = torch.full((len(encoded), seq_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((len(encoded), seq_len), dtype=torch.long)
    opt_idx = torch.full((len(encoded), num_options), -1, dtype=torch.long)
    for i, e in enumerate(encoded):
        input_ids[i, : len(e.input_ids)] = torch.tensor(e.input_ids)
        attention_mask[i, : len(e.input_ids)] = 1
        opt_idx[i, : len(e.opt_idx)] = torch.tensor(e.opt_idx)
    batch = {"input_ids": input_ids, "attention_mask": attention_mask, "opt_idx": opt_idx,
             "answer_idx": torch.tensor([e.answer_idx for e in encoded])}
    return {k: v.to(device) for k, v in batch.items()}


# The padding mask


def test_stock_backbones_get_the_mask_and_leaving_it_out_is_exact(tmp_path):
    _save_tokenizer(tmp_path)
    tokenizer = AutoTokenizer.from_pretrained(tmp_path)
    torch.manual_seed(0)
    text_config = transformers.Qwen3Config(
        vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=2,
        num_key_value_heads=1, head_dim=16,
    )
    model = DecisionModel(DecisionConfig(text_config=text_config, decision_head_dim=16)).eval()
    assert uses_padding_mask(model.model) and model.pass_attention_mask

    # Two prompts of different lengths, so the shorter one is right-padded
    batch = _batch(_encode(tokenizer, [_row(0), _row(4)]), tokenizer.pad_token_id, "cpu")
    assert batch["attention_mask"].min() == 0
    with torch.no_grad():
        with_mask = model(**batch).log_probs
        model.pass_attention_mask = False
        without_mask = model(**batch).log_probs
    torch.testing.assert_close(without_mask, with_mask, rtol=1e-5, atol=1e-5)


# A GKA hybrid (GPU)


@pytest.fixture
def gka_hybrid(tmp_path) -> dict:
    """A tiny HMF hybrid causal LM with a GKA layer and an attention layer, and its decision model."""
    pytest.importorskip("fla")
    from hmf.model.hybrid_zoo.models.hybrid_qwen3.configuration_hybrid_qwen3 import HybridQwen3Config
    from hmf.model.hybrid_zoo.models.hybrid_qwen3.modeling_hybrid_qwen3 import HybridQwen3ForCausalLM
    from hmf.priming.decision_init import DecisionInitConfig, init_decision_model

    source_dir, decision_dir = tmp_path / "source", tmp_path / "decision"
    _save_tokenizer(source_dir)
    torch.manual_seed(0)
    config = HybridQwen3Config(
        vocab_size=64, hidden_size=128, intermediate_size=256, num_hidden_layers=2, num_attention_heads=2,
        num_key_value_heads=1, head_dim=64, hybrid_override_pattern="GKA-*", tie_word_embeddings=True,
    )
    HybridQwen3ForCausalLM(config).to(torch.bfloat16).save_pretrained(source_dir)
    init_decision_model(
        DecisionInitConfig(
            base_model_name_or_path=str(source_dir), output_dir=str(decision_dir), dtype=torch.bfloat16,
            decision_head_dim=16, seed=0,
        )
    )
    return {"root": tmp_path, "source_dir": str(source_dir), "decision_dir": str(decision_dir)}


@needs_cuda
def test_gka_decision_model_without_the_mask(gka_hybrid):
    model = AutoModel.from_pretrained(gka_hybrid["decision_dir"], dtype=torch.bfloat16).to("cuda").eval()
    assert not uses_padding_mask(model.model) and not model.pass_attention_mask

    tokenizer = AutoTokenizer.from_pretrained(gka_hybrid["decision_dir"])
    encoded = _encode(tokenizer, [_row(0), _row(4)])
    with torch.no_grad():
        # In eval mode, where GKA would refuse a padding mask
        padded = model(**_batch(encoded, tokenizer.pad_token_id, "cuda")).log_probs
        alone = model(**_batch(encoded[:1], tokenizer.pad_token_id, "cuda")).log_probs
    # The right-padded row reads as it does alone (bf16)
    num_options = len(encoded[0].opt_idx)
    torch.testing.assert_close(padded[0, :num_options].exp(), alone[0].exp(), rtol=0, atol=2e-2)


@needs_cuda
def test_gka_training_evaluation_and_labeling(gka_hybrid):
    from hmf.data.decision_teachers import CausalLMScorer, DecisionModelTeacher, ScoringQuery
    from hmf.train.tuner import run_exp

    root = gka_hybrid["root"]
    data_dir = root / "data"
    data_dir.mkdir()
    rows = [_row(i) for i in range(24)]
    (data_dir / "toy.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    (data_dir / "dataset_info.json").write_text(
        json.dumps({"toy_decision": {"file_name": "toy.jsonl", "formatting": "decision"}})
    )
    output_dir = root / "output"

    # LoRA on every linear layer but the head (GKA's projections included), with evaluation in eval mode
    run_exp(
        {
            "model_name_or_path": gka_hybrid["decision_dir"],
            "stage": "decision",
            "do_train": True,
            "do_eval": True,
            "finetuning_type": "lora",
            "lora_target": "all",
            "lora_rank": 4,
            "dataset": "toy_decision",
            "dataset_dir": str(data_dir),
            "cutoff_len": 512,
            "val_size": 0.25,
            "max_steps": 2,
            "per_device_train_batch_size": 4,
            "per_device_eval_batch_size": 4,
            "learning_rate": 1e-3,
            "eval_strategy": "steps",
            "eval_steps": 2,
            "save_strategy": "no",
            "logging_steps": 1,
            "output_dir": str(output_dir),
            "overwrite_output_dir": True,
            "overwrite_cache": True,
            "report_to": "none",
            "bf16": True,
            "seed": 0,
        }
    )
    with open(output_dir / "trainer_state.json") as f:
        logs = json.load(f)["log_history"]
    assert any("eval_loss" in entry for entry in logs)

    # The hybrid causal LM as the anchor, which reads right-padded prompts without the mask
    tokenizer = AutoTokenizer.from_pretrained(gka_hybrid["source_dir"])
    scorer = CausalLMScorer(gka_hybrid["source_dir"], "cuda")
    assert not scorer.pass_attention_mask
    encoded = _encode(tokenizer, rows[:6])
    queries = [ScoringQuery(e.input_ids, e.answer_idx, [str(k + 1) for k in range(len(e.opt_idx))]) for e in encoded]
    for probs in scorer(queries):
        assert probs is not None and sum(probs) == pytest.approx(1.0, abs=1e-4)

    # The trained hybrid decision model as a teacher (replay)
    teacher = DecisionModelTeacher(gka_hybrid["decision_dir"], "cuda", adapter_name_or_path=str(output_dir))
    for row, probs in zip(rows[:6], teacher(rows[:6])):
        num_options = 2 if row["kind"] == "noul" else len(row["options"])
        assert probs is not None and len(probs) == num_options and sum(probs) == pytest.approx(1.0, abs=1e-4)
