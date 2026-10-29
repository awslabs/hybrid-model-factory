"""
Fixtures for the decision model training tests, which are tiny decision models converted with
hmf decision-init, tiny decision datasets, and collated batches. Everything is built locally, so
the tests need no network access.
"""

import json

import pytest
import torch
import transformers
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

import hmf.model.hybrid_zoo.models.model_register  # noqa: F401  # Registers hybrid and decision models
from hmf.data import DecisionDataCollator
from hmf.data.decision_prompt import StrandsPromptFormat, encode_decision_prompt
from hmf.priming.decision_init import DecisionInitConfig, init_decision_model


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Gated DeltaNet's CUDA kernels expect half precision; on CPU, float32
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32
VOCAB_SIZE = 128
HIDDEN_SIZE = 64
HEAD_DIM = 16

# LoRA targets of each tiny source, which are the attention and MLP projections, plus Gated DeltaNet's for Qwen3.5
ATTENTION_MLP_TARGETS = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
LORA_TARGETS = {
    "qwen3": ATTENTION_MLP_TARGETS,
    "qwen3_5": ATTENTION_MLP_TARGETS + ",in_proj_qkv,in_proj_z,in_proj_a,in_proj_b,out_proj",
}


@pytest.fixture(scope="session")
def lora_targets() -> dict:
    """The LoRA targets of each tiny source."""
    return LORA_TARGETS


@pytest.fixture(scope="session")
def device_and_dtype() -> tuple:
    """The device the tests run on, and the dtype of the tiny decision models."""
    return DEVICE, DTYPE


def _tiny_source(source: str) -> "transformers.PreTrainedModel":
    if source == "qwen3":
        config = transformers.Qwen3Config(
            vocab_size=VOCAB_SIZE, hidden_size=HIDDEN_SIZE, intermediate_size=128, num_hidden_layers=2,
            num_attention_heads=4, num_key_value_heads=2, head_dim=16, tie_word_embeddings=True,
        )
        return transformers.Qwen3ForCausalLM(config)

    config_cls = getattr(transformers, "Qwen3_5TextConfig", None)
    model_cls = getattr(transformers, "Qwen3_5ForCausalLM", None)
    if config_cls is None or model_cls is None:
        pytest.skip(f"transformers {transformers.__version__} has no Qwen3.5")
    config = config_cls(
        vocab_size=VOCAB_SIZE, hidden_size=HIDDEN_SIZE, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=1, layer_types=["linear_attention", "full_attention"],
        tie_word_embeddings=True,
    )
    return model_cls(config)


def _save_tiny_tokenizer(path) -> None:
    """A word-level tokenizer, in which each word or punctuation run is one token, unknown unless listed."""
    words = ["state", "question", "options", "answer", "true", "false", "refund", "ignore", "escalate"]
    vocab = {"[UNK]": 0, "[PAD]": 1, **{word: i + 2 for i, word in enumerate(words)}}
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="[UNK]", pad_token="[PAD]").save_pretrained(path)


@pytest.fixture(scope="session")
def decision_model_dir(tmp_path_factory):
    """Returns the directory of a tiny decision model converted from a tiny `source` causal LM."""
    cache = {}

    def make(source: str) -> str:
        if source not in cache:
            root = tmp_path_factory.mktemp(f"decision_{source}")
            torch.manual_seed(0)
            _tiny_source(source).save_pretrained(root / "source")
            _save_tiny_tokenizer(root / "source")
            init_decision_model(
                DecisionInitConfig(
                    base_model_name_or_path=str(root / "source"),
                    output_dir=str(root / "decision"),
                    dtype=DTYPE,
                    decision_head_dim=HEAD_DIM,
                    decision_head_dropout=0.0,
                    seed=0,
                )
            )
            cache[source] = str(root / "decision")
        return cache[source]

    return make


def decision_row(i: int, task: str) -> dict:
    """A decision row; the kind cycles through noul, choice, and score."""
    kind = ("noul", "choice", "score")[i % 3]
    if kind == "noul":
        options, label = [["false", None], ["true", "it holds"]], i % 2
    elif kind == "choice":
        options, label = [["refund", "return it"], ["ignore", None], ["escalate", "to a human"]], i % 3
    else:
        options, label = [["0", "low"], ["1", "mid"], ["2", "high"], ["3", "top"]], i % 4
    return {
        "kind": kind,
        "state": f"state {' '.join(['words'] * (1 + i % 7))} {i}",
        "instructions": f"question {i}?",
        "instruction_variants": [f"other question {i}?"],
        "options": options,
        "label": label,
        "task": task,
    }


@pytest.fixture
def decision_dataset_dir(tmp_path) -> str:
    """A dataset folder with a tiny decision dataset named toy_decision."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    rows = [decision_row(i, task=("a", "b")[i % 2]) for i in range(30)]
    (data_dir / "toy.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    (data_dir / "dataset_info.json").write_text(
        json.dumps({"toy_decision": {"file_name": "toy.jsonl", "formatting": "decision"}})
    )
    return str(data_dir)


@pytest.fixture
def make_batch():
    """Returns a collated batch of four decision rows (with 2 to 4 options), on DEVICE."""

    def make(tokenizer) -> dict:
        features = []
        for i in range(4):
            row = decision_row(i, task="a")
            rendered = StrandsPromptFormat.render(row["state"], row["instructions"], row["kind"], row["options"])
            encoded = encode_decision_prompt(tokenizer, rendered)
            num_options = len(rendered.option_spans)
            features.append(
                {
                    "input_ids": encoded.input_ids,
                    "opt_idx": encoded.opt_idx,
                    "answer_idx": encoded.answer_idx,
                    "label": row["label"],
                    "label_dist": [float(k == row["label"]) for k in range(num_options)],
                    "weight": 1.0,
                    "kind": row["kind"],
                }
            )
        batch = DecisionDataCollator(tokenizer=tokenizer)(features)
        return {k: v.to(DEVICE) for k, v in batch.items()}

    return make
