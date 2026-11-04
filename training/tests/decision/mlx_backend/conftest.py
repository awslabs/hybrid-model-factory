"""
Fixtures for the tests of the MLX backend of decision models (hmf/train/decision/mlx/), which are tiny Qwen3 and
Qwen3.5 decision models converted with hmf decision-init, in float32. The test modules are skipped where MLX and mlx-lm
are not installed (they run on Apple silicon).

The tiny models are initialized with larger weights than transformers' default (initializer_range 0.2 rather than
0.02). With the default, the queries and keys of Gated DeltaNet are so small that the epsilon of their normalization,
which mlx-lm and transformers add in slightly different places, changes the outputs by about 1e-3, while with weights
of a realistic size the two agree to about 1e-5. The key and value dimensions of Gated DeltaNet are 32, the smallest
size mlx-lm's Metal kernel supports.
"""

import pytest
import torch
import transformers
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

import hmf.model.hybrid_zoo.models.model_register  # noqa: F401  # Registers decision models
from hmf.priming.decision_init import DecisionInitConfig, init_decision_model


VOCAB_SIZE = 128
HIDDEN_SIZE = 64
HEAD_DIM = 16
PAD_TOKEN_ID = 1


def tiny_source(source: str) -> "transformers.PreTrainedModel":
    """A tiny random causal LM of the source architecture."""
    common = dict(
        vocab_size=VOCAB_SIZE, hidden_size=HIDDEN_SIZE, intermediate_size=128, num_hidden_layers=2,
        tie_word_embeddings=True, initializer_range=0.2,
    )
    if source == "qwen3":
        config = transformers.Qwen3Config(num_attention_heads=4, num_key_value_heads=2, head_dim=16, **common)
        return transformers.Qwen3ForCausalLM(config)
    config = transformers.Qwen3_5TextConfig(
        num_attention_heads=2, num_key_value_heads=1, layer_types=["linear_attention", "full_attention"],
        linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=32, linear_value_head_dim=32,
        **common,
    )
    return transformers.Qwen3_5ForCausalLM(config)


def _save_tiny_tokenizer(path) -> None:
    words = ["state", "question", "options", "answer", "true", "false", "refund", "ignore", "escalate", "words"]
    words += [f"tok{i}" for i in range(30)]
    vocab = {"[UNK]": 0, "[PAD]": PAD_TOKEN_ID, **{word: i + 2 for i, word in enumerate(words)}}
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="[UNK]", pad_token="[PAD]").save_pretrained(path)


@pytest.fixture(scope="session")
def decision_model_dir(tmp_path_factory):
    """Returns the folder of a tiny float32 decision model converted from a tiny causal LM of the source."""
    cache = {}

    def make(source: str) -> str:
        if source not in cache:
            root = tmp_path_factory.mktemp(f"mlx_decision_{source}")
            torch.manual_seed(0)
            tiny_source(source).save_pretrained(root / "source")
            _save_tiny_tokenizer(root / "source")
            init_decision_model(
                DecisionInitConfig(
                    base_model_name_or_path=str(root / "source"), output_dir=str(root / "decision"),
                    dtype=torch.float32, decision_head_dim=HEAD_DIM, decision_head_dropout=0.0, seed=0,
                )
            )
            cache[source] = str(root / "decision")
        return cache[source]

    return make


@pytest.fixture(scope="session")
def batch() -> dict:
    """A right-padded batch of three prompts of different lengths, with padded options in two of them."""
    generator = torch.Generator().manual_seed(1)
    lengths, seq_len = [150, 97, 40], 150
    input_ids = torch.randint(2, VOCAB_SIZE, (len(lengths), seq_len), generator=generator)
    attention_mask = torch.zeros(len(lengths), seq_len, dtype=torch.long)
    for row, length in enumerate(lengths):
        input_ids[row, length:] = PAD_TOKEN_ID
        attention_mask[row, :length] = 1
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "opt_idx": torch.tensor([[10, 50, 120, 140], [5, 60, 90, -1], [3, 20, -1, -1]]),
        "answer_idx": torch.tensor([149, 96, 39]),
    }
