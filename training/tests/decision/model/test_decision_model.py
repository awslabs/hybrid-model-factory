"""
Tests for the decision model (model/hybrid_zoo/models/decision/) and for converting a causal LM
into one (priming/decision_init.py).

The conversion tests build tiny random causal LMs and a tiny tokenizer locally, so they need no
network access. Each source model is converted once per module, and every conversion test runs
against each source:

    - qwen3_tied / qwen3_untied: a stock Transformer, with and without tied embeddings
    - qwen3_5_text: a stock Qwen3.5 text model (Gated DeltaNet + Attention)
    - qwen3_5_multimodal: a stock Qwen3.5 checkpoint with a vision tower, as published on the Hub

Sources that the installed transformers cannot build are skipped, with the reason.
"""

import copy
import glob
import os

import pytest
import torch
import torch.nn.functional as F
import transformers
import yaml
from safetensors import safe_open
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import AutoConfig, AutoModel, PreTrainedTokenizerFast

import hmf.model.hybrid_zoo.models.model_register  # noqa: F401  # Registers hybrid and decision models
from hmf.model.hybrid_zoo.models.decision.configuration_decision import DecisionConfig
from hmf.model.hybrid_zoo.models.decision.modeling_decision import (
    MASK_VALUE,
    DecisionModel,
    PointerHead,
    masked_log_softmax,
)
from hmf.priming.decision_init import DecisionInitConfig, init_decision_model, load_config


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
AUTOCAST_DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])

HIDDEN_SIZE = 64
VOCAB_SIZE = 128
HEAD_DIM = 16
SEED = 0


# PointerHead


def _head(use_norm: bool = True, dropout: float = 0.0) -> PointerHead:
    head = PointerHead(HIDDEN_SIZE, dim=HEAD_DIM, use_norm=use_norm, dropout=dropout)
    if use_norm:
        # Move the LayerNorm away from its identity init so the tests exercise its affine params
        with torch.no_grad():
            head.norm.weight.normal_(1.0, 0.1)
            head.norm.bias.normal_(0.0, 0.1)
    return head.eval()


def _head_inputs(batch: int = 3, num_options: int = 4):
    answer_states = torch.randn(batch, HIDDEN_SIZE)
    option_states = torch.randn(batch, num_options, HIDDEN_SIZE)
    option_mask = torch.ones(batch, num_options, dtype=torch.bool)
    option_mask[1, 2:] = False
    option_mask[2, 3:] = False
    return answer_states, option_states, option_mask


@pytest.mark.parametrize("use_norm", [True, False])
def test_pointer_head_matches_formula(use_norm: bool):
    torch.manual_seed(SEED)
    head = _head(use_norm=use_norm)
    answer_states, option_states, option_mask = _head_inputs()

    logits = head(answer_states, option_states, option_mask)

    def norm(x):
        if not use_norm:
            return x
        return F.layer_norm(x, (HIDDEN_SIZE,), head.norm.weight, head.norm.bias, head.norm.eps)

    q = norm(answer_states) @ head.q.weight.T + head.q.bias  # [batch, dim]
    k = norm(option_states) @ head.k.weight.T + head.k.bias  # [batch, num_options, dim]
    expected = (k @ q.unsqueeze(-1)).squeeze(-1) / HEAD_DIM**0.5

    assert logits.shape == option_mask.shape
    torch.testing.assert_close(logits[option_mask], expected[option_mask])
    assert (logits[~option_mask] == MASK_VALUE).all()


def test_masked_log_softmax():
    torch.manual_seed(SEED)
    _, _, option_mask = _head_inputs()
    logits = torch.randn(option_mask.shape)

    log_probs = masked_log_softmax(logits, option_mask)

    assert log_probs.dtype == torch.float32
    assert torch.isneginf(log_probs[~option_mask]).all()
    torch.testing.assert_close(log_probs.exp().sum(-1), torch.ones(option_mask.size(0)))
    for i in range(option_mask.size(0)):
        row = option_mask[i]
        torch.testing.assert_close(log_probs[i, row], torch.log_softmax(logits[i, row], dim=-1))

    # A fully masked row is malformed, but must not produce NaNs
    all_masked = masked_log_softmax(logits, torch.zeros_like(option_mask))
    assert not torch.isnan(all_masked).any()


@pytest.mark.parametrize("device", AUTOCAST_DEVICES)
def test_pointer_head_computes_in_fp32(device: str):
    torch.manual_seed(SEED)
    head_bf16 = _head().to(device=device, dtype=torch.bfloat16)
    answer_states, option_states, option_mask = (t.to(device) for t in _head_inputs())
    answer_states, option_states = answer_states.bfloat16(), option_states.bfloat16()

    # The reference is the same (bf16-rounded) weights and inputs, computed in fp32 without autocast
    reference = copy.deepcopy(head_bf16).float()(answer_states.float(), option_states.float(), option_mask)

    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        logits = head_bf16(answer_states, option_states, option_mask)

    assert logits.dtype == torch.float32
    torch.testing.assert_close(logits, reference)


def test_pointer_head_dropout_only_in_training():
    torch.manual_seed(SEED)
    head = _head(dropout=0.5)
    inputs = _head_inputs()

    head.eval()
    assert torch.equal(head(*inputs), head(*inputs))
    head.train()
    assert not torch.equal(head(*inputs), head(*inputs))


def test_reset_parameters_is_pytorch_default():
    head = PointerHead(HIDDEN_SIZE, dim=HEAD_DIM)
    with torch.no_grad():
        for p in head.parameters():
            p.fill_(7.0)

    head.reset_parameters()

    # nn.Linear's default initialization draws the weight and bias uniformly in +-1/sqrt(fan_in)
    bound = HIDDEN_SIZE**-0.5
    for layer in (head.q, head.k):
        assert layer.weight.abs().max() <= bound
        assert layer.bias.abs().max() <= bound
        assert layer.weight.std() > 0
    assert torch.equal(head.norm.weight, torch.ones(HIDDEN_SIZE))
    assert torch.equal(head.norm.bias, torch.zeros(HIDDEN_SIZE))

    # A seeded reset is reproducible
    torch.manual_seed(SEED)
    head.reset_parameters()
    first = {k: v.clone() for k, v in head.state_dict().items()}
    torch.manual_seed(SEED)
    head.reset_parameters()
    for k, v in head.state_dict().items():
        assert torch.equal(v, first[k]), k


# DecisionConfig


def _tiny_qwen3_config(tie_word_embeddings: bool = True) -> "transformers.Qwen3Config":
    return transformers.Qwen3Config(
        vocab_size=VOCAB_SIZE,
        hidden_size=HIDDEN_SIZE,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        tie_word_embeddings=tie_word_embeddings,
    )


def test_decision_config_validation():
    with pytest.raises(ValueError):
        DecisionConfig(decision_head_dim=0)
    with pytest.raises(ValueError):
        DecisionConfig(decision_head_dropout=1.0)


def test_decision_config_round_trip(tmp_path):
    config = DecisionConfig(
        text_config=_tiny_qwen3_config(),
        decision_head_dim=HEAD_DIM,
        decision_head_norm=False,
        decision_head_dropout=0.1,
        source_model_name_or_path="some/model",
    )
    config.save_pretrained(tmp_path)

    loaded = AutoConfig.from_pretrained(tmp_path)

    assert isinstance(loaded, DecisionConfig)
    assert isinstance(loaded.text_config, transformers.Qwen3Config)
    assert loaded.text_config.hidden_size == HIDDEN_SIZE
    assert loaded.decision_head_dim == HEAD_DIM
    assert loaded.decision_head_norm is False
    assert loaded.decision_head_dropout == 0.1
    assert loaded.decision_head_init == "random"
    assert loaded.source_model_name_or_path == "some/model"
    assert loaded.tie_word_embeddings is False


# Options of hmf decision-init


def test_load_config(tmp_path):
    path = tmp_path / "init.yaml"
    path.write_text(yaml.safe_dump({"base_model_name_or_path": "a", "output_dir": "b", "dtype": "bf16", "seed": 3}))

    config = load_config(str(path))

    assert config.dtype == torch.bfloat16
    assert config.seed == 3
    assert config.decision_head_init == "random"

    path.write_text(yaml.safe_dump({"base_model_name_or_path": "a", "output_dir": "b", "head_dim": 8}))
    with pytest.raises(ValueError, match="head_dim"):
        load_config(str(path))


def test_unknown_head_init_rejected_before_loading(tmp_path):
    # The source path does not exist, so the error must come from validation, not from loading
    config = DecisionInitConfig(
        base_model_name_or_path=str(tmp_path / "missing"),
        output_dir=str(tmp_path / "out"),
        decision_head_init="not_an_init",
    )
    with pytest.raises(ValueError, match="decision_head_init"):
        init_decision_model(config)


# Conversion


def _qwen3_5_text_kwargs() -> dict:
    return dict(
        vocab_size=VOCAB_SIZE,
        hidden_size=HIDDEN_SIZE,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        layer_types=["linear_attention", "full_attention"],
        tie_word_embeddings=True,
    )


def _require(*names: str) -> list:
    classes = [getattr(transformers, name, None) for name in names]
    missing = [name for name, cls in zip(names, classes) if cls is None]
    if missing:
        pytest.skip(f"transformers {transformers.__version__} has no {', '.join(missing)}")
    return classes


def _tiny_qwen3_5_text():
    config_cls, model_cls = _require("Qwen3_5TextConfig", "Qwen3_5ForCausalLM")
    return model_cls(config_cls(**_qwen3_5_text_kwargs()))


def _tiny_qwen3_5_multimodal():
    config_cls, model_cls = _require("Qwen3_5Config", "Qwen3_5ForConditionalGeneration")
    config = config_cls(
        text_config=_qwen3_5_text_kwargs(),
        vision_config=dict(
            depth=1,
            hidden_size=32,
            intermediate_size=64,
            num_heads=2,
            out_hidden_size=HIDDEN_SIZE,
            num_position_embeddings=64,
        ),
        tie_word_embeddings=True,
    )
    return model_cls(config)


SOURCES = {
    "qwen3_tied": lambda: transformers.Qwen3ForCausalLM(_tiny_qwen3_config(tie_word_embeddings=True)),
    "qwen3_untied": lambda: transformers.Qwen3ForCausalLM(_tiny_qwen3_config(tie_word_embeddings=False)),
    "qwen3_5_text": _tiny_qwen3_5_text,
    "qwen3_5_multimodal": _tiny_qwen3_5_multimodal,
}


def _save_tiny_tokenizer(path: str) -> None:
    vocab = {"[UNK]": 0, "[PAD]": 1, **{f"tok{i}": i + 2 for i in range(30)}}
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="[UNK]", pad_token="[PAD]").save_pretrained(path)


@pytest.fixture(scope="module", params=list(SOURCES))
def converted(request, tmp_path_factory) -> dict:
    """Builds a tiny source causal LM, saves it, and converts it with init_decision_model."""
    name = request.param
    root = tmp_path_factory.mktemp(name)
    source_dir, output_dir = str(root / "source"), str(root / "decision")

    torch.manual_seed(SEED)
    try:
        source = SOURCES[name]()
    except Exception as e:  # a skip raised by _require is not an Exception and passes through
        pytest.skip(f"could not build a tiny {name} model: {e!r}")
    source.save_pretrained(source_dir)
    _save_tiny_tokenizer(source_dir)

    init_decision_model(
        DecisionInitConfig(
            base_model_name_or_path=source_dir,
            output_dir=output_dir,
            dtype=torch.bfloat16,
            decision_head_dim=HEAD_DIM,
            seed=SEED,
        )
    )
    return {"name": name, "source_dir": source_dir, "output_dir": output_dir}


def _saved_keys(output_dir: str) -> set:
    keys = set()
    for path in glob.glob(os.path.join(output_dir, "*.safetensors")):
        with safe_open(path, framework="pt") as f:
            keys |= set(f.keys())
    return keys


def _batch(device: str) -> dict:
    """Two rows of three options each, the second row right-padded and with only two options."""
    torch.manual_seed(SEED)
    attention_mask = torch.ones(2, 12, dtype=torch.long)
    attention_mask[1, 9:] = 0
    batch = {
        "input_ids": torch.randint(2, VOCAB_SIZE, (2, 12)),
        "attention_mask": attention_mask,
        "opt_idx": torch.tensor([[3, 5, 7], [2, 4, -1]]),
        "answer_idx": torch.tensor([11, 8]),
    }
    return {k: v.to(device) for k, v in batch.items()}


def _forward_dtype() -> torch.dtype:
    return torch.bfloat16 if DEVICE == "cuda" else torch.float32


def test_saved_checkpoint_has_no_lm_head(converted):
    keys = _saved_keys(converted["output_dir"])

    assert keys, "no safetensors files were saved"
    assert not any(k.startswith("lm_head") for k in keys)
    assert all(k.startswith(("model.", "decision_head.")) for k in keys), sorted(keys)[:10]
    assert {"decision_head.q.weight", "decision_head.k.weight"} <= keys


def test_config(converted):
    config = AutoConfig.from_pretrained(converted["output_dir"])
    source_config = AutoConfig.from_pretrained(converted["source_dir"]).get_text_config()

    assert isinstance(config, DecisionConfig)
    assert config.text_config.model_type == source_config.model_type
    assert config.source_model_name_or_path == converted["source_dir"]
    assert config.decision_head_dim == HEAD_DIM
    assert config.decision_head_init == "random"
    assert getattr(config.text_config, "quantization_config", None) is None


def test_backbone_matches_source(converted):
    decision = AutoModel.from_pretrained(converted["output_dir"], dtype=torch.bfloat16)
    source = transformers.AutoModelForCausalLM.from_pretrained(converted["source_dir"], dtype=torch.bfloat16)
    source_backbone = source.base_model

    assert isinstance(decision, DecisionModel)
    assert type(decision.model) is type(source_backbone)
    decision_state, source_state = decision.model.state_dict(), source_backbone.state_dict()
    assert decision_state.keys() == source_state.keys()
    for k in source_state:
        assert torch.equal(decision_state[k], source_state[k]), k


def test_head_restored_exactly(converted):
    # Loading in fp32 isolates the round trip from the fp32-storage question
    decision = AutoModel.from_pretrained(converted["output_dir"], dtype=torch.float32)
    config = decision.config

    # decision-init seeds right before reset_parameters, so this reproduces its head
    expected = PointerHead(
        config.text_config.hidden_size,
        dim=config.decision_head_dim,
        use_norm=config.decision_head_norm,
        dropout=config.decision_head_dropout,
    )
    torch.manual_seed(SEED)
    expected.reset_parameters()

    loaded_state = decision.decision_head.state_dict()
    for k, v in expected.state_dict().items():
        assert torch.equal(loaded_state[k], v), f"{k} was re-initialized or not restored"


def test_head_kept_in_fp32_when_loading_bf16(converted):
    decision = AutoModel.from_pretrained(converted["output_dir"], dtype=torch.bfloat16)

    assert decision.get_input_embeddings().weight.dtype == torch.bfloat16
    head_dtypes = {name: p.dtype for name, p in decision.decision_head.named_parameters()}
    assert all(d == torch.float32 for d in head_dtypes.values()), (
        f"head loaded as {head_dtypes}; check DecisionModel._keep_in_fp32_modules_strict in "
        f"transformers {transformers.__version__}"
    )


def test_forward(converted):
    decision = AutoModel.from_pretrained(converted["output_dir"], dtype=_forward_dtype()).to(DEVICE).eval()
    batch = _batch(DEVICE)

    with torch.no_grad():
        out = decision(**batch)
        hidden_states = decision.model(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        ).last_hidden_state

    assert out.logits.shape == (2, 3)
    assert out.logits.dtype == out.log_probs.dtype == torch.float32
    assert torch.isfinite(out.logits).all()
    assert out.logits[1, 2] == MASK_VALUE
    assert torch.isneginf(out.log_probs[1, 2])
    torch.testing.assert_close(out.log_probs.exp().sum(-1), torch.ones(2, device=DEVICE))

    # The head reads the answer and option positions from the backbone's hidden states
    rows = torch.arange(2, device=DEVICE)
    expected = decision.decision_head(
        hidden_states[rows, batch["answer_idx"]],
        hidden_states[rows.unsqueeze(-1), batch["opt_idx"].clamp_min(0)],
        batch["opt_idx"] >= 0,
    )
    torch.testing.assert_close(out.logits, expected)


@pytest.mark.parametrize("gradient_checkpointing", [False, True])
def test_backward(converted, gradient_checkpointing: bool):
    decision = AutoModel.from_pretrained(converted["output_dir"], dtype=_forward_dtype()).to(DEVICE).train()
    if gradient_checkpointing:
        decision.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    batch = _batch(DEVICE)

    out = decision(**batch)
    loss = F.nll_loss(out.log_probs, torch.tensor([1, 0], device=DEVICE))
    loss.backward()

    assert torch.isfinite(loss)
    for name, p in decision.decision_head.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
    assert decision.decision_head.q.weight.grad.abs().sum() > 0
    assert decision.get_input_embeddings().weight.grad is not None


def test_load_with_sdpa(converted):
    decision = AutoModel.from_pretrained(converted["output_dir"], attn_implementation="sdpa")

    assert decision.model.config._attn_implementation == "sdpa"
