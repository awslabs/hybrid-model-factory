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
    with pytest.raises(ValueError, match="decision_attention"):
        DecisionConfig(decision_attention="non_causal")


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
    assert loaded.decision_attention == "causal"
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


def _tiny_gemma4_multimodal():
    """
    A Gemma 4 checkpoint as Google releases it, multimodal (here without its vision and audio towers), with
    sliding window and full attention layers of different head sizes, shared keys and values, and per-layer
    embeddings, as in E2B. Its last five layers share the keys and values of the last layer of their kind before
    them, layer 3 (sliding) or 4 (full).
    """
    config_cls, model_cls = _require("Gemma4Config", "Gemma4ForConditionalGeneration")
    text_config = dict(
        vocab_size=VOCAB_SIZE,
        hidden_size=HIDDEN_SIZE,
        intermediate_size=128,
        num_hidden_layers=10,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=16,
        global_head_dim=32,
        layer_types=(["sliding_attention"] * 4 + ["full_attention"]) * 2,
        sliding_window=4,
        num_kv_shared_layers=5,
        hidden_size_per_layer_input=8,
        vocab_size_per_layer_input=VOCAB_SIZE,
        max_position_embeddings=256,
        pad_token_id=1,
    )
    return model_cls(config_cls(text_config=text_config, vision_config=None, audio_config=None))


SOURCES = {
    "qwen3_tied": lambda: transformers.Qwen3ForCausalLM(_tiny_qwen3_config(tie_word_embeddings=True)),
    "qwen3_untied": lambda: transformers.Qwen3ForCausalLM(_tiny_qwen3_config(tie_word_embeddings=False)),
    "qwen3_5_text": _tiny_qwen3_5_text,
    "qwen3_5_multimodal": _tiny_qwen3_5_multimodal,
    "gemma4_multimodal": _tiny_gemma4_multimodal,
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
    if converted["name"].startswith("gemma4"):
        # Gemma 4's text weights load only through the multimodal class, whose text decoder is the backbone
        source = transformers.Gemma4ForConditionalGeneration.from_pretrained(
            converted["source_dir"], dtype=torch.bfloat16
        )
        source_backbone = source.model.language_model
    else:
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


# Bidirectional attention (decision_attention)


ATTENTION_BACKBONES = {
    "qwen3": lambda: _tiny_qwen3_config(),
    "qwen3_5_text": lambda: _require("Qwen3_5TextConfig")[0](**_qwen3_5_text_kwargs()),
}


def _decision_models(name: str):
    """
    A causal and a bidirectional decision model with the same weights, in float32, on the GPU if there is one.
    Qwen3.5's Gated DeltaNet kernels run on the GPU only.
    """
    if name.startswith("qwen3_5") and DEVICE != "cuda":
        pytest.skip("Qwen3.5's Gated DeltaNet kernels need a GPU")
    torch.manual_seed(SEED)
    causal = DecisionModel(DecisionConfig(text_config=ATTENTION_BACKBONES[name](), decision_head_dim=HEAD_DIM)).eval()
    config = DecisionConfig(text_config=causal.config.text_config, decision_head_dim=HEAD_DIM,
                            decision_attention="bidirectional")
    bidirectional = DecisionModel(config).eval()
    bidirectional.load_state_dict(causal.state_dict())
    return causal.to(DEVICE), bidirectional.to(DEVICE)


def _hidden_states(model: DecisionModel, input_ids, attention_mask) -> torch.Tensor:
    input_ids, attention_mask = input_ids.to(DEVICE), attention_mask.to(DEVICE)
    with torch.no_grad():
        if model.config.decision_attention == "bidirectional":
            out = model._bidirectional_backbone(input_ids, attention_mask, None)
        else:
            out = model.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False, return_dict=True)
    return out.last_hidden_state


@pytest.mark.parametrize("name", list(ATTENTION_BACKBONES))
def test_bidirectional_attention_sees_later_tokens(name):
    causal, bidirectional = _decision_models(name)
    input_ids = torch.randint(2, VOCAB_SIZE, (1, 10), generator=torch.Generator().manual_seed(SEED))
    changed = input_ids.clone()
    changed[0, -1] = (changed[0, -1] + 1) % VOCAB_SIZE
    mask = torch.ones_like(input_ids)

    # A causal model's early states ignore a change to the last token, but a bidirectional model's do not
    torch.testing.assert_close(
        _hidden_states(causal, input_ids, mask)[:, :9],
        _hidden_states(causal, changed, mask)[:, :9],
        rtol=1e-4,
        atol=1e-4,
    )
    assert not torch.allclose(_hidden_states(bidirectional, input_ids, mask)[:, 0],
                              _hidden_states(bidirectional, changed, mask)[:, 0])


@pytest.mark.parametrize("name", list(ATTENTION_BACKBONES))
def test_bidirectional_attention_ignores_padding(name):
    _, bidirectional = _decision_models(name)
    input_ids = torch.randint(2, VOCAB_SIZE, (1, 8), generator=torch.Generator().manual_seed(SEED))
    padded = torch.cat([input_ids, torch.full((1, 4), 1)], dim=1)
    padded_mask = torch.cat([torch.ones(1, 8, dtype=torch.long), torch.zeros(1, 4, dtype=torch.long)], dim=1)

    # The real tokens' states are the same with and without right padding, up to the GPU kernels, which split
    # prompts of different lengths differently
    torch.testing.assert_close(
        _hidden_states(bidirectional, padded, padded_mask)[:, :8],
        _hidden_states(bidirectional, input_ids, torch.ones_like(input_ids)),
        rtol=1e-3,
        atol=1e-3,
    )


def test_bidirectional_forward_and_saved_setting(tmp_path):
    _, bidirectional = _decision_models("qwen3")
    batch = _batch(DEVICE)
    with torch.no_grad():
        out = bidirectional(**batch)
    assert torch.isfinite(out.log_probs[batch["opt_idx"] >= 0]).all()

    bidirectional.save_pretrained(tmp_path)
    assert AutoConfig.from_pretrained(tmp_path).decision_attention == "bidirectional"


# Running from prompt lengths (seq_lens)

def _flash_attention_models(name: str):
    """
    A causal and a bidirectional decision model with the same weights, in bfloat16, with flash attention, on
    the GPU. Skipped without a GPU or without the flash_attn package.
    """
    if DEVICE != "cuda":
        pytest.skip("flash attention needs a GPU")
    pytest.importorskip("flash_attn")
    torch.manual_seed(SEED)
    text_config = ATTENTION_BACKBONES[name]()
    causal = DecisionModel(DecisionConfig(text_config=text_config, decision_head_dim=HEAD_DIM)).eval()
    bidirectional = DecisionModel(DecisionConfig(text_config=text_config, decision_head_dim=HEAD_DIM,
                                                 decision_attention="bidirectional")).eval()
    bidirectional.load_state_dict(causal.state_dict())
    models = [model.to(DEVICE, torch.bfloat16) for model in (causal, bidirectional)]
    # Set on the backbone once it is on the GPU in bfloat16: DecisionConfig rebuilds the text config it is
    # given, which drops an attention implementation set on it
    for model in models:
        model.model.set_attn_implementation("flash_attention_2")
    return models


def _padded_batch(lengths, seq_len: int) -> dict:
    """Right-padded prompts of the given lengths, each with three options and its answer marker last."""
    generator = torch.Generator().manual_seed(SEED)
    input_ids = torch.ones(len(lengths), seq_len, dtype=torch.long)
    attention_mask = torch.zeros(len(lengths), seq_len, dtype=torch.long)
    for row, length in enumerate(lengths):
        input_ids[row, :length] = torch.randint(2, VOCAB_SIZE, (length,), generator=generator)
        attention_mask[row, :length] = 1
    batch = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "opt_idx": torch.tensor([[length - 4, length - 3, length - 2] for length in lengths]),
        "answer_idx": torch.tensor([length - 1 for length in lengths]),
        "seq_lens": torch.tensor(lengths, dtype=torch.int32),
    }
    return {k: v.to(DEVICE) for k, v in batch.items()}


def _log_probs(model: DecisionModel, batch: dict, varlen: bool) -> torch.Tensor:
    inputs = dict(batch) if varlen else {k: v for k, v in batch.items() if k != "seq_lens"}
    with torch.no_grad():
        return model(**inputs).log_probs


@pytest.mark.parametrize("name", list(ATTENTION_BACKBONES))
@pytest.mark.parametrize("attention", ["causal", "bidirectional"])
def test_varlen_matches_the_masked_forward(name, attention):
    model = dict(zip(["causal", "bidirectional"], _flash_attention_models(name)))[attention]
    batch = _padded_batch([16, 11, 7], seq_len=16)
    torch.testing.assert_close(_log_probs(model, batch, varlen=True), _log_probs(model, batch, varlen=False),
                               rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("name", list(ATTENTION_BACKBONES))
@pytest.mark.parametrize("attention", ["causal", "bidirectional"])
def test_varlen_ignores_padding(name, attention):
    model = dict(zip(["causal", "bidirectional"], _flash_attention_models(name)))[attention]
    # The same prompts padded to two lengths, as length buckets pad them, give the same answers
    torch.testing.assert_close(_log_probs(model, _padded_batch([16, 11, 7], seq_len=32), varlen=True),
                               _log_probs(model, _padded_batch([16, 11, 7], seq_len=16), varlen=True),
                               rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("name", list(ATTENTION_BACKBONES))
def test_varlen_makes_the_cpu_wait_for_nothing(name):
    _, model = _flash_attention_models(name)
    batch = _padded_batch([16, 11, 7], seq_len=16)
    _log_probs(model, batch, varlen=True)  # The first pass may compile kernels and wait for them
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        _log_probs(model, batch, varlen=True)
    finally:
        torch.cuda.set_sync_debug_mode("default")


def test_varlen_needs_flash_attention():
    model = DecisionModel(DecisionConfig(text_config=_tiny_qwen3_config(), decision_head_dim=HEAD_DIM)).eval()
    batch = {k: v.cpu() for k, v in _batch("cpu").items()}
    with pytest.raises(ValueError, match="flash attention"):
        model(**batch, seq_lens=batch["attention_mask"].sum(dim=1))


# Fewer active experts without renormalization (moe_topk_renormalize)

def _tiny_moe_config(**overrides):
    config_cls = _require("Qwen3_5MoeTextConfig")[0]
    kwargs = dict(
        vocab_size=VOCAB_SIZE, hidden_size=HIDDEN_SIZE, num_hidden_layers=2, num_attention_heads=2,
        num_key_value_heads=1, head_dim=16, moe_intermediate_size=32, shared_expert_intermediate_size=32,
        num_experts=8, num_experts_per_tok=4, layer_types=["full_attention", "full_attention"],
        tie_word_embeddings=True,
    )
    return config_cls(**{**kwargs, **overrides})


def _moe_decision_model(**text_overrides) -> DecisionModel:
    torch.manual_seed(SEED)
    config = DecisionConfig(text_config=_tiny_moe_config(**text_overrides), decision_head_dim=HEAD_DIM)
    return DecisionModel(config).eval()


def _routers(model: DecisionModel) -> list:
    return [m for m in model.model.modules() if type(m).__name__.endswith("TopKRouter")]


def test_keeping_topk_weights_at_the_trained_k_changes_nothing():
    plain = _moe_decision_model()
    kept = _moe_decision_model(moe_topk_renormalize=False, moe_topk_norm_k=4)
    kept.load_state_dict(plain.state_dict())
    batch = {k: v.cpu() for k, v in _batch("cpu").items()}
    with torch.no_grad():
        torch.testing.assert_close(kept(**batch).log_probs, plain(**batch).log_probs, rtol=0, atol=0)


def test_keeping_topk_weights_at_the_trained_k_changes_nothing_in_bfloat16():
    # The experts multiply their outputs by the routing weights, so the weights must come back in the dtype
    # transformers' router gives them, or the product runs in another precision and the answers move
    plain = _moe_decision_model().to(torch.bfloat16)
    kept = _moe_decision_model(moe_topk_renormalize=False, moe_topk_norm_k=4).to(torch.bfloat16)
    kept.load_state_dict(plain.state_dict())
    hidden = torch.randn(5, HIDDEN_SIZE, generator=torch.Generator().manual_seed(SEED)).to(torch.bfloat16)
    expected, got = _routers(plain)[0](hidden), _routers(kept)[0](hidden)
    assert got[1].dtype == expected[1].dtype
    # All three outputs match transformers' router: the logits (which its auxiliary loss reads), weights and indices
    assert all(torch.equal(g, e) for g, e in zip(got, expected))
    batch = {k: v.cpu() for k, v in _batch("cpu").items()}
    with torch.no_grad():
        torch.testing.assert_close(kept(**batch).log_probs, plain(**batch).log_probs, rtol=0, atol=0)


def test_fewer_active_experts_keep_their_weights_at_the_trained_k():
    model = _moe_decision_model(num_experts_per_tok=2, moe_topk_renormalize=False, moe_topk_norm_k=4)
    router = _routers(model)[0]
    hidden = torch.randn(5, HIDDEN_SIZE, generator=torch.Generator().manual_seed(SEED))
    logits, weights, indices = router(hidden)
    top4, top4_indices = torch.topk(torch.softmax(hidden @ router.weight.T, dim=-1), 4, dim=-1)
    # The two experts kept get the weights they have among the top four, which sum to less than 1
    torch.testing.assert_close(weights, top4[:, :2] / top4.sum(dim=-1, keepdim=True))
    assert torch.equal(indices, top4_indices[:, :2])
    assert (weights.sum(dim=-1) < 1).all() and weights.dtype == torch.float32
    torch.testing.assert_close(logits, hidden @ router.weight.T)


def test_transformers_renormalizes_fewer_active_experts_by_default():
    router = _routers(_moe_decision_model(num_experts_per_tok=2))[0]
    _, weights, _ = router(torch.randn(5, HIDDEN_SIZE, generator=torch.Generator().manual_seed(SEED)))
    torch.testing.assert_close(weights.sum(dim=-1), torch.ones(5))


def test_keeping_topk_weights_survives_saving_and_loading(tmp_path):
    model = _moe_decision_model(num_experts_per_tok=2, moe_topk_renormalize=False, moe_topk_norm_k=4)
    model.save_pretrained(tmp_path)
    loaded = AutoModel.from_pretrained(tmp_path).eval()
    assert all(router.norm_k == 4 for router in _routers(loaded)) and len(_routers(loaded)) == 2
    batch = {k: v.cpu() for k, v in _batch("cpu").items()}
    with torch.no_grad():
        torch.testing.assert_close(loaded(**batch).log_probs, model(**batch).log_probs)


def test_keeping_topk_weights_needs_a_mixture_of_experts():
    config = _tiny_qwen3_config()
    config.moe_topk_renormalize = False
    with pytest.raises(ValueError, match="no TopK routers"):
        DecisionModel(DecisionConfig(text_config=config, decision_head_dim=HEAD_DIM))
