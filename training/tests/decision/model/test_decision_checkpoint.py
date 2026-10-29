"""
Checks a decision model checkpoint written by hmf decision-init against the causal LM it was
converted from. Unlike test_decision_model.py, which converts tiny random models, these tests run
on a real converted checkpoint:

    hmf decision-init examples/decision/init/qwen3_5_2b/pointer.yaml
    HMF_DECISION_MODEL=./models/Qwen3.5-2B-Base-Decision \
        pytest tests/decision/model/test_decision_checkpoint.py -v -rs

The source model is read from the checkpoint's config (source_model_name_or_path). Set
HMF_DECISION_SOURCE to load it from elsewhere, e.g. if the checkpoint was converted from a relative
path on another machine. All tests are skipped when HMF_DECISION_MODEL is not set.
"""

import glob
import os

import pytest
import torch
import torch.nn.functional as F
from safetensors import safe_open
from transformers import AutoModel, AutoTokenizer

import hmf.model.hybrid_zoo.models.model_register  # noqa: F401  # Registers hybrid and decision models
from hmf.model.hybrid_zoo.models.decision.configuration_decision import DecisionConfig
from hmf.model.hybrid_zoo.models.decision.modeling_decision import MASK_VALUE, DecisionModel
from hmf.priming.hybridize_model import load_pretrained_model


DECISION_MODEL = os.getenv("HMF_DECISION_MODEL")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

pytestmark = pytest.mark.skipif(DECISION_MODEL is None, reason="set HMF_DECISION_MODEL to a decision-init output")


# Two prompts in the layout the decision model is trained on, a 3-option choice and a yes/no
# question, so the batch is right-padded and the second row has a padded option slot.
PROMPTS = [
    {
        "state": "Order 1182 was charged twice. The customer asked for the duplicate charge to be removed.",
        "kind": "choice",
        "header": "Select exactly one option.",
        "instructions": "What should support do next?",
        "options": [
            "1. refund — return the duplicate payment",
            "2. ignore — take no action",
            "3. escalate — send the case to a human agent",
        ],
        "label": 0,
    },
    {
        "state": "The meeting was moved from Tuesday to Thursday.",
        "kind": "noul",
        "header": "Decide whether the statement is true of the state.",
        "instructions": "The meeting takes place on Tuesday.",
        "options": [
            "1. false — the statement does not hold for this state",
            "2. true — the statement holds for this state",
        ],
        "label": 0,
    },
]


def _render(prompt: dict) -> str:
    return (
        f"<state>\n{prompt['state']}\n</state>\n"
        f"<question type=\"{prompt['kind']}\">\n{prompt['header']}\n{prompt['instructions']}\n<options>\n"
        + "\n".join(prompt["options"])
        + "\n</options>\n</question>\n<answer>"
    )


def _last_token_in_span(offsets: list, start: int, end: int) -> int:
    """Index of the last token lying entirely within characters [start, end)."""
    last = -1
    for j, (lo, hi) in enumerate(offsets):
        if hi > lo and lo >= start and hi <= end:
            last = j
    assert last >= 0, f"no token lies within characters ({start}, {end})"
    return last


def _batch(tokenizer) -> dict:
    """Tokenizes PROMPTS (right-padded) and locates each option line and the answer marker."""
    texts = [_render(p) for p in PROMPTS]
    enc = tokenizer(texts, padding=True, padding_side="right", return_offsets_mapping=True, return_tensors="pt")

    opt_idx, answer_idx = [], []
    for text, prompt, offsets in zip(texts, PROMPTS, enc["offset_mapping"].tolist()):
        positions = []
        for line in prompt["options"]:
            start = text.index(line)
            positions.append(_last_token_in_span(offsets, start, start + len(line)))
        opt_idx.append(positions)
        answer_start = text.rindex("<answer>")
        answer_idx.append(_last_token_in_span(offsets, answer_start, len(text)))

    width = max(len(p) for p in opt_idx)
    batch = {
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        "opt_idx": torch.tensor([p + [-1] * (width - len(p)) for p in opt_idx]),
        "answer_idx": torch.tensor(answer_idx),
    }
    return {k: v.to(DEVICE) for k, v in batch.items()}


@pytest.fixture(scope="module")
def decision() -> DecisionModel:
    return AutoModel.from_pretrained(DECISION_MODEL, dtype=torch.bfloat16).to(DEVICE).eval()


@pytest.fixture(scope="module")
def source(decision):
    """The source causal LM, loaded exactly as decision-init loads it."""
    path = os.getenv("HMF_DECISION_SOURCE") or decision.config.source_model_name_or_path
    model, _, _ = load_pretrained_model(path, dtype=torch.bfloat16)
    return model.to(DEVICE).eval()


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained(DECISION_MODEL)


def test_config(decision, source):
    config = decision.config

    assert isinstance(config, DecisionConfig)
    assert config.text_config.model_type == source.config.model_type
    assert config.source_model_name_or_path
    assert getattr(config.text_config, "quantization_config", None) is None


def test_saved_weights_have_no_lm_head():
    keys = set()
    for path in glob.glob(os.path.join(DECISION_MODEL, "*.safetensors")):
        with safe_open(path, framework="pt") as f:
            keys |= set(f.keys())

    assert keys, "no safetensors files were saved"
    assert not any(k.startswith("lm_head") for k in keys)
    assert all(k.startswith(("model.", "decision_head.")) for k in keys), sorted(keys)[:10]


def test_parameter_counts(decision, source):
    config = decision.config
    hidden_size = config.text_config.hidden_size
    dim = config.decision_head_dim

    backbone_params = sum(p.numel() for p in decision.model.parameters())
    source_backbone_params = sum(p.numel() for p in source.base_model.parameters())
    head_params = sum(p.numel() for p in decision.decision_head.parameters())
    expected_head_params = 2 * (hidden_size * dim + dim) + (2 * hidden_size if config.decision_head_norm else 0)

    print(
        f"\nsource: {sum(p.numel() for p in source.parameters()):,} | backbone: {backbone_params:,} | "
        f"head: {head_params:,} | decision model: {sum(p.numel() for p in decision.parameters()):,}"
    )
    assert backbone_params == source_backbone_params
    assert head_params == expected_head_params


def test_backbone_weights_match_source(decision, source):
    decision_state, source_state = decision.model.state_dict(), source.base_model.state_dict()

    assert decision_state.keys() == source_state.keys()
    for k in source_state:
        assert torch.equal(decision_state[k], source_state[k]), k


def test_head_stored_in_fp32(decision):
    head_dtypes = {name: p.dtype for name, p in decision.decision_head.named_parameters()}

    assert decision.get_input_embeddings().weight.dtype == torch.bfloat16
    assert all(d == torch.float32 for d in head_dtypes.values()), head_dtypes


def test_tokenizer_matches_source(decision, tokenizer):
    source_path = os.getenv("HMF_DECISION_SOURCE") or decision.config.source_model_name_or_path
    source_tokenizer = AutoTokenizer.from_pretrained(source_path)

    assert tokenizer.get_vocab() == source_tokenizer.get_vocab()


def test_no_processor_saved():
    # A processor config makes HMF's load_tokenizer load a processor and treat the model as
    # multimodal; the decision model is text-only.
    assert not os.path.exists(os.path.join(DECISION_MODEL, "preprocessor_config.json")), (
        "preprocessor_config.json was copied (copy_auxiliary_files: true?)"
    )


def test_hidden_states_reproduce_source_logits(decision, source, tokenizer):
    # The source LM's own logits, recomputed from the decision backbone's hidden states through
    # the source's LM head, must match, so that nothing was lost or remapped in the conversion.
    batch = _batch(tokenizer)
    real = batch["attention_mask"].bool()
    # As in DecisionModel, HMF's hybrids get no padding mask (exact for right-padded prompts)
    mask = batch["attention_mask"] if decision.pass_attention_mask else None

    with torch.no_grad():
        hidden_states = decision.model(input_ids=batch["input_ids"], attention_mask=mask).last_hidden_state
        source_logits = source(input_ids=batch["input_ids"], attention_mask=mask, use_cache=False).logits
        reconstructed = source.get_output_embeddings()(hidden_states)

    softcap = getattr(source.config, "final_logit_softcapping", None)
    if softcap:
        reconstructed = softcap * torch.tanh(reconstructed / softcap)

    torch.testing.assert_close(reconstructed[real].float(), source_logits[real].float(), rtol=1.6e-2, atol=1e-2)


def test_forward_and_backward(decision, tokenizer):
    batch = _batch(tokenizer)
    labels = torch.tensor([p["label"] for p in PROMPTS], device=DEVICE)

    decision.train()
    try:
        out = decision(**batch)
        loss = F.nll_loss(out.log_probs, labels)
        loss.backward()

        print(f"\noption probabilities (untrained head): {out.log_probs.exp().tolist()}")
        assert out.logits.shape == (2, 3)
        assert out.logits.dtype == out.log_probs.dtype == torch.float32
        assert torch.isfinite(out.logits).all()
        assert out.logits[1, 2] == MASK_VALUE
        assert torch.isneginf(out.log_probs[1, 2])
        torch.testing.assert_close(out.log_probs.exp().sum(-1).detach(), torch.ones(2, device=DEVICE))

        assert torch.isfinite(loss)
        for name, p in decision.decision_head.named_parameters():
            assert p.grad is not None and torch.isfinite(p.grad).all(), name
        assert decision.get_input_embeddings().weight.grad is not None
    finally:
        decision.zero_grad(set_to_none=True)
        decision.eval()
