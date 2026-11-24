"""
Tests that gradient checkpointing keeps the gradients of a model whose layers share keys and values, such as Gemma 4.
A source layer stores its keys and values in a dict that the later layers read, outside the inputs and outputs of
the checkpointed layers, so reentrant checkpointing, which runs each layer's first forward without a graph, drops
the gradients through them. prepare_model_for_training therefore switches such models to non-reentrant
checkpointing. The tests use the tiny hybrid Gemma 4 of test_hybrid_gemma4.py, whose last four layers share keys and
values, and run on the CPU.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from hmf.model.hybrid_zoo.models.decision.configuration_decision import DecisionConfig
from hmf.model.hybrid_zoo.models.hybrid_gemma4.configuration_hybrid_gemma4 import HybridGemma4TextConfig
from hmf.model.hybrid_zoo.models.hybrid_gemma4.modeling_hybrid_gemma4 import HybridGemma4ForCausalLM
from hmf.model.model_utils.checkpointing import prepare_model_for_training, shares_keys_and_values

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from test_hybrid_gemma4 import ALL_ATTENTION, TINY, _ids  # noqa: E402


def _model_args(**overrides):
    return SimpleNamespace(**{
        "upcast_layernorm": False, "disable_gradient_checkpointing": False, "use_unsloth_gc": False,
        "use_reentrant_gc": True, "upcast_lmhead_output": False, **overrides,
    })


def _gemma4():
    torch.manual_seed(0)
    config = HybridGemma4TextConfig(**TINY, hybrid_override_pattern=ALL_ATTENTION)
    config._attn_implementation = "eager"
    return HybridGemma4ForCausalLM(config).train()


def _first_layer_grads(model):
    ids = _ids()
    model(input_ids=ids, labels=ids).loss.backward()
    return {n: p.grad for n, p in model.named_parameters() if ".layers.0." in n}


def _largest_difference(grads, reference):
    return max((grads[n] - reference[n]).abs().max().item() for n in reference)


def test_reentrant_checkpointing_changes_the_gradients_of_layers_before_the_sources():
    reference = _first_layer_grads(_gemma4())
    model = _gemma4()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": True})
    assert _largest_difference(_first_layer_grads(model), reference) > 1e-3


def test_prepare_model_for_training_switches_gemma4_to_non_reentrant_checkpointing():
    reference = _first_layer_grads(_gemma4())
    model, model_args = _gemma4(), _model_args()
    prepare_model_for_training(model, model_args)
    assert model_args.use_reentrant_gc is False
    assert model.model.gradient_checkpointing
    assert _largest_difference(_first_layer_grads(model), reference) == 0.0


def test_prepare_model_for_training_refuses_unsloth_checkpointing_for_gemma4():
    with pytest.raises(ValueError, match="use_unsloth_gc"):
        prepare_model_for_training(_gemma4(), _model_args(use_unsloth_gc=True))


def test_prepare_model_for_training_keeps_reentrant_checkpointing_for_models_without_shared_keys_and_values():
    config = Qwen3Config(vocab_size=97, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                         num_attention_heads=2, num_key_value_heads=1, head_dim=16)
    model_args = _model_args()
    prepare_model_for_training(Qwen3ForCausalLM(config), model_args)
    assert model_args.use_reentrant_gc is True


def test_prepare_model_for_training_leaves_the_setting_when_checkpointing_is_disabled():
    model_args = _model_args(disable_gradient_checkpointing=True)
    prepare_model_for_training(_gemma4(), model_args)
    assert model_args.use_reentrant_gc is True


def test_shares_keys_and_values_reads_the_backbone_config_of_a_decision_model():
    decision_config = DecisionConfig(text_config=HybridGemma4TextConfig(**TINY, hybrid_override_pattern=ALL_ATTENTION))
    assert shares_keys_and_values(SimpleNamespace(config=decision_config))
