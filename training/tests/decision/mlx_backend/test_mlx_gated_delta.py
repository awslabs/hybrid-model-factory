"""
Tests for the chunked Gated DeltaNet rule of the MLX backend (hmf/train/decision/mlx/gated_delta.py), against mlx-lm's
reference, gated_delta_ops, which loops over the tokens one at a time: the outputs, the final state, and the gradients,
with and without a mask, and that only the layers of the models passed to install_training_kernels use the chunked
rule, and only in training mode.
"""

import pytest


pytest.importorskip("mlx.core", reason="needs MLX (Apple silicon)")
pytest.importorskip("mlx_lm", reason="needs mlx-lm")

import mlx.core as mx
import numpy as np
from mlx_lm.models import qwen3_5
from mlx_lm.models.gated_delta import gated_delta_ops, gated_delta_update

from hmf.train.decision.mlx import gated_delta
from hmf.train.decision.mlx.gated_delta import gated_delta_chunked, gated_delta_update_train, install_training_kernels


BATCH, KEY_HEADS, VALUE_HEADS, KEY_DIM, VALUE_DIM = 2, 2, 4, 16, 8


def _inputs(seq_len: int, seed: int = 0) -> dict:
    """Random inputs of the rule, with q and k normalized and scaled as mlx-lm's layers do."""
    mx.random.seed(seed)
    q = mx.random.normal((BATCH, seq_len, KEY_HEADS, KEY_DIM))
    k = mx.random.normal((BATCH, seq_len, KEY_HEADS, KEY_DIM))
    scale = KEY_DIM**-0.5
    return {
        "q": scale**2 * mx.fast.rms_norm(q, None, 1e-6),
        "k": scale * mx.fast.rms_norm(k, None, 1e-6),
        "v": mx.random.normal((BATCH, seq_len, VALUE_HEADS, VALUE_DIM)),
        "g": mx.random.uniform(0.5, 1.0, (BATCH, seq_len, VALUE_HEADS)),
        "beta": mx.random.uniform(0.0, 1.0, (BATCH, seq_len, VALUE_HEADS)),
        "state": 0.1 * mx.random.normal((BATCH, VALUE_HEADS, VALUE_DIM, KEY_DIM)),
    }


def _close(a: mx.array, b: mx.array, atol: float) -> None:
    np.testing.assert_allclose(np.array(a), np.array(b), atol=atol, rtol=1e-4)


@pytest.mark.parametrize("seq_len", [1, 7, 64, 150])
def test_chunked_rule_matches_the_reference(seq_len):
    inputs = _inputs(seq_len)

    y, state = gated_delta_chunked(**inputs)
    y_ref, state_ref = gated_delta_ops(**inputs)

    assert y.shape == y_ref.shape and state.shape == state_ref.shape
    _close(y, y_ref, atol=1e-5)
    _close(state, state_ref, atol=1e-5)


@pytest.mark.parametrize("chunk_size", [8, 16])
def test_chunk_size_does_not_change_the_result(chunk_size):
    inputs = _inputs(50)

    y, state = gated_delta_chunked(**inputs, chunk_size=chunk_size)
    y_ref, state_ref = gated_delta_ops(**inputs)

    _close(y, y_ref, atol=1e-5)
    _close(state, state_ref, atol=1e-5)


def test_chunk_size_must_be_a_power_of_two():
    with pytest.raises(ValueError, match="power of two"):
        gated_delta_chunked(**_inputs(10), chunk_size=12)


def test_masked_steps_match_the_reference():
    inputs = _inputs(70)
    mask = mx.array(np.random.default_rng(0).random((BATCH, 70)) > 0.3)

    y, state = gated_delta_chunked(**inputs, mask=mask)
    y_ref, state_ref = gated_delta_ops(**inputs, mask=mask)

    _close(y, y_ref, atol=1e-5)
    _close(state, state_ref, atol=1e-5)


@pytest.mark.parametrize("masked", [False, True])
def test_gradients_match_the_reference(masked):
    inputs = _inputs(40, seed=1)
    names = list(inputs)
    mask = mx.array(np.random.default_rng(1).random((BATCH, 40)) > 0.3) if masked else None
    weights = mx.random.normal((BATCH, 40, VALUE_HEADS, VALUE_DIM))
    state_weights = mx.random.normal((BATCH, VALUE_HEADS, VALUE_DIM, KEY_DIM))

    def loss(rule):
        def f(*args):
            y, state = rule(**dict(zip(names, args)), mask=mask)
            return (y * weights).sum() + (state * state_weights).sum()

        return f

    grads = mx.grad(loss(gated_delta_chunked), argnums=list(range(len(names))))(*inputs.values())
    grads_ref = mx.grad(loss(gated_delta_ops), argnums=list(range(len(names))))(*inputs.values())

    for name, grad, grad_ref in zip(names, grads, grads_ref):
        np.testing.assert_allclose(np.array(grad), np.array(grad_ref), atol=1e-4, rtol=1e-3, err_msg=name)


def test_update_matches_mlx_lm():
    inputs = _inputs(90, seed=2)
    a = mx.random.normal((BATCH, 90, VALUE_HEADS))
    b = mx.random.normal((BATCH, 90, VALUE_HEADS))
    A_log = mx.random.normal((VALUE_HEADS,))
    dt_bias = mx.random.normal((VALUE_HEADS,))
    args = (inputs["q"], inputs["k"], inputs["v"], a, b, A_log, dt_bias, inputs["state"])

    y, state = gated_delta_update_train(*args)
    y_ref, state_ref = gated_delta_update(*args, use_kernel=False)

    _close(y, y_ref, atol=1e-5)
    _close(state, state_ref, atol=1e-5)


def _tiny_qwen3_5() -> "qwen3_5.TextModel":
    args = qwen3_5.TextModelArgs(
        model_type="qwen3_5_text", hidden_size=64, intermediate_size=128, num_hidden_layers=4,
        num_attention_heads=2, num_key_value_heads=1, vocab_size=128, linear_num_key_heads=2,
        linear_num_value_heads=4, linear_key_head_dim=32, linear_value_head_dim=32, full_attention_interval=2,
        tie_word_embeddings=True,
    )
    return qwen3_5.TextModel(args)


def test_only_installed_layers_in_training_mode_use_the_chunked_rule(monkeypatch):
    calls = []
    original = gated_delta.gated_delta_update_train

    def counting(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(gated_delta, "gated_delta_update_train", counting)
    installed, other = _tiny_qwen3_5(), _tiny_qwen3_5()
    tokens = mx.array(np.arange(20)[None] % 128)

    assert install_training_kernels(installed) == 2  # the linear attention layers of the four
    assert install_training_kernels(installed) == 2  # installing again changes nothing

    installed.eval()
    eval_out = installed(tokens)
    mx.eval(eval_out)
    assert not calls
    installed.train()
    train_out = installed(tokens)
    mx.eval(train_out)
    assert len(calls) == 2
    _close(train_out, eval_out, atol=1e-4)

    other.train()
    mx.eval(other(tokens))
    assert len(calls) == 2  # a model that was not passed keeps mlx-lm's rule
