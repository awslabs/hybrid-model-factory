"""
A chunked Gated DeltaNet rule in MLX, which MLX can differentiate, for training Qwen3.5 backbones with mlx-lm.

mlx-lm's Qwen3.5 layers call `gated_delta_update(..., use_kernel=not self.training)`. Its Metal kernel has no
backward pass, so in training mode mlx-lm falls back to `gated_delta_ops`, a Python loop over every token, which is
very slow on long prompts. This module computes the same recurrence chunk by chunk, as transformers'
`torch_chunk_gated_delta_rule` and fla's `chunk_gated_delta_rule` do, with plain MLX operations, so that `mx.grad`
differentiates it and the Python loop runs once per chunk rather than once per token.

The shapes follow mlx-lm's `mlx_lm/models/gated_delta.py`. The queries and keys are [batch, seq_len, key_heads,
key_dim], the values are [batch, seq_len, value_heads, value_dim], the gates g and beta are [batch, seq_len,
value_heads], and the state is [batch, value_heads, value_dim, key_dim]. Value head h reads key head
h // (value_heads // key_heads), and the queries and keys arrive already normalized and scaled by the layer. Per
token, with S the [value_dim, key_dim] state, the rule is

    S_t = g_t S_{t-1} + delta_t k_t^T,   delta_t = beta_t (v_t - g_t S_{t-1} k_t),   y_t = S_t q_t.

install_training_kernels makes the Gated DeltaNet layers of a model use this rule in training mode only, so that
evaluation and inference keep mlx-lm's Metal kernel.
"""

import contextvars
from typing import Optional, Tuple

import mlx.core as mx
from mlx import nn


__all__ = ["CHUNK_SIZE", "gated_delta_chunked", "gated_delta_update_train", "install_training_kernels"]

# The number of tokens per chunk, a power of two
CHUNK_SIZE = 64
# The smallest normal float32
_FLOAT32_TINY = 1.1754944e-38
# Decay factors below e^-80 (about 2e-35) count as 0
_LOG_FLUSH = -80.0


def _flush(x: mx.array) -> mx.array:
    return mx.where(mx.abs(x) < _FLOAT32_TINY, 0, x)


@mx.custom_function
def _ftz(x: mx.array) -> mx.array:
    """
    Flushes float32 denormals to zero, in the forward pass and in the gradient of the backward pass. Strong decays
    multiply many tiny factors together, and denormal operands slow MLX's CPU matmuls down about 70 times, while
    values below 1e-38 are irrelevant next to the terms of order 1 they are added to.
    """
    return _flush(x)


@_ftz.vjp
def _ftz_vjp(primals, cotangent, output):
    return _flush(cotangent)


def _decay(log_decay: mx.array) -> mx.array:
    return mx.where(log_decay > _LOG_FLUSH, mx.exp(log_decay), 0)


def _unit_lower_inverse(a: mx.array) -> mx.array:
    """
    Computes (I + a)^-1 for a strictly lower triangular a of shape [..., n, n], with n a power of two, by inverting
    blocks recursively. For the halves [[L1, 0], [A21, L2]], the inverse is [[X1, 0], [-X2 A21 X1, X2]], with
    X1 = L1^-1 and X2 = L2^-1 inverted together one level down. The recursion has log2(n) levels of two matmuls,
    each restricted to the block that couples the two halves, and every product is of exact inverses, which stay
    bounded (a Neumann series, the sum of (-a)^k, overflows in float32 instead).
    """
    n = a.shape[-1]
    if n == 1:
        return mx.ones_like(a)
    h = n // 2
    a21 = a[..., h:, :h]
    diag = _unit_lower_inverse(mx.stack([a[..., :h, :h], a[..., h:, h:]], axis=-3))
    x11, x22 = diag[..., 0, :, :], diag[..., 1, :, :]
    top = mx.concatenate([x11, mx.zeros_like(a21)], axis=-1)
    bottom = mx.concatenate([_ftz(-(x22 @ a21) @ x11), x22], axis=-1)
    return mx.concatenate([top, bottom], axis=-2)


def _heads_first(x: mx.array, pad: int) -> mx.array:
    """
    Moves the heads before the tokens, from [batch, seq_len, heads, ...] to [batch, heads, seq_len + pad, ...], in
    float32, with pad zeros appended to the tokens.
    """
    x = mx.swapaxes(x.astype(mx.float32), 1, 2)
    if pad:
        x = mx.pad(x, [(0, 0), (0, 0), (0, pad)] + [(0, 0)] * (x.ndim - 3))
    return x


def _chunked(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    log_g: mx.array,
    beta: mx.array,
    state: mx.array,
    state_dtype: mx.Dtype,
    mask: Optional[mx.array],
    chunk_size: int,
) -> Tuple[mx.array, mx.array]:
    """The chunked rule on the logarithms of the gates, which gated_delta_chunked documents."""
    B, T, Hk, _ = q.shape
    Hv = v.shape[-2]
    y_dtype = q.dtype
    if log_g.ndim != 3:
        raise ValueError(f"Only scalar gates g of shape [batch, seq_len, heads] are supported, not {log_g.shape}.")
    if chunk_size < 1 or chunk_size & (chunk_size - 1):
        raise ValueError(f"The chunk size must be a power of two, not {chunk_size}.")
    if (repeat := Hv // Hk) > 1:
        q = mx.repeat(q, repeat, -2)
        k = mx.repeat(k, repeat, -2)
    if mask is not None:
        # gated_delta_ops leaves the state of a masked step unchanged, which g = 1 (log g = 0) and beta = 0 do
        g_true, beta_true = mx.exp(log_g), beta
        log_g = mx.where(mask[..., None], log_g, 0)
        beta = mx.where(mask[..., None], beta, 0)

    # Short inputs use the smallest power of two at least as long as they are. The padded steps have beta = 0 and
    # log g = 0, so they leave the state unchanged.
    c = min(chunk_size, 1 << (T - 1).bit_length())
    n = -(-T // c)
    pad = n * c - T
    q, k, v = (_heads_first(x, pad).reshape(B, Hv, n, c, -1) for x in (q, k, v))
    log_g, beta = (_heads_first(x, pad).reshape(B, Hv, n, c) for x in (log_g, beta))

    # Within a chunk, Gamma_i is the product of g_s for s <= i, and decay[i, j] = Gamma_i / Gamma_j for j <= i. The
    # decays come from sums of log g over the segments j < s <= i, as in Mamba-2, rather than from differences of
    # cumulative sums, which lose precision when the two sums are close.
    idx = mx.arange(c)
    strict = idx[:, None] > idx[None, :]
    lower = idx[:, None] >= idx[None, :]
    seg = mx.cumsum(mx.where(strict, log_g[..., :, None], 0), axis=-2)
    decay = mx.where(lower, _decay(seg), 0)  # seg is 0 above the diagonal, so nothing overflows there
    gamma = _decay(mx.cumsum(log_g, axis=-1))[..., None]
    to_end = decay[..., -1, :, None]  # Gamma_last / Gamma_j
    chunk_decay = gamma[..., -1:, :]  # Gamma_last, [batch, heads, num_chunks, 1, 1]

    # With S0 the chunk's initial state, delta_i = beta_i (v_i - Gamma_i S0 k_i - sum_{j<i} decay_ij (k_i.k_j)
    # delta_j). In matrix form, (I + A) delta = beta v - beta Gamma K S0^T, with A_ij = beta_i decay_ij k_i.k_j
    # strictly lower triangular, so delta = u - w S0^T with u = (I + A)^-1 beta v and w = (I + A)^-1 beta Gamma k.
    kk = _ftz((k @ mx.swapaxes(k, -1, -2)) * mx.where(strict, decay, 0))
    inv = _unit_lower_inverse(_ftz(beta[..., None] * kk))
    u = _ftz(inv @ (v * beta[..., None]))
    w = _ftz(inv @ (k * (beta[..., None] * gamma)))
    attn = _ftz((q @ mx.swapaxes(k, -1, -2)) * decay)
    q_gamma = _ftz(q * gamma)
    k_to_end = _ftz(mx.swapaxes(k * to_end, -1, -2))
    if mask is not None:
        k_gamma = _ftz(k * gamma)

    # The state is carried transposed, [batch, heads, key_dim, value_dim], so that every product below is a matmul
    s = mx.swapaxes(state.astype(mx.float32), -1, -2)
    ys, ks = [], []
    for i in range(n):
        delta = _ftz(u[:, :, i] - w[:, :, i] @ s)
        # y_i = Gamma_i S0 q_i + sum_{j<=i} decay_ij q_i.k_j delta_j
        ys.append(q_gamma[:, :, i] @ s + attn[:, :, i] @ delta)
        if mask is not None:
            # S_{i-1} k_i, the same sum with k_i as the query
            ks.append(k_gamma[:, :, i] @ s + kk[:, :, i] @ delta)
        s = _ftz(s * chunk_decay[:, :, i] + k_to_end[:, :, i] @ delta)
    y = mx.stack(ys, axis=2).reshape(B, Hv, n * c, -1)[:, :, :T]

    if mask is not None:
        # At a masked step, gated_delta_ops still outputs y_t = q_t . (g_t S_{t-1} + k_t beta_t (v_t - g_t S_{t-1}
        # k_t)), as if the step had been applied. y holds q_t . S_{t-1} at those steps, so that output is rebuilt
        # from it.
        sk = mx.stack(ks, axis=2).reshape(B, Hv, n * c, -1)[:, :, :T]
        g_t = _heads_first(g_true, 0)[..., None]
        beta_t = _heads_first(beta_true, 0)[..., None]
        qk = (q * k).sum(-1).reshape(B, Hv, n * c)[:, :, :T, None]
        v = v.reshape(B, Hv, n * c, -1)[:, :, :T]
        y = mx.where(mask[:, None, :, None], y, g_t * y + beta_t * qk * (v - g_t * sk))
    return mx.swapaxes(y, 1, 2).astype(y_dtype), mx.swapaxes(s, -1, -2).astype(state_dtype)


def _initial_state(q: mx.array, v: mx.array, state: Optional[mx.array], *step_inputs: mx.array):
    """
    Returns the initial state, which is zeros as in mlx-lm when none is given, and the dtype in which
    gated_delta_ops would return the final state.
    """
    if state is None:
        B, _, _, Dk = q.shape
        Hv, Dv = v.shape[-2:]
        state = mx.zeros((B, Hv, Dv, Dk), dtype=mx.float32)
    return state, mx.result_type(state, *step_inputs)


def gated_delta_chunked(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    g: mx.array,
    beta: mx.array,
    state: Optional[mx.array] = None,
    mask: Optional[mx.array] = None,
    chunk_size: int = CHUNK_SIZE,
) -> Tuple[mx.array, mx.array]:
    """
    Computes what mlx-lm's gated_delta_ops computes, with the same arguments, outputs, and final state, chunk by
    chunk and differentiably. The math runs in float32. Where the mask is False, the state is left unchanged, and y
    is the output gated_delta_ops gives there, as if the step had been applied. The rule works on log g, so a g
    that underflowed to 0 gets no gradient here, which gated_delta_update_train avoids by forming log g directly.

    Args:
        q: Queries of shape [batch, seq_len, key_heads, key_dim], already normalized and scaled.
        k: Keys of shape [batch, seq_len, key_heads, key_dim], already normalized and scaled.
        v: Values of shape [batch, seq_len, value_heads, value_dim], with value_heads a multiple of key_heads.
        g: Decays in (0, 1] of shape [batch, seq_len, value_heads]. Gates per key dimension are not supported.
        beta: Write strengths of shape [batch, seq_len, value_heads].
        state: The initial state, of shape [batch, value_heads, value_dim, key_dim]. Defaults to zeros.
        mask: A boolean mask of shape [batch, seq_len], False at the steps that must not change the state.
            Defaults to none.
        chunk_size: The number of tokens per chunk, a power of two. Defaults to CHUNK_SIZE.

    Returns:
        The outputs y of shape [batch, seq_len, value_heads, value_dim] in q's dtype, and the final state.
    """
    state, state_dtype = _initial_state(q, v, state, g, k, v, beta)
    # A g of 0 (exp of a large negative number, rounded to 0) would give log g = -inf, so it is raised to the
    # smallest normal float first
    log_g = mx.log(mx.maximum(g.astype(mx.float32), _FLOAT32_TINY))
    return _chunked(q, k, v, log_g, beta, state, state_dtype, mask, chunk_size)


def gated_delta_update_train(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    a: mx.array,
    b: mx.array,
    A_log: mx.array,
    dt_bias: mx.array,
    state: Optional[mx.array] = None,
    mask: Optional[mx.array] = None,
    use_kernel: bool = True,
) -> Tuple[mx.array, mx.array]:
    """
    Takes the place of mlx-lm's gated_delta_update, with the same arguments, and always runs the chunked rule. As in
    gated_delta_update, beta = sigmoid(b) and g = exp(-exp(A_log) softplus(a + dt_bias)), except that log g is
    computed directly, without taking the exponential and then the logarithm, so it stays finite where g underflows.

    Args:
        q: Queries of shape [batch, seq_len, key_heads, key_dim].
        k: Keys of shape [batch, seq_len, key_heads, key_dim].
        v: Values of shape [batch, seq_len, value_heads, value_dim].
        a: Decay logits of shape [batch, seq_len, value_heads].
        b: Write strength logits of shape [batch, seq_len, value_heads].
        A_log: The logarithms of the decay rates of the heads, of shape [value_heads].
        dt_bias: The decay logit biases of the heads, of shape [value_heads].
        state: The initial state, of shape [batch, value_heads, value_dim, key_dim]. Defaults to zeros.
        mask: A boolean padding mask of shape [batch, seq_len]. Defaults to none.
        use_kernel: Ignored, and kept so that the arguments match gated_delta_update's.

    Returns:
        The outputs y of shape [batch, seq_len, value_heads, value_dim] in q's dtype, and the final state.
    """
    del use_kernel
    beta = mx.sigmoid(b)
    log_g = -mx.exp(A_log.astype(mx.float32)) * nn.softplus(a + dt_bias)
    state, state_dtype = _initial_state(q, v, state, log_g, k, v, beta)
    return _chunked(q, k, v, log_g, beta, state, state_dtype, mask, CHUNK_SIZE)


# True while a layer passed to install_training_kernels runs in training mode
_USE_CHUNKED = contextvars.ContextVar("hmf_gated_delta_use_chunked", default=False)


def _layer_classes() -> Tuple[type, ...]:
    """The mlx-lm layer classes whose __call__ passes scalar gates to their module's gated_delta_update."""
    from mlx_lm.models import qwen3_5

    return (qwen3_5.GatedDeltaNet,)


def _patch_class(cls: type) -> None:
    """
    Routes the gated_delta_update calls of a layer class's __call__ through a function that picks the chunked rule
    when _USE_CHUNKED is set. A class that is already patched is left as it is.
    """
    cls = next(c for c in cls.__mro__ if "__call__" in c.__dict__)  # the class that defines the __call__ in use
    call = cls.__call__
    if getattr(call, "_hmf_chunked", False):
        return
    namespace = call.__globals__  # the module globals in which that __call__ looks gated_delta_update up
    original = namespace["gated_delta_update"]
    if not getattr(original, "_hmf_chunked", False):

        def dispatch(*args, **kwargs):
            return (gated_delta_update_train if _USE_CHUNKED.get() else original)(*args, **kwargs)

        dispatch._hmf_chunked = True
        namespace["gated_delta_update"] = dispatch

    def __call__(self, *args, **kwargs):
        token = _USE_CHUNKED.set(self.training and getattr(self, "_hmf_chunked_train", False))
        try:
            return call(self, *args, **kwargs)
        finally:
            _USE_CHUNKED.reset(token)

    __call__._hmf_chunked = True
    cls.__call__ = __call__


def install_training_kernels(model: nn.Module) -> int:
    """
    Makes the Gated DeltaNet layers of an mlx-lm model use the chunked rule in training mode. The layers look
    gated_delta_update up in their module's globals, so that global is replaced by a function that picks the rule,
    and the layers' own code stays mlx-lm's. The global is shared by every model of the architecture, so the choice
    is made per layer, by a flag that only the layers of the models passed here have and that only counts while the
    layer is in training mode. Evaluation and inference therefore keep mlx-lm's Metal kernel, as do models never
    passed here. Calling it again on the same model changes nothing.

    Args:
        model: An mlx-lm model, e.g. a Qwen3.5 text model.

    Returns:
        The number of layers that now train with the chunked rule.
    """
    classes = _layer_classes()
    count = 0
    for module in model.modules():
        if isinstance(module, classes):
            _patch_class(type(module))
            module._hmf_chunked_train = True
            count += 1
    return count
