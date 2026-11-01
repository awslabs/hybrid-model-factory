"""
Single-GPU tests of the GKA solver and of the state passing that its sequence parallel (SP) path relies on:

    - test_whole_matches_exact_reference: the Triton solver on a whole sequence against an exact float64
      PyTorch reference (an exact solve at every position, differentiated by autograd).
    - test_split_matches_whole: two halves, the second started from the first's states with the functions
      of the SP path, against the whole sequence. This is what SP computes across ranks, without the
      communication.
    - test_chunk_h_decays_with_natural_log_gates: the fla_compat wrappers of fla's chunk_fwd_h decay with
      exp(g), whichever convention the installed fla uses.

The comparisons are in fp32 without TF32, so that tolerances are tight. Run with:

    TRITON_F32_DEFAULT=ieee NVIDIA_TF32_OVERRIDE=0 pytest src/hmf/model/hybrid_zoo/layers/gated_kalmanet/tests/test_gka_sp_split.py -v
"""

import os

import pytest
import torch
import torch.nn.functional as F


os.environ.setdefault("TRITON_F32_DEFAULT", "ieee")  # exact fp32 dot products in the Triton kernels
pytest.importorskip("fla")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="GKA's kernels need CUDA")

from fla.ops.simple_gla import chunk_simple_gla  # noqa: E402
from fla.ops.utils import chunk_local_cumsum  # noqa: E402

from hmf.model.hybrid_zoo.layers.fla_compat import chunk_fwd_h  # noqa: E402
from hmf.model.hybrid_zoo.layers.gated_kalmanet.ops.chebyshev.chunk_dk_backward_gating_sp import (  # noqa: E402
    compute_kk_final_with_gating,
)
from hmf.model.hybrid_zoo.layers.gated_kalmanet.ops.chebyshev.gka_chebyshev_solve import (  # noqa: E402
    gka_chebyshev_gla,
)
from hmf.model.hybrid_zoo.layers.gated_kalmanet.sequence_parallel.gka_sp_utils import (  # noqa: E402
    recompute_with_initial_states,
)


HEADS, HEAD_DIM, CHUNK_SIZE, RIDGE = 2, 64, 64, 0.02
INPUTS = ("q", "k", "v", "alpha", "g")


@pytest.fixture(autouse=True)
def no_tf32():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def make_inputs(seq_len: int, gating: bool, seed: int = 0) -> dict:
    """Random GKA inputs: unit-norm q and k, alpha in (0, 1), and log decays g <= 0 (or 0 without gating)."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    randn = lambda *shape: torch.randn(*shape, generator=gen, device="cuda")  # noqa: E731
    shape = (1, seq_len, HEADS)
    inputs = {
        "q": F.normalize(randn(*shape, HEAD_DIM), dim=-1),
        "k": F.normalize(randn(*shape, HEAD_DIM), dim=-1),
        "v": randn(*shape, HEAD_DIM),
        "alpha": torch.sigmoid(randn(*shape)),
        "g": -F.softplus(randn(*shape) - 3.0) if gating else torch.zeros(*shape, device="cuda"),
    }
    return {name: x.requires_grad_() for name, x in inputs.items()}


def solver_kwargs(bp_lambda: bool, num_iter: int) -> dict:
    return dict(gla_scale=HEAD_DIM**-0.5, ridge_strength=RIDGE, solver_type="chebyshev", num_iter=num_iter,
                bp_lambda=bp_lambda, chunk_size=CHUNK_SIZE)


def whole(x: dict, kw: dict, kk_gate: bool) -> torch.Tensor:
    gk = x["g"] if kk_gate else None
    o, _, _ = gka_chebyshev_gla(x["q"], x["k"], x["v"], x["alpha"], x["g"], gk, **kw)
    return o


def split(x: dict, split_at: int, kw: dict, kk_gate: bool) -> torch.Tensor:
    """The two halves as the SP path runs them: the first from zero states, the second from the first's."""
    first = {n: t[:, :split_at] for n, t in x.items()}
    second = {n: t[:, split_at:] for n, t in x.items()}
    gk1, gk2 = (first["g"], second["g"]) if kk_gate else (None, None)
    zeros = x["q"].new_zeros(1, HEADS, HEAD_DIM, HEAD_DIM)
    o1, _, _ = recompute_with_initial_states(
        first["q"], first["k"], first["v"], first["alpha"], first["g"], gk1,
        prev_kk=zeros, prev_h_kv=zeros, **kw,
    )
    kk_final = compute_kk_final_with_gating(first["k"], gk1, chunk_size=CHUNK_SIZE)
    _, h_kv_final = chunk_simple_gla(
        q=first["q"], k=first["k"], v=first["v"], scale=kw["gla_scale"], g=first["g"], output_final_state=True
    )
    o2, _, _ = recompute_with_initial_states(
        second["q"], second["k"], second["v"], second["alpha"], second["g"], gk2,
        prev_kk=kk_final, prev_h_kv=h_kv_final, **kw,
    )
    return torch.cat([o1, o2], dim=1)


def exact(x: dict, kw: dict, kk_gate: bool) -> torch.Tensor:
    """
    GKA in float64 PyTorch, with an exact solve at every position:
        H_t = exp(g_t) H_{t-1} + k_t k_t^T (no decay without kk_gate),
        λ_t = ridge * ||H_t||_F (a constant for autograd unless bp_lambda),
        x_t = (H_t + λ_t I)^-1 q_t,  q'_t = q_t + alpha_t (x_t - q_t),
        S_t = exp(g_t) S_{t-1} + k_t v_t^T,  o_t = scale * q'_t S_t.
    """
    q, k, v, alpha, g = (x[n].double() for n in INPUTS)
    eye = torch.eye(HEAD_DIM, dtype=torch.float64, device=q.device)
    h_kk = q.new_zeros(1, HEADS, HEAD_DIM, HEAD_DIM)
    s_kv = q.new_zeros(1, HEADS, HEAD_DIM, HEAD_DIM)
    outputs = []
    for t in range(q.shape[1]):
        decay = g[:, t].exp()[..., None, None]
        h_kk = (decay if kk_gate else 1.0) * h_kk + k[:, t, :, :, None] * k[:, t, :, None, :]
        s_kv = decay * s_kv + k[:, t, :, :, None] * v[:, t, :, None, :]
        lam = RIDGE * torch.linalg.matrix_norm(h_kk)
        if not kw["bp_lambda"]:
            lam = lam.detach()
        x_t = torch.linalg.solve(h_kk + lam[..., None, None] * eye, q[:, t])
        q_t = q[:, t] + alpha[:, t, :, None] * (x_t - q[:, t])
        outputs.append(kw["gla_scale"] * torch.einsum("bhk,bhkv->bhv", q_t, s_kv))
    return torch.stack(outputs, dim=1)


def output_and_grads(fn, x: dict, do: torch.Tensor):
    for t in x.values():
        t.grad = None
    o = fn(x)
    (o.float() * do).sum().backward()
    return o.detach().float(), {n: x[n].grad.detach().float().clone() for n in INPUTS}


def rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).norm() / b.norm().clamp_min(1e-30))


def assert_close(got, ref, tol: float) -> None:
    (o_got, g_got), (o_ref, g_ref) = got, ref
    errors = {"output": rel(o_got, o_ref), **{f"d{n}": rel(g_got[n], g_ref[n]) for n in INPUTS}}
    worst = max(errors, key=errors.get)
    assert errors[worst] <= tol, f"{worst} is off by {errors[worst]:.2e} (tolerance {tol:.0e}): {errors}"


# (gating, kk_gate): decays on both states; on the kv state only (no gate on the key-key state, as with
# use_forgetting_gate_kk=False, which runs the ungated kernels); and zero decays through the gated kernels
GATES = {"kk_and_kv": (True, True), "kv_only": (True, False), "zero": (False, True)}


@pytest.mark.parametrize("bp_lambda", [True, False])
@pytest.mark.parametrize("gates", list(GATES))
def test_whole_matches_exact_reference(gates, bp_lambda):
    """
    The solver's 30 iterations (its default) leave a convergence error of ~1e-4 at ridge 0.02. More
    iterations would be closer, but num_iter is a compile-time constant that Triton unrolls, so every new
    value compiles the solver kernels again, for each autotuning config.
    """
    gating, kk_gate = GATES[gates]
    x = make_inputs(256, gating)
    do = torch.randn(1, 256, HEADS, HEAD_DIM, device="cuda")
    kw = solver_kwargs(bp_lambda, num_iter=30)
    assert_close(output_and_grads(lambda t: whole(t, kw, kk_gate), x, do),
                 output_and_grads(lambda t: exact(t, kw, kk_gate), x, do), 1e-3)


@pytest.mark.parametrize("bp_lambda", [True, False])
@pytest.mark.parametrize("gates", list(GATES))
def test_split_matches_whole(gates, bp_lambda):
    """Exact up to fp32 rounding: a missing or wrong gradient across the split shows at 1e-3 or more."""
    gating, kk_gate = GATES[gates]
    x = make_inputs(1024, gating)
    do = torch.randn(1, 1024, HEADS, HEAD_DIM, device="cuda")
    kw = solver_kwargs(bp_lambda, num_iter=30)
    assert_close(output_and_grads(lambda t: split(t, 512, kw, kk_gate), x, do),
                 output_and_grads(lambda t: whole(t, kw, kk_gate), x, do), 1e-5)


def test_chunk_h_decays_with_natural_log_gates():
    """fla_compat.chunk_fwd_h takes natural-log gates: its chunk-start states decay by exp(g)."""
    x = {n: t.detach() for n, t in make_inputs(256, gating=True).items()}
    k, g = x["k"], x["g"]
    states, final = chunk_fwd_h(k=k, v=k, g=chunk_local_cumsum(g, chunk_size=CHUNK_SIZE), gk=None, gv=None, h0=None,
                                output_final_state=True, states_in_fp32=True, cu_seqlens=None, chunk_size=CHUNK_SIZE)
    h, starts = k.new_zeros(1, HEADS, HEAD_DIM, HEAD_DIM, dtype=torch.float64), []
    for t in range(k.shape[1]):
        if t % CHUNK_SIZE == 0:
            starts.append(h.clone())
        h = g[:, t].double().exp()[..., None, None] * h + k[:, t, :, :, None].double() * k[:, t, :, None, :].double()
    assert rel(states.double(), torch.stack(starts, dim=1)) <= 1e-5
    assert rel(final.double(), h) <= 1e-5
