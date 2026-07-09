# Adapted from:
# https://github.com/triton-lang/triton/blob/main/python/tutorials/06-fused-attention.py
#
# Part of the Triton project, licensed under the MIT License.
# See THIRD_PARTY_LICENSES/triton-MIT.txt.

"""
Q-Sparse Flash Attention Kernel Implementation
===============

This is a Triton implementation of our Q-Sparse Attention based on the Flash Attention v2 triton kernel at https://github.com/triton-lang/triton/blob/main/python/tutorials/06-fused-attention.py

Credits: OpenAI kernel team

Extra Credits:

* Original flash attention paper (https://arxiv.org/abs/2205.14135)
* Rabe and Staats (https://arxiv.org/pdf/2112.05682v2.pdf)

"""

import torch
import math
import os

import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor  # requires Triton >= 3.4.0


def is_hip():
    return triton.runtime.driver.active.get_current_target().backend == "hip"


def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"


def supports_host_descriptor():
    return is_cuda() and torch.cuda.get_device_capability()[0] >= 9


def is_blackwell():
    return is_cuda() and torch.cuda.get_device_capability()[0] == 10


def is_hopper():
    return is_cuda() and torch.cuda.get_device_capability()[0] == 9


@triton.jit
def _attn_fwd_inner(
    acc,
    l_i,
    m_i,
    q,  #
    desc_k,
    desc_v,  #
    offset_y,
    dtype: tl.constexpr,
    start_m,
    qk_scale,  #
    BLOCK_M: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,  #
    offs_m: tl.constexpr,
    offs_n: tl.constexpr,  #
    qi_vals: tl.constexpr,
    N_CTX: tl.constexpr,
    warp_specialize: tl.constexpr,
    IS_HOPPER: tl.constexpr,
):
    # figure out which key range to process for this chunk of queries
    max_qi = tl.max(qi_vals, axis=0)  # largest query index in this block
    hi = min(
        tl.cdiv(max_qi + 1, BLOCK_N) * BLOCK_N, N_CTX
    )  # round up for correct loading and clamp

    offsetk_y = offset_y
    if dtype == tl.float8e5:
        offsetv_y = offset_y * HEAD_DIM
    else:
        offsetv_y = offset_y

    # loop over k, v and update accumulator
    for start_n in tl.range(0, hi, BLOCK_N, warp_specialize=warp_specialize):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        key_idx = start_n + offs_n[None, :]  # shape [1, BLOCK_N]

        # -- compute qk ----
        k = desc_k.load([offsetk_y, 0]).T
        qk = tl.dot(q, k)
        # use qi_vals, the original q indices for ensuring causality in q-sparse variant
        mask = qi_vals[:, None] >= key_idx  # [BLOCK_M, BLOCK_N]
        qk = tl.where(mask, qk * qk_scale, float("-inf"))

        m_ij_ = tl.maximum(m_i, tl.max(qk, 1))
        m_ij = tl.where(m_ij_ != float("-inf"), m_ij_, float(0.0))
        qk -= m_ij[:, None]

        p = tl.math.exp2(qk)
        # -- compute correction factor
        alpha = tl.math.exp2(m_i - m_ij)
        l_ij = tl.sum(p, 1)

        # -- update output accumulator --
        if (
            not IS_HOPPER and warp_specialize and BLOCK_M == 128 and HEAD_DIM == 128
        ):  # not true in our case because warp_specialize=False
            BM: tl.constexpr = acc.shape[0]
            BN: tl.constexpr = acc.shape[1]
            acc0, acc1 = acc.reshape([BM, 2, BN // 2]).permute(0, 2, 1).split()
            acc0 = acc0 * alpha[:, None]
            acc1 = acc1 * alpha[:, None]
            acc = tl.join(acc0, acc1).permute(0, 2, 1).reshape([BM, BN])
        else:
            acc = acc * alpha[:, None]
        # prepare p and v for the dot
        if dtype == tl.float8e5:
            v = desc_v.load([0, offsetv_y]).T
        else:
            v = desc_v.load([offsetv_y, 0])
        p = p.to(dtype)
        acc = tl.dot(p, v, acc)
        # update m_i and l_i, place this at the end of the loop to reduce register pressure
        l_i = l_i * alpha + l_ij
        m_i = m_ij

        offsetk_y += BLOCK_N
        offsetv_y += BLOCK_N
    return acc, l_i, m_i


def _host_descriptor_pre_hook(nargs):
    BLOCK_M = nargs["BLOCK_M"]
    BLOCK_N = nargs["BLOCK_N"]
    HEAD_DIM = nargs["HEAD_DIM"]
    if nargs["FP8_OUTPUT"]:
        nargs["desc_v"].block_shape = [HEAD_DIM, BLOCK_N]
    else:
        nargs["desc_v"].block_shape = [BLOCK_N, HEAD_DIM]
    nargs["desc_k"].block_shape = [BLOCK_N, HEAD_DIM]


if is_hip():
    NUM_STAGES_OPTIONS = [1]
elif supports_host_descriptor():
    NUM_STAGES_OPTIONS = [2, 3, 4]
else:
    NUM_STAGES_OPTIONS = [2, 3, 4]

# Tip for debugging if sticking to small seq lengths: make sure these BLOCK_M & BLOCK_N
# are greater than seq_len otherwise things won't work because of no masked loads for K & V.
# In other words, always ensure K & V lengths (N_CTX) is a multiple of these chosen block sizes
configs = [
    triton.Config(
        {"BLOCK_M": BM, "BLOCK_N": BN},
        num_stages=s,
        num_warps=w,
        pre_hook=_host_descriptor_pre_hook,
    )
    for BM in [64, 128]
    for BN in [32, 64, 128]
    for s in NUM_STAGES_OPTIONS
    for w in [4, 8]
]
if "PYTEST_VERSION" in os.environ:
    # Use a single config in testing for reproducibility
    configs = [
        triton.Config(
            dict(BLOCK_M=128, BLOCK_N=64),
            num_stages=2,
            num_warps=4,
            pre_hook=_host_descriptor_pre_hook,
        )
    ]


def keep(conf):
    BLOCK_M = conf.kwargs["BLOCK_M"]
    BLOCK_N = conf.kwargs["BLOCK_N"]
    return not (
        is_cuda()
        and torch.cuda.get_device_capability()[0] == 9
        and BLOCK_M * BLOCK_N < 128 * 128
        and conf.num_warps == 8
    )


def prune_invalid_configs(configs, named_args, **kwargs):
    N_CTX = kwargs["N_CTX"]

    # Filter out configs where BLOCK_M > N_CTX
    return [conf for conf in configs if conf.kwargs.get("BLOCK_M", 0) <= N_CTX]


@triton.jit
def _maybe_make_tensor_desc(desc_or_ptr, shape, strides, block_shape):
    if isinstance(desc_or_ptr, tl.tensor_descriptor):
        return desc_or_ptr
    else:
        return tl.make_tensor_descriptor(desc_or_ptr, shape, strides, block_shape)


# This kernel is run over a grid of (bsz*num_heads, num of query blocks)
@triton.autotune(configs=configs, key=["HEAD_DIM", "FP8_OUTPUT", "warp_specialize"])
@triton.jit
def _attn_fwd(
    sm_scale,
    M,  #
    Z,
    H,
    q,
    desc_k,
    desc_v,
    o,
    q_idx,
    sqih,
    sqim,
    sqz,
    sqh,
    sqm,
    sqd,  # Q strides
    soz,
    soh,
    som,
    sod,  # O strides
    N_CTX_Q,
    N_CTX,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    FP8_OUTPUT: tl.constexpr,
    STAGE: tl.constexpr,
    warp_specialize: tl.constexpr,
    IS_HOPPER: tl.constexpr,
):
    dtype = tl.float8e5 if FP8_OUTPUT else tl.bfloat16
    tl.static_assert(BLOCK_N <= HEAD_DIM)
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H

    y_dim_q = Z * H * N_CTX_Q
    y_dim_kv = Z * H * N_CTX
    if FP8_OUTPUT:
        desc_v = _maybe_make_tensor_desc(
            desc_v,
            shape=[HEAD_DIM, y_dim_kv],
            strides=[N_CTX, 1],
            block_shape=[HEAD_DIM, BLOCK_N],
        )
    else:
        desc_v = _maybe_make_tensor_desc(
            desc_v,
            shape=[y_dim_kv, HEAD_DIM],
            strides=[HEAD_DIM, 1],
            block_shape=[BLOCK_N, HEAD_DIM],
        )
    desc_k = _maybe_make_tensor_desc(
        desc_k,
        shape=[y_dim_kv, HEAD_DIM],
        strides=[HEAD_DIM, 1],
        block_shape=[BLOCK_N, HEAD_DIM],
    )

    offset_y = off_z * (N_CTX_Q * H) + off_h * N_CTX_Q
    offset_y_kv = off_z * (N_CTX * H) + off_h * N_CTX
    qo_offset_y = offset_y + start_m * BLOCK_M
    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    # needed for q-sparse logic
    offs_qi = off_hz * sqih + offs_m * sqim
    # load the actual query indices
    qi_vals = tl.load(q_idx + offs_qi, mask=offs_m < N_CTX_Q, other=-1)
    offs_n = tl.arange(0, BLOCK_N)
    # initialize pointer to m and l, and all accumulation buffers in fp32 to avoid precision issues
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    # load scales
    qk_scale = sm_scale
    qk_scale *= 1.44269504  # 1/log(2)
    # load q: it will stay in SRAM throughout
    offs_q = off_hz * sqh + offs_m[:, None] * sqm + offs_d[None, :] * sqd
    q_vals = tl.load(q + offs_q, mask=offs_m[:, None] < N_CTX_Q, other=0)
    # Removed Stage logic for simplicity, which means only causal=True works
    acc, l_i, m_i = _attn_fwd_inner(
        acc,
        l_i,
        m_i,
        q_vals,
        desc_k,
        desc_v,
        offset_y_kv,
        dtype,
        start_m,
        qk_scale,
        BLOCK_M,
        HEAD_DIM,
        BLOCK_N,
        offs_m,
        offs_n,
        qi_vals,
        N_CTX,
        warp_specialize,
        IS_HOPPER,
    )
    m_i += tl.math.log2(l_i)
    # replace -inf + log2(0) with safe neutral (0)
    m_i = tl.where(l_i > 0, m_i, 0.0)
    acc = tl.where(l_i[:, None] > 0, acc / l_i[:, None], 0.0)
    m_ptrs = M + off_hz * N_CTX_Q + offs_m
    tl.store(m_ptrs, m_i, mask=offs_m < N_CTX_Q)
    offs_o = off_hz * soh + offs_m[:, None] * som + offs_d[None, :] * sod
    tl.store(
        o + offs_o, acc, mask=offs_m[:, None] < N_CTX_Q
    )  # masked stores are essential here


# this kernel is run over a grid of (bsz*seq_len, chunks of queries) to preprocess
# and store delta for dq, dk calculation later (Refer to FA-2 paper backward pass implementation)
@triton.jit
def _attn_bwd_preprocess(
    O, DO, Delta, Z, H, N_CTX_Q, BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr
):
    off_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    off_hz = tl.program_id(1)
    off_n = tl.arange(0, HEAD_DIM)
    # load
    o = tl.load(
        O + off_hz * HEAD_DIM * N_CTX_Q + off_m[:, None] * HEAD_DIM + off_n[None, :],
        mask=off_m[:, None] < N_CTX_Q,
        other=0.0,
    )
    do = tl.load(
        DO + off_hz * HEAD_DIM * N_CTX_Q + off_m[:, None] * HEAD_DIM + off_n[None, :],
        mask=off_m[:, None] < N_CTX_Q,
        other=0.0,
    ).to(tl.float32)
    delta = tl.sum(o * do, axis=1)
    # write-back
    tl.store(Delta + off_hz * N_CTX_Q + off_m, delta, mask=off_m < N_CTX_Q)


# The main inner-loop logic for computing dK and dV. This kernel is run over a grid
# of (bsz * num_heads, chunks of k/v) and the iterations are over query
@triton.jit
def _attn_bwd_dkdv(
    dk,
    dv,
    Q,
    k,
    v,
    sm_scale,
    DO,
    M,
    D,
    # shared by Q/K/V/DO.
    stride_tokq,
    stride_dq,
    H,
    q_idx,
    sqim,
    BLOCK_M1: tl.constexpr,
    BLOCK_N1: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    # Filled in by the wrapper.
    start_n,
    num_steps,
    N_CTX_Q,
):
    offs_m = tl.arange(0, BLOCK_M1)
    offs_n = start_n + tl.arange(0, BLOCK_N1)
    offs_k = tl.arange(0, HEAD_DIM)
    qT_ptrs = Q + offs_m[None, :] * stride_tokq + offs_k[:, None] * stride_dq
    do_ptrs = DO + offs_m[:, None] * stride_tokq + offs_k[None, :] * stride_dq
    # BLOCK_N1 must be a multiple of BLOCK_M1, otherwise the code wouldn't work.
    tl.static_assert(BLOCK_N1 % BLOCK_M1 == 0)
    curr_m = 0
    step_m = BLOCK_M1

    # Skip over query blocks that shouldn't be attended to based on causality.
    # This saves unnecessary for loop iterations and makes things go faster
    min_ki = tl.min(offs_n, axis=0)  # first key in this tile
    # figure out where query loop should start
    start_blockidx_m = 0
    for blk_m in range(0, N_CTX_Q, BLOCK_M1):
        offs_m_probe = blk_m + tl.arange(0, BLOCK_M1)
        offs_qi_probe = offs_m_probe * sqim
        qi_vals_probe = tl.load(
            q_idx + offs_qi_probe, mask=offs_m_probe < N_CTX_Q, other=-1
        )
        max_qi_probe = tl.max(qi_vals_probe, axis=0)
        if (max_qi_probe < min_ki) & (max_qi_probe != -1):
            start_blockidx_m += 1

    curr_m = start_blockidx_m * BLOCK_M1
    for blk_idx in range(start_blockidx_m, num_steps):
        offs_m = curr_m + tl.arange(0, BLOCK_M1)
        offs_qi = offs_m * sqim
        # load the actual query indices
        qi_vals = tl.load(q_idx + offs_qi, mask=offs_m < N_CTX_Q, other=-1)
        mask_m = offs_m < N_CTX_Q
        qT = tl.load(
            Q + offs_m[None, :] * stride_tokq + offs_k[:, None] * stride_dq,
            mask=offs_m[None, :] < N_CTX_Q,
            other=0.0,
        )
        # Load m before computing qk to reduce pipeline stall.
        m = tl.load(M + offs_m, mask=mask_m, other=0.0)
        qkT = tl.dot(k, qT)
        pT = tl.math.exp2(qkT - m[None, :])
        # Autoregressive masking.
        mask = (
            qi_vals[None, :] >= offs_n[:, None]
        )  # use original query indices qi_vals to establish causality
        pT = tl.where(mask, pT, 0.0)
        do = tl.load(
            DO + offs_m[:, None] * stride_tokq + offs_k[None, :] * stride_dq,
            mask=offs_m[:, None] < N_CTX_Q,
            other=0.0,
        )
        # Compute dV.
        ppT = pT
        ppT = ppT.to(tl.bfloat16)
        dv += tl.dot(ppT, do)
        Di = tl.load(D + offs_m, mask=mask_m, other=0.0)

        # Compute dP and dS.
        dpT = tl.dot(v, tl.trans(do)).to(tl.float32)
        dsT = pT * (dpT - Di[None, :])
        dsT = dsT.to(tl.bfloat16)
        dk += tl.dot(dsT, tl.trans(qT))
        # Increment pointers.
        curr_m += step_m

    return dk, dv


# Main inner-loop logic for computing dQ. This kernel has a grid over (bsz*num_heads, query chunks)
# and iterates over key and value blocks.
@triton.jit
def _attn_bwd_dq(
    Q,
    K,
    V,
    DO,
    DQ,
    M,
    D,
    H,
    stride_z,
    stride_h,
    stride_tok,
    stride_d,
    stride_zq,
    stride_hq,
    stride_tokq,
    stride_dq,
    stride_zdq,
    stride_hdq,
    stride_tokdq,
    stride_ddq,  # DQ strides
    q_idx,
    sqih,
    sqim,
    num_steps,
    N_CTX,
    N_CTX_Q,
    BLOCK_M2: tl.constexpr,
    BLOCK_N2: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    LN2: tl.constexpr = 0.6931471824645996  # = ln(2)

    bhid = tl.program_id(2)
    off_chz = (bhid * N_CTX_Q).to(tl.int64)
    adj = (stride_h * (bhid % H) + stride_z * (bhid // H)).to(tl.int64)
    adjq = (stride_hq * (bhid % H) + stride_zq * (bhid // H)).to(tl.int64)
    adjdq = (stride_hdq * (bhid % H) + stride_zdq * (bhid // H)).to(tl.int64)
    pid = tl.program_id(0)

    # offset pointers for batch/head
    Q += adjq
    K += adj
    V += adj
    DO += adjq
    DQ += adjdq
    M += off_chz
    D += off_chz

    start_m = pid * BLOCK_M2
    offs_m = start_m + tl.arange(0, BLOCK_M2)
    offs_n = tl.arange(0, BLOCK_N2)
    offs_k = tl.arange(0, HEAD_DIM)

    # need to be careful here, do masked loading because of q being compact
    q = tl.load(
        Q + offs_m[:, None] * stride_tokq + offs_k[None, :] * stride_dq,
        mask=offs_m[:, None] < N_CTX_Q,
        other=0.0,
    )
    dq = tl.zeros([BLOCK_M2, HEAD_DIM], dtype=tl.float32)
    do = tl.load(
        DO + offs_m[:, None] * stride_tokq + offs_k[None, :] * stride_dq,
        mask=offs_m[:, None] < N_CTX_Q,
        other=0.0,
    )

    m = tl.load(M + offs_m, mask=offs_m < N_CTX_Q, other=0)
    m = m[:, None]

    kT_ptrs = K + offs_n[None, :] * stride_tok + offs_k[:, None] * stride_d
    vT_ptrs = V + offs_n[None, :] * stride_tok + offs_k[:, None] * stride_d
    # D (= delta) is pre-divided by ds_scale.
    Di = tl.load(D + offs_m, mask=offs_m < N_CTX_Q, other=0.0)

    offs_qi = bhid * sqih + offs_m * sqim
    # load the actual query indices
    qi_vals = tl.load(q_idx + offs_qi, mask=offs_m < N_CTX_Q, other=-1)
    max_qi = tl.max(qi_vals, axis=0)  # max query index in this tile
    num_kv_blocks = tl.cdiv(
        min(max_qi + 1, N_CTX), BLOCK_N2
    )  # only run for kv blocks that are causally correct

    # BLOCK_M2 must be a multiple of BLOCK_N2, otherwise the code wouldn't work.
    tl.static_assert(BLOCK_M2 % BLOCK_N2 == 0)
    curr_n = 0
    step_n = BLOCK_N2
    for blk_idx in range(num_kv_blocks):
        kT = tl.load(kT_ptrs)
        vT = tl.load(vT_ptrs)
        qk = tl.dot(q, kT)
        p = tl.math.exp2(qk - m)
        # Autoregressive masking.
        offs_n = curr_n + tl.arange(0, BLOCK_N2)
        mask = (
            qi_vals[:, None] >= offs_n[None, :]
        )  # use true query indices to establish causality
        p = tl.where(mask, p, 0.0)
        # Compute dP and dS.
        dp = tl.dot(do, vT).to(tl.float32)  # ensure accumulations in fp32
        ds = p * (dp - Di[:, None])
        ds = ds.to(tl.bfloat16)
        # Compute dQ.
        # NOTE: We need to de-scale dq in the end, because kT was pre-scaled.
        dq += tl.dot(ds, tl.trans(kT))
        # Increment pointers.
        curr_n += step_n
        kT_ptrs += step_n * stride_tok
        vT_ptrs += step_n * stride_tok

    dq_ptrs = DQ + offs_m[:, None] * stride_tokdq + offs_k[None, :] * stride_ddq
    dq *= LN2
    tl.store(dq_ptrs, dq, mask=offs_m[:, None] < N_CTX_Q)


@triton.jit
def _attn_bwd(
    Q,
    K,
    V,
    sm_scale,
    DO,
    DK,
    DV,
    M,
    D,
    stride_z,
    stride_h,
    stride_tok,
    stride_d,  # shared by K,V
    stride_zq,
    stride_hq,
    stride_tokq,
    stride_dq,  # shared by Q,DO
    q_idx,
    sqih,
    sqim,
    H,
    N_CTX,
    N_CTX_Q,
    BLOCK_M1: tl.constexpr,
    BLOCK_N1: tl.constexpr,
    BLOCK_M2: tl.constexpr,
    BLOCK_N2: tl.constexpr,
    BLK_SLICE_FACTOR: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    bhid = tl.program_id(2)
    off_chz = (bhid * N_CTX_Q).to(tl.int64)
    adj = (stride_h * (bhid % H) + stride_z * (bhid // H)).to(tl.int64)
    adjq = (stride_hq * (bhid % H) + stride_zq * (bhid // H)).to(tl.int64)
    off_hz_qi = bhid * sqih
    pid = tl.program_id(0)

    # offset pointers for batch/head
    Q += adjq
    K += adj
    V += adj
    DO += adjq
    DK += adj
    DV += adj
    M += off_chz
    D += off_chz
    q_idx += off_hz_qi

    # load scales
    offs_k = tl.arange(0, HEAD_DIM)

    start_n = pid * BLOCK_N1
    start_m = 0

    offs_n = start_n + tl.arange(0, BLOCK_N1)

    dv = tl.zeros([BLOCK_N1, HEAD_DIM], dtype=tl.float32)
    dk = tl.zeros([BLOCK_N1, HEAD_DIM], dtype=tl.float32)

    # load K and V: they stay in SRAM throughout the inner loop.
    k = tl.load(K + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d)
    v = tl.load(V + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d)

    num_steps = tl.cdiv(
        N_CTX_Q, BLOCK_M1
    )  # loop over all query blocks, so this should be N_CTX_Q
    dk, dv = _attn_bwd_dkdv(
        dk,
        dv,
        Q,
        k,
        v,
        sm_scale,
        DO,
        M,
        D,
        stride_tokq,
        stride_dq,
        H,
        q_idx,
        sqim,
        BLOCK_M1,
        BLOCK_N1,
        HEAD_DIM,
        start_n,
        num_steps,
        N_CTX_Q,
    )

    dv_ptrs = DV + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d
    tl.store(dv_ptrs, dv)

    # Write back dK.
    dk *= sm_scale
    dk_ptrs = DK + offs_n[:, None] * stride_tok + offs_k[None, :] * stride_d
    tl.store(dk_ptrs, dk)


class _attention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, q_idx, causal, sm_scale, warp_specialize=True):
        # shape constraints
        HEAD_DIM_Q, HEAD_DIM_K = q.shape[-1], k.shape[-1]
        # when v is in float8_e5m2 it is transposed.
        HEAD_DIM_V = v.shape[-1]
        assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
        assert HEAD_DIM_K in {16, 32, 64, 128, 256}
        o = torch.empty_like(q)
        stage = 3 if causal else 1
        extra_kern_args = {}
        # Tuning for AMD target
        if is_hip():
            waves_per_eu = 3 if HEAD_DIM_K <= 64 else 2
            extra_kern_args = {"waves_per_eu": waves_per_eu, "allow_flush_denorm": True}

        M = torch.zeros(
            (q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32
        )
        # Use device_descriptor for Hopper + warpspec.
        if supports_host_descriptor() and not (is_hopper() and warp_specialize):
            # Note that on Hopper we cannot perform a FP8 dot with a non-transposed second tensor
            y_dim_q = q.shape[0] * q.shape[1] * q.shape[2]
            y_dim_kv = k.shape[0] * k.shape[1] * k.shape[2]

            dummy_block = [1, 1]
            desc_q = q
            if q.dtype == torch.float8_e5m2:
                desc_v = TensorDescriptor(
                    v,
                    shape=[HEAD_DIM_K, y_dim_kv],
                    strides=[v.shape[2], 1],
                    block_shape=dummy_block,
                )
            else:
                desc_v = TensorDescriptor(
                    v,
                    shape=[y_dim_kv, HEAD_DIM_K],
                    strides=[HEAD_DIM_K, 1],
                    block_shape=dummy_block,
                )
            desc_k = TensorDescriptor(
                k,
                shape=[y_dim_kv, HEAD_DIM_K],
                strides=[HEAD_DIM_K, 1],
                block_shape=dummy_block,
            )
            desc_o = o
        else:
            desc_q = q
            desc_v = v
            desc_k = k
            desc_o = o

        # q_idx originally: (B, H, N_CTX_Q)
        q_idx_flat = q_idx.view(q.shape[0] * q.shape[1], q.shape[2]).contiguous()
        sqih, sqim = q_idx_flat.stride()
        _alloc_device = q.device

        def alloc_fn(size: int, align: int, _):
            return torch.empty(size, dtype=torch.int8, device=_alloc_device)

        triton.set_allocator(alloc_fn)

        def grid(META):
            return (
                triton.cdiv(q.shape[2], META["BLOCK_M"]),
                q.shape[0] * q.shape[1],
                1,
            )

        ctx.grid = grid
        if is_blackwell() and warp_specialize:
            if HEAD_DIM_K == 128 and q.dtype == torch.bfloat16:
                extra_kern_args["maxnreg"] = 168
            else:
                extra_kern_args["maxnreg"] = 80

        _attn_fwd[grid](
            sm_scale,
            M,
            q.shape[0],
            q.shape[1],  # batch-size, #heads same for q,k,v
            desc_q,
            desc_k,
            desc_v,
            desc_o,  #
            q_idx_flat,
            sqih,
            sqim,  # added for q-sparse logic to track original q indices for causality
            desc_q.stride(0),
            desc_q.stride(1),
            desc_q.stride(2),
            desc_q.stride(3),
            desc_o.stride(0),
            desc_o.stride(1),
            desc_o.stride(2),
            desc_o.stride(3),
            N_CTX_Q=q.shape[2],
            N_CTX=k.shape[2],  # only q-sparse so k-dim>=q-dim
            HEAD_DIM=HEAD_DIM_K,
            FP8_OUTPUT=q.dtype == torch.float8_e5m2,
            STAGE=stage,
            warp_specialize=warp_specialize,
            IS_HOPPER=is_hopper(),
            **extra_kern_args
        )

        ctx.save_for_backward(q, k, v, o, M, q_idx)
        ctx.sm_scale = sm_scale
        ctx.HEAD_DIM = HEAD_DIM_K
        ctx.causal = causal
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, M, q_idx = ctx.saved_tensors
        do = do.contiguous()
        assert do.is_contiguous()
        assert k.stride() == v.stride()
        assert q.stride() == o.stride() == do.stride()

        dk = torch.empty_like(k)
        dv = torch.empty_like(v)

        BATCH, N_HEAD, N_CTX_Q = q.shape[:3]
        N_CTX = k.shape[2]
        PRE_BLOCK = 128
        NUM_WARPS, NUM_STAGES = 8, 4
        BLOCK_M1, BLOCK_N1, BLOCK_M2, BLOCK_N2 = 64, 128, 128, 64
        BLK_SLICE_FACTOR = 1
        RCP_LN2 = 1.4426950408889634  # = 1.0 / ln(2)
        arg_k = k
        arg_k = arg_k * (ctx.sm_scale * RCP_LN2)

        q_idx_flat = q_idx.view(q.shape[0] * q.shape[1], q.shape[2]).contiguous()
        sqih, sqim = q_idx_flat.stride()
        pre_grid = (
            math.ceil(N_CTX_Q / PRE_BLOCK),
            BATCH * N_HEAD,
        )  # should be N_CTX_Q, and change to cdiv() to avoid ignoring remaining blocks
        delta = torch.zeros_like(M)
        _attn_bwd_preprocess[pre_grid](
            o,
            do,
            delta,
            BATCH,
            N_HEAD,
            N_CTX_Q,
            BLOCK_M=PRE_BLOCK,
            HEAD_DIM=ctx.HEAD_DIM,
        )
        grid = (
            math.ceil(N_CTX / BLOCK_N1),
            1,
            BATCH * N_HEAD,
        )  # this grid is over keys/values

        _attn_bwd[grid](
            q,
            arg_k,
            v,
            ctx.sm_scale,
            do,
            dk,
            dv,
            M,
            delta,
            k.stride(0),
            k.stride(1),
            k.stride(2),
            k.stride(3),
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q.stride(3),
            q_idx_flat,
            sqih,
            sqim,
            N_HEAD,
            N_CTX,
            N_CTX_Q,
            BLOCK_M1=BLOCK_M1,
            BLOCK_N1=BLOCK_N1,
            BLOCK_M2=BLOCK_M2,
            BLOCK_N2=BLOCK_N2,
            BLK_SLICE_FACTOR=BLK_SLICE_FACTOR,
            HEAD_DIM=ctx.HEAD_DIM,
            num_warps=NUM_WARPS,
            num_stages=NUM_STAGES,
        )

        dq = torch.zeros_like(q).contiguous()
        num_steps = math.ceil(N_CTX / BLOCK_N2)  # to loop through k & v
        grid_dq = (
            math.ceil(N_CTX_Q / BLOCK_M2),
            1,
            BATCH * N_HEAD,
        )  # this grid is over queries
        _attn_bwd_dq[grid_dq](
            q,
            arg_k,
            v,  #
            do,
            dq,
            M,
            delta,
            N_HEAD,
            k.stride(0),
            k.stride(1),
            k.stride(2),
            k.stride(3),
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q.stride(3),
            # DQ strides (for stores)
            dq.stride(0),
            dq.stride(1),
            dq.stride(2),
            dq.stride(3),
            q_idx_flat,
            sqih,
            sqim,
            num_steps,
            N_CTX,
            N_CTX_Q,
            BLOCK_M2=BLOCK_M2,
            BLOCK_N2=BLOCK_N2,
            HEAD_DIM=ctx.HEAD_DIM,
            num_warps=NUM_WARPS,
            num_stages=NUM_STAGES,
        )
        return dq, dk, dv, None, None, None, None


attention = _attention.apply

def compact(alphas_mask, v, index=None):
    """ v.shape = B, N_CTX, H, dim_head
        alphas.shape = B, N_CTX, H
    """
    B, T, H, dim_head = v.shape
    if index is None:
        with torch.no_grad():
            indices_per_head = alphas_mask.sum(dim=-2)
            buffer_size = (
                indices_per_head.max().int()
            )  # first sum computes the num of non-killed elem per head, we take to max of that
            # sorting: it is very important that the sorting is stable, otherwise we cannot use causal masking
            sorted = alphas_mask.sort(
                dim=-2, descending=True, stable=True
            )  # sorted.indices.shape == (B x T x H) , now sorted over sequence T
            index = sorted.indices[
                :, :buffer_size, :
            ]  # (B x buffer_size x H) expand indices to cover all the dimensions for each heads
    else:
        indices_per_head = None
    compact_v = v.gather(
        dim=-3, index=index.unsqueeze(-1).expand(-1, -1, -1, dim_head)
    )  # (B x buffer_size x H x dim_head) / expand indices to cover all the dimensions for each heads
    return compact_v, index, indices_per_head


# this is needed if q_c is different across batch or head dimension
@torch.no_grad()
def pad_index(index, indices_per_head, pad_idx=-1):
    """ index.shape = B, buffer_size, H  <- index given by `compact`, represents for each batch and timestep the head idx it's originating from
        indices_per_head.shape = B, H  <- for each head, number of "active" timesteps
    """
    B, buffer_size, H = index.shape
    index_copy = torch.clone(index).type(torch.int32)
    mask = torch.arange(buffer_size, device=index.device).view(1, -1, 1).expand(
        B, buffer_size, H
    ) >= indices_per_head.view(B, 1, -1)
    index_copy[mask] = pad_idx
    return index_copy


def attention_prefill(q, k, v, alphas_q):
    BATCH, N_CTX, H, D_HEAD = q.shape
    sm_scale = 1.0 / math.sqrt(D_HEAD)

    # Building compact representations
    q_c, index_q, iph_q = compact(alphas_q, q)
    index_q_padded = pad_index(index_q, iph_q, pad_idx=-1)  # (B, compact_T_q, nh)

    compact_N_CTX_Q = q_c.shape[1]

    # We need to transpose everything
    q_c = (
        q_c.view(BATCH, compact_N_CTX_Q, H, D_HEAD).transpose(1, 2).contiguous()
    )  # (BATCH, H, compact_N_CTX_Q, D_HEAD)
    k = k.transpose(1, 2).contiguous()  # (BATCH, H, N_CTX_KV, D_HEAD)
    v = v.transpose(1, 2).contiguous()  # (BATCH, H, N_CTX_KV, D_HEAD)
    index_q_padded = index_q_padded.transpose(
        1, 2
    ).contiguous()  # (BATCH, H, compact_N_CTX_Q)

    y_c = attention(q_c, k, v, index_q_padded, True, sm_scale, False).transpose(1, 2)
    y = torch.zeros(
        (BATCH, N_CTX, H, D_HEAD), dtype=torch.bfloat16, device=q.device
    ).scatter(
        dim=1,
        index=index_q.long().view(BATCH, -1, H, 1).expand(BATCH, -1, H, D_HEAD),
        src=y_c,
    )

    return y
