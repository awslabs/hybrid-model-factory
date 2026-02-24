"""
Primed GKA layer for vLLM V1 engine.

Adapted from PrimedGDNLayer to be checkpoint-compatible with Hybrid Model Factory
GKA implementation while maintaining vLLM optimizations.

Uses merged conv1d (like GDN) for simpler state management.
Checkpoint weights (q_conv1d, k_conv1d, v_conv1d) are merged at load time.
"""

from typing import Optional

import torch
import torch.nn as nn
import triton
import triton.language as tl
from einops import rearrange
from transformers.activations import ACT2FN

from vllm.v1.attention.backend import AttentionBackend, AttentionMetadata
from vllm.config import CacheConfig, ModelConfig, SpeculativeConfig
from vllm.distributed import (
    divide,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
)
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.layernorm import RMSNormGated
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.mamba.abstract import MambaBase
from vllm.model_executor.layers.mamba.mamba_utils import MambaStateDtypeCalculator
from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
    causal_conv1d_fn,
    causal_conv1d_update,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.model_loader.weight_utils import sharded_weight_loader
from vllm.model_executor.models.utils import extract_layer_index
from vllm.model_executor.utils import set_weight_attrs
from vllm.platforms import current_platform
from vllm.utils.torch_utils import direct_register_custom_op
from .gdn.primed_gdn_attn import PrimedGDNAttentionMetadata
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata

# Import GKA kernels
from .gka.gka_forward import (
    gka_chebyshev_gla_prefill,
    gka_chebyshev_gla_decode,
)



class LinearLowRank(nn.Module):
    """
    Low-rank projection: x -> linearA(rank) -> [act_fn] -> linearB(out).
    B is zero-initialized in HF so the projection starts as identity (via residual).
    Mirrors HF's LinearLowRank used for GQA kv_proj_rank.

    TP strategy:
      linearA is ReplicatedLinear (full weights on every rank) since the
      input is all-gathered k/v and we need the same [L, rank] intermediate
      on every rank.
      linearB is ColumnParallelLinear which shards the output dimension,
      so each rank only computes its local Q-head columns directly.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        act_fn: Optional[nn.Module] = None,
        bias: bool = False,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.act_fn = act_fn if act_fn is not None else nn.Identity()

        self.linearA = ReplicatedLinear(
            input_size=in_features,
            output_size=rank,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.linearA",
        )
        self.linearB = ColumnParallelLinear(
            input_size=rank,
            output_size=out_features,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.linearB",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.linearA(x)
        out = self.act_fn(out)
        out, _ = self.linearB(out)
        return out


def gka_conv_weight_loader(
    key_dim: int,
    kv_k_dim: int,
    kv_v_dim: int,
    tp_size: int,
    tp_rank: int,
):
    """
    Custom weight loader for GKA conv1d that merges separate q_conv1d, k_conv1d, v_conv1d
    weights from checkpoint into a single merged conv1d weight.

    NOTE: Identical to GDN's gdn_conv_weight_loader - can be shared.
    """
    q_shard_size = key_dim // tp_size
    k_shard_size = kv_k_dim // tp_size
    v_shard_size = kv_v_dim // tp_size

    def loader(param: torch.Tensor, loaded_weight: torch.Tensor, shard_id: int) -> None:
        if shard_id == 0:  # q_conv1d
            start_idx = 0
            shard_size = q_shard_size
            full_size = key_dim
        elif shard_id == 1:  # k_conv1d
            start_idx = q_shard_size
            shard_size = k_shard_size
            full_size = kv_k_dim
        elif shard_id == 2:  # v_conv1d
            start_idx = q_shard_size + k_shard_size
            shard_size = v_shard_size
            full_size = kv_v_dim
        else:
            raise ValueError(f"Invalid shard_id: {shard_id}. Expected 0, 1, or 2.")

        loaded_start = tp_rank * shard_size
        loaded_end = loaded_start + shard_size

        if loaded_weight.dim() == 2:
            loaded_weight = loaded_weight.unsqueeze(1)

        param.data[start_idx:start_idx + shard_size, ...] = loaded_weight[
            loaded_start:loaded_end, ...
        ]

    return loader


def gka_in_proj_weight_loader(
    output_sizes: list[int],
    shard_names: list[str],
    tp_size: int,
    tp_rank: int,
    last_shard_pad: int = 0,
):
    """
    Custom weight loader for GKA merged input projections.

    Handles dynamic number of projections based on config flags.
    The last shard may be padded for CUDA stride alignment; the checkpoint
    has the unpadded size.

    Args:
        output_sizes: List of output sizes (before TP, last may include padding)
        shard_names: List of checkpoint weight names in order
        tp_size: Tensor parallel world size
        tp_rank: Tensor parallel rank
        last_shard_pad: Padding added to last shard for alignment.
    """
    shard_sizes = [size // tp_size for size in output_sizes]
    shard_offsets = [0]
    for size in shard_sizes[:-1]:
        shard_offsets.append(shard_offsets[-1] + size)

    # Checkpoint sizes (unpadded)
    ckpt_sizes = list(output_sizes)
    if last_shard_pad > 0:
        ckpt_sizes[-1] -= last_shard_pad

    name_to_idx = {name: idx for idx, name in enumerate(shard_names)}

    def loader(param: torch.Tensor, loaded_weight: torch.Tensor, shard_id: int | str) -> None:
        if isinstance(shard_id, str):
            if shard_id not in name_to_idx:
                raise ValueError(f"Unknown shard name: {shard_id}. Expected one of {shard_names}")
            shard_id = name_to_idx[shard_id]

        if shard_id >= len(output_sizes):
            raise ValueError(f"Invalid shard_id: {shard_id}. Expected 0-{len(output_sizes)-1}.")

        start_idx = shard_offsets[shard_id]
        ckpt_full = ckpt_sizes[shard_id]
        load_size = ckpt_full // tp_size

        loaded_start = tp_rank * load_size
        loaded_end = loaded_start + load_size

        if loaded_weight.shape[0] != ckpt_full:
            raise ValueError(
                f"Loaded weight has wrong size: expected {ckpt_full}, got {loaded_weight.shape[0]} "
                f"(shard_id={shard_id}, shard_names={shard_names})"
            )

        param.data[start_idx:start_idx + load_size, :] = loaded_weight[
            loaded_start:loaded_end, :
        ].contiguous()

    return loader


# =============================================================================
# GKA-SPECIFIC GATING KERNEL
# =============================================================================

@triton.jit
def _fused_gka_gating_kernel(
    # Outputs
    g_ptr,
    alpha_ptr,
    beta_ptr,  # Optional, may be None
    # Inputs
    A_log_ptr,
    a_ptr,
    alpha_in_ptr,
    b_ptr,  # Optional, may be None
    dt_bias_ptr,
    # Dimensions
    num_tokens,
    NUM_HEADS: tl.constexpr,
    HAS_BETA: tl.constexpr,
    # Softplus parameters
    softplus_beta: tl.constexpr,
    softplus_threshold: tl.constexpr,
    # Block size
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused triton kernel for GKA gating computation.

    Computes:
        g = -exp(A_log) * softplus(a + dt_bias)
        alpha = sigmoid(alpha_in)
        beta = sigmoid(b) if HAS_BETA else None

    Input shapes:
        A_log: [num_heads]
        a: [num_tokens, num_heads]
        alpha_in: [num_tokens, num_heads]
        b: [num_tokens, num_heads] (optional)
        dt_bias: [num_heads]

    Output shapes:
        g: [num_tokens, num_heads]
        alpha: [num_tokens, num_heads]
        beta: [num_tokens, num_heads] (optional)
    """
    token_idx = tl.program_id(0)
    head_offsets = tl.arange(0, BLOCK_SIZE)

    for head_start in range(0, NUM_HEADS, BLOCK_SIZE):
        head_idx = head_start + head_offsets
        mask = head_idx < NUM_HEADS

        offset_2d = token_idx * NUM_HEADS + head_idx

        # Load inputs
        A_log = tl.load(A_log_ptr + head_idx, mask=mask).to(tl.float32)
        a = tl.load(a_ptr + offset_2d, mask=mask).to(tl.float32)
        alpha_in = tl.load(alpha_in_ptr + offset_2d, mask=mask).to(tl.float32)
        dt_bias = tl.load(dt_bias_ptr + head_idx, mask=mask).to(tl.float32)

        # Compute g = -exp(A_log) * softplus(a + dt_bias)
        x = a + dt_bias
        softplus_x = tl.where(
            softplus_beta * x <= softplus_threshold,
            (1.0 / softplus_beta) * tl.log(1.0 + tl.exp(softplus_beta * x)),
            x
        )
        g = -tl.exp(A_log) * softplus_x

        # Compute alpha = sigmoid(alpha_in)
        alpha = tl.sigmoid(alpha_in)

        # Store outputs
        tl.store(g_ptr + offset_2d, g, mask=mask)
        tl.store(alpha_ptr + offset_2d, alpha, mask=mask)

        # Compute beta if enabled
        if HAS_BETA:
            b = tl.load(b_ptr + offset_2d, mask=mask).to(tl.float32)
            beta = tl.sigmoid(b)
            tl.store(beta_ptr + offset_2d, beta, mask=mask)


def fused_gka_gating(
    A_log: torch.Tensor,
    a: torch.Tensor,
    alpha_in: torch.Tensor,
    b: Optional[torch.Tensor],
    dt_bias: torch.Tensor,
    beta: float = 1.0,
    threshold: float = 20.0,
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """
    Fused computation of g, alpha, and beta for GKA using triton kernel.

    Args:
        A_log: A_log tensor of shape [num_heads]
        a: a_proj output of shape [num_tokens, num_heads]
        alpha_in: alpha_proj output of shape [num_tokens, num_heads]
        b: b_proj output of shape [num_tokens, num_heads] (optional)
        dt_bias: dt_bias tensor of shape [num_heads]

    Returns:
        g: gating tensor of shape [1, num_tokens, num_heads]
        alpha: alpha tensor of shape [1, num_tokens, num_heads]
        beta_out: beta tensor of shape [1, num_tokens, num_heads] or None
    """
    num_tokens, num_heads = a.shape

    a = a.contiguous()
    alpha_in = alpha_in.contiguous()

    g = torch.empty(num_tokens, num_heads, dtype=a.dtype, device=a.device)
    alpha_out = torch.empty(num_tokens, num_heads, dtype=a.dtype, device=a.device)

    has_beta = b is not None
    if has_beta:
        b = b.contiguous()
        beta_out = torch.empty(num_tokens, num_heads, dtype=a.dtype, device=a.device)
    else:
        beta_out = None

    BLOCK_SIZE = triton.next_power_of_2(min(num_heads, 128))
    grid = (num_tokens,)

    _fused_gka_gating_kernel[grid](
        g, alpha_out, beta_out if has_beta else g,  # Dummy ptr if no beta
        A_log, a, alpha_in, b if has_beta else a,  # Dummy ptr if no beta
        dt_bias,
        num_tokens, num_heads, has_beta,
        beta, threshold,
        BLOCK_SIZE,
    )

    return g.unsqueeze(0), alpha_out.unsqueeze(0), beta_out.unsqueeze(0) if has_beta else None


# =============================================================================
# CUSTOM OP REGISTRATION
# =============================================================================

def gka_attention_core(
    mixed_qkv: torch.Tensor,
    a: torch.Tensor,
    alpha_in: torch.Tensor,
    b: torch.Tensor,
    core_attn_out: torch.Tensor,
    layer_name: str,
) -> None:
    """Custom op for GKA attention core computation.

    Note: gk (h_kk decay gate) is currently derived from g inside _forward_core.
    Independent h_kk gating (gk != g) would require adding a separate gk parameter
    to this op signature and relaxing the `assert g is gk` in gka_chebyshev_gla_decode.
    """
    forward_context = get_forward_context()
    self = forward_context.no_compile_layers[layer_name]
    self._forward_core(mixed_qkv, a, alpha_in, b, core_attn_out)


def gka_attention_core_fake(
    mixed_qkv: torch.Tensor,
    a: torch.Tensor,
    alpha_in: torch.Tensor,
    b: torch.Tensor,
    core_attn_out: torch.Tensor,
    layer_name: str,
) -> None:
    return


direct_register_custom_op(
    op_name="gka_attention_core",
    op_func=gka_attention_core,
    mutates_args=["core_attn_out"],
    fake_impl=gka_attention_core_fake,
)


# =============================================================================
# PRIMED GKA LAYER
# =============================================================================

class PrimedGKALayer(nn.Module, MambaBase):
    """
    GKA layer adapted for vLLM V1 engine with checkpoint compatibility.

    Key differences from PrimedGDNLayer:
    - Uses Chebyshev iterative solver for query refinement
    - Additional alpha_proj for residual connection gating
    - Optional beta_proj for key/value scaling
    - Dual recurrent state: h_kk (sketch covariance) + h_kv (key-value state)
    - Prefill uses chunk_simple_gla_gqa (instead of chunk_gated_delta_rule)
    - Decode uses fused recurrent simple GLA via chebyshev_iteration_gqa

    Uses merged conv1d (like GDN) for simpler state management.
    Checkpoint weights are merged at load time via custom weight loaders.
    """

    @property
    def mamba_type(self) -> str:
        return "gdn_attention"

    def get_state_dtype(self) -> tuple[torch.dtype, torch.dtype, torch.dtype]:
        # GKA has 3 states: conv_state, h_kk, h_kv
        # h_kk must be float32 to avoid numerical drift in Chebyshev solver
        # h_kv can use model dtype (typically bfloat16) - GLA is more robust
        conv_dtype, ssm_dtype = MambaStateDtypeCalculator.gated_delta_net_state_dtype(
            self.model_config.dtype, self.cache_config.mamba_cache_dtype
        )
        return conv_dtype, torch.float32, ssm_dtype  # conv, h_kk, h_kv

    def get_state_shape(self) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        """
        GKA state shape: (conv_state_shape, h_kk_shape, h_kv_shape)
        """
        from .primed_utils import PrimedStateShapeCalculator
        return PrimedStateShapeCalculator.gka_state_shape(
            tp_world_size=self.tp_size,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            hidden_size=self.hidden_size,
            head_dim=self.head_k_dim,
            conv_kernel_size=self.conv_kernel_size,
            num_spec=self.num_spec,
        )

    def __init__(
        self,
        config,
        model_config: ModelConfig | None = None,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        speculative_config: SpeculativeConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()

        # Extract GKA config (new-style: must contain num_q_heads, num_k_heads, etc.)
        gka_config = getattr(config, "gka_config", {}) or {}

        # Apply runtime overrides from --hf-overrides '{"gka_overrides": {"num_iter": 10}}'
        # This avoids the vLLM dict-replacement issue with nested configs.
        gka_overrides = getattr(config, "gka_overrides", None)
        if gka_overrides is not None:
            gka_config.update(gka_overrides)

        if "num_q_heads" not in gka_config:
            raise ValueError(
                "gka_config must contain 'num_q_heads'. "
                "Use construct_gka_config() from gka_config.py to build the config."
            )
        if gka_config.get("num_k_heads") != gka_config.get("num_v_heads"):
            raise ValueError(
                f"GKA config requires num_k_heads == num_v_heads, "
                f"got {gka_config.get('num_k_heads')} != {gka_config.get('num_v_heads')}"
            )

        # =================================================================
        # GKA-SPECIFIC CONFIGURATION PARAMETERS
        # Defaults aligned with GKAConfig dataclass in training code.
        # =================================================================
        self.use_alpha_connection = gka_config.get("use_alpha_connection", True)
        self.use_v_conv = gka_config.get("use_v_conv", True)
        self.use_forgetting_gate = gka_config.get("use_forgetting_gate", True)
        self.use_forgetting_gate_kk = gka_config.get("use_forgetting_gate_kk", True)
        self.gla_rescale = gka_config.get("gla_rescale", True)
        self.solver_type = gka_config.get("solver_type", "chebyshev")
        self.num_iter = gka_config.get("num_iter", 30)
        self.ridge_strength = gka_config.get("ridge_strength", 0.02)
        self.use_gate = gka_config.get("use_gate", True)
        self.use_beta_gate = gka_config.get("use_beta_gate", True)
        self.conv_kernel_size = gka_config.get("conv_size", 4)
        conv_bias = gka_config.get("conv_bias", False)
        self.norm_eps = gka_config.get("norm_eps", 1e-6)
        self.kv_proj_rank = gka_config.get("kv_proj_rank", None)
        self.kv_learnable_residual = gka_config.get("kv_learnable_residual", False)

        # =================================================================
        # DIMENSION CALCULATION
        # =================================================================
        self.hidden_size = config.hidden_size
        self.num_heads = gka_config["num_q_heads"]
        self.num_kv_heads = gka_config["num_k_heads"]
        head_dim = gka_config["head_dim"]
        self.num_kv_groups = self.num_heads // self.num_kv_heads

        expand_k = (self.num_heads * head_dim) / self.hidden_size
        expand_v = (self.num_heads * head_dim) / self.hidden_size

        self.key_dim = int(self.hidden_size * expand_k)
        self.value_dim = int(self.hidden_size * expand_v)

        assert self.key_dim % self.num_heads == 0
        assert self.value_dim % self.num_heads == 0

        self.head_k_dim = self.key_dim // self.num_heads
        self.head_v_dim = self.value_dim // self.num_heads

        # GQA dimensions
        self.kv_k_dim = self.key_dim // self.num_kv_groups
        self.kv_v_dim = self.value_dim // self.num_kv_groups

        # GLA scale factor
        if self.gla_rescale:
            self.gla_scale = self.head_k_dim ** -0.5
        else:
            self.gla_scale = 1.0

        self.layer_idx = extract_layer_index(prefix)
        self.activation = config.hidden_act
        self.act = ACT2FN[config.hidden_act]
        self.prefix = prefix

        self.config = config
        self.model_config = model_config
        self.cache_config = cache_config
        if cache_config is not None and cache_config.mamba_cache_mode == "all":
            raise NotImplementedError(
                "GKA layers do not support 'all' prefix caching, "
                "please use '--mamba-cache-mode=align' instead"
            )
        self.quant_config = quant_config
        self.speculative_config = speculative_config
        self.num_spec = (
            self.speculative_config.num_speculative_tokens
            if self.speculative_config
            else 0
        )

        # =================================================================
        # MERGED INPUT PROJECTION (single GEMM for ALL projections)
        # =================================================================
        # Build output sizes and shard names dynamically based on config
        # Always present: q, k, v, a
        # Optional: alpha (if use_alpha_connection), b (if use_beta_gate), g (if use_gate)
        # Biases for alpha and b are handled separately after the GEMM

        self.in_proj_output_sizes = [self.key_dim, self.kv_k_dim, self.kv_v_dim, self.num_heads]
        self.in_proj_shard_names = ["q_proj", "k_proj", "v_proj", "a_proj"]

        # Track indices for optional projections
        self._alpha_idx = None
        self._b_idx = None
        self._g_idx = None

        if self.use_alpha_connection:
            self._alpha_idx = len(self.in_proj_output_sizes)
            self.in_proj_output_sizes.append(self.num_heads)
            self.in_proj_shard_names.append("alpha_proj")

        if self.use_beta_gate:
            self._b_idx = len(self.in_proj_output_sizes)
            self.in_proj_output_sizes.append(self.num_heads)
            self.in_proj_shard_names.append("b_proj")

        if self.use_gate:
            self._g_idx = len(self.in_proj_output_sizes)
            self.in_proj_output_sizes.append(self.value_dim)
            self.in_proj_shard_names.append("g_proj")

        # Pad the per-rank total to 64-alignment to avoid stride mismatch
        # between torch.compile trace and CUDA execution. The CUDA allocator
        # pads tensor strides to 64-element boundaries, which can differ
        # between tracing and execution if we don't align explicitly.
        # We pad via the last projection (like Mamba2 does with dt).
        per_rank_total = sum(s // self.tp_size for s in self.in_proj_output_sizes)
        per_rank_pad = (64 - (per_rank_total % 64)) % 64
        self._in_proj_pad = per_rank_pad * self.tp_size
        if self._in_proj_pad > 0:
            self.in_proj_output_sizes[-1] += self._in_proj_pad

        self.in_proj = MergedColumnParallelLinear(
            input_size=self.hidden_size,
            output_sizes=self.in_proj_output_sizes,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.in_proj",
        )

        # Custom weight loader for merged projection
        # Pass padded output sizes so offsets are correct, but the loader
        # knows the checkpoint has unpadded sizes via the extra parameter
        delattr(self.in_proj.weight, "weight_loader")
        set_weight_attrs(
            self.in_proj.weight,
            {
                "weight_loader": gka_in_proj_weight_loader(
                    self.in_proj_output_sizes,
                    self.in_proj_shard_names,
                    self.tp_size,
                    self.tp_rank,
                    last_shard_pad=self._in_proj_pad,
                )
            },
        )

        # Pre-compute split sizes for forward pass (after TP sharding)
        self.in_proj_split_sizes = [size // self.tp_size for size in self.in_proj_output_sizes]

        # =================================================================
        # SEPARATE BIAS PARAMETERS (for alpha_proj and b_proj)
        # =================================================================
        # These are added manually after splitting the merged projection output

        if self.use_alpha_connection:
            self.alpha_bias = nn.Parameter(torch.zeros(self.num_heads // self.tp_size))
            set_weight_attrs(self.alpha_bias, {"weight_loader": sharded_weight_loader(0)})
        else:
            self.register_parameter("alpha_bias", None)

        if self.use_beta_gate:
            self.b_bias = nn.Parameter(torch.zeros(self.num_heads // self.tp_size))
            set_weight_attrs(self.b_bias, {"weight_loader": sharded_weight_loader(0)})
        else:
            self.register_parameter("b_bias", None)

        # =================================================================
        # MERGED CONV1D
        # =================================================================
        self.conv_dim = self.key_dim + self.kv_k_dim + self.kv_v_dim
        self.conv1d = ColumnParallelLinear(
            input_size=self.conv_kernel_size,
            output_size=self.conv_dim,
            bias=conv_bias,
            prefix=f"{prefix}.conv1d",
        )
        self.conv1d.weight.data = self.conv1d.weight.data.unsqueeze(1)

        # Custom weight loader to merge q_conv1d, k_conv1d, v_conv1d from checkpoint
        delattr(self.conv1d.weight, "weight_loader")
        set_weight_attrs(
            self.conv1d.weight,
            {
                "weight_loader": gka_conv_weight_loader(
                    self.key_dim,
                    self.kv_k_dim,
                    self.kv_v_dim,
                    self.tp_size,
                    self.tp_rank,
                )
            },
        )

        # =================================================================
        # KV LOW-RANK PROJECTION (kv_proj_rank for GQA upsampling)
        # =================================================================
        # When kv_proj_rank is set, k/v are upsampled from num_kv_heads to
        # num_heads via a low-rank projection + residual, instead of simple
        # repeat_interleave. This matches HF's LinearLowRank(proj_k, proj_v).
        if self.kv_proj_rank is not None and self.num_kv_groups > 1:
            act_fn = nn.SiLU() 
            self.proj_k = LinearLowRank(
                in_features=self.kv_k_dim,
                out_features=self.key_dim,
                rank=self.kv_proj_rank,
                act_fn=act_fn,
                quant_config=quant_config,
                prefix=f"{prefix}.proj_k",
            )
            self.proj_v = LinearLowRank(
                in_features=self.kv_v_dim,
                out_features=self.value_dim,
                rank=self.kv_proj_rank,
                act_fn=act_fn,
                quant_config=quant_config,
                prefix=f"{prefix}.proj_v",
            )

            if self.kv_learnable_residual:
                # Learnable mixing: k_rep is [num_heads, num_kv_heads].
                # Replicated on every rank (tiny matrix).
                self.k_rep = nn.Parameter(torch.empty(self.num_heads, self.num_kv_heads))
                self.v_rep = nn.Parameter(torch.empty(self.num_heads, self.num_kv_heads))

                def _kv_rep_loader(param, loaded_weight, *args, **kwargs):
                    param.data.copy_(loaded_weight)

                set_weight_attrs(self.k_rep, {"weight_loader": _kv_rep_loader})
                set_weight_attrs(self.v_rep, {"weight_loader": _kv_rep_loader})
            else:
                self.k_rep = None
                self.v_rep = None
        else:
            self.proj_k = None
            self.proj_v = None
            self.k_rep = None
            self.v_rep = None

        # =================================================================
        # GATING PARAMETERS (similar to GDN)
        # =================================================================

        # A_log parameter (same initialization pattern as GDN)
        self.A = nn.Parameter(
            torch.empty(divide(self.num_heads, self.tp_size)),
        )
        set_weight_attrs(self.A, {"weight_loader": sharded_weight_loader(0)})

        # dt_bias parameter
        # HF GKA: dt initialized with exp(uniform) then inverse softplus
        self.dt_bias = nn.Parameter(
            torch.zeros(self.num_heads // self.tp_size),
        )
        set_weight_attrs(self.dt_bias, {"weight_loader": sharded_weight_loader(0)})

        # =================================================================
        # OUTPUT PROJECTION
        # =================================================================
        # Note: g_proj (output gate) is now part of merged in_proj when use_gate=True

        # o_norm: output normalization
        # When use_gate=False, we pass z=ones_like() to RMSNormGated which
        # effectively makes it behave like RMSNorm (gate of 1 = no gating).
        self.o_norm = RMSNormGated(
            self.head_v_dim,
            eps=self.norm_eps,
            group_size=None,
            norm_before_gate=True,
            device=current_platform.current_device(),
            dtype=None,
        )

        # o_proj: output projection
        self.o_proj = RowParallelLinear(
            self.value_dim,
            self.hidden_size,
            bias=False,
            input_is_parallel=True,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        # =================================================================
        # REGISTER IN COMPILATION CONTEXT
        # =================================================================
        from vllm.config import get_current_vllm_config
        compilation_config = get_current_vllm_config().compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

    # =========================================================================
    # HELPER METHODS
    # =========================================================================

    def rearrange_qkv(self, mixed_qkv):
        """
        Rearrange mixed QKV for FLA ops.
        When kv_proj_rank is set, applies low-rank projection + residual
        to upsample k/v from num_kv_heads to num_heads.
        Otherwise, K/V keep HKV heads for native GQA kernels.
        """
        if mixed_qkv is None:
            return None, None, None

        num_heads_tp = self.num_heads // self.tp_size
        num_kv_heads_tp = self.num_kv_heads // self.tp_size

        query, key, value = torch.split(
            mixed_qkv,
            [
                self.key_dim // self.tp_size,
                self.kv_k_dim // self.tp_size,
                self.kv_v_dim // self.tp_size,
            ],
            dim=-1,
        )

        # Reshape query: [num_tokens, key_dim_tp] → [1, num_tokens, num_heads_tp, head_k_dim]
        query = rearrange(query, "l (h d) -> 1 l h d", h=num_heads_tp, d=self.head_k_dim)

        if self.proj_k is not None:
            # Low-rank projection path: upsample k/v to full num_heads.
            # All-gather k/v so proj sees all KV heads.
            k_full = tensor_model_parallel_all_gather(key.contiguous(), dim=-1)  # [l, kv_k_dim]
            v_full = tensor_model_parallel_all_gather(value.contiguous(), dim=-1)  # [l, kv_v_dim]

            q_start = self.tp_rank * num_heads_tp

            # 1. Compute residual (repeat or learnable mixing),
            #    sliced to local Q heads only
            if self.kv_learnable_residual:
                k_rep_local = self.k_rep[q_start:q_start + num_heads_tp, :]  # [num_heads_tp, num_kv_heads]
                v_rep_local = self.v_rep[q_start:q_start + num_heads_tp, :]
                k_rearranged = rearrange(k_full, "l (h d) -> l h d", h=self.num_kv_heads)
                k_res = torch.einsum("kh,lhd->lkd", k_rep_local, k_rearranged)  # [l, num_heads_tp, D]
                v_rearranged = rearrange(v_full, "l (h d) -> l h d", h=self.num_kv_heads)
                v_res = torch.einsum("kh,lhd->lkd", v_rep_local, v_rearranged)  # [l, num_heads_tp, D]
            else:
                k_res = rearrange(
                    k_full, "l (h d) -> l h d", h=self.num_kv_heads
                ).repeat_interleave(self.num_kv_groups, dim=1)
                v_res = rearrange(
                    v_full, "l (h d) -> l h d", h=self.num_kv_heads
                ).repeat_interleave(self.num_kv_groups, dim=1)
                k_res = k_res[:, q_start:q_start + num_heads_tp, :]
                v_res = v_res[:, q_start:q_start + num_heads_tp, :]

            # 2. Low-rank projection (ColumnParallelLinear produces local shard)
            key = self.proj_k(k_full)  # [l, key_dim_tp]
            key = rearrange(key, "l (h d) -> l h d", h=num_heads_tp, d=self.head_k_dim)
            key = key + k_res

            value = self.proj_v(v_full)  # [l, value_dim_tp]
            value = rearrange(value, "l (h d) -> l h d", h=num_heads_tp, d=self.head_v_dim)
            value = value + v_res

            # Add batch dim: [l, h, d] -> [1, l, h, d]
            key = key.unsqueeze(0)
            value = value.unsqueeze(0)
        else:
            # Standard path: K/V keep HKV heads (GQA handled by kernels)
            key = rearrange(key, "l (h d) -> 1 l h d", h=num_kv_heads_tp, d=self.head_k_dim)
            value = rearrange(value, "l (h d) -> 1 l h d", h=num_kv_heads_tp, d=self.head_v_dim)

        return query.contiguous(), key.contiguous(), value.contiguous()

    # =========================================================================
    # FORWARD PASS
    # =========================================================================

    def forward(
        self,
        hidden_states: torch.Tensor,
        output: torch.Tensor,
        positions: torch.Tensor | None = None,
    ):
        """
        Forward pass with three parts:
        1. Input projection (single merged GEMM)
        2. Core attention (custom op)
        3. Output projection

        Args:
            hidden_states: Input tensor [num_tokens, hidden_size]
            output: Output tensor (modified in-place)
            positions: Position indices (unused)
        """
        num_tokens = hidden_states.size(0)

        # =================================================================
        # Part 1: Input Projection (single merged GEMM)
        # =================================================================

        # Merged projection: q, k, v, a, [alpha], [b], [g] in one GEMM
        projected, _ = self.in_proj(hidden_states)

        # Split all projections (returns tuple, directly indexable)
        # Last split may include padding for 64-element CUDA stride alignment
        proj_splits = torch.split(projected, self.in_proj_split_sizes, dim=-1)

        # Always present: q, k, v, a at indices 0, 1, 2, 3
        q, k, v, a_out = proj_splits[0], proj_splits[1], proj_splits[2], proj_splits[3]

        # Optional projections accessed by index
        if self._alpha_idx is not None:
            alpha_out = proj_splits[self._alpha_idx]
            if self.alpha_bias is not None:
                alpha_out = alpha_out + self.alpha_bias
        else:
            alpha_out = torch.zeros(num_tokens, self.num_heads // self.tp_size,
                                   device=hidden_states.device, dtype=hidden_states.dtype)

        if self._b_idx is not None:
            b_out = proj_splits[self._b_idx]
            if self.b_bias is not None:
                b_out = b_out + self.b_bias
        else:
            b_out = None

        if self._g_idx is not None:
            g_out = proj_splits[self._g_idx]
            # Slice off alignment padding if present
            if self._in_proj_pad > 0:
                g_out = g_out[..., :self.value_dim // self.tp_size]
        else:
            g_out = None

        # Slice off alignment padding from last split when g is not the last
        if self._in_proj_pad > 0 and self._g_idx is None:
            if self._b_idx is not None:
                b_out = b_out[..., :self.num_heads // self.tp_size]
            elif self._alpha_idx is not None:
                alpha_out = alpha_out[..., :self.num_heads // self.tp_size]
            else:
                a_out = a_out[..., :self.num_heads // self.tp_size]

        # Create mixed_qkv for conv operation
        mixed_qkv = torch.cat((q, k, v), dim=-1)

        # =================================================================
        # Part 2: Core Attention (Custom Op)
        # =================================================================
        core_attn_out = torch.zeros(
            (num_tokens, self.num_heads // self.tp_size, self.head_v_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

        torch.ops.vllm.gka_attention_core(
            mixed_qkv,
            a_out,
            alpha_out,
            b_out if b_out is not None else a_out,  # Dummy if no beta
            core_attn_out,
            self.prefix,
        )

        # =================================================================
        # Part 3: Output Projection
        # =================================================================

        # Output gate (from merged projection if use_gate=True)
        if g_out is not None:
            z = rearrange(g_out, "l (h d) -> l h d", h=self.num_heads // self.tp_size, d=self.head_v_dim)
        else:
            # When use_gate=False, pass ones to make RMSNormGated behave like RMSNorm
            z = torch.ones_like(core_attn_out)

        z_shape_og = z.shape

        # Reshape for o_norm
        core_attn_out = core_attn_out.reshape(-1, core_attn_out.shape[-1])
        z = z.reshape(-1, z.shape[-1])

        # Apply gated norm
        core_attn_out = self.o_norm(core_attn_out, z)

        # Reshape back and flatten heads
        core_attn_out = core_attn_out.reshape(z_shape_og)
        core_attn_out = rearrange(core_attn_out, "l h d -> l (h d)")

        # Output projection
        output[:num_tokens], _ = self.o_proj(core_attn_out)

    # =========================================================================
    # CORE ATTENTION COMPUTATION
    # =========================================================================

    def _forward_core(
        self,
        mixed_qkv: torch.Tensor,
        a: torch.Tensor,
        alpha_in: torch.Tensor,
        b: torch.Tensor,
        core_attn_out: torch.Tensor,
    ):
        """
        Core attention computation (called by custom op).
        Handles convolution, gating, and GKA-specific recurrent computation.
        """
        forward_context = get_forward_context()
        attn_metadata: AttentionMetadata = forward_context.attn_metadata

        if attn_metadata is None:
            # V1 profile run
            return

        assert isinstance(attn_metadata, dict)
        attn_metadata = attn_metadata[self.prefix]
        assert isinstance(attn_metadata, GDNAttentionMetadata)

        # Extract metadata (same as GDN)
        has_initial_state = attn_metadata.has_initial_state
        spec_query_start_loc = attn_metadata.spec_query_start_loc
        non_spec_query_start_loc = attn_metadata.non_spec_query_start_loc
        spec_sequence_masks = attn_metadata.spec_sequence_masks
        spec_token_indx = attn_metadata.spec_token_indx
        non_spec_token_indx = attn_metadata.non_spec_token_indx
        spec_state_indices_tensor = attn_metadata.spec_state_indices_tensor
        non_spec_state_indices_tensor = attn_metadata.non_spec_state_indices_tensor

        # Cache access: GKA has 3 states
        self_kv_cache = self.kv_cache[forward_context.virtual_engine]
        conv_state = self_kv_cache[0].transpose(-1, -2)
        h_kk_state = self_kv_cache[1]  # Chebyshev solver state
        h_kv_state = self_kv_cache[2]  # Simple GLA state

        num_actual_tokens = attn_metadata.num_actual_tokens

        mixed_qkv = mixed_qkv[:num_actual_tokens]
        a = a[:num_actual_tokens]
        alpha_in = alpha_in[:num_actual_tokens]
        if self.use_beta_gate:
            b = b[:num_actual_tokens]

        # =================================================================
        # 1. CONVOLUTION (same as GDN)
        # =================================================================
        conv_weights = self.conv1d.weight.view(
            self.conv1d.weight.size(0), self.conv1d.weight.size(2)
        )

        # Split spec/non-spec tokens
        if spec_sequence_masks is not None:
            if attn_metadata.num_prefills == 0 and attn_metadata.num_decodes == 0:
                mixed_qkv_spec = mixed_qkv
                mixed_qkv_non_spec = None
            else:
                mixed_qkv_spec = mixed_qkv.index_select(0, spec_token_indx)
                mixed_qkv_non_spec = mixed_qkv.index_select(0, non_spec_token_indx)
        else:
            mixed_qkv_spec = None
            mixed_qkv_non_spec = mixed_qkv

        # Process speculative decoding part
        num_accepted_tokens = attn_metadata.num_accepted_tokens
        if spec_sequence_masks is not None:
            mixed_qkv_spec = causal_conv1d_update(
                mixed_qkv_spec,
                conv_state,
                conv_weights,
                None,  # bias
                self.activation,
                conv_state_indices=spec_state_indices_tensor[:, 0][
                    : attn_metadata.num_spec_decodes
                ],
                num_accepted_tokens=num_accepted_tokens,
                query_start_loc=spec_query_start_loc,
                max_query_len=spec_state_indices_tensor.size(-1),
                validate_data=False,
            )

        # Process prefill/decode part
        # In mixed batches, split non-spec tokens into decode and prefill
        # subsets so each uses its optimal kernel. Follows Mamba2/GDN pattern.
        num_non_spec_decodes = attn_metadata.num_decodes
        num_non_spec_decode_tokens = attn_metadata.num_decode_tokens
        num_non_spec_prefills = attn_metadata.num_prefills
        has_non_spec_prefill = num_non_spec_prefills > 0
        has_non_spec_decode = num_non_spec_decodes > 0
        is_mixed = has_non_spec_prefill and has_non_spec_decode

        if is_mixed:
            # Split tokens (decodes first in batch)
            mixed_qkv_non_spec_d = mixed_qkv_non_spec[:num_non_spec_decode_tokens]
            mixed_qkv_non_spec_p = mixed_qkv_non_spec[num_non_spec_decode_tokens:]
            # Split state indices along batch dim
            state_indices_d = non_spec_state_indices_tensor[:num_non_spec_decodes]
            state_indices_p = non_spec_state_indices_tensor[num_non_spec_decodes:]

            # Conv for decodes (recurrent update)
            mixed_qkv_non_spec_d = causal_conv1d_update(
                mixed_qkv_non_spec_d,
                conv_state,
                conv_weights,
                None,  # bias
                self.activation,
                conv_state_indices=state_indices_d,
                validate_data=True,
            )
            # Conv for prefills (chunked)
            # metadata=attn_metadata works because builder overwrote
            # nums_dict/batch_ptr/token_chunk_offset_ptr with prefill-only versions
            mixed_qkv_non_spec_p = causal_conv1d_fn(
                mixed_qkv_non_spec_p.transpose(0, 1),
                conv_weights,
                None,  # bias
                activation=self.activation,
                conv_states=conv_state,
                has_initial_state=has_initial_state,
                cache_indices=state_indices_p,
                query_start_loc=attn_metadata.non_spec_query_start_loc_p,
                metadata=attn_metadata,
            ).transpose(0, 1)
            # Recombine
            mixed_qkv_non_spec = torch.cat(
                [mixed_qkv_non_spec_d, mixed_qkv_non_spec_p], dim=0
            )
        elif has_non_spec_prefill:
            mixed_qkv_non_spec = causal_conv1d_fn(
                mixed_qkv_non_spec.transpose(0, 1),
                conv_weights,
                None,  # bias
                activation=self.activation,
                conv_states=conv_state,
                has_initial_state=has_initial_state,
                cache_indices=non_spec_state_indices_tensor,
                query_start_loc=non_spec_query_start_loc,
                metadata=attn_metadata,
            ).transpose(0, 1)
        elif has_non_spec_decode:
            mixed_qkv_non_spec = causal_conv1d_update(
                mixed_qkv_non_spec,
                conv_state,
                conv_weights,
                None,  # bias
                self.activation,
                conv_state_indices=non_spec_state_indices_tensor[
                    : attn_metadata.num_actual_tokens
                ],
                validate_data=True,
            )
        else:
            mixed_qkv_non_spec = None

        # Rearrange for FLA ops
        query_spec, key_spec, value_spec = self.rearrange_qkv(mixed_qkv_spec)
        query_non_spec, key_non_spec, value_non_spec = self.rearrange_qkv(mixed_qkv_non_spec)

        # =================================================================
        # 2. COMPUTE GATING PARAMETERS (GKA-specific)
        # =================================================================
        # 
        # GKA gating differs from GDN:
        # - g: forgetting gate (same formula as GDN)
        # - alpha: residual connection weight (GKA-specific)
        # - beta: optional k/v scaling (GKA-specific)
        #
        # HF reference:
        #   g = -A_log.exp() * softplus(a_proj(x) + dt_bias)
        #   alpha = alpha_proj(x).sigmoid()
        #   beta = b_proj(x).sigmoid() if use_beta_gate else None
        #   if beta: k = (beta + eps) * k, v = (beta + eps) * v

        if self.use_forgetting_gate:
            g, alpha, beta = fused_gka_gating(
                self.A, a, alpha_in, 
                b if self.use_beta_gate else None,
                self.dt_bias
            )
        else:
            # When use_forgetting_gate=False, g should be None (match HF)
            g = None
            alpha = alpha_in.sigmoid().unsqueeze(0)
            beta = b.sigmoid().unsqueeze(0) if self.use_beta_gate else None

        if self.num_iter == 0:
            alpha = torch.zeros_like(alpha)
        elif self._alpha_idx is None:
            alpha = alpha + 0.5

        # Split g, alpha, beta for spec/non-spec
        if spec_sequence_masks is not None:
            if attn_metadata.num_prefills == 0 and attn_metadata.num_decodes == 0:
                g_spec, alpha_spec = g, alpha
                beta_spec = beta
                g_non_spec, alpha_non_spec = None, None
                beta_non_spec = None
            else:
                g_spec = g.index_select(1, spec_token_indx) if g is not None else None
                alpha_spec = alpha.index_select(1, spec_token_indx)
                g_non_spec = g.index_select(1, non_spec_token_indx) if g is not None else None
                alpha_non_spec = alpha.index_select(1, non_spec_token_indx)
                if beta is not None:
                    beta_spec = beta.index_select(1, spec_token_indx)
                    beta_non_spec = beta.index_select(1, non_spec_token_indx)
                else:
                    beta_spec, beta_non_spec = None, None
        else:
            g_spec, alpha_spec, beta_spec = None, None, None
            g_non_spec, alpha_non_spec = g, alpha
            beta_non_spec = beta

        # L2 normalization and beta scaling are handled inside the GKA kernels
        # (fused_l2_norm in gka_forward.py for prefill, inline in decode kernel)

        # Compute gk for h_kk forgetting (if use_forgetting_gate_kk)
        gk_spec = g_spec if self.use_forgetting_gate_kk else None
        gk_non_spec = g_non_spec if self.use_forgetting_gate_kk else None

        # =================================================================
        # 4. GKA RECURRENT ATTENTION
        # =================================================================

        # 4.1: Process speculative decoding part
        if spec_sequence_masks is not None:
            # Use GKA decode for speculative tokens
            core_attn_out_spec, _, _ = gka_chebyshev_gla_decode(
                q=query_spec,
                k=key_spec,
                v=value_spec,
                alpha=alpha_spec,
                g=g_spec,
                gk=gk_spec,
                beta=beta_spec,
                gla_scale=self.gla_scale,
                ridge_ratio=self.ridge_strength,
                num_iter=self.num_iter,
                h_kk=h_kk_state,
                h_kv=h_kv_state,
                cu_seqlens=spec_query_start_loc[: attn_metadata.num_spec_decodes + 1],
                ssm_state_indices=spec_state_indices_tensor,
            )
        else:
            core_attn_out_spec = None

        # 4.2: Process prefill/decode part
        if is_mixed:
            # Split q/k/v along token dim (dim=1, decodes first)
            q_d = query_non_spec[:, :num_non_spec_decode_tokens]
            q_p = query_non_spec[:, num_non_spec_decode_tokens:]
            k_d = key_non_spec[:, :num_non_spec_decode_tokens]
            k_p = key_non_spec[:, num_non_spec_decode_tokens:]
            v_d = value_non_spec[:, :num_non_spec_decode_tokens]
            v_p = value_non_spec[:, num_non_spec_decode_tokens:]
            # Split gating along token dim
            g_d = g_non_spec[:, :num_non_spec_decode_tokens] if g_non_spec is not None else None
            g_p = g_non_spec[:, num_non_spec_decode_tokens:] if g_non_spec is not None else None
            alpha_d = alpha_non_spec[:, :num_non_spec_decode_tokens]
            alpha_p = alpha_non_spec[:, num_non_spec_decode_tokens:]
            beta_d = beta_non_spec[:, :num_non_spec_decode_tokens] if beta_non_spec is not None else None
            beta_p = beta_non_spec[:, num_non_spec_decode_tokens:] if beta_non_spec is not None else None
            # gk must be the same object as g when use_forgetting_gate_kk
            # (gka_chebyshev_gla_decode asserts `g is gk`), so reuse slices.
            gk_d = g_d if gk_non_spec is g_non_spec else (gk_non_spec[:, :num_non_spec_decode_tokens] if gk_non_spec is not None else None)
            gk_p = g_p if gk_non_spec is g_non_spec else (gk_non_spec[:, num_non_spec_decode_tokens:] if gk_non_spec is not None else None)

            # Decode: recurrent (O(1) per token)
            core_attn_out_d, _, _ = gka_chebyshev_gla_decode(
                q=q_d, k=k_d, v=v_d,
                alpha=alpha_d, g=g_d, gk=gk_d, beta=beta_d,
                gla_scale=self.gla_scale,
                ridge_ratio=self.ridge_strength,
                num_iter=self.num_iter,
                h_kk=h_kk_state,
                h_kv=h_kv_state,
                cu_seqlens=attn_metadata.non_spec_query_start_loc_d,
                ssm_state_indices=state_indices_d,
            )

            # Prefill: chunked
            h_kk_initial = h_kk_state[state_indices_p].contiguous()
            h_kv_initial = h_kv_state[state_indices_p].contiguous()
            has_initial_state_p = has_initial_state[num_non_spec_decodes:]
            h_kk_initial[~has_initial_state_p, ...] = 0
            h_kv_initial[~has_initial_state_p, ...] = 0

            core_attn_out_p, h_kk_final, h_kv_final = gka_chebyshev_gla_prefill(
                q=q_p, k=k_p, v=v_p,
                alpha=alpha_p, g=g_p, gk=gk_p, beta=beta_p,
                gla_scale=self.gla_scale,
                ridge_ratio=self.ridge_strength,
                num_iter=self.num_iter,
                h_kk_initial=h_kk_initial,
                h_kv_initial=h_kv_initial,
                output_final_state=True,
                cu_seqlens=attn_metadata.non_spec_query_start_loc_p,
            )
            if self.num_iter == 0:
                h_kk_state[state_indices_p] = h_kk_initial
            else:
                h_kk_state[state_indices_p] = h_kk_final.to(h_kk_state.dtype)
            h_kv_state[state_indices_p] = h_kv_final.to(h_kv_state.dtype)

            # Merge decode + prefill outputs
            core_attn_out_non_spec = torch.cat(
                [core_attn_out_d, core_attn_out_p], dim=1
            )
        elif has_non_spec_prefill:
            # GKA prefill using gka_chebyshev_gla_prefill
            h_kk_initial = h_kk_state[non_spec_state_indices_tensor].contiguous()
            h_kv_initial = h_kv_state[non_spec_state_indices_tensor].contiguous()
            h_kk_initial[~has_initial_state, ...] = 0
            h_kv_initial[~has_initial_state, ...] = 0

            core_attn_out_non_spec, h_kk_final, h_kv_final = gka_chebyshev_gla_prefill(
                q=query_non_spec,
                k=key_non_spec,
                v=value_non_spec,
                alpha=alpha_non_spec,
                g=g_non_spec,
                gk=gk_non_spec,
                beta=beta_non_spec,
                gla_scale=self.gla_scale,
                ridge_ratio=self.ridge_strength,
                num_iter=self.num_iter,
                h_kk_initial=h_kk_initial,
                h_kv_initial=h_kv_initial,
                output_final_state=True,
                cu_seqlens=non_spec_query_start_loc,
            )
            # Update cache
            if self.num_iter == 0:
                h_kk_state[non_spec_state_indices_tensor] = h_kk_initial
            else:
                h_kk_state[non_spec_state_indices_tensor] = h_kk_final.to(h_kk_state.dtype)
            h_kv_state[non_spec_state_indices_tensor] = h_kv_final.to(h_kv_state.dtype)

        elif has_non_spec_decode:
            # GKA decode using gka_chebyshev_gla_decode
            core_attn_out_non_spec, _, _ = gka_chebyshev_gla_decode(
                q=query_non_spec,
                k=key_non_spec,
                v=value_non_spec,
                alpha=alpha_non_spec,
                beta=beta_non_spec,
                g=g_non_spec,
                gk=gk_non_spec,
                gla_scale=self.gla_scale,
                ridge_ratio=self.ridge_strength,
                num_iter=self.num_iter,
                h_kk=h_kk_state,
                h_kv=h_kv_state,
                cu_seqlens=non_spec_query_start_loc[: num_non_spec_decodes + 1],
                ssm_state_indices=non_spec_state_indices_tensor,
            )
        else:
            core_attn_out_non_spec = None

        # =================================================================
        # 5. MERGE OUTPUTS (same as GDN)
        # =================================================================
        if spec_sequence_masks is not None and core_attn_out_non_spec is not None:
            merged_out = torch.empty(
                (1, num_actual_tokens, *core_attn_out_spec.shape[2:]),
                dtype=core_attn_out_non_spec.dtype,
                device=core_attn_out_non_spec.device,
            )
            merged_out.index_copy_(1, spec_token_indx, core_attn_out_spec)
            merged_out.index_copy_(1, non_spec_token_indx, core_attn_out_non_spec)
            core_attn_out[:num_actual_tokens] = merged_out.squeeze(0)
        elif spec_sequence_masks is not None:
            core_attn_out[:num_actual_tokens] = core_attn_out_spec.squeeze(0)
        else:
            core_attn_out[:num_actual_tokens] = core_attn_out_non_spec.squeeze(0)
