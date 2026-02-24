"""
Primed GatedDeltaNet layer for vLLM V1 engine.

Adapted from vLLM's native Qwen3NextGatedDeltaNet to be checkpoint-compatible
with Hybrid Model Factory's GDN implementation while maintaining vLLM optimizations.
"""

from typing import Optional

import torch
from einops import rearrange
from torch import nn
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
from vllm.model_executor.layers.fla.ops import (
    chunk_gated_delta_rule,
    fused_recurrent_gated_delta_rule,
)
from .gdn.ops import fused_recurrent_gated_delta_rule_gqa, fused_gdn_gating, chunk_gated_delta_rule_gqa
from vllm.model_executor.layers.layernorm import RMSNormGated
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.mamba.abstract import MambaBase
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
)
from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
    causal_conv1d_fn,
    causal_conv1d_update,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    sharded_weight_loader,
)
from vllm.model_executor.models.utils import extract_layer_index
from vllm.model_executor.utils import set_weight_attrs
from vllm.platforms import current_platform
from vllm.utils.torch_utils import direct_register_custom_op
from .gdn.primed_gdn_attn import PrimedGDNAttentionMetadata
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata

from .primed_gka_layer import LinearLowRank

def custom_gdn_attention_core(
    mixed_qkv: torch.Tensor,
    b: torch.Tensor,
    a: torch.Tensor,
    core_attn_out: torch.Tensor,
    layer_name: str,
) -> None:
    """Custom op for the GDN core attention computation.

    Handles convolution, gating, and recurrent/chunk delta rule.
    Input/output projections are handled outside this op.
    """
    forward_context = get_forward_context()
    self = forward_context.no_compile_layers[layer_name]
    self._forward_core(mixed_qkv, b, a, core_attn_out)


def custom_gdn_attention_core_fake(
    mixed_qkv: torch.Tensor,
    b: torch.Tensor,
    a: torch.Tensor,
    core_attn_out: torch.Tensor,
    layer_name: str,
) -> None:
    return


direct_register_custom_op(
    op_name="custom_gdn_attention_core",
    op_func=custom_gdn_attention_core,
    mutates_args=["core_attn_out"],
    fake_impl=custom_gdn_attention_core_fake,
)


def gdn_conv_weight_loader(
    key_dim: int,
    kv_k_dim: int,
    kv_v_dim: int,
    tp_size: int,
    tp_rank: int,
):
    """
    Custom weight loader for GDN conv1d that merges separate q_conv1d, k_conv1d, v_conv1d
    weights from checkpoint into a single merged conv1d weight.

    Args:
        key_dim: Full query dimension (before TP)
        kv_k_dim: Full key dimension (before TP, for GQA)
        kv_v_dim: Full value dimension (before TP, for GQA)
        tp_size: Tensor parallel world size
        tp_rank: Tensor parallel rank

    Returns:
        Loader function that accepts (param, loaded_weight, shard_id)
    """
    # Calculate shard sizes
    q_shard_size = key_dim // tp_size
    k_shard_size = kv_k_dim // tp_size
    v_shard_size = kv_v_dim // tp_size

    def loader(param: torch.Tensor, loaded_weight: torch.Tensor, shard_id: int) -> None:
        """
        Load a shard of conv weights into the merged conv1d parameter.

        Args:
            param: The merged conv1d.weight parameter [total_dim, 1, kernel_size]
            loaded_weight: The checkpoint weight [shard_dim, 1, kernel_size]
            shard_id: 0=q_conv1d, 1=k_conv1d, 2=v_conv1d
        """
        # Determine which shard and its size
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

        # Calculate which part of the loaded weight to take for this TP rank
        loaded_start = tp_rank * shard_size
        loaded_end = loaded_start + shard_size

        # Ensure loaded_weight has the right shape [full_size, 1, kernel_size]
        if loaded_weight.dim() == 2:
            # If checkpoint has [full_size, kernel_size], add middle dim
            loaded_weight = loaded_weight.unsqueeze(1)

        # Copy the appropriate shard into the merged parameter
        param.data[start_idx:start_idx + shard_size, ...] = loaded_weight[
            loaded_start:loaded_end, ...
        ]

    return loader


def gdn_linear_weight_loader(
    output_sizes: list[int],
    tp_size: int,
    tp_rank: int,
):
    """
    Custom weight loader for GDN merged linear projections.

    Args:
        output_sizes: List of output sizes for each shard (before TP)
        tp_size: Tensor parallel world size
        tp_rank: Tensor parallel rank

    Returns:
        Loader function that accepts (param, loaded_weight, shard_id)
    """
    # Calculate shard sizes and offsets
    shard_sizes = [size // tp_size for size in output_sizes]
    shard_offsets = [0]
    for size in shard_sizes[:-1]:
        shard_offsets.append(shard_offsets[-1] + size)

    def loader(param: torch.Tensor, loaded_weight: torch.Tensor, shard_id: int) -> None:
        """
        Load a shard of linear weights into the merged parameter.

        Args:
            param: The merged parameter [total_output_dim, input_dim]
            loaded_weight: The checkpoint weight [output_dim, input_dim]
            shard_id: Index into output_sizes list
        """
        if shard_id >= len(output_sizes):
            raise ValueError(f"Invalid shard_id: {shard_id}. Expected 0-{len(output_sizes)-1}.")

        # Get the shard info
        start_idx = shard_offsets[shard_id]
        shard_size = shard_sizes[shard_id]
        full_size = output_sizes[shard_id]

        # Calculate which part of the loaded weight to take for this TP rank
        loaded_start = tp_rank * shard_size
        loaded_end = loaded_start + shard_size

        # Validate dimensions
        if loaded_weight.shape[0] != full_size:
            raise ValueError(
                f"Loaded weight has wrong size: expected {full_size}, got {loaded_weight.shape[0]} "
                f"(shard_id={shard_id}, output_sizes={output_sizes})"
            )

        # Copy the appropriate shard into the merged parameter
        param.data[start_idx:start_idx + shard_size, :] = loaded_weight[
            loaded_start:loaded_end, :
        ].contiguous()

    return loader


class PrimedGDNLayer(nn.Module, MambaBase):
    """
    GatedDeltaNet layer adapted for vLLM V1 engine with checkpoint compatibility.

    Key differences from native vLLM GDN:
    - Uses expand_k/expand_v dimension calculation (from config)
    - Parameters sized by num_heads (not num_v_heads)
    - Supports allow_neg_eigval for beta scaling
    - Maintains checkpoint compatibility with Hybrid Model Factory
    """

    @property
    def mamba_type(self) -> str:
        return "gdn_attention"

    def get_state_dtype(self) -> tuple[torch.dtype, torch.dtype]:
        return MambaStateDtypeCalculator.gated_delta_net_state_dtype(
            self.model_config.dtype, self.cache_config.mamba_cache_dtype
        )

    def get_state_shape(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        return MambaStateShapeCalculator.gated_delta_net_state_shape(
            self.tp_size,
            self.num_kv_heads,  # Use num_kv_heads (your style)
            self.num_heads,     # Use num_heads (your style)
            self.head_k_dim,
            self.head_v_dim,
            self.conv_kernel_size,
            self.num_spec,
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

        # Extract GDN config
        gdn_config = getattr(config, "gdn_config", {}) or {}
        self.allow_neg_eigval = gdn_config.get("allow_neg_eigval", False)
        self.conv_kernel_size = gdn_config.get("conv_size", 4)
        conv_bias = gdn_config.get("conv_bias", False)

        self.kv_proj_rank = gdn_config.get("kv_proj_rank", None)
        self.kv_learnable_residual = gdn_config.get("kv_learnable_residual", False)

        # Dimension calculation
        self.hidden_size = config.hidden_size
        if "num_q_heads" in gdn_config:
            self.num_heads = gdn_config["num_q_heads"]
            self.num_kv_heads = gdn_config["num_k_heads"]
            head_dim = gdn_config["head_dim"]
        else:
            self.num_heads = config.num_attention_heads
            self.num_kv_heads = config.num_key_value_heads
            head_dim = getattr(config, "head_dim", self.hidden_size // self.num_heads)
        self.num_kv_groups = self.num_heads // self.num_kv_heads

        expand_k = (self.num_heads * head_dim) / self.hidden_size
        expand_v = (self.num_heads * head_dim) / self.hidden_size

        # Calculate dimensions
        self.key_dim = int(self.hidden_size * expand_k)
        self.value_dim = int(self.hidden_size * expand_v)

        # Validate divisibility
        assert self.key_dim % self.num_heads == 0, \
            f"key_dim ({self.key_dim}) must be divisible by num_heads ({self.num_heads})"
        assert self.value_dim % self.num_heads == 0, \
            f"value_dim ({self.value_dim}) must be divisible by num_heads ({self.num_heads})"

        # Derive head dimensions
        self.head_k_dim = self.key_dim // self.num_heads
        self.head_v_dim = self.value_dim // self.num_heads

        # GQA dimensions (for conv and projections)
        self.kv_k_dim = self.key_dim // self.num_kv_groups
        self.kv_v_dim = self.value_dim // self.num_kv_groups

        # For vLLM compatibility (used in state shape calculation)
        self.num_k_heads = self.num_kv_heads
        self.num_v_heads = self.num_heads

        self.layer_idx = extract_layer_index(prefix)
        self.activation = config.hidden_act
        self.act = ACT2FN[config.hidden_act]
        # Use 1e-5 for GDN o_norm to match HF's FusedRMSNormGated default
        # (config.rms_norm_eps may be 1e-6 for other norms, but GDN uses 1e-5)
        self.layer_norm_epsilon = 1e-5
        self.prefix = prefix

        self.config = config
        self.model_config = model_config
        self.cache_config = cache_config
        if cache_config is not None and cache_config.mamba_cache_mode == "all":
            raise NotImplementedError(
                "GDN layers do not support 'all' prefix caching, "
                "please use '--mamba-cache-mode=align' instead"
            )
        self.quant_config = quant_config
        self.speculative_config = speculative_config
        self.num_spec = (
            self.speculative_config.num_speculative_tokens
            if self.speculative_config
            else 0
        )

        # Merged Conv1d layer (like native qwen3_next)
        # Checkpoint has separate q_conv1d, k_conv1d, v_conv1d weights
        # We merge them at load time using custom weight loader
        self.conv_dim = self.key_dim + self.kv_k_dim + self.kv_v_dim
        self.conv1d = ColumnParallelLinear(
            input_size=self.conv_kernel_size,
            output_size=self.conv_dim,
            bias=conv_bias,
            prefix=f"{prefix}.conv1d",
        )
        self.conv1d.weight.data = self.conv1d.weight.data.unsqueeze(1)

        # Set up custom weight loader to merge separate checkpoint weights
        # This loader accepts (param, loaded_weight, shard_id) where:
        # shard_id=0 for q_conv1d, 1 for k_conv1d, 2 for v_conv1d
        delattr(self.conv1d.weight, "weight_loader")
        set_weight_attrs(
            self.conv1d.weight,
            {
                "weight_loader": gdn_conv_weight_loader(
                    self.key_dim,
                    self.kv_k_dim,
                    self.kv_v_dim,
                    self.tp_size,
                    self.tp_rank,
                )
            },
        )

        # Fused projection: q, k, v, g (output gate z), b, gk
        # Single GEMM instead of two separate projections
        self.in_proj_all = MergedColumnParallelLinear(
            input_size=self.hidden_size,
            output_sizes=[self.key_dim, self.kv_k_dim, self.kv_v_dim, self.value_dim,
                          self.num_heads, self.num_heads],
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.in_proj_all",
        )
        delattr(self.in_proj_all.weight, "weight_loader")
        set_weight_attrs(
            self.in_proj_all.weight,
            {
                "weight_loader": gdn_linear_weight_loader(
                    [self.key_dim, self.kv_k_dim, self.kv_v_dim, self.value_dim,
                     self.num_heads, self.num_heads],
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

        # Gating parameters (sized by num_heads)
        # Note: Checkpoint has A_log but weight loading renames it to A
        self.dt_bias = nn.Parameter(
            torch.ones(self.num_heads // self.tp_size),
        )
        self.A = nn.Parameter(
            torch.empty(divide(self.num_heads, self.tp_size)),
        )

        set_weight_attrs(self.A, {"weight_loader": sharded_weight_loader(0)})
        set_weight_attrs(self.dt_bias, {"weight_loader": sharded_weight_loader(0)})

        # Output norm and projection (named o_norm/o_proj to match checkpoint)
        self.o_norm = RMSNormGated(
            self.head_v_dim,
            eps=self.layer_norm_epsilon,
            group_size=None,
            norm_before_gate=True,
            device=current_platform.current_device(),
            dtype=None,
        )

        self.o_proj = RowParallelLinear(
            self.value_dim,
            self.hidden_size,
            bias=False,
            input_is_parallel=True,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        # Register in compilation context
        from vllm.config import get_current_vllm_config
        compilation_config = get_current_vllm_config().compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

    def fix_query_key_value_ordering(self, projected_all):
        """Split fused projection into q, k, v, z, b, a tensors."""
        q, k, v, z, b, a = torch.split(
            projected_all,
            [
                self.key_dim // self.tp_size,
                self.kv_k_dim // self.tp_size,
                self.kv_v_dim // self.tp_size,
                self.value_dim // self.tp_size,
                self.num_heads // self.tp_size,
                self.num_heads // self.tp_size,
            ],
            dim=-1,
        )
        # Make contiguous to avoid stride mismatch with torch.compile
        return q.contiguous(), k.contiguous(), v.contiguous(), z.contiguous(), b, a

    def rearrange_mixed_qkv(self, mixed_qkv):
        """
        Rearrange mixed QKV for FLA ops.

        Input: mixed_qkv with dimensions [query (key_dim), key (kv_k_dim), value (kv_v_dim)]
        Output: Reshaped tensors for FLA ops (NO repeat_interleave - GQA handled in kernel)
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

    def _expand_kv_for_gqa(self, key, value):
        """Expand KV for kernels that don't support native GQA (e.g., chunk_gated_delta_rule)."""
        if key is None:
            return None, None
        if self.num_kv_groups > 1:
            key = torch.repeat_interleave(key, self.num_kv_groups, dim=2)
            value = torch.repeat_interleave(value, self.num_kv_groups, dim=2)
        return key, value

    def forward(
        self,
        hidden_states: torch.Tensor,
        output: torch.Tensor,
        positions: torch.Tensor | None = None,
    ):
        """
        Forward pass with three parts:
        1. Input projection
        2. Core attention (custom op)
        3. Output projection

        Args:
            hidden_states: Input tensor
            output: Output tensor (modified in-place)
            positions: Position indices (unused, for signature compatibility with Mamba2)
        """
        num_tokens = hidden_states.size(0)

        # ============================================================
        # Part 1: Input Projection (single fused GEMM)
        # ============================================================
        projected_all, _ = self.in_proj_all(hidden_states)
        query, key, value, z, b, a = self.fix_query_key_value_ordering(projected_all)

        # Create mixed_qkv for conv operation
        # Note: key and value are in grouped dimensions (kv_k_dim, kv_v_dim)
        # They will be repeated AFTER convolution
        mixed_qkv = torch.cat((query, key, value), dim=-1)

        # ============================================================
        # Part 2: Core Attention (Custom Op)
        # ============================================================
        core_attn_out = torch.zeros(
            (num_tokens, self.num_heads // self.tp_size, self.head_v_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

        torch.ops.vllm.custom_gdn_attention_core(
            mixed_qkv,
            b,
            a,
            core_attn_out,
            self.prefix,
        )

        # ============================================================
        # Part 3: Output Projection
        # ============================================================

        # Reshape z from [num_tokens, value_dim] to [num_tokens, num_heads, head_v_dim]
        z = rearrange(z, "l (h d) -> l h d", h=self.num_heads // self.tp_size, d=self.head_v_dim)
        z_shape_og = z.shape

        # Reshape both tensors to 2D for o_norm: [num_tokens * num_heads, head_v_dim]
        core_attn_out = core_attn_out.reshape(-1, core_attn_out.shape[-1])
        z = z.reshape(-1, z.shape[-1])

        # Apply gated norm
        core_attn_out = self.o_norm(core_attn_out, z)

        # Reshape back and flatten heads
        core_attn_out = core_attn_out.reshape(z_shape_og)
        core_attn_out = rearrange(core_attn_out, "l h d -> l (h d)")

        # Output projection
        output[:num_tokens], _ = self.o_proj(core_attn_out)

    def _forward_core(
        self,
        mixed_qkv: torch.Tensor,
        b: torch.Tensor,
        a: torch.Tensor,
        core_attn_out: torch.Tensor,
    ):
        """
        Core attention computation (called by custom op).
        Handles convolution, gating, and recurrent/chunk delta rule.
        """
        forward_context = get_forward_context()
        attn_metadata: AttentionMetadata = forward_context.attn_metadata

        if attn_metadata is None:
            # V1 profile run
            return

        assert isinstance(attn_metadata, dict)
        attn_metadata = attn_metadata[self.prefix]
        assert isinstance(attn_metadata, GDNAttentionMetadata)

        # Extract metadata
        has_initial_state = attn_metadata.has_initial_state
        spec_query_start_loc = attn_metadata.spec_query_start_loc
        non_spec_query_start_loc = attn_metadata.non_spec_query_start_loc
        spec_sequence_masks = attn_metadata.spec_sequence_masks
        spec_token_indx = attn_metadata.spec_token_indx
        non_spec_token_indx = attn_metadata.non_spec_token_indx
        spec_state_indices_tensor = attn_metadata.spec_state_indices_tensor
        non_spec_state_indices_tensor = attn_metadata.non_spec_state_indices_tensor

        # Get cache
        self_kv_cache = self.kv_cache[forward_context.virtual_engine]
        conv_state = self_kv_cache[0].transpose(-1, -2)
        ssm_state = self_kv_cache[1]
        num_actual_tokens = attn_metadata.num_actual_tokens
        num_accepted_tokens = attn_metadata.num_accepted_tokens

        mixed_qkv = mixed_qkv[:num_actual_tokens]
        b = b[:num_actual_tokens]
        a = a[:num_actual_tokens]

        # ============================================================
        # 1. Convolution sequence transformation
        # ============================================================
        # Use merged conv weights (already combined at load time)
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

        # 1.1: Process speculative decoding part
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

        # 1.2: Process prefill/decode part
        # In mixed batches, split non-spec tokens into decode and prefill
        # subsets so each uses its optimal kernel. Follows Mamba2's pattern:
        # layer does cheap GPU splits, builder pre-computed qsl and conv metadata.
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

            # 1.2a: Conv for decodes (recurrent update)
            mixed_qkv_non_spec_d = causal_conv1d_update(
                mixed_qkv_non_spec_d,
                conv_state,
                conv_weights,
                None,  # bias
                self.activation,
                conv_state_indices=state_indices_d,
                validate_data=True,
            )
            # 1.2b: Conv for prefills (chunked)
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
        query_spec, key_spec, value_spec = self.rearrange_mixed_qkv(mixed_qkv_spec)
        query_non_spec, key_non_spec, value_non_spec = self.rearrange_mixed_qkv(
            mixed_qkv_non_spec
        )

        # ============================================================
        # 2. Compute gating parameters
        # ============================================================
        g, beta = fused_gdn_gating(self.A, a, b, self.dt_bias)

        # Apply beta scaling if allow_neg_eigval is enabled
        if self.allow_neg_eigval:
            beta = beta * 2.0

        # Split g and beta for spec/non-spec
        if spec_sequence_masks is not None:
            if attn_metadata.num_prefills == 0 and attn_metadata.num_decodes == 0:
                g_spec = g
                beta_spec = beta
                g_non_spec = None
                beta_non_spec = None
            else:
                g_spec = g.index_select(1, spec_token_indx)
                beta_spec = beta.index_select(1, spec_token_indx)
                g_non_spec = g.index_select(1, non_spec_token_indx)
                beta_non_spec = beta.index_select(1, non_spec_token_indx)
        else:
            g_spec = None
            beta_spec = None
            g_non_spec = g
            beta_non_spec = beta

        # ============================================================
        # 3. Recurrent attention (FLA ops)
        # ============================================================

        # 3.1: Process speculative decoding part (uses GQA kernel)
        if spec_sequence_masks is not None:
            core_attn_out_spec, _ = fused_recurrent_gated_delta_rule_gqa(
                q=query_spec,
                k=key_spec,
                v=value_spec,
                g=g_spec,
                beta=beta_spec,
                initial_state=ssm_state,
                inplace_final_state=True,
                cu_seqlens=spec_query_start_loc[: attn_metadata.num_spec_decodes + 1],
                ssm_state_indices=spec_state_indices_tensor,
                num_accepted_tokens=num_accepted_tokens,
                use_qk_l2norm_in_kernel=True,
            )
        else:
            core_attn_out_spec = None

        # 3.2: Process prefill/decode part
        if is_mixed:
            # Split q/k/v/g/beta along token dim (dim=1, decodes first)
            q_d = query_non_spec[:, :num_non_spec_decode_tokens]
            q_p = query_non_spec[:, num_non_spec_decode_tokens:]
            k_d = key_non_spec[:, :num_non_spec_decode_tokens]
            k_p = key_non_spec[:, num_non_spec_decode_tokens:]
            v_d = value_non_spec[:, :num_non_spec_decode_tokens]
            v_p = value_non_spec[:, num_non_spec_decode_tokens:]
            g_d = g_non_spec[:, :num_non_spec_decode_tokens]
            g_p = g_non_spec[:, num_non_spec_decode_tokens:]
            beta_d = beta_non_spec[:, :num_non_spec_decode_tokens]
            beta_p = beta_non_spec[:, num_non_spec_decode_tokens:]

            # Decode: fused recurrent (O(1) per token)
            core_attn_out_d, _ = fused_recurrent_gated_delta_rule_gqa(
                q=q_d, k=k_d, v=v_d, g=g_d, beta=beta_d,
                initial_state=ssm_state,
                inplace_final_state=True,
                cu_seqlens=attn_metadata.non_spec_query_start_loc_d,
                ssm_state_indices=state_indices_d,
                use_qk_l2norm_in_kernel=True,
            )

            # Prefill: chunked (O(chunk_size) per token)
            initial_state = ssm_state[state_indices_p].contiguous()
            has_initial_state_p = has_initial_state[num_non_spec_decodes:]
            initial_state[~has_initial_state_p, ...] = 0
            core_attn_out_p, last_recurrent_state = chunk_gated_delta_rule_gqa(
                q=q_p, k=k_p, v=v_p, g=g_p, beta=beta_p,
                initial_state=initial_state,
                output_final_state=True,
                cu_seqlens=attn_metadata.non_spec_query_start_loc_p,
                use_qk_l2norm_in_kernel=True,
            )
            ssm_state[state_indices_p] = last_recurrent_state.to(ssm_state.dtype)

            # Merge decode + prefill outputs
            core_attn_out_non_spec = torch.cat(
                [core_attn_out_d, core_attn_out_p], dim=1
            )
        elif has_non_spec_prefill:
            initial_state = ssm_state[non_spec_state_indices_tensor].contiguous()
            initial_state[~has_initial_state, ...] = 0
            (
                core_attn_out_non_spec,
                last_recurrent_state,
            ) = chunk_gated_delta_rule_gqa(
                q=query_non_spec,
                k=key_non_spec,
                v=value_non_spec,
                g=g_non_spec,
                beta=beta_non_spec,
                initial_state=initial_state,
                output_final_state=True,
                cu_seqlens=non_spec_query_start_loc,
                use_qk_l2norm_in_kernel=True,
            )
            ssm_state[non_spec_state_indices_tensor] = last_recurrent_state.to(
                ssm_state.dtype
            )
        elif has_non_spec_decode:
            core_attn_out_non_spec, _ = fused_recurrent_gated_delta_rule_gqa(
                q=query_non_spec,
                k=key_non_spec,
                v=value_non_spec,
                g=g_non_spec,
                beta=beta_non_spec,
                initial_state=ssm_state,
                inplace_final_state=True,
                cu_seqlens=non_spec_query_start_loc[: num_non_spec_decodes + 1],
                ssm_state_indices=non_spec_state_indices_tensor,
                use_qk_l2norm_in_kernel=True,
            )
        else:
            core_attn_out_non_spec = None

        # ============================================================
        # 4. Merge outputs
        # ============================================================
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
