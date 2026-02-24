"""BMOJO-F Attention Layer for vLLM V1.

This module implements the unified BMOJO-F attention layer with:
- QKV projections at layer level (matches state dict structure)
- Native V1 Attention for s-stream (standard sliding window)
- Custom BMOJOFAttentionC for c-stream (offset sliding window)
- Log-sum-exp merging of stream outputs
"""

import torch
from torch import nn


from primed_vllm.bmojof.bmojof_attn_backend import (
    BMOJOFAttentionSBackend,
    BMOJOFAttentionBackend
)

from primed_vllm.primed_mamba2_layer import PrimedMamba2Layer
from vllm.v1.attention.backend import AttentionType
from vllm.attention.layer import Attention
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.platforms import current_platform
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from vllm.model_executor.layers.linear import UnquantizedLinearMethod
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.utils.torch_utils import kv_cache_dtype_str_to_dtype
from vllm.v1.kv_cache_interface import SlidingWindowSpec


def merge_attn_states_torch(
    output: torch.Tensor,  # [NUM_TOKENS, NUM_HEADS, HEAD_SIZE]
    prefix_output: torch.Tensor,  # [NUM_TOKENS, NUM_HEADS, HEAD_SIZE]
    prefix_lse: torch.Tensor,  # [NUM_HEADS, NUM_TOKENS]
    suffix_output: torch.Tensor,  # [NUM_TOKENS, NUM_HEADS, HEAD_SIZE]
    suffix_lse: torch.Tensor,  # [NUM_HEADS, NUM_TOKENS]
    output_lse: torch.Tensor | None = None,  # [NUM_HEADS, NUM_TOKENS]
):
    p_lse = prefix_lse
    s_lse = suffix_lse
    # inf -> -inf
    p_lse[p_lse == torch.inf] = -torch.inf
    s_lse[s_lse == torch.inf] = -torch.inf
    # max_lse [NUM_HEADS, NUM_TOKENS]
    max_lse = torch.maximum(p_lse, s_lse)
    p_lse = p_lse - max_lse
    s_lse = s_lse - max_lse
    p_lse_exp = torch.exp(p_lse)
    s_lse_exp = torch.exp(s_lse)
    out_se = p_lse_exp + s_lse_exp
    if output_lse is not None:
        output_lse = torch.log(out_se) + max_lse
    p_scale = p_lse_exp / out_se  # [NUM_HEADS, NUM_TOKENS]
    s_scale = s_lse_exp / out_se  # [NUM_HEADS, NUM_TOKENS]
    p_scale = torch.transpose(p_scale, 0, 1).unsqueeze(2)  # [NUM_TOKENS, NUM_HEADS, 1]
    s_scale = torch.transpose(s_scale, 0, 1).unsqueeze(2)  # [NUM_TOKENS, NUM_HEADS, 1]
    output = prefix_output * p_scale + suffix_output * s_scale
    return output


def should_load_quant_weights(quant_method: QuantizeMethodBase | None) -> bool:
    """Returns whether the quantization method should load quantized weights."""
    return quant_method is not None and not isinstance(
        quant_method, UnquantizedLinearMethod
    )


def set_default_quant_scales(layer: nn.Module, register_buffer: bool = False) -> None:
    """Sets default quantization scales for the layer."""
    if register_buffer:
        layer.register_buffer("_k_scale", torch.tensor(1.0, dtype=torch.float32))
        layer.register_buffer("_v_scale", torch.tensor(1.0, dtype=torch.float32))
        layer.register_buffer("_q_scale", torch.tensor(1.0, dtype=torch.float32))
    else:
        layer._k_scale.fill_(1.0)
        layer._v_scale.fill_(1.0)
        layer._q_scale.fill_(1.0)

    # We also keep q/k/v_scale on host (cpu) memory for attention
    # backends that require the scales to be on host instead of on device.
    # e.g. Flashinfer
    layer._q_scale_float = 1.0
    layer._k_scale_float = 1.0
    layer._v_scale_float = 1.0


class BMOJOFAttentionLayer(nn.Module):
    """Unified BMOJO-F attention layer with QKV projections at layer level.

    This layer implements dual-stream attention:
    - S-stream: Standard sliding window attention over recent tokens
    - C-stream: Offset sliding window attention over older tokens

    QKV projections are at the layer level (not inside stream instances)
    to match the state dict structure from trained models.

    Args:
        config: Model configuration with bmojo_config
        hidden_size: Hidden dimension size
        num_heads: Number of attention heads
        num_kv_heads: Number of key-value heads
        max_position: Maximum position for RoPE
        rope_theta: RoPE theta parameter
        cache_config: Cache configuration (optional)
        quant_config: Quantization configuration (optional)
        rope_scaling: RoPE scaling configuration (optional)
        prefix: Layer name prefix for weight loading
        attn_type: Attention type (default: DECODER)
        qk_norm: QK normalization layer class (optional)
        rms_norm_eps: RMS norm epsilon (default: 1e-6)
    """

    def __init__(
        self,
        config,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position: int = 4096 * 32,
        rope_parameters: dict | None = None,
        cache_config=None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        attn_type: str = AttentionType.DECODER,
        qk_norm: type | None = None,
        rms_norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()

        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads

        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0
        else:
            assert tp_size % self.total_num_kv_heads == 0

        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = getattr(config, "head_dim", hidden_size // self.total_num_heads)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5

        # Use rope_parameters from config if provided, otherwise construct from legacy parameters
        if rope_parameters is None:
            rope_parameters = getattr(config, "rope_parameters", {
                "rope_type": "default",
                "rope_theta": getattr(config, "rope_theta", 10000),
            })

        # Get BMOJO-F configuration
        self.tie_attn_weights = config.bmojo_config["tie_attn_weights"]
        self.window_size = config.bmojo_config["window_size"]

        # Q projection (shared by both streams)
        self.q_proj = ColumnParallelLinear(
            input_size=hidden_size,
            output_size=self.total_num_heads * self.head_dim,
            bias=getattr(config, 'attention_bias', True),
            quant_config=quant_config,
            prefix=f"{prefix}.q_proj",
        )

        # Merged KV projection for s-stream (or tied) - uses MergedColumnParallelLinear
        # This allows loading separate k_proj and v_proj weights from state dict
        self.kv_proj = MergedColumnParallelLinear(
            input_size=hidden_size,
            output_sizes=[self.total_num_kv_heads * self.head_dim] * 2,
            bias=getattr(config, 'attention_bias', True),
            quant_config=quant_config,
            prefix=f"{prefix}.kv_proj",
        )

        # Merged KV projection for c-stream (if untied)
        if not self.tie_attn_weights:
            self.kv_proj_ = MergedColumnParallelLinear(
                input_size=hidden_size,
                output_sizes=[self.total_num_kv_heads * self.head_dim] * 2,
                bias=getattr(config, 'attention_bias', True),
                quant_config=quant_config,
                prefix=f"{prefix}.kv_proj_",
            )

        # QK normalization (optional, for backward compatibility)
        self.q_norm = None
        self.k_norm = None
        self.q_norm_ = None
        self.k_norm_ = None

        if qk_norm is not None:
            self.q_norm = qk_norm(self.head_dim, eps=rms_norm_eps)
            self.k_norm = qk_norm(self.head_dim, eps=rms_norm_eps)

            if not self.tie_attn_weights:
                self.q_norm_ = qk_norm(self.head_dim, eps=rms_norm_eps)
                self.k_norm_ = qk_norm(self.head_dim, eps=rms_norm_eps)

        # RoPE (shared by both streams)
        # Ensure rope_parameters has 'rope_type' for vLLM v0.12.0+ compatibility
        if "rope_type" not in rope_parameters:
            # Default to 'default' if not specified (no RoPE scaling)
            rope_parameters = dict(rope_parameters)  # Make a copy to avoid modifying input
            rope_parameters["rope_type"] = "default"

        self.rotary_emb = get_rope(
            self.head_dim,
            max_position=max_position,
            rope_parameters=rope_parameters,
        )

        # S-stream: Custom attention with standard sliding window (returns LSE)
        # Note: We need LSE from s-stream for merge_attn_states
        # Use numeric suffix (.0, .1) for extract_layer_index compatibility with num_attn_module=2
        # This ensures s-stream and c-stream get separate KV caches
        self.s_stream = BMOJOFAttentionS(
            num_heads=self.num_heads,
            head_size=self.head_dim,
            num_kv_heads=self.num_kv_heads,
            window_size=self.window_size,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.0",  # s-stream gets sub-index 0
            attn_type=attn_type,
        )

        # C-stream: Custom attention with offset window
        self.c_stream = BMOJOFAttentionC(
            num_heads=self.num_heads,
            head_size=self.head_dim,
            num_kv_heads=self.num_kv_heads,
            window_size=self.window_size,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.1",  # c-stream gets sub-index 1
            attn_type=attn_type,
        )

        # Output projection (shared)
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )
        self.prefix = prefix

    def _apply_qkv_proj(self, hidden_states: torch.Tensor, kv_states: torch.Tensor, use_untied: bool = False):
        """Apply QKV projections with optional normalization.

        Args:
            hidden_states: Input for Q projection
            kv_states: Input for KV projection (hidden_states for s-stream, fading_tokens for c-stream)
            use_untied: Whether to use untied weights (for c-stream when tie_attn_weights=False)

        Returns:
            Tuple of (q, k, v) tensors
        """
        # Q projection (always from hidden_states)
        q, _ = self.q_proj(hidden_states)

        # KV projection
        if use_untied and not self.tie_attn_weights:
            kv, _ = self.kv_proj_(kv_states)
            q_norm = self.q_norm_
            k_norm = self.k_norm_
        else:
            kv, _ = self.kv_proj(kv_states)
            q_norm = self.q_norm
            k_norm = self.k_norm

        k, v = kv.split([self.kv_size, self.kv_size], dim=-1)

        # Apply QK normalization if available
        if q_norm is not None:
            q_by_head = q.view(*q.shape[:-1], q.shape[-1] // self.head_dim, self.head_dim)
            q = q_norm(q_by_head).view(q.shape)

            k_by_head = k.view(*k.shape[:-1], k.shape[-1] // self.head_dim, self.head_dim)
            k = k_norm(k_by_head).view(k.shape)

        return q, k, v

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        fading_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass for dual-stream BMOJO-F attention.

        Args:
            positions: Position indices for RoPE
            hidden_states: Input hidden states
            fading_tokens: Fading tokens from Mamba2 (used for c-stream KV)

        Returns:
            Attention output tensor
        """ 
        # 1. Apply QKV projections for s-stream (hidden_states -> hidden_states)
        q_s, k_s, v_s = self._apply_qkv_proj(hidden_states, hidden_states, use_untied=False)

        # 2. Apply QKV projections for c-stream (hidden_states -> fading_tokens)
        # Note: q_c uses hidden_states, k_c/v_c use fading_tokens
        q_c, k_c, v_c = self._apply_qkv_proj(hidden_states, fading_tokens, use_untied=not self.tie_attn_weights)

        # S-stream: apply RoPE to q_s and k_s
        q_s, k_s = self.rotary_emb(positions, q_s, k_s)
        # C-stream: apply RoPE to q_c and k_c (separate from s-stream)
        q_c, k_c = self.rotary_emb(positions, q_c, k_c)

        # 4. S-stream attention (standard sliding window, returns LSE)
        s_output, s_lse = self.s_stream(q_s, k_s, v_s)

        # 5. C-stream attention (offset sliding window, returns LSE)
        c_output, c_lse = self.c_stream(q_c, k_c, v_c)

        # 6. Merge outputs using log-sum-exp
        # merge_attn_states combines two attention outputs weighted by their LSE values
        # This is mathematically equivalent to computing attention over the combined KV sets
        # 
        # The function signature is:
        #   merge_attn_states(output, prefix_output, prefix_lse, suffix_output, suffix_lse)
        # where:
        #   - output: destination tensor (will be written to)
        #   - prefix_output/lse: first attention output and its LSE (s-stream)
        #   - suffix_output/lse: second attention output and its LSE (c-stream)
        #
        # Note: s_output and c_output are in 2D format [num_tokens, num_heads * head_size]
        # but merge_attn_states expects 3D format [num_tokens, num_heads, head_size]
        #
        # CUDA GRAPH COMPATIBILITY:
        # Both s_output and c_output are bucket-sized (padded) tensors.
        # We merge on the full bucket size, then slice to num_actual_tokens at the end.
        num_tokens = s_output.shape[0]  # Bucket size (same for both streams)
        s_output_3d = s_output.view(num_tokens, self.num_heads, self.head_dim)
        c_output_3d = c_output.view(num_tokens, self.num_heads, self.head_dim)

        merged_3d = torch.empty_like(s_output_3d)
        merged_3d = merge_attn_states_torch(merged_3d, s_output_3d, s_lse, c_output_3d, c_lse)

        merged = merged_3d.to(s_output_3d.dtype).view(num_tokens, self.num_heads * self.head_dim)
        # 7. Output projection
        output, _ = self.o_proj(merged)

        return output


class BMOJOFAttentionS(nn.Module, AttentionLayerBase):
    """S-stream attention with standard sliding window.

    This class implements the s-stream attention which attends to the most
    recent window_size tokens. It uses the native FlashAttentionBackend
    but returns LSE for merging with c-stream.

    Note: This class does NOT contain QKV projections. It receives
    pre-projected query, key, value tensors from BMOJOFAttentionLayer.

    Args:
        num_heads: Number of attention heads
        head_size: Size of each attention head
        num_kv_heads: Number of key-value heads
        window_size: Size of the attention window
        cache_config: Cache configuration (optional)
        quant_config: Quantization configuration (optional)
        prefix: Layer name prefix
        attn_type: Attention type (default: DECODER)
    """

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        num_kv_heads: int,
        window_size: int,
        cache_config=None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        attn_type: str = AttentionType.DECODER,
    ) -> None:
        super().__init__()

        self.num_heads = num_heads
        self.head_size = head_size
        self.num_kv_heads = num_kv_heads
        self.window_size = window_size
        self.quant_config = quant_config
        self.layer_name = prefix
        self.attn_type = attn_type
        self.scaling = head_size**-0.5

        # Use custom BMOJOFAttentionSBackend for s-stream (adds return_lse support)
        self.attn_backend = BMOJOFAttentionSBackend

        # Get cache configuration
        if cache_config is not None:
            kv_cache_dtype = cache_config.cache_dtype
            block_size = cache_config.block_size
        else:
            kv_cache_dtype = "auto"
            block_size = 16

        self.kv_cache_dtype = kv_cache_dtype
        self.block_size = block_size

        # Convert string dtype to torch dtype for SlidingWindowSpec
        vllm_config = get_current_vllm_config()
        self.kv_cache_torch_dtype = kv_cache_dtype_str_to_dtype(
            kv_cache_dtype, vllm_config.model_config
        )

        # Create implementation from backend
        impl_cls = self.attn_backend.get_impl_cls()
        self.impl = impl_cls(
            num_heads,
            head_size,
            self.scaling,
            num_kv_heads,
            None,  # alibi_slopes
            window_size,  # sliding_window
            kv_cache_dtype,
            None,  # logits_soft_cap
            attn_type,
            None,  # kv_sharing_target_layer_name
        )

        # Initialize KV cache scaling factors (for FP8 quantization)
        # Note [Register q/k/v scales in state dict]
        # When calling model.to(device), only parameters/buffers in state dict are
        # moved. Registering scales as buffers ensures they are properly moved to device.
        set_default_quant_scales(self, register_buffer=True)

        # Register with forward_context
        vllm_config.compilation_config.static_forward_context[prefix] = self

        # KV cache placeholder (will be bound by V1's cache manager)
        self.kv_cache = [
            torch.tensor([])
            for _ in range(vllm_config.parallel_config.pipeline_parallel_size)
        ]

        # For torch.compile + cudagraph compatibility:
        # When use_direct_call=False, we use custom ops that are registered as
        # splitting_ops (graph break points) to prevent torch.compile from
        # tracing through the attention computation.
        self.use_direct_call = not current_platform.opaque_attention_op()

    def process_weights_after_loading(self, act_dtype: torch.dtype):
        """Process weights after loading, including setting default scales if needed."""
        # If we should not load quant weights, we initialize the scales to 1.0
        # as the default value. See [Note: Register q/k/v scales in state dict]
        # for more details.
        quant_method = (
            self.quant_config.get_quant_method(self, prefix=self.layer_name)
            if self.quant_config
            else None
        )
        if not should_load_quant_weights(quant_method):
            set_default_quant_scales(self, register_buffer=False)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for s-stream attention with LSE return.

        Args:
            query: Pre-projected query tensor
            key: Pre-projected key tensor
            value: Pre-projected value tensor

        Returns:
            Tuple of (output tensor, LSE tensor)
        """
        # Create output tensors
        output = torch.empty_like(query)
        lse = torch.empty(
            (self.num_heads, query.shape[0]),
            dtype=torch.float32,
            device=query.device,
        )

        if self.use_direct_call:
            # Direct call path (non-CUDA platforms or when opaque_attention_op is False)
            forward_context = get_forward_context()
            attn_metadata = forward_context.attn_metadata

            # Handle dict-based metadata (for hybrid models)
            if isinstance(attn_metadata, dict):
                attn_metadata = attn_metadata[self.layer_name]

            kv_cache = self.kv_cache[forward_context.virtual_engine]

            # Call implementation forward with return_lse=True
            output, lse = self.impl.forward(
                layer=self,
                query=query,
                key=key,
                value=value,
                kv_cache=kv_cache,
                attn_metadata=attn_metadata,
                output=output,
                return_lse=True,
            )
        else:
            # Custom op path for torch.compile + cudagraph compatibility
            # The custom op is registered as a splitting_op, so torch.compile
            # treats it as a graph break point and doesn't trace through it
            torch.ops.vllm.bmojof_attention_s_forward(
                query, key, value, output, lse, self.layer_name
            )

        return output, lse

    def get_kv_cache_spec(self, vllm_config: VllmConfig):
        """Return KV cache specification for s-stream.

        S-stream uses standard sliding window with window_size capacity.

        Args:
            vllm_config: vLLM configuration

        Returns:
            SlidingWindowSpec with 2*window_size capacity (same as c-stream)
        """
        spec = SlidingWindowSpec(
            block_size=vllm_config.cache_config.block_size,
            num_kv_heads=self.num_kv_heads,
            head_size=self.head_size,
            dtype=self.kv_cache_torch_dtype,
            sliding_window=2 * self.window_size,  # Must match c-stream to share KV cache group
        )
        return spec

    def get_attn_backend(self):
        """Return the attention backend."""
        return self.attn_backend


class BMOJOFAttentionC(nn.Module, AttentionLayerBase):
    """C-stream attention with hollow sliding window using FlashAttention.

    This class implements the c-stream attention which attends to tokens
    from [seq_len - 2×window : seq_len - window]. It uses FlashAttention
    with a custom hollow window indexing structure with a unified handling of all
    scenarios (prefill, decode, short/long sequences).

    Key Design:
    - Uses BMOJOFAttentionBackend (registered as "BMOJOF_ATTN")
    - Indexes qkv and constructs seq-len objects for compatibility with FlashAttention
    - Handles prefill and decode
    - Returns LSE values for merging with s-stream

    Note: This class does NOT contain QKV projections. It receives
    pre-projected query, key, value tensors from BMOJOFAttentionLayer.

    Args:
        num_heads: Number of attention heads
        head_size: Size of each attention head
        num_kv_heads: Number of key-value heads
        window_size: Size of the attention window (passed to metadata, not impl)
        cache_config: Cache configuration (optional)
        quant_config: Quantization configuration (optional)
        prefix: Layer name prefix
        attn_type: Attention type (default: DECODER)
    """

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        num_kv_heads: int,
        window_size: int,
        cache_config=None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        attn_type: str = AttentionType.DECODER,
    ) -> None:
        super().__init__()

        self.num_heads = num_heads
        self.head_size = head_size
        self.num_kv_heads = num_kv_heads
        self.window_size = window_size
        self.quant_config = quant_config
        self.layer_name = prefix
        self.attn_type = attn_type
        self.scaling = head_size**-0.5

        # Use custom BMOJOF backend for c-stream (FlashAttention-based)
        self.attn_backend = BMOJOFAttentionBackend

        # Get cache configuration
        if cache_config is not None:
            kv_cache_dtype = cache_config.cache_dtype
            block_size = cache_config.block_size
        else:
            kv_cache_dtype = "auto"
            block_size = 16

        self.kv_cache_dtype = kv_cache_dtype
        self.block_size = block_size

        # Convert string dtype to torch dtype for SlidingWindowSpec
        vllm_config = get_current_vllm_config()
        self.kv_cache_torch_dtype = kv_cache_dtype_str_to_dtype(
            kv_cache_dtype, vllm_config.model_config
        )

        # Create implementation from backend
        # Note: c_stream_window_size is passed to the implementation for the
        # "lie about sequence length" trick used in FlashAttention-based c-stream.
        impl_cls = self.attn_backend.get_impl_cls()
        self.impl = impl_cls(
            num_heads,
            head_size,
            self.scaling,
            num_kv_heads,
            None,  # alibi_slopes
            None,  # sliding_window (handled via c_stream_window_size)
            kv_cache_dtype,
            None,  # logits_soft_cap
            attn_type,
            None,  # kv_sharing_target_layer_name
            c_stream_window_size=window_size,  # Pass window_size for hollow window logic
            prefix=prefix
        )

        # Initialize KV cache scaling factors (for FP8 quantization)
        # Note [Register q/k/v scales in state dict]
        # When calling model.to(device), only parameters/buffers in state dict are
        # moved. Registering scales as buffers ensures they are properly moved to device.
        set_default_quant_scales(self, register_buffer=True)

        # Register with forward_context
        vllm_config.compilation_config.static_forward_context[prefix] = self

        # KV cache placeholder (will be bound by V1's cache manager)
        self.kv_cache = [
            torch.tensor([])
            for _ in range(vllm_config.parallel_config.pipeline_parallel_size)
        ]

        # For torch.compile + cudagraph compatibility:
        # When use_direct_call=False, we use custom ops that are registered as
        # splitting_ops (graph break points) to prevent torch.compile from
        # tracing through the attention computation.
        self.use_direct_call = not current_platform.opaque_attention_op()

    def process_weights_after_loading(self, act_dtype: torch.dtype):
        """Process weights after loading, including setting default scales if needed."""
        # If we should not load quant weights, we initialize the scales to 1.0
        # as the default value. See [Note: Register q/k/v scales in state dict]
        # for more details.
        quant_method = (
            self.quant_config.get_quant_method(self, prefix=self.layer_name)
            if self.quant_config
            else None
        )
        if not should_load_quant_weights(quant_method):
            set_default_quant_scales(self, register_buffer=False)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for c-stream attention with LSE return.

        Args:
            query: Pre-projected query tensor
            key: Pre-projected key tensor
            value: Pre-projected value tensor

        Returns:
            Tuple of (output tensor, LSE tensor) for merging with s-stream
        """
        # Create output tensors
        output = torch.zeros_like(query)
        lse = torch.empty(
            (self.num_heads, query.shape[0]),
            dtype=torch.float32,
            device=query.device,
        )

        if self.use_direct_call:
            # Direct call path (non-CUDA platforms or when opaque_attention_op is False)
            forward_context = get_forward_context()
            attn_metadata = forward_context.attn_metadata

            # Handle dict-based metadata (for hybrid models)
            if isinstance(attn_metadata, dict):
                attn_metadata = attn_metadata[self.layer_name]

            kv_cache = self.kv_cache[forward_context.virtual_engine]

            # Call implementation with return_lse=True for merging
            output, lse = self.impl.forward(
                layer=self,
                query=query,
                key=key,
                value=value,
                kv_cache=kv_cache,
                attn_metadata=attn_metadata,
                output=output,
                return_lse=True,
            )
        else:
            # Custom op path for torch.compile + cudagraph compatibility
            # The custom op is registered as a splitting_op, so torch.compile
            # treats it as a graph break point and doesn't trace through it
            torch.ops.vllm.bmojof_attention_c_forward(
                query, key, value, output, lse, self.layer_name
            )

        return output, lse

    def get_kv_cache_spec(self, vllm_config: VllmConfig):
        """Return KV cache specification for c-stream.

        C-stream needs 2×window_size capacity to store tokens for the
        offset window [seq_len - 2×window : seq_len - window].

        Args:
            vllm_config: vLLM configuration

        Returns:
            SlidingWindowSpec with 2×window_size capacity
        """
        spec = SlidingWindowSpec(
            block_size=vllm_config.cache_config.block_size,
            num_kv_heads=self.num_kv_heads,
            head_size=self.head_size,
            dtype=self.kv_cache_torch_dtype,
            sliding_window=2 * self.window_size,  # 2×window_size capacity
        )
        return spec

    def get_attn_backend(self):
        """Return the attention backend for this layer.

        Returns:
            BMOJOFAttentionBackend class
        """
        return self.attn_backend


class BMOJOFLayer(nn.Module):
    """BMOJO-F layer combining Mamba2 and dual-stream attention.

    This layer contains:
    - Mamba2 layer that produces fading tokens
    - BMOJOFAttentionLayer with dual-stream attention
    - Residual connection handling

    Args:
        config: Model configuration
        layer_idx: Layer index
        prefix: Layer name prefix
        model_config: Model configuration (optional)
        cache_config: Cache configuration (optional)
        quant_config: Quantization configuration (optional)
    """

    def __init__(
        self,
        config,
        layer_idx: int,
        prefix: str = "",
        model_config=None,
        cache_config=None,
        quant_config: QuantizationConfig | None = None,
    ) -> None:
        super().__init__()


        # Get BMOJO-F configuration
        bmojo_config = config.bmojo_config
        self.prefix = prefix
        # Store configuration
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        rms_norm_eps = getattr(config, "rms_norm_eps", 1e-6)

        # Get SSM mixer type from config
        self.ssm_mixer_type = bmojo_config.get("ssm_mixer", "mamba2").lower()

        # Determine qk_norm for attention - Qwen3 uses QK normalization
        # Check if model type indicates Qwen3 or if explicitly configured
        attn_qk_norm = None
        model_type = getattr(config, "model_type", "")
        if "qwen3" in model_type.lower() or bmojo_config.get("use_qk_norm", False):
            attn_qk_norm = RMSNorm

        # Instantiate appropriate SSM mixer based on config
        if self.ssm_mixer_type == "mamba2":
            self.mamba = PrimedMamba2Layer(
                config=config,
                model_config=model_config,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=f"{prefix}.mamba",
            )
        elif self.ssm_mixer_type in ["gated_deltanet", "gdn"]:
            from ..primed_gdn_layer import PrimedGDNLayer
            self.gdn = PrimedGDNLayer(
                config=config,
                model_config=model_config,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=f"{prefix}.gdn",
            )
        elif self.ssm_mixer_type in ["parallel_latent_sketch", "gka"]:
            from ..primed_gka_layer import PrimedGKALayer
            self.gka = PrimedGKALayer(
                config=config,
                model_config=model_config,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=f"{prefix}.gka",
            )
        else:
            raise ValueError(
                f"Unknown SSM mixer type: {self.ssm_mixer_type}. "
                f"Supported types: 'mamba2', 'gated_deltanet'/'gdn', 'parallel_latent_sketch'/'gka'"
            )

        # BMOJO-F attention layer
        self.bmojo_attn = BMOJOFAttentionLayer(
            config=config,
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position=getattr(config, "max_position_embeddings", 4096 * 32),
            rope_parameters=getattr(config, "rope_parameters", None),
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.bmojo_attn",
            attn_type=AttentionType.DECODER,
            qk_norm=attn_qk_norm,
            rms_norm_eps=rms_norm_eps,
        )

    @property
    def ssm_mixer(self):
        """Access the SSM mixer regardless of type.

        This property provides a unified interface to access the SSM mixer,
        matching the HuggingFace implementation pattern.
        """
        if self.ssm_mixer_type == "mamba2":
            return self.mamba
        elif self.ssm_mixer_type in ["gated_deltanet", "gdn"]:
            return self.gdn
        elif self.ssm_mixer_type in ["parallel_latent_sketch", "gka"]:
            return self.gka

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:

        """Forward pass for BMOJO-F layer.

        Args:
            positions: Position indices
            hidden_states: Input hidden states

        Returns:
            Output tensor
        """
        # SSM mixer produces fading tokens
        # All SSM mixers (Mamba2, GDN, GKA) use in-place output modification
        if self.ssm_mixer_type != "mamba2":
            fading_tokens = torch.empty_like(hidden_states)
            self.ssm_mixer(hidden_states, fading_tokens, positions)
        else:
            fading_tokens = self.ssm_mixer(hidden_states, positions)

        hidden_states = self.bmojo_attn(positions, hidden_states, fading_tokens)

        return hidden_states
