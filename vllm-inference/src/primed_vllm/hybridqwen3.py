from collections.abc import Iterable
import re

import torch
from torch import nn

from vllm.v1.attention.backend import AttentionType
from vllm.attention.layer import Attention
from vllm.compilation.decorators import support_torch_compile
from vllm.v1.attention.backend import AttentionMetadata
from vllm.config import CacheConfig, ModelConfig, VllmConfig
from vllm.distributed import divide, get_tensor_model_parallel_world_size
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.distributed.parallel_state import get_pp_group
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    ColumnParallelLinear,
    RowParallelLinear,
    QKVParallelLinear
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.rotary_embedding import get_rope
from .primed_mamba2_layer import PrimedMamba2Layer, extra_groups_for_head_shards
from .primed_utils import PrimedStateShapeCalculator
from .primed_gdn_layer import PrimedGDNLayer
from .primed_gka_layer import PrimedGKALayer
from .bmojof.bmojof_layer import BMOJOFLayer

from vllm.model_executor.layers.mamba.mamba_utils import MambaStateDtypeCalculator
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.interfaces import (
    HasInnerState, IsHybrid, SupportsPP, SupportsLoRA, SupportsMambaPrefixCaching
)
from vllm.model_executor.models.utils import extract_layer_index
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.config import set_default_rope_theta
from vllm.model_executor.models.utils import (
    is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
    AutoWeightsLoader,
)

from .utils import parse_override_pattern, pattern_has_symbols, Symbols
from .hybridqwen2 import Qwen2MLP as Qwen3MLP

from primed_vllm.hf_configs import HybridQwen3Config


# Copy verbatim from official Qwen3 implementation
class Qwen3Attention(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        rope_parameters: dict,
        max_position: int = 4096 * 32,
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-06,
        qkv_bias: bool = False,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        attn_type: str = AttentionType.DECODER,
        dual_chunk_attention_config: dict | None = None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = head_dim or hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.dual_chunk_attention_config = dual_chunk_attention_config

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=qkv_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        self.rotary_emb = get_rope(
            self.head_dim,
            max_position=max_position,
            rope_parameters=rope_parameters,
            dual_chunk_attention_config=dual_chunk_attention_config,
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
            attn_type=attn_type,
            **{
                "layer_idx": extract_layer_index(prefix),
                "dual_chunk_attention_config": dual_chunk_attention_config,
            } if dual_chunk_attention_config else {},
        )
        self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        # Add qk-norm
        q_by_head = q.view(*q.shape[:-1], q.shape[-1] // self.head_dim,
                           self.head_dim)
        q_by_head = self.q_norm(q_by_head)
        q = q_by_head.view(q.shape)
        k_by_head = k.view(*k.shape[:-1], k.shape[-1] // self.head_dim,
                           self.head_dim)
        k_by_head = self.k_norm(k_by_head)
        k = k_by_head.view(k.shape)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output


class Qwen3SWAttention(nn.Module):
    """Sliding Window Attention layer for HybridQwen3.

    This is a wrapper around the standard Qwen3Attention that configures
    it with sliding window attention via the sliding_window parameter.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        rope_parameters: dict,
        max_position: int = 4096 * 32,
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-06,
        qkv_bias: bool = False,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        attn_type: str = AttentionType.DECODER,
        swa_config: dict | None = None,
    ) -> None:
        super().__init__()

        # Extract sliding window size from swa_config
        if swa_config is None or "window_size" not in swa_config:
            raise ValueError("swa_config must specify window_size for SWA layers")

        sliding_window = swa_config["window_size"]

        # Initialize standard attention with sliding window
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
        self.head_dim = head_dim or hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=qkv_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        self.rotary_emb = get_rope(
            self.head_dim,
            max_position=max_position,
            rope_parameters=rope_parameters,
        )

        # Configure attention with sliding window
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
            attn_type=attn_type,
            per_layer_sliding_window=sliding_window,  # Key difference from regular attention
        )
        self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        # Add qk-norm
        q_by_head = q.view(*q.shape[:-1], q.shape[-1] // self.head_dim,
                           self.head_dim)
        q_by_head = self.q_norm(q_by_head)
        q = q_by_head.view(q.shape)
        k_by_head = k.view(*k.shape[:-1], k.shape[-1] // self.head_dim,
                           self.head_dim)
        k_by_head = self.k_norm(k_by_head)
        k = k_by_head.view(k.shape)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output



class HybridQwen3BMOJOFDecoderLayer(nn.Module):
    """BMOJO-F decoder layer for HybridQwen3.

    This layer uses the new V1 architecture with:
    - BMOJOFLayer containing SSM + dual-stream attention
    - No V0 cache parameters
    - Simplified forward signature matching V1 patterns
    """

    def __init__(
        self,
        config: HybridQwen3Config,
        layer_idx: int,
        model_config: ModelConfig | None = None,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size

        # BMOJO-F layer (Mamba2 + dual-stream attention)
        self.bmojo_f = BMOJOFLayer(
            config=config,
            layer_idx=layer_idx,
            prefix=f"{prefix}.bmojo_f",
            model_config=model_config,
            cache_config=cache_config,
            quant_config=quant_config,
        )

        # MLP layer
        self.mlp = Qwen3MLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )

        # Layer normalization
        self.input_layernorm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for BMOJO-F decoder layer.

        Args:
            positions: Position indices for RoPE
            hidden_states: Input hidden states
            residual: Residual connection tensor (optional)
            **kwargs: Additional arguments (ignored for V1 compatibility)

        Returns:
            Tuple of (hidden_states, residual)
        """
        # Input layer normalization
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(
                hidden_states, residual)

        # BMOJO-F layer (Mamba2 + dual-stream attention)
        # No cache params needed - layers access cache via forward_context
        hidden_states = self.bmojo_f(
            positions=positions,
            hidden_states=hidden_states,
        )

        # Post-attention layer normalization
        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual)

        # MLP
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class HybridQwen3AttnDecoderLayer(nn.Module):

    def __init__(
        self,
        config: HybridQwen3Config,
        layer_idx: int,
        model_config: ModelConfig | None = None,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size

        set_default_rope_theta(config, default_theta=1000000)
        dual_chunk_attention_config = getattr(config, "dual_chunk_attention_config", None)

        if getattr(config, "is_causal", True):
            attn_type = AttentionType.DECODER
        else:
            attn_type = AttentionType.ENCODER_ONLY

        self.self_attn = Qwen3Attention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position=config.max_position_embeddings,
            head_dim=getattr(config, 'head_dim', None),
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, 'attention_bias', False),
            rope_parameters=config.rope_parameters,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
            attn_type=attn_type,
            dual_chunk_attention_config=dual_chunk_attention_config,
        )
        self.mlp = Qwen3MLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )
        self.input_layernorm = RMSNorm(config.hidden_size,
                                       eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size,
                                                eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(
                hidden_states, residual)

        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
        )

        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class HybridQwen3SWADecoderLayer(nn.Module):
    """Sliding Window Attention decoder layer for HybridQwen3."""

    def __init__(
        self,
        config: HybridQwen3Config,
        layer_idx: int,
        model_config: ModelConfig | None = None,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size

        set_default_rope_theta(config, default_theta=1000000)
        swa_config = getattr(config, "swa_config", None)

        if getattr(config, "is_causal", True):
            attn_type = AttentionType.DECODER
        else:
            attn_type = AttentionType.ENCODER_ONLY

        self.swa = Qwen3SWAttention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position=config.max_position_embeddings,
            head_dim=getattr(config, 'head_dim', None),
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, 'attention_bias', False),
            rope_parameters=config.rope_parameters,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.swa",
            attn_type=attn_type,
            swa_config=swa_config,
        )
        self.mlp = Qwen3MLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )
        self.input_layernorm = RMSNorm(config.hidden_size,
                                       eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size,
                                                eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(
                hidden_states, residual)

        hidden_states = self.swa(
            positions=positions,
            hidden_states=hidden_states,
        )

        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class HybridQwen3Mamba2DecoderLayer(nn.Module):
    def __init__(
        self,
        config: HybridQwen3Config,
        layer_idx: int,
        model_config: ModelConfig | None = None,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size

        self.mamba = PrimedMamba2Layer(
            config=config,
            model_config=model_config,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.mixer",
        )

        self.mlp = Qwen3MLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )

        self.input_layernorm = RMSNorm(config.hidden_size,
                                       eps=config.mamba2_config.get("rms_norm_eps", False),
                                       )
        self.post_attention_layernorm = RMSNorm(config.hidden_size,
                                                eps=config.mamba2_config.get("rms_norm_eps", False))

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        positions: torch.Tensor,
        **kwargs,
    ):
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        output = self.mamba(
            hidden_states=hidden_states,
            positions=positions
        )

        hidden_states, residual = self.post_attention_layernorm(
            output, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class HybridQwen3GDNDecoderLayer(nn.Module):
    """GatedDeltaNet decoder layer for HybridQwen3.

    Uses PrimedGDNLayer which provides:
    - vLLM V1 optimizations (custom ops, cache management)
    - Checkpoint compatibility with Hybrid Model Factory
    - Beta scaling support (allow_neg_eigval)
    """

    def __init__(
        self,
        config: HybridQwen3Config,
        layer_idx: int,
        model_config: ModelConfig | None = None,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size

        self.gdn = PrimedGDNLayer(
            config=config,
            model_config=model_config,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.gdn",
        )

        self.mlp = Qwen3MLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )

        # Get norm eps from gdn_config or use default
        gdn_config = getattr(config, "gdn_config", {})
        norm_eps = gdn_config.get("norm_eps", config.rms_norm_eps)

        self.input_layernorm = RMSNorm(config.hidden_size, eps=norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        positions: torch.Tensor,
        **kwargs,
    ):
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        output = torch.empty_like(hidden_states)
        self.gdn(
            hidden_states=hidden_states,
            output=output,
        )

        hidden_states, residual = self.post_attention_layernorm(output, residual)

        hidden_states = self.mlp(hidden_states)

        return hidden_states, residual


class HybridQwen3GKADecoderLayer(nn.Module):
    """GKA decoder layer for HybridQwen3."""

    def __init__(
        self,
        config: HybridQwen3Config,
        layer_idx: int,
        model_config: ModelConfig | None = None,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size

        self.gka = PrimedGKALayer(
            config=config,
            model_config=model_config,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.gka",
        )

        self.mlp = Qwen3MLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )

        gka_config = getattr(config, "gka_config", {})
        norm_eps = gka_config.get("norm_eps", config.rms_norm_eps)

        self.input_layernorm = RMSNorm(config.hidden_size, eps=norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        positions: torch.Tensor,
        **kwargs,
    ):
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        output = torch.empty_like(hidden_states)
        self.gka(hidden_states=hidden_states, output=output)

        hidden_states, residual = self.post_attention_layernorm(output, residual)
        hidden_states = self.mlp(hidden_states)

        return hidden_states, residual


ALL_DECODER_LAYER_TYPES = {
    "attention": HybridQwen3AttnDecoderLayer,
    "mamba": HybridQwen3Mamba2DecoderLayer,
    "bmojo-f": HybridQwen3BMOJOFDecoderLayer,
    "gdn": HybridQwen3GDNDecoderLayer,
    "gka": HybridQwen3GKADecoderLayer,
    "swa": HybridQwen3SWADecoderLayer,
}

@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        "positions": 0,
        "intermediate_tensors": 0,
        "inputs_embeds": 0,
    }
)
class HybridQwen3Model(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_config.get_text_config()
        config: HybridQwen3Config
        model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        self.config = config
        self.vocab_size = config.vocab_size

        self.embed_tokens = VocabParallelEmbedding(
            self.vocab_size,
            config.hidden_size,
        )

        self.layer_allocation = parse_override_pattern(config.hybrid_override_pattern)
        self.config.layers_block_type = []

        # Validate: cannot mix GDN and Mamba2 layers (incompatible state shapes)
        has_gdn = any(layer_type == Symbols.GDN for layer_type in self.layer_allocation.values())
        has_mamba = any(layer_type == Symbols.MAMBA for layer_type in self.layer_allocation.values())
        if has_gdn and has_mamba:
            raise ValueError(
                f"Cannot mix GDN and Mamba2 layers in the same model. "
                f"They have incompatible state shapes (GDN uses 2D recurrent state, "
                f"Mamba2 uses 3D temporal state). "
                f"Pattern: {config.hybrid_override_pattern}\n"
                f"Please use either GDN or Mamba2, not both."
            )

        def get_layer(prefix: str):
            layer_idx = int(prefix.rsplit(".", 1)[1])
            layer_symbol = self.layer_allocation[layer_idx]
            match layer_symbol:
                case Symbols.MAMBA:
                    self.config.layers_block_type.append("mamba")
                    return HybridQwen3Mamba2DecoderLayer(
                        config,
                        layer_idx,
                        model_config,
                        cache_config,
                        quant_config=quant_config,
                        prefix=prefix,
                    )
                case Symbols.ATTENTION:
                    self.config.layers_block_type.append("attention")
                    return HybridQwen3AttnDecoderLayer(
                        config,
                        layer_idx,
                        model_config,
                        cache_config,
                        quant_config=quant_config,
                        prefix=prefix,
                    )
                case Symbols.BMOJO_F:
                    # Mamba has it's own implementation of MambaCache, which is of constant size, we will do the same for bmojo-f.
                    self.config.layers_block_type.append("bmojo-f")
                    return HybridQwen3BMOJOFDecoderLayer(
                        config,
                        layer_idx,
                        model_config,
                        cache_config,
                        quant_config=quant_config,
                        prefix=prefix,
                    )
                case Symbols.GDN:
                    self.config.layers_block_type.append("gdn")
                    return HybridQwen3GDNDecoderLayer(
                        config,
                        layer_idx,
                        model_config,
                        cache_config,
                        quant_config=quant_config,
                        prefix=prefix,
                    )
                case Symbols.GKA:
                    self.config.layers_block_type.append("gka")
                    return HybridQwen3GKADecoderLayer(
                        config,
                        layer_idx,
                        model_config,
                        cache_config,
                        quant_config=quant_config,
                        prefix=prefix,
                    )
                case Symbols.SWA:
                    self.config.layers_block_type.append("swa")
                    return HybridQwen3SWADecoderLayer(
                        config,
                        layer_idx,
                        model_config,
                        cache_config,
                        quant_config=quant_config,
                        prefix=prefix,
                    )

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            get_layer,
            prefix=f"{prefix}.layers")

        self.norm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.make_empty_intermediate_tensors = (
            make_empty_intermediate_tensors_factory(
                ["hidden_states", "residual"], config.hidden_size))

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load weights for HybridQwen3Model.

        This method handles weight loading for all layer types including:
        - Attention layers with qkv_proj packing
        - SWA layers with qkv_proj packing
        - Mamba2 layers
        - BMOJO-F layers with kv_proj packing
        - GDN layers with merged conv and projections
        - GKA layers with merged conv and in_proj
        - MLP layers with gate_up_proj packing
        """
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("self_attn.qkv_proj", "self_attn.q_proj", "q"),
            ("self_attn.qkv_proj", "self_attn.k_proj", "k"),
            ("self_attn.qkv_proj", "self_attn.v_proj", "v"),
            ("swa.qkv_proj", "swa.q_proj", "q"),
            ("swa.qkv_proj", "swa.k_proj", "k"),
            ("swa.qkv_proj", "swa.v_proj", "v"),
            ("bmojo_attn.kv_proj", "bmojo_attn.k_proj", 0),
            ("bmojo_attn.kv_proj", "bmojo_attn.v_proj", 1),
            ("bmojo_attn.kv_proj_", "bmojo_attn.k_proj_", 0),
            ("bmojo_attn.kv_proj_", "bmojo_attn.v_proj_", 1),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
            # GDN weight remapping: separate → fused
            # Conv weights: q_conv1d, k_conv1d, v_conv1d → conv1d (merged at load time)
            ("gdn.conv1d", "gdn.q_conv1d", 0),
            ("gdn.conv1d", "gdn.k_conv1d", 1),
            ("gdn.conv1d", "gdn.v_conv1d", 2),
            # Linear projections: in_proj_all contains q, k, v, g, b, gk
            ("gdn.in_proj_all", "gdn.q_proj", 0),
            ("gdn.in_proj_all", "gdn.k_proj", 1),
            ("gdn.in_proj_all", "gdn.v_proj", 2),
            ("gdn.in_proj_all", "gdn.g_proj", 3),
            ("gdn.in_proj_all", "gdn.b_proj", 4),
            ("gdn.in_proj_all", "gdn.gk_proj", 5),
            # GKA conv weights: q_conv1d, k_conv1d, v_conv1d → conv1d (merged at load time)
            ("gka.conv1d", "gka.q_conv1d", 0),
            ("gka.conv1d", "gka.k_conv1d", 1),
            ("gka.conv1d", "gka.v_conv1d", 2),
            # GKA merged in_proj: q, k, v, a, [alpha], [b], [g]
            # All weights merged, biases for alpha/b loaded separately
            # The weight loader uses string shard_ids to handle dynamic config
            ("gka.in_proj", "gka.q_proj", "q_proj"),
            ("gka.in_proj", "gka.k_proj", "k_proj"),
            ("gka.in_proj", "gka.v_proj", "v_proj"),
            ("gka.in_proj", "gka.a_proj", "a_proj"),
            ("gka.in_proj", "gka.alpha_proj", "alpha_proj"),  # Weight only, bias separate
            ("gka.in_proj", "gka.b_proj", "b_proj"),  # Weight only, bias separate
            ("gka.in_proj", "gka.g_proj", "g_proj"),
        ]

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        for name, loaded_weight in weights:
            # Handle self_attn -> swa renaming for SWA layers.
            # Original Qwen3 checkpoints use "self_attn" for all layers,
            # but our SWA decoder layers store the module as "swa".
            if "self_attn" in name:
                match = re.search(r"layers\.(\d+)\.", name)
                if match:
                    layer_idx = int(match.group(1))
                    if self.layer_allocation.get(layer_idx) == Symbols.SWA:
                        name = name.replace("self_attn", "swa")

            # Handle Mamba2/GDN/GKA A_log -> A renaming
            if "A_log" in name:
                name = name.replace("A_log", "A")

            # Handle GKA alpha_proj.bias and b_proj.bias -> alpha_bias, b_bias
            if "gka.alpha_proj.bias" in name:
                name = name.replace("gka.alpha_proj.bias", "gka.alpha_bias")
            elif "gka.b_proj.bias" in name:
                name = name.replace("gka.b_proj.bias", "gka.b_bias")

            # Try stacked params mapping first
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                if 'experts' in name:
                    continue
                name = name.replace(weight_name, param_name)

                # Skip loading extra bias for GPTQ models
                if name.endswith(".bias") and name not in params_dict:
                    continue
                # Skip layers on other devices
                if is_pp_missing_parameter(name, self):
                    continue

                param = params_dict[name]
                weight_loader = param.weight_loader
                # default_weight_loader doesn't accept shard_id, but custom loaders do
                if weight_loader == default_weight_loader:
                    weight_loader(param, loaded_weight)
                else:
                    weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # No mapping found - load directly by name
                # Skip loading extra bias for GPTQ models
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if is_pp_missing_parameter(name, self):
                    continue

                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)

            loaded_params.add(name)
        return loaded_params

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass for HybridQwen3Model.

        Args:
            input_ids: Input token IDs
            positions: Position indices
            intermediate_tensors: Intermediate tensors for pipeline parallelism
            inputs_embeds: Input embeddings (optional)

        Returns:
            Hidden states tensor
        """
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_input_ids(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        for i, layer in enumerate(self.layers[self.start_layer:self.end_layer]):
            # All layer types (Attention, Mamba2, BMOJO-F) use the same interface
            hidden_states, residual = layer(
                positions=positions,
                hidden_states=hidden_states,
                residual=residual,
            )

        if not get_pp_group().is_last_rank:
            return IntermediateTensors({
                "hidden_states": hidden_states,
                "residual": residual
            })
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class HybridQwen3ForCausalLM(nn.Module, HasInnerState, SupportsMambaPrefixCaching, IsHybrid, SupportsLoRA, SupportsPP):
    packed_modules_mapping = {
        "self_attn.qkv_proj": [
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
        ],
        "swa.qkv_proj": [
            "swa.q_proj",
            "swa.k_proj",
            "swa.v_proj",
        ],
        "bmojo_attn.kv_proj": [
            "bmojo_attn.k_proj",
            "bmojo_attn.v_proj",
        ],
        "bmojo_attn.kv_proj_": [
            "bmojo_attn.k_proj_",
            "bmojo_attn.v_proj_",
        ],
        "in_proj": ["in_proj"],
        "gate_up_proj": [
            "gate_proj",
            "up_proj",
        ],
        # GDN packed modules (all merged at load time)
        "gdn.conv1d": [
            "gdn.q_conv1d",
            "gdn.k_conv1d",
            "gdn.v_conv1d",
        ],
        "gdn.in_proj_all": [
            "gdn.q_proj",
            "gdn.k_proj",
            "gdn.v_proj",
            "gdn.g_proj",
            "gdn.b_proj",
            "gdn.gk_proj",
        ],
        # GKA packed modules (conv and in_proj merged at load time)
        "gka.conv1d": [
            "gka.q_conv1d",
            "gka.k_conv1d",
            "gka.v_conv1d",
        ],
        "gka.in_proj": [
            "gka.q_proj",
            "gka.k_proj",
            "gka.v_proj",
            "gka.a_proj",
            "gka.alpha_proj",  # Weight only, bias loaded as gka.alpha_bias
            "gka.b_proj",      # Weight only, bias loaded as gka.b_bias
            "gka.g_proj",
        ],
    }

    # LoRA specific attributes
    embedding_modules = {
        "embed_tokens": "input_embeddings",
        "lm_head": "output_embeddings",
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        config = vllm_config.model_config.hf_config.get_text_config()
        if hasattr(config, "bmojo_config"):
            if "window_size" in config.bmojo_config:
                if config.bmojo_config["window_size"] % 2:  # we probably have power of 2 - 1 let's adjust this up 1
                    config.bmojo_config["window_size"] = config.bmojo_config["window_size"] + 1
        cache_config = vllm_config.cache_config
        scheduler_config = vllm_config.scheduler_config

        super().__init__()
        self.config = config
        self.vllm_config = vllm_config
        self.scheduler_config = scheduler_config
        self.model_config = vllm_config.model_config
        self.model = HybridQwen3Model(vllm_config=vllm_config,
                                    prefix=maybe_prefix(prefix, "model"))

        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
        )

        self.logits_processor = LogitsProcessor(config.vocab_size)

        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Apply token embeddings to input_ids (required for V1 compatibility)."""
        return self.model.embed_input_ids(input_ids)

    def forward(self,
                input_ids: torch.Tensor,
                positions: torch.Tensor,
                intermediate_tensors: IntermediateTensors | None = None,
                inputs_embeds: torch.Tensor | None = None,
                **kwargs):
        """Forward pass for HybridQwen3 model.

        Args:
            input_ids: Input token IDs
            positions: Position indices
            intermediate_tensors: Intermediate tensors for pipeline parallelism
            inputs_embeds: Input embeddings (optional)
            **kwargs: Additional arguments

        Returns:
            Hidden states tensor
        """
        # V1 pattern: no cache params needed, layers access cache via forward_context
        hidden_states = self.model(
            input_ids,
            positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )
        return hidden_states

    @classmethod
    def get_mamba_state_shape_from_config(
        cls,
        vllm_config: "VllmConfig",
    ) -> tuple[tuple[int, int], tuple[int, int, int]] | tuple[tuple[int, int], tuple[int, int]]:
        """Calculate shapes for inner state caches (Mamba2 or GDN).

        Args:
            vllm_config: vLLM config

        Returns:
            For Mamba2:
                - conv_state_shape: Shape for convolutional state cache
                - temporal_state_shape: Shape for state space model cache (3D)
            For GDN:
                - conv_state_shape: Shape for convolutional state cache
                - ssm_state_shape: Shape for recurrent state cache (2D)
        """
        parallel_config = vllm_config.parallel_config
        hf_config = vllm_config.model_config.hf_config.get_text_config()

        pattern = getattr(hf_config, "hybrid_override_pattern", "")
        bmojo_config = getattr(hf_config, "bmojo_config", {})
        syms = pattern_has_symbols(pattern, bmojo_config)
        has_gdn = Symbols.GDN in syms
        has_gka = Symbols.GKA in syms
        has_mamba = Symbols.MAMBA in syms

        # Check for attention-only models (only *, SWA layers, and BMF with attention-only mixers)
        # Inner state layers are: GDN, GKA, M2, and BMF with stateful mixers
        has_inner_state_layers = has_gdn or has_gka or has_mamba

        if not has_inner_state_layers:
            # This is an attention-only model (*, SWA layers, BMF with attention-only mixers)
            # Return empty shapes/dtypes so MambaSpec.page_size_bytes == 0
            # This will result in mamba_page_size == 0 and early return in config.py
            return (0,), (0,)

        # Validate: cannot mix different state shape types
        # - GDN: 2-tuple state (conv_state, ssm_state)
        # - GKA: 3-tuple state (conv_state, h_kk, h_kv)
        # - Mamba2: different state structure entirely
        bmojo_ssm_mixer = bmojo_config.get("ssm_mixer", "").lower() if bmojo_config else ""
        if has_gdn and has_gka:
            raise ValueError(
                f"Cannot mix GDN and GKA layers in the same model. "
                f"They have incompatible state shapes (2-tuple vs 3-tuple). "
                f"Pattern: {pattern}"
                + (f", BMOJO-F SSM mixer: {bmojo_ssm_mixer}" if bmojo_ssm_mixer else "")
            )
        if (has_gdn or has_gka) and has_mamba:
            raise ValueError(
                f"Cannot mix GDN/GKA and Mamba2 layers in the same model. "
                f"They have incompatible state shapes. "
                f"Pattern: {pattern}"
                + (f", BMOJO-F SSM mixer: {bmojo_ssm_mixer}" if bmojo_ssm_mixer else "")
                + "\nPlease use either GDN, GKA, or Mamba2, not mixed."
            )

        # If we have GKA layers, return GKA state shape (3-tuple)
        if has_gka:
            tp_size = parallel_config.tensor_parallel_size
            num_spec = (
                vllm_config.speculative_config.num_speculative_tokens
                if vllm_config.speculative_config
                else 0
            )

            hidden_size = hf_config.hidden_size
            gka_config = getattr(hf_config, "gka_config", {}) or {}
            if "num_q_heads" in gka_config:
                num_heads = gka_config["num_q_heads"]
                num_kv_heads = gka_config["num_k_heads"]
                head_dim = gka_config["head_dim"]
            else:
                num_heads = hf_config.num_attention_heads
                num_kv_heads = hf_config.num_key_value_heads
                head_dim = getattr(hf_config, "head_dim", hidden_size // num_heads)

            conv_kernel_size = gka_config.get("conv_size", 4)
            num_sketches_per_head = gka_config.get("num_sketches_per_head", 1)

            return PrimedStateShapeCalculator.gka_state_shape(
                tp_world_size=tp_size,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                hidden_size=hidden_size,
                head_dim=head_dim,
                conv_kernel_size=conv_kernel_size,
                num_spec=num_spec,
                num_sketches_per_head=num_sketches_per_head,
            )

        # If we have GDN layers, return GDN state shape
        if has_gdn:
            # GDN state shape calculation (checkpoint-compatible with GQA)
            tp_size = parallel_config.tensor_parallel_size
            num_spec = (
                vllm_config.speculative_config.num_speculative_tokens
                if vllm_config.speculative_config
                else 0
            )

            # Get dimensions from config
            hidden_size = hf_config.hidden_size

            # Get config from gdn_config
            layer_config = getattr(hf_config, "gdn_config", {}) or {}

            if "num_q_heads" in layer_config:
                num_heads = layer_config["num_q_heads"]
                num_kv_heads = layer_config["num_k_heads"]
                head_dim = layer_config["head_dim"]
            else:
                num_heads = hf_config.num_attention_heads
                num_kv_heads = hf_config.num_key_value_heads
                head_dim = getattr(hf_config, "head_dim", hidden_size // num_heads)

            conv_kernel_size = layer_config.get("conv_size", 4)

            # Use PrimedStateShapeCalculator for checkpoint-compatible GDN
            return PrimedStateShapeCalculator.gated_delta_net_state_shape(
                tp_world_size=tp_size,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                hidden_size=hidden_size,
                head_dim=head_dim,
                conv_kernel_size=conv_kernel_size,
                num_spec=num_spec,
            )

        # Otherwise, return Mamba2 state shape
        if has_mamba:
            intermediate_size = hf_config.mamba2_config.get("d_inner", 128)
            state_size = hf_config.mamba2_config.get("ssm_cfg", {}).get("d_state", 64)
            d_conv = hf_config.mamba2_config.get("d_conv", 4)
            n_groups = hf_config.mamba2_config.get("ssm_cfg", {}).get("ngroups", 1)
            num_heads = hf_config.mamba2_config.get("ssm_cfg", {}).get("ngroups", 1)
            head_dim = hf_config.mamba2_config.get("ssm_cfg", {}).get("d_state", 64)
            repeat_kv_before_conv = hf_config.mamba2_config.get("ssm_cfg", {}).get("repeat_kv_before_conv", False)
            xb_size = hf_config.mamba2_config.get("d_xb", 128)
            if repeat_kv_before_conv:
                conv_dim = intermediate_size + intermediate_size + intermediate_size
            else:
                conv_dim = intermediate_size + xb_size + xb_size

            return PrimedStateShapeCalculator.mamba2_state_shape(
                intermediate_size=intermediate_size,
                tp_world_size=parallel_config.tensor_parallel_size,
                n_groups=n_groups,
                num_heads=num_heads,
                head_dim=head_dim,
                state_size=state_size,
                conv_kernel=d_conv,
                conv_dim=conv_dim,
            )

        # If we reach here, we should have handled all cases above
        raise RuntimeError(f"Unexpected state in get_mamba_state_shape_from_config for pattern: {pattern}")

    @classmethod
    def get_mamba_state_dtype_from_config(
        cls,
        vllm_config: "VllmConfig",
    ) -> tuple[torch.dtype, torch.dtype] | tuple[torch.dtype, torch.dtype, torch.dtype]:
        hf_config = vllm_config.model_config.hf_config.get_text_config()
        pattern = getattr(hf_config, "hybrid_override_pattern", "")
        bmojo_config = getattr(hf_config, "bmojo_config", {})
        syms = pattern_has_symbols(pattern, bmojo_config)
        has_gdn = Symbols.GDN in syms
        has_gka = Symbols.GKA in syms
        has_mamba = Symbols.MAMBA in syms

        if has_gka:
            # GKA needs 3 dtypes: (conv_dtype, h_kk_dtype, h_kv_dtype)
            # h_kk must be float32 to avoid numerical drift in Chebyshev solver
            # h_kv can use ssm_dtype (typically bfloat16) - GLA is more robust
            conv_dtype, ssm_dtype = MambaStateDtypeCalculator.gated_delta_net_state_dtype(
                vllm_config.model_config.dtype,
                vllm_config.cache_config.mamba_cache_dtype,
            )
            return conv_dtype, torch.float32, ssm_dtype

        if has_gdn:
            # GDN uses 2 dtypes
            return MambaStateDtypeCalculator.gated_delta_net_state_dtype(
                vllm_config.model_config.dtype,
                vllm_config.cache_config.mamba_cache_dtype,
            )

        if has_mamba:
            # Mamba2 uses 2 dtypes
            return MambaStateDtypeCalculator.mamba2_state_dtype(
                vllm_config.model_config.dtype,
                vllm_config.cache_config.mamba_cache_dtype,
                vllm_config.cache_config.mamba_ssm_cache_dtype,
            )

        # Attention-only model
        return (vllm_config.model_config.dtype,)

    def get_mamba_state_copy_func(self) -> tuple:
        """
        Returns copy functions for each mamba state to support prefix caching.

        Returns a tuple of MambaStateCopyFunc callables, one per state tensor.
        The order must match get_mamba_state_shape_from_config.
        """
        from vllm.model_executor.layers.mamba.mamba_utils import (
            MambaStateCopyFuncCalculator, get_conv_copy_spec,
            get_temporal_copy_spec,
        )

        pattern = getattr(self.config, "hybrid_override_pattern", "")
        bmojo_config = getattr(self.config, "bmojo_config", {})
        syms = pattern_has_symbols(pattern, bmojo_config)

        has_mamba2 = Symbols.MAMBA in syms
        has_gdn = Symbols.GDN in syms
        has_gka = Symbols.GKA in syms

        if has_gka:
            # 3 states: conv_state, h_kk, h_kv
            return (get_conv_copy_spec, get_temporal_copy_spec, get_temporal_copy_spec)
        if has_gdn:
            # 2 states: conv_state, ssm_state
            return MambaStateCopyFuncCalculator.gated_delta_net_state_copy_func()
        if has_mamba2:
            # 2 states: conv_state, ssm_state
            return MambaStateCopyFuncCalculator.mamba2_state_copy_func()

        # Attention-only model — no mamba states
        return ()

    # V1: BMOJO-F cache shape and dtype methods removed
    # Cache management is now handled automatically by V1's cache system
    # via get_kv_cache_spec() in BMOJOFAttentionC

    def _get_mamba_cache_shape(
            self) -> tuple[tuple[int, int], tuple[int, int]]:
        world_size = get_tensor_model_parallel_world_size()

        conv_state_shape, temporal_state_shape = None, None

        intermediate_size = self.config.mamba2_config.get("d_inner")
        xb_size = self.config.mamba2_config.get("d_xb")

        # if n_groups is not divisible by world_size, need to extend the shards
        # to ensure all groups needed by a head is sharded along with it
        n_groups = (
            self.config.mamba2_config.get("ssm_cfg", {}).get("ngroups") +
            extra_groups_for_head_shards(self.config.mamba2_config.get("ssm_cfg", {}).get("ngroups"), world_size))

        # - heads and n_groups are TP-ed
        if self.config.mamba2_config.get("ssm_cfg", {}).get("repeat_kv_before_conv", False):
            conv_dim = intermediate_size + intermediate_size + intermediate_size
        else:
            conv_dim = intermediate_size + xb_size + xb_size
        conv_kernel_size = self.config.mamba2_config.get("d_conv", 4)
        conv_state_shape = (
            divide(conv_dim, world_size),
            conv_kernel_size - 1,
        )

        # These are not TP-ed as they depend on A, dt_bias, D
        # - they are typically small
        #   e.g., (h_heads, d_head, d_state) = (128, 64, 128)
        temporal_state_shape = (
            divide(self.config.mamba2_config.get("ssm_cfg", {}).get("ngroups", 1), world_size),
            self.config.mamba2_config.get("ssm_cfg", {}).get("d_state", 64),
            self.config.mamba2_config.get("ssm_cfg", {}).get("d_state", 64),
        )
        return conv_state_shape, temporal_state_shape

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        logits = self.logits_processor(self.lm_head, hidden_states)
        return logits

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load weights for HybridQwen3ForCausalLM.

        Uses AutoWeightsLoader which delegates to HybridQwen3Model.load_weights()
        for the actual weight loading logic.
        """
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=(["lm_head."] if self.config.tie_word_embeddings else None),
        )
        return loader.load_weights(weights)
