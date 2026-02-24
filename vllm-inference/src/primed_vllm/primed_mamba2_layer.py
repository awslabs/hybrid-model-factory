
import torch
from torch import nn

from einops import rearrange

from vllm.config import CacheConfig, ModelConfig, get_current_vllm_config
from vllm.v1.attention.backend import AttentionMetadata
from vllm.distributed import (divide, get_tensor_model_parallel_rank,
                              get_tensor_model_parallel_world_size,
                              tensor_model_parallel_all_gather,
                              tensor_model_parallel_all_reduce)
from vllm.forward_context import ForwardContext, get_forward_context
from vllm.model_executor.custom_op import CustomOp
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               MergedColumnParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
    causal_conv1d_fn, causal_conv1d_update)
from vllm.model_executor.layers.mamba.ops.mamba_ssm import (
    selective_state_update)
from vllm.model_executor.layers.mamba.ops.ssd_combined import (
    mamba_chunk_scan_combined_varlen)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.model_loader.weight_utils import (
    LoaderFunction, composed_weight_loader, sharded_weight_loader)

from vllm.model_executor.layers.mamba.ops.layernorm_gated import rms_norm_gated
from vllm.model_executor.utils import set_weight_attrs
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.mamba.abstract import MambaBase
from vllm.platforms import current_platform
from vllm.utils.torch_utils import direct_register_custom_op
from vllm.v1.attention.backends.mamba2_attn import Mamba2AttentionMetadata
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateDtypeCalculator,)

# Import state shape calculator from primed_utils
from .primed_utils import PrimedStateShapeCalculator


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, None, :, :].expand(
        num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(num_key_value_heads * n_rep, slen, head_dim)


# Adapted from transformers.models.mamba2.modeling_mamba2.MambaRMSNormGated
@CustomOp.register("mixer2_gated_rms_norm_primed")
class Mixer2RMSNormGated(CustomOp):

    def __init__(self,
                 full_hidden_size: int,
                 full_n_groups: int,
                 use_rms_norm: bool = True,
                 eps: float = 1e-6):
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()
        self.full_hidden_size = full_hidden_size
        self.group_size = full_hidden_size // full_n_groups
        self.per_rank_hidden_size = full_hidden_size // self.tp_size
        self.n_groups = full_hidden_size // self.group_size

        self.variance_epsilon = eps
        self.use_rms_norm = use_rms_norm
        if self.use_rms_norm:
            # Register norm weight only if we're actually applying RMSNorm
            self.weight = nn.Parameter(torch.ones(self.per_rank_hidden_size))
            set_weight_attrs(self.weight,
                             {"weight_loader": sharded_weight_loader(0)})
        else:
            # Avoid checkpoint mismatch by skipping unused parameter
            self.register_parameter("weight", None)
        assert (self.full_hidden_size % self.tp_size == 0
                ), "Tensor parallel world size must divide hidden size."

    def forward_native(
        self,
        x: torch.Tensor,
        gate: torch.Tensor,
    ):
        # Three tensor-parallel cases:
        #   1. n_groups is 1
        #      In this case we parallelize along the reduction dim.
        #      Each rank computes a local sum of squares followed by AllReduce
        #   2. tp_size divides n_groups
        #      Each rank only reduces within its local group(s).
        #      No collective ops necessary.
        #   3. The general case can be pretty complicated so we AllGather
        #      the input and then redundantly compute the RMSNorm.
        input_dtype = x.dtype
        x = x * nn.functional.silu(gate.to(torch.float32))
        if not self.use_rms_norm:
            return x.to(input_dtype)

        if self.n_groups == 1:
            if self.tp_size > 1:
                # Compute local sum and then reduce to obtain global sum
                local_sums = x.pow(2).sum(dim=-1, keepdim=True)
                global_sums = tensor_model_parallel_all_reduce(local_sums)
                # Calculate the variance
                count = self.tp_size * x.shape[-1]
                variance = global_sums / count

            else:
                variance = x.pow(2).mean(-1, keepdim=True)
            x = x * torch.rsqrt(variance + self.variance_epsilon)
        else:
            redundant_tp: bool = self.n_groups % self.tp_size != 0
            if redundant_tp:
                # To handle the general case, redundantly apply the variance
                x = tensor_model_parallel_all_gather(x, -1)

            *prefix_dims, hidden_dim = x.shape
            group_count = hidden_dim // self.group_size
            x_grouped = x.view(*prefix_dims, group_count, self.group_size)
            variance = x_grouped.pow(2).mean(-1, keepdim=True)
            x_grouped = x_grouped * torch.rsqrt(variance +
                                                self.variance_epsilon)
            x = x_grouped.view(*prefix_dims, hidden_dim)

            if redundant_tp:
                start = self.per_rank_hidden_size * self.tp_rank
                end = start + self.per_rank_hidden_size
                x = x[..., start:end]

        return self.weight * x.to(input_dtype)

    def forward_cuda(
        self,
        x: torch.Tensor,
        gate: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        input_dtype = x.dtype
        if not self.use_rms_norm:
            # Keep gate in float32 for numerical stability during silu
            return x * nn.functional.silu(gate.to(
                torch.float32)).to(input_dtype)

        if (((self.n_groups % self.tp_size) != 0) or self.n_groups != 1):
            return self.forward_native(x, gate)

        return rms_norm_gated(x,
                              self.weight.data,
                              bias=None,
                              z=gate,
                              eps=self.variance_epsilon,
                              norm_before_gate=False)


def extra_groups_for_head_shards(ngroups: int, tp_size: int):
    """Compute the increase in group numbers to account for 
    replication in order to accompany the head shards."""

    # in the case ngoups % tp_size == 0, this will be zero
    if ngroups % tp_size == 0:
        return 0

    # for n_groups == 1, this is exactly tp_size - n_groups
    return tp_size - ngroups


def mamba_v2_sharded_weight_loader(
    shard_spec: list[tuple[int, int, float]],
    tp_size: int,
    tp_rank: int,
) -> LoaderFunction:
    """Create a weight loader for mamba v2. This ensures that the projections 
    are correctly sharded so that they can be split into x, B, C. It also 
    ensures the the all the groups corresponding to a head shard is placed 
    together with it.
    """

    def loader(param: torch.Tensor, loaded_weight: torch.Tensor) -> None:

        # - track boundary of (sharded) param, and loaded_weight, respectively
        boundary, loaded_boundary = 0, 0

        # - iterate over the shard specs
        for full_dim, extra, duplicate_groups in shard_spec:
            # - full dim is the model dim (before TP).
            # - extra > 0, means there is expected overall increase
            #   of dimensions. This is so because of replication.
            # - ratio is used map the tp_rank to the actual shard
            #   rank. This is useful when there is replication of
            #   groups to accompany head shards.

            # - size of the loaded shard (includes any padding)
            shard_size = full_dim // tp_size

            # - compute the rank into the loaded shard.
            # - if there is replication, different TP shards will
            #   take from the same rank.
            # NOTE: currently we only support duplication
            # in the case where num_groups == 1
            rank = 0 if duplicate_groups else tp_rank

            # - real (unpadded) per-rank size for checkpoint indexing.
            #   When extra > 0 (e.g. stride-alignment padding), the
            #   checkpoint only has (full_dim - extra) real values.
            real_per_rank = (full_dim - extra) // tp_size

            # - leftmost boundary index into loaded weight.
            loaded_skip = rank * real_per_rank
            loaded_start_idx = loaded_boundary + loaded_skip

            # - take these many dims from the loaded weight.
            take = min(real_per_rank, full_dim - extra - loaded_skip)

            # - always shard on dim 0
            # - the ignore is for a mundane mypy error as it does not
            #   seem to handle slices well.
            # https://github.com/python/mypy/issues/2410
            param.data[
                boundary:(boundary + take),  # type: ignore[misc]
                ...] = loaded_weight[loaded_start_idx:(  # type: ignore[misc]
                    loaded_start_idx + take)]  # type: ignore[misc]

            # move indexing boundaries
            boundary += shard_size
            loaded_boundary += (full_dim - extra)

    return loader


# Adapted from transformers.models.mamba.modeling_mamba.MambaMixer
@CustomOp.register("mamba_mixer2_primed")
class PrimedMamba2Layer(CustomOp, MambaBase):
    """
    Mamba2 selective SSM layer for Hybrid Model Factory models.

    Adapted from vLLM's MambaMixer2 with additional support for QK normalization,
    rotary/positional embeddings, and configurable xB projections used by
    Hybrid Model Factory Primed architectures.

    Accepts the full model config and extracts mamba2_config internally,
    consistent with PrimedGDNLayer/PrimedGKALayer.
    """

    def __init__(
            self,
            config,
            model_config: ModelConfig | None = None,
            cache_config: CacheConfig | None = None,
            quant_config: QuantizationConfig | None = None,
            prefix: str = "",
    ):
        super().__init__()

        # Extract Mamba2 config (follows GDN/GKA pattern)
        mamba2_config = getattr(config, "mamba2_config", {}) or {}
        ssm_cfg = mamba2_config.get("ssm_cfg", {})

        hidden_size = mamba2_config.get("d_model", 2048)
        ssm_state_size = ssm_cfg.get("d_state", 64)
        conv_kernel_size = mamba2_config.get("d_conv", 4)
        intermediate_size = mamba2_config.get("d_inner", 128)
        xb_size = mamba2_config.get("d_xb", 128)
        use_conv_bias = mamba2_config.get("use_conv_bias", True)
        use_bias = mamba2_config.get("use_bias", False)
        n_groups = ssm_cfg.get("ngroups", 1)
        num_heads = ssm_cfg.get("ngroups", 1)
        head_dim = ssm_cfg.get("d_state", 64)
        rms_norm_eps = mamba2_config.get("rms_norm_eps", 1e-6)
        activation = mamba2_config.get("hidden_act", "silu")
        chunk_size = mamba2_config.get("chunk_size", 256)
        repeat_kv_before_conv = ssm_cfg.get("repeat_kv_before_conv", False)
        use_rotary_emb = mamba2_config.get("use_rotary_emb", False)
        use_pos_embed = mamba2_config.get("use_pos_embed", False)
        xb_proj_rank = mamba2_config.get("xb_proj_rank", None)
        xb_proj_activation = mamba2_config.get("xb_proj_activation", False)
        xb_learnable_residual = mamba2_config.get("xb_learnable_residual", False)
        use_qk_norm = mamba2_config.get("use_qk_norm", True)

        # Check for unsupported features
        if xb_proj_rank is not None:
            raise NotImplementedError("xb_proj_rank is not implemented in vLLM")
        if xb_learnable_residual:
            raise NotImplementedError("xb_learnable_residual is not implemented in vLLM")

        # For TP, the sharding plan is as follows:
        # - for the conv modules, since
        #   conv_dim = intermediate_size * 2 * n_groups * ssm_state_size,
        #   we shard intermediate_size and n_groups
        # - since intermediate_size = n_heads * head_dim, sharding on
        #   intermediate_size is achieved by sharding on n_heads.
        # - IF, world_size divides groups, then sharding
        #   (n_groups / world_size, n_heads / world_size)
        #   also maintains the invariant n_heads % n_groups == 0
        # - HOWEVER IF, world_size DOES NOT divide groups, then we need
        #   to allocate extra space in the shard, such that groups
        #   may be replicated to follow the head shard.
        # - NOTE: currently for the world size DOES NOT divide groups
        #   case, we only support the case when n_groups == 1
        self.tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()

        assert num_heads % self.tp_size == 0, \
            f"Tensor parallel world size must divide num heads: {tp_rank} {num_heads}"

        assert (n_groups % self.tp_size) == 0 or n_groups == 1, \
            (
                "If tensor parallel world size does not divide num_heads, "
                "then num_groups must equal 1."
            )

        assert self.tp_size == 1 or quant_config is None, \
            "Tensor parallel currently not supported for quantized models."
        assert not repeat_kv_before_conv, "repeat_kv_before_conv not implemented yet"

        self.ssm_state_size = ssm_state_size
        self.activation = activation
        self.conv_kernel_size = conv_kernel_size

        self.chunk_size = chunk_size
        self.intermediate_size = intermediate_size
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.repeat_kv_before_conv = repeat_kv_before_conv
        self.xb_size = xb_size
        self.n_groups = n_groups
        self.repeat_group = self.intermediate_size // self.xb_size

        assert self.n_groups == self.num_heads
        if n_groups % self.tp_size != 0:
            # - for TP we shard conv_dim by sharding on n_groups,
            # - but if n_groups cannot divide tp_size, we need to
            #   extend some extra groups
            self.n_groups = n_groups + extra_groups_for_head_shards(
                n_groups, self.tp_size)

        if self.repeat_kv_before_conv:
            self.conv_dim = intermediate_size + intermediate_size + intermediate_size
            conv1d_output_sizes = [intermediate_size, intermediate_size, intermediate_size]
        else:
            self.conv_dim = intermediate_size + xb_size + xb_size
            conv1d_output_sizes = [intermediate_size, xb_size, xb_size]
        self.conv1d = MergedColumnParallelLinear(
            input_size=conv_kernel_size,
            output_sizes=conv1d_output_sizes,
            bias=use_conv_bias,
            quant_config=None,
        )
        self.d_in_proj = self.intermediate_size + self.xb_size + self.xb_size + self.intermediate_size + self.num_heads
        # Pad d_in_proj so the PER-RANK size is a multiple of 8.
        # CUDA memory allocator can pad unaligned dimensions inconsistently
        # between torch.compile trace and execution, causing stride assertion
        # failures. At TP=1 the full size matters; at TP=N the sharded size
        # (d_in_proj / N) must be 8-aligned. We pad via dt (the smallest
        # component) by computing the deficit on the per-rank size and scaling
        # back up to the full size.
        per_rank = self.d_in_proj // self.tp_size
        per_rank_pad = (8 - (per_rank % 8)) % 8
        self.dt_pad = per_rank_pad * self.tp_size
        self.d_in_proj += self.dt_pad
        # unsqueeze to fit conv1d weights shape into the linear weights shape.
        # Can't do this in `weight_loader` since it already exists in
        # `ColumnParallelLinear` and `set_weight_attrs`
        # doesn't allow to override it
        self.conv1d.weight.data = self.conv1d.weight.data.unsqueeze(1)

        self.in_proj = MergedColumnParallelLinear(
            input_size=hidden_size,
            output_sizes=[
                self.intermediate_size,  # gate
                self.xb_size,            # B
                self.xb_size,            # C
                self.intermediate_size,  # hidden states
                self.num_heads + self.dt_pad,  # dt (padded for alignment)
            ],
            bias=use_bias,
            quant_config=quant_config,
        )

        # - because in_proj is a concatenation of 3 weights, we
        #   need to interleave them before sharding
        # - use the custom weight loader mamba_v2_sharded_weight_loader
        #   for conv1d.bias, covn1d.weight and in_proj.weight
        # - need to set these settings, to assign the groups to the head shards
        group_shard_settings = (
            self.n_groups * self.ssm_state_size,  # expected model size
            (self.n_groups - n_groups) *
            self.ssm_state_size,  # extra dims assigned
            n_groups == 1,  # if there was only one group
        )
        xb_settings = (self.xb_size, 0, False)
        # head_settings includes dt_pad: model has num_heads + dt_pad dims,
        # but checkpoint only has num_heads dims. The extra dims are padding.
        head_settings = (self.num_heads + self.dt_pad, self.dt_pad, False)
        intermediate_settings = (intermediate_size, 0, False)

        # - the weight already has a "weight_loader" attribute
        #   which set_weight_attrs will raise if we do not
        #   delete before trying to override it
        # - ditto for the otther two weights below
        delattr(self.conv1d.bias, "weight_loader")
        set_weight_attrs(
            self.conv1d.bias, {
                "weight_loader":
                mamba_v2_sharded_weight_loader(
                    [
                        xb_settings,
                        xb_settings,
                        group_shard_settings,
                    ],
                    self.tp_size,
                    tp_rank,
                )
            })

        delattr(self.conv1d.weight, "weight_loader")
        set_weight_attrs(
            self.conv1d.weight, {
                "weight_loader":
                mamba_v2_sharded_weight_loader([
                    xb_settings,
                    xb_settings,
                    group_shard_settings,
                ], self.tp_size, tp_rank)
            })

        if quant_config is None:
            # - quant layers do not have a weight loader
            delattr(self.in_proj.weight, "weight_loader")
            set_weight_attrs(
                self.in_proj.weight,
                {
                    "weight_loader":
                    mamba_v2_sharded_weight_loader(
                        [
                            intermediate_settings,  # for gate
                            xb_settings,
                            xb_settings,
                            intermediate_settings,
                            head_settings,  # for dt
                        ],
                        self.tp_size,
                        tp_rank)
                })

        # - these are TPed by heads to reduce the size of the
        #   temporal shape
        self.A = nn.Parameter(
            torch.empty(
                divide(num_heads, self.tp_size),
                dtype=torch.float32,
            ))
        self.D = nn.Parameter(torch.ones(num_heads // self.tp_size))
        self.dt_bias = nn.Parameter(torch.ones(num_heads // self.tp_size))

        set_weight_attrs(self.D, {"weight_loader": sharded_weight_loader(0)})
        a_weight_loader = composed_weight_loader(
            sharded_weight_loader(0), lambda x: -torch.exp(x.float()))
        set_weight_attrs(self.A, {"weight_loader": a_weight_loader})
        set_weight_attrs(self.dt_bias,
                         {"weight_loader": sharded_weight_loader(0)})

        self.out_proj = RowParallelLinear(intermediate_size,
                                          hidden_size,
                                          bias=use_bias,
                                          input_is_parallel=True,
                                          quant_config=quant_config)

        self.norm = Mixer2RMSNormGated(intermediate_size,
                                       n_groups,
                                       eps=1e-5)

        # Register conv_weights buffer for efficient access
        conv_weights = self.conv1d.weight.view(
            self.conv1d.weight.size(0), self.conv1d.weight.size(2)
        )
        self.register_buffer("conv_weights", conv_weights, persistent=False)

        # Pre-compute sizes for forward pass
        self.tped_intermediate_size = self.intermediate_size // self.tp_size
        self.tped_conv_size = self.conv_dim // self.tp_size
        self.tped_dt_size = (self.num_heads + self.dt_pad) // self.tp_size

        groups_time_state_size = self.n_groups * self.ssm_state_size
        # Lambda for splitting hidden_states_B_C
        self.split_hidden_states_B_C_fn = lambda hidden_states_B_C: torch.split(
            hidden_states_B_C,
            [
                self.xb_size // self.tp_size,
                self.xb_size // self.tp_size,
                groups_time_state_size // self.tp_size,
            ],
            dim=-1,
        )

        compilation_config = get_current_vllm_config().compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self
        # The tuple is (conv_state, ssm_state)
        self.kv_cache = (torch.tensor([]), torch.tensor([]))

        self.use_rotary_emb = use_rotary_emb
        if self.use_rotary_emb:
            hf_config = model_config.hf_config
            rope_theta = getattr(hf_config, "rope_theta", 1000000)
            rope_scaling = getattr(hf_config, "rope_scaling", None)
            max_position = getattr(hf_config, "max_position_embeddings", 4096 * 32)

            # Build rope_parameters dict for new get_rope signature
            rope_parameters = {"rope_theta": rope_theta, "rope_type": "default"}
            if rope_scaling is not None:
                # Convert old rope_scaling format to new rope_parameters format
                rope_type = rope_scaling.get("type", "default")
                rope_parameters["rope_type"] = rope_type
                # Copy other rope_scaling params as-is
                for key, value in rope_scaling.items():
                    if key != "type":
                        rope_parameters[key] = value

            self.rotary_emb = get_rope(
                self.head_dim,
                max_position=max_position,
                rope_parameters=rope_parameters,
            )

        self.model_config = model_config
        self.cache_config = cache_config
        self.prefix = prefix

        # Positional embeddings
        self.use_pos_embed = use_pos_embed
        if use_pos_embed:
            hf_config = model_config.hf_config if model_config else None
            rope_theta = getattr(hf_config, "rope_theta", 1000000) if hf_config else 1000000
            rope_scaling = getattr(hf_config, "rope_scaling", None) if hf_config else None
            max_position = getattr(hf_config, "max_position_embeddings", 4096 * 32) if hf_config else 4096 * 32

            # Build rope_parameters dict for new get_rope signature
            rope_parameters = {"rope_theta": rope_theta, "rope_type": "default"}
            if rope_scaling is not None:
                # Convert old rope_scaling format to new rope_parameters format
                rope_type = rope_scaling.get("type", "default")
                rope_parameters["rope_type"] = rope_type
                # Copy other rope_scaling params as-is
                for key, value in rope_scaling.items():
                    if key != "type":
                        rope_parameters[key] = value

            self.rotary_emb = get_rope(
                self.head_dim,
                max_position=max_position,
                rope_parameters=rope_parameters,
            )

        # QK normalization
        self.C_norm = None
        self.B_norm = None
        if use_qk_norm:
            self.C_norm = RMSNorm(self.head_dim, rms_norm_eps)
            self.B_norm = RMSNorm(self.head_dim, rms_norm_eps)

        # Positional embeddings
        self.use_pos_embed = use_pos_embed

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        mup_vector: torch.Tensor | None = None,
    ):
        # 1. Gated MLP's linear projection
        projected_states, _ = self.in_proj(hidden_states)
        if mup_vector is not None:
            projected_states = projected_states * mup_vector

        # 2. Prepare inputs for conv + SSM
        ssm_output = torch.empty(
            [
                hidden_states.shape[0],
                (self.num_heads // self.tp_size) * self.head_dim,
            ],
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

        # 3. conv + SSM
        # (split `projected_states` into hidden_states_B_C, dt in the custom op to
        # ensure it is not treated as an intermediate tensor by torch compile)
        torch.ops.vllm.mamba_mixer2_primed(
            projected_states,
            ssm_output,
            self.prefix,
            positions,
        )

        # 4. gated MLP
        # GatedRMSNorm internally applying SiLU to the gate
        # SiLU is applied internally before normalization, unlike standard
        # norm usage
        gate = projected_states[..., : self.tped_intermediate_size]
        hidden_states = self.norm(ssm_output, gate)

        # 5. Final linear projection
        output, _ = self.out_proj(hidden_states)

        return output


    def conv_ssm_forward(
        self,
        projected_states: torch.Tensor,
        output: torch.Tensor,
        positions: torch.Tensor,
    ):
        # Split projected_states into gate, hidden_states_B_C, dt
        gate, hidden_states_B_C, dt = torch.split(
            projected_states,
            [
                self.tped_intermediate_size,
                self.tped_conv_size,
                self.tped_dt_size,
            ],
            dim=-1,
        )
        # Slice off padding from dt (added for memory alignment)
        if self.dt_pad > 0:
            dt = dt[..., : self.num_heads // self.tp_size]

        forward_context = get_forward_context()
        # attn_metadata contains metadata necessary for the mamba2 triton
        # kernels to operate in continuous batching and in chunked prefill
        # modes; they are computed at top-level model forward since they
        # stay the same and reused for all mamba layers in the same iteration
        attn_metadata: AttentionMetadata = forward_context.attn_metadata

        assert self.cache_config is not None
        mamba_block_size = self.cache_config.mamba_block_size
        is_mamba_cache_all = self.cache_config.mamba_cache_mode == "all"
        if attn_metadata is not None:
            assert isinstance(attn_metadata, dict)
            attn_metadata = attn_metadata[self.prefix]
            assert isinstance(attn_metadata, Mamba2AttentionMetadata)
            self_kv_cache = self.kv_cache[forward_context.virtual_engine]
            # conv_state = (..., dim, width-1) yet contiguous along 'dim'
            conv_state = self_kv_cache[0].transpose(-1, -2)
            ssm_state = self_kv_cache[1]
            state_indices_tensor = attn_metadata.state_indices_tensor
            has_initial_states_p = attn_metadata.has_initial_states_p
            prep_initial_states = attn_metadata.prep_initial_states
            chunk_size = attn_metadata.chunk_size
            seq_idx_p = attn_metadata.seq_idx_p
            query_start_loc_p = attn_metadata.query_start_loc_p
            cu_chunk_seqlen_p = attn_metadata.cu_chunk_seqlen_p
            last_chunk_indices_p = attn_metadata.last_chunk_indices_p


        groups_time_state_size = self.n_groups * self.ssm_state_size
        if (self.B_norm is not None) or self.use_pos_embed:
            hidden_states, B, C = self.split_hidden_states_B_C_fn(hidden_states_B_C)
            if self.B_norm is not None:
                B = self.B_norm(B.reshape(-1, self.head_dim)).view(B.shape)
                C = self.C_norm(C.reshape(-1, self.head_dim)).view(C.shape)
            if self.use_pos_embed:
                C, B = self.rotary_emb(positions, C, B)
            hidden_states_B_C = torch.cat([hidden_states, B, C], dim=-1)

        # gate: L x intermediate_size
        # hidden_states_B_C: L x intermediate_size + xb_size + xb_size
        # dt: L x self.num_heads
        if attn_metadata is None:
            # profile run - write to output tensor to match execution path
            # Use transpose().clone().transpose() pattern to match upstream vLLM
            # and ensure consistent strides between profile and execution runs
            # 
            # CRITICAL: hidden_states_B_C is a slice of projected_states and inherits
            # its stride. The linear layer output stride can vary between profile and
            # execution runs due to memory allocation patterns. We must normalize
            # the stride BEFORE any operations to ensure torch.compile sees consistent
            # tensor layouts.
            hidden_states_B_C = (
                hidden_states_B_C.transpose(0, 1).clone().transpose(0, 1)
            ).contiguous()

            hidden_states, _B, _C = self.split_hidden_states_B_C_fn(
                hidden_states_B_C)

            xb_group = self.xb_size // (self.ssm_state_size * self.tp_size)

            hidden_states = rearrange(hidden_states, "l (group_size ssm_size) -> group_size l ssm_size", group_size=xb_group)
            hidden_states = repeat_kv(hidden_states, self.repeat_group)

            hidden_states = rearrange(hidden_states, "rep_group_size seq_len d -> seq_len (rep_group_size d)", rep_group_size=self.num_heads // self.tp_size)
            return hidden_states

        num_prefills = attn_metadata.num_prefills  # request count
        num_decodes = attn_metadata.num_decode_tokens  # token count (=request)
        num_prefill_tokens = attn_metadata.num_prefill_tokens  # token count
        has_prefill = num_prefills > 0
        has_decode = num_decodes > 0
        num_actual_tokens = num_prefill_tokens + num_decodes

        # Separate prefill and decode by splitting varlen input
        # Split along token dimension
        hidden_states_B_C_d, hidden_states_B_C_p = torch.split(
            hidden_states_B_C[:num_actual_tokens],
            [num_decodes, num_prefill_tokens],
            dim=0,
        )
        dt_d, dt_p = torch.split(
            dt[:num_actual_tokens],
            [num_decodes, num_prefill_tokens],
            dim=0,
        )
        # Split along batch dimension
        state_indices_tensor_d, state_indices_tensor_p = torch.split(
            state_indices_tensor[:num_actual_tokens],
            [num_decodes, num_prefills],
            dim=0,
        )

        if is_mamba_cache_all:
            # If prefix caching is enabled, retrieve the relevant variables
            # for prefill and decode
            block_idx_last_computed_token_d, block_idx_last_computed_token_p = (
                torch.split(
                    attn_metadata.block_idx_last_computed_token,
                    [num_decodes, num_prefills],
                    dim=0,
                )
            )
            block_idx_last_scheduled_token_d, block_idx_last_scheduled_token_p = (
                torch.split(
                    attn_metadata.block_idx_last_scheduled_token,
                    [num_decodes, num_prefills],
                    dim=0,
                )
            )
            # Prefill-only variables:
            block_idx_first_scheduled_token_p = (
                attn_metadata.block_idx_first_scheduled_token_p
            )
            num_computed_tokens_p = attn_metadata.num_computed_tokens_p
        else:
            block_idx_last_computed_token_d = None
            block_idx_last_computed_token_p = None
            block_idx_last_scheduled_token_d = None
            block_idx_last_scheduled_token_p = None
            block_idx_first_scheduled_token_p = None
            num_computed_tokens_p = None

        # Split output tensor directly to avoid stride mismatches with torch.compile
        # Use output[:num_actual_tokens] directly instead of creating a separate tensor
        preallocated_ssm_out_d, preallocated_ssm_out_p = torch.split(
            output[:num_actual_tokens],
            [num_decodes, num_prefill_tokens],
            dim=0,
        )

        # Process prefill requests
        if has_prefill:
            # 2. Convolution sequence transformation
            # - It will read the initial states for every sequence,
            #   that has "has_initial_states_p" == True,
            #   from "cache_indices", using "state_indices_tensor_p".
            # - It updates the "conv_state" cache in positions pointed
            #   to by "state_indices_tensor_p".
            #   In particular, it will always write the state at the
            #   sequence end.
            #   In addition, "block_idx_first_scheduled_token_p" and
            #   "block_idx_last_scheduled_token_p"
            #   are provided (which are pointers into
            #   "state_indices_tensor_p"), it will write additional cache
            #   states aligned at "block_size_to_align".
            x = hidden_states_B_C_p.transpose(
                0, 1)  # this is the form that causal-conv see
            hidden_states_B_C_p = causal_conv1d_fn(
                x,
                self.conv_weights,
                self.conv1d.bias,
                activation=self.activation,
                conv_states=conv_state,
                has_initial_state=has_initial_states_p,
                cache_indices=state_indices_tensor_p,
                block_idx_first_scheduled_token=block_idx_first_scheduled_token_p,
                block_idx_last_scheduled_token=block_idx_last_scheduled_token_p,
                initial_state_idx=block_idx_last_computed_token_p,
                num_computed_tokens=num_computed_tokens_p,
                block_size_to_align=mamba_block_size,
                metadata=attn_metadata,
                query_start_loc=query_start_loc_p,
            ).transpose(0, 1)[:num_prefill_tokens]

            hidden_states_p, B_p, C_p = self.split_hidden_states_B_C_fn(
                hidden_states_B_C_p)

            xb_group = self.xb_size // (self.ssm_state_size * self.tp_size)

            hidden_states_p = rearrange(hidden_states_p, "l (group_size ssm_size) -> group_size l ssm_size", group_size=xb_group)
            hidden_states_p = repeat_kv(hidden_states_p, self.repeat_group)

            # For B
            B_p = rearrange(B_p, "l (group_size ssm_size) -> group_size l ssm_size", group_size=xb_group)
            B_p = repeat_kv(B_p, self.repeat_group)

            # 3. State Space Model sequence transformation
            initial_states = None
            if has_initial_states_p is not None and prep_initial_states:
                kernel_ssm_indices = state_indices_tensor_p
                if is_mamba_cache_all:
                    kernel_ssm_indices = state_indices_tensor_p.gather(
                        1, block_idx_last_computed_token_p.unsqueeze(1)
                    ).squeeze(1)
                # Ensure initial_states has consistent dtype
                initial_states = torch.where(
                    has_initial_states_p[:, None, None, None],
                    ssm_state[kernel_ssm_indices],
                    0,
                )

            # NOTE: final output is an in-place update of out tensor
            # Reshape for varlen function: expects (seqlen, nheads, headdim)
            hidden_states_p_reshaped = rearrange(
                hidden_states_p, 
                "rep_group_size seq_len d -> seq_len rep_group_size d", 
                rep_group_size=self.num_heads // self.tp_size
            )
            B_p_reshaped = rearrange(
                B_p, 
                "rep_group_size seq_len d -> seq_len rep_group_size d", 
                rep_group_size=self.num_heads // self.tp_size
            )
            C_p_reshaped = rearrange(
                C_p, 
                "seq_len (h d) -> seq_len h d", 
                h=self.num_heads // self.tp_size
            )

            varlen_states = mamba_chunk_scan_combined_varlen(
                hidden_states_p_reshaped,
                dt_p,
                self.A,
                B_p_reshaped,
                C_p_reshaped,
                chunk_size,
                cu_seqlens=query_start_loc_p,
                cu_chunk_seqlens=cu_chunk_seqlen_p,
                last_chunk_indices=last_chunk_indices_p,
                seq_idx=seq_idx_p,
                out=preallocated_ssm_out_p.view(num_prefill_tokens, -1, self.head_dim),
                D=self.D,
                z=None,
                dt_bias=self.dt_bias,
                initial_states=initial_states,
                dt_softplus=True,
                dt_limit=(0.0, float("inf")),
                return_intermediate_states=is_mamba_cache_all,
                state_dtype=ssm_state.dtype,
            )

            if is_mamba_cache_all:
                # The chunk_stride is the number of chunks per mamba block
                # e.g., if mamba_block_size = 512 and chunk_size = 256,
                # then chunk_stride = 2
                chunk_stride = mamba_block_size // chunk_size

                # Save state for sequences with more than just final state
                for seq_idx in range(num_prefills):
                    # Block index for the first scheduled token
                    block_idx_first_scheduled_token = block_idx_first_scheduled_token_p[
                        seq_idx
                    ]

                    # Block index for the last scheduled token
                    block_idx_last_scheduled_token = block_idx_last_scheduled_token_p[
                        seq_idx
                    ]

                    # Number of blocks that need to be written
                    n_blocks_to_fill = (
                        block_idx_last_scheduled_token - block_idx_first_scheduled_token
                    )

                    # Skip sequences that don't have any blocks to fill
                    if n_blocks_to_fill == 0:
                        continue

                    # Look up cache blocks to fill
                    cache_blocks_to_fill = state_indices_tensor_p[
                        seq_idx,
                        block_idx_first_scheduled_token:block_idx_last_scheduled_token,
                    ]

                    # First chunk index for this sequence
                    if seq_idx == 0:
                        first_chunk = 0
                    else:
                        first_chunk = 1 + last_chunk_indices_p[seq_idx - 1]

                    # First chunk that is aligned on the mamba block boundary
                    first_aligned_chunk = first_chunk + chunk_stride - 1

                    # Calculate the number of computed tokens that were not
                    # already cached
                    num_unaligned_computed_tokens = (
                        num_computed_tokens_p[seq_idx] % mamba_block_size
                    )

                    if num_unaligned_computed_tokens > 0:
                        # If the number of computed tokens is not block aligned,
                        # then we need to shift the index accordingly
                        first_aligned_chunk -= (
                            num_unaligned_computed_tokens // chunk_size
                        )

                    # Get states to write
                    from_where = varlen_states[
                        first_aligned_chunk : first_aligned_chunk
                        + n_blocks_to_fill * chunk_stride : chunk_stride
                    ]

                    # Write the states
                    ssm_state[cache_blocks_to_fill] = from_where

                # For all seqs, store the last state (note: might be partial):
                ssm_state[
                    state_indices_tensor_p.gather(
                        1, block_idx_last_scheduled_token_p.unsqueeze(1)
                    ).squeeze(1)
                ] = varlen_states[last_chunk_indices_p]

            else:
                # update ssm states
                # - varlen state is a (num_prefills, nheads, headdim, dstate)
                #   tensor
                ssm_state[state_indices_tensor_p] = varlen_states.to(ssm_state.dtype)

        # Process decode requests
        if has_decode:
            if is_mamba_cache_all:
                state_indices_tensor_d_input = state_indices_tensor_d.gather(
                    1, block_idx_last_computed_token_d.unsqueeze(1)
                ).squeeze(1)
                state_indices_tensor_d_output = state_indices_tensor_d.gather(
                    1, block_idx_last_scheduled_token_d.unsqueeze(1)
                ).squeeze(1)
                # for decode:
                #   block_idx_first_scheduled_token_d ==
                #       block_idx_last_scheduled_token_d
                # at block boundaries:
                #   block_idx_first_scheduled_token_d >
                #       block_idx_last_computed_token_d
            else:
                # Without caching, read and write in-place to the same blocks:
                state_indices_tensor_d_input = state_indices_tensor_d
                state_indices_tensor_d_output = state_indices_tensor_d

            # 2. Convolution sequence transformation
            hidden_states_B_C_d = causal_conv1d_update(
                hidden_states_B_C_d,
                conv_state,
                self.conv_weights,
                self.conv1d.bias,
                self.activation,
                conv_state_indices=state_indices_tensor_d,
                block_idx_last_scheduled_token=block_idx_last_scheduled_token_d,
                initial_state_idx=block_idx_last_computed_token_d,
            )

            hidden_states_d, B_d, C_d = self.split_hidden_states_B_C_fn(
                hidden_states_B_C_d)

            xb_group = self.xb_size // (self.ssm_state_size * self.tp_size)

            hidden_states_d = rearrange(hidden_states_d, "l (group_size ssm_size) -> group_size l ssm_size", group_size=xb_group)
            hidden_states_d = repeat_kv(hidden_states_d, self.repeat_group)

            # For B
            B_d = rearrange(B_d, "l (group_size ssm_size) -> group_size l ssm_size", group_size=xb_group)
            B_d = repeat_kv(B_d, self.repeat_group)

            # 3. State Space Model sequence transformation
            n_groups = self.n_groups // self.tp_size
            A_d = (
                self.A[:, None, ...][:, :, None]
                .expand(-1, self.head_dim, self.ssm_state_size)
                .to(dtype=torch.float32)
            )
            dt_d = dt_d[:, :, None].expand(-1, -1, self.head_dim)
            dt_bias = self.dt_bias[:, None, ...].expand(-1, self.head_dim)
            D_d = self.D[:, None, ...].expand(-1, self.head_dim)
            B_d = B_d.permute(1, 0, 2).contiguous()
            hidden_states_d = hidden_states_d.permute(1, 0, 2).contiguous()
            C_d = C_d.view(-1, n_groups, C_d.shape[1] // n_groups)
            hidden_states_reshaped_d = hidden_states_d.view(
                -1, self.num_heads // self.tp_size, self.head_dim)

            # - the hidden is reshaped into (bs, num_heads, head_dim)
            # - mamba_cache_params.ssm_state's slots will be selected
            #   using state_indices_tensor_d
            # NOTE: final output is an in-place update of out tensor
            selective_state_update(
                ssm_state,
                hidden_states_reshaped_d,
                dt_d,
                A_d,
                B_d,
                C_d,
                D_d,
                z=None,
                dt_bias=dt_bias,
                dt_softplus=True,
                state_batch_indices=state_indices_tensor_d_input,
                dst_state_batch_indices=state_indices_tensor_d_output,
                out=preallocated_ssm_out_d.view(num_decodes, -1,
                                                self.head_dim),
            )



    def get_state_shape(self) -> tuple[tuple[int, int], tuple[int, int, int]]:
        return PrimedStateShapeCalculator.mamba2_state_shape(
            intermediate_size=self.intermediate_size,
            tp_world_size=get_tensor_model_parallel_world_size(),
            n_groups=self.n_groups,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            state_size=self.ssm_state_size,
            conv_kernel=self.conv_kernel_size,
            conv_dim=self.conv_dim,
        )

    def get_state_dtype(self) -> tuple[torch.dtype, torch.dtype]:
        assert self.model_config is not None
        assert self.cache_config is not None
        return MambaStateDtypeCalculator.mamba2_state_dtype(
            self.model_config.dtype,
            self.cache_config.mamba_cache_dtype,
            self.cache_config.mamba_ssm_cache_dtype,
        )


    @property
    def mamba_type(self) -> str:
        return "mamba2"


def mamba_mixer2(
    projected_states: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
    positions: torch.Tensor | None = None,
) -> None:
    forward_context: ForwardContext = get_forward_context()
    self = forward_context.no_compile_layers[layer_name]
    self.conv_ssm_forward(projected_states=projected_states,
                          output=output,
                          positions=positions)

def mamba_mixer2_fake(
    projected_states: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
    positions: torch.Tensor | None = None,
) -> None:
    return

direct_register_custom_op(
    op_name="mamba_mixer2_primed",
    op_func=mamba_mixer2,
    mutates_args=["output"],
    fake_impl=mamba_mixer2_fake
)
