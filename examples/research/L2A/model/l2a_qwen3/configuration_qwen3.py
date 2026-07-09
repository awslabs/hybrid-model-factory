# Copyright 2024 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""L2AQwen3 model configuration"""

from transformers.configuration_utils import PreTrainedConfig, layer_type_validation
from transformers.modeling_rope_utils import RopeParameters
from transformers.utils import logging


logger = logging.get_logger(__name__)


class L2AQwen3Config(PreTrainedConfig):
    r"""
    Configuration class for the memory-augmented (L2A) Qwen3 model. In addition to the standard Qwen3
    arguments it exposes the memory-augmentation controls:

    Args:
        l2a_initfrombase (`bool`, *optional*, defaults to `False`):
            Whether the l2a sublayers were initialized from their corresponding base layers.
        sig_input_type (`str`, *optional*, defaults to `"hidden_linearproj"`):
            Input used by the router / straight-through sigmoid gate.
        sigmoid_temp (`float`, *optional*, defaults to 1.0):
            Temperature of the straight-through sigmoid gate.
        sigmoid_linear (`bool`, *optional*, defaults to `True`):
            Whether the router uses a learned linear projection.
        sigmoid_threshold (`float`, *optional*, defaults to 0.5):
            Threshold for the hard decision of the straight-through sigmoid gate.
    """

    model_type = "l2a_qwen3"
    keys_to_ignore_at_inference = ["past_key_values"]

    # Default tensor parallel plan for base model `Qwen3`
    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.q_norm": "replicated_with_grad_allreduce",
        "layers.*.self_attn.k_norm": "replicated_with_grad_allreduce",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }

    def __init__(
        self,
        vocab_size: int | None = 151936,
        hidden_size: int | None = 4096,
        intermediate_size: int | None = 22016,
        num_hidden_layers: int | None = 32,
        num_attention_heads: int | None = 32,
        num_key_value_heads: int | None = 32,
        head_dim: int | None = 128,
        hidden_act: str | None = "silu",
        max_position_embeddings: int | None = 32768,
        initializer_range: float | None = 0.02,
        rms_norm_eps: float | None = 1e-6,
        use_cache: bool | None = True,
        tie_word_embeddings: bool | None = False,
        rope_parameters: RopeParameters | dict[str, RopeParameters] | None = None,
        attention_bias: bool | None = False,
        use_sliding_window: bool | None = False,
        sliding_window: int | None = 4095,
        max_window_layers: int | None = 28,
        layer_types: list[str] | None = None,
        attention_dropout: float | None = 0.0,
        pad_token_id: int | None = None,
        bos_token_id: int | None = None,
        eos_token_id: int | None = None,
        # L2A specific fields ---
        l2a_initfrombase: bool | None = False,
        sig_input_type: str | None = "hidden_linearproj",
        sigmoid_temp: float | None = 1.0,
        sigmoid_linear: bool | None = True,
        sigmoid_threshold: float | None = 0.5,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.use_sliding_window = use_sliding_window
        # NOTE: unlike stock Qwen3Config we keep `sliding_window` as provided (do not null it
        # when `use_sliding_window` is False). The l2a attention modules perform their own
        # per-layer gating and `Qwen3SWAttention` requires a concrete window size.
        self.sliding_window = sliding_window
        self.max_window_layers = max_window_layers

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout

        self.layer_types = layer_types
        if self.layer_types is None:
            self.layer_types = [
                "sliding_attention"
                if self.use_sliding_window
                and self.sliding_window is not None
                and i >= self.max_window_layers
                else "full_attention"
                for i in range(self.num_hidden_layers)
            ]
        layer_type_validation(self.layer_types, self.num_hidden_layers)

        # L2A specific fields
        self.l2a_initfrombase = l2a_initfrombase
        self.sig_input_type = sig_input_type
        self.sigmoid_temp = sigmoid_temp
        self.sigmoid_linear = sigmoid_linear
        self.sigmoid_threshold = sigmoid_threshold

        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.tie_word_embeddings = tie_word_embeddings
        self.rope_parameters = rope_parameters

        super().__init__(**kwargs)


__all__ = ["L2AQwen3Config"]
