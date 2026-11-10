# Copyright 2026 the HuggingFace Team. All rights reserved.
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
"""HybridGemma4 text model configuration"""

from huggingface_hub.dataclasses import strict
from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig

from ..utils import hybrid_layer_types


@strict
class HybridGemma4TextConfig(Gemma4TextConfig):
    r"""
    Configures a hybrid Gemma 4 text model. It extends Gemma4TextConfig with a hybrid_override_pattern that chooses
    each decoder layer's class (see LAYER_TYPE_PATTERNS in models/utils.py).

    hybrid_override_pattern (`str`, *optional*):
        A string denoting the hybridization pattern, e.g., '*-GKA-*-*-SWA'.
    base_layer_types (`list[str]`, *optional*):
        The attention kind of each layer in the base Gemma 4 model ("sliding_attention" or "full_attention"). When
        it is not given, the config takes it from layer_types, which a base Gemma 4 config holds. The base kinds
        decide each layer's head size (per_layer_config), its RoPE, and which layers share keys and values, while
        layer_types gets transformers' names for the hybrid layers (see hybrid_layer_types).
    """

    model_type = "hybrid_gemma4_text"
    # Gemma 4 names its activation hidden_activation, while HMF's Mamba2 (and B'MOJO with Mamba2 as its SSM) reads
    # the hidden_act that the other base models have
    attribute_map = {"hidden_act": "hidden_activation"}

    hybrid_override_pattern: str | None = None
    base_layer_types: list[str] | None = None

    def __post_init__(self, **kwargs):
        if self.base_layer_types is None:
            self.base_layer_types = self.layer_types
        unknown = sorted(set(self.base_layer_types or []) - {"sliding_attention", "full_attention"})
        if unknown:
            raise ValueError(
                f"base_layer_types must hold only 'sliding_attention' and 'full_attention', got {unknown}. A hybrid "
                "config saves base_layer_types, so pass it along with layer_types."
            )

        # Gemma4TextConfig derives the larger head size of the full attention layers (per_layer_config) from
        # layer_types, so it gets the base model's layout here. It also fills in the default layout when there is
        # none, and makes the last layer full attention.
        self.layer_types = list(self.base_layer_types) if self.base_layer_types is not None else None
        super().__post_init__(**kwargs)
        self.base_layer_types = list(self.layer_types)

        if self.hybrid_override_pattern is not None:
            self.layer_types = hybrid_layer_types(self.hybrid_override_pattern, self.base_layer_types)

    def kv_sharing_sources(self) -> dict[str, int]:
        """
        Returns the layer whose keys and values each kind of attention ("sliding_attention" or "full_attention")
        shares with the layers from first_kv_shared_layer on, which is the last layer of that kind before them. It
        returns an empty dict when no layers share.
        """
        first_shared = self.first_kv_shared_layer
        if first_shared >= self.num_hidden_layers:
            return {}
        earlier = self.base_layer_types[:first_shared]
        return {kind: len(earlier) - 1 - earlier[::-1].index(kind) for kind in set(earlier)}

    @property
    def first_kv_shared_layer(self) -> int:
        """Returns the first layer that reuses an earlier layer's keys and values, or num_hidden_layers if none do."""
        return self.num_hidden_layers - (self.num_kv_shared_layers or 0)

    def kinds_read_from_sources(self) -> set[str]:
        """
        Returns the attention kinds whose shared keys and values a layer from first_kv_shared_layer on still reads.
        A '*' layer there reads them, and so does an SWA layer, which only limits them to its window, while a GKA,
        GDN, or Mamba2 layer there has key and value projections of its own.
        """
        entries = (self.hybrid_override_pattern or "").split("-")
        return {
            self.base_layer_types[i]
            for i in range(self.first_kv_shared_layer, self.num_hidden_layers)
            if self.hybrid_override_pattern is None or entries[i] in ("*", "SWA")
        }

    def keeps_source_keys_and_values(self, layer_idx: int) -> bool:
        """
        Returns whether the layer at layer_idx is a hybrid sequence mixer (GKA, GDN, or Mamba2) at a source layer
        whose shared keys and values later layers still read (see kinds_read_from_sources). Such a layer keeps the
        source's key and value projections to compute them, while its mixer computes its output.
        """
        if self.hybrid_override_pattern is None:
            return False
        entry = self.hybrid_override_pattern.split("-")[layer_idx]
        kind = self.base_layer_types[layer_idx]
        return (
            entry not in ("*", "SWA")
            and self.kv_sharing_sources().get(kind) == layer_idx
            and kind in self.kinds_read_from_sources()
        )

__all__ = ["HybridGemma4TextConfig"]
