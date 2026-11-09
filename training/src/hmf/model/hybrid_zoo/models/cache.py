"""Cache implementations for Hybrid models with multiple layer types.

This module provides cache classes for storing and retrieving hidden states
during autoregressive generation for various sequence mixing layers.

Each layer type has its own cache class (Mamba2Cache, BMojoCache, etc.) that
manages layer-specific states. The HybridCache class provides a unified interface
that coordinates all cache types within a single model.
"""

import weakref
from typing import Any, Dict, Optional, Tuple

import torch
from transformers.cache_utils import Cache


# The layer types whose tokens HybridCache counts, in the order in which get_seq_length searches them for a layer.
# A B'MOJO layer and its SSM keep separate counts under the same layer index, and the two always agree.
_COUNTED_PARTS = ("attention", "bmojo_f", "mamba2", "gated_deltanet", "gated_kalmanet")


class _HybridCachePart:
    """
    Base class of the per-layer-type caches (Mamba2Cache, BMojoCache, GatedDeltaNetCache, and GatedKalmaNetCache),
    which store their states in the HybridCache that creates them and access it through their parent property. They
    keep only a weak reference to it, because a strong one would form a reference cycle with the HybridCache's
    references to them. Python frees such a cycle only when its garbage collector runs, so a finished cache's GPU
    memory would stay allocated until then.
    """

    def __init__(self, parent_cache):
        self._parent = weakref.ref(parent_cache)

    @property
    def parent(self) -> "HybridCache":
        parent = self._parent()
        if parent is None:
            raise ReferenceError(f"the HybridCache that created this {type(self).__name__} has been freed")
        return parent

    def seen_tokens(self, layer_idx: int) -> int:
        """Returns the number of tokens that this layer type has processed at layer_idx, or 0 before any update."""
        return self.parent._seen_tokens[self.part].get(layer_idx, 0)

    def record_tokens(self, layer_idx: int, count: int):
        """Adds count to the number of tokens that this layer type has processed at layer_idx."""
        self.parent._record_tokens(self.part, layer_idx, count)


class Mamba2Cache(_HybridCachePart):
    """Cache for Mamba2 layer.

    Attributes:
        cache_aprods: If True, store cumulative A products (log_aprods) for PICASO composition.
        log_aprods: Dict mapping layer_idx to log(A_prod) tensors for state composition.
    """

    part = "mamba2"

    def __init__(
        self,
        parent_cache,
        max_sequence_length: Optional[int] = None,
        max_batch_size: Optional[int] = None,
    ):
        super().__init__(parent_cache)

        # Enable caching of cumulative A products for PICASO state composition
        self.cache_aprods = True
        self.log_aprods: Dict[int, Any] = {}

        if max_sequence_length is None:
            self.max_sequence_length = parent_cache.config.max_position_embeddings
        else:
            self.max_sequence_length = max_sequence_length

        if max_batch_size is None:
            self.max_batch_size = getattr(parent_cache.config, "batch_size", 1)
        else:
            self.max_batch_size = max_batch_size

    @property
    def key_value_memory_dict(self) -> Dict[int, Any]:
        return self.parent._mamba_memory_dict

    @key_value_memory_dict.setter
    def key_value_memory_dict(self, value: Dict[int, Any]):
        self.parent._mamba_memory_dict = value



class BMojoCache(_HybridCachePart):
    """Cache for B'MOJO-F layer."""

    part = "bmojo_f"

    @property
    def key_value_memory_dict(self) -> Dict[int, Any]:
        return self.parent._bmojo_f_memory_dict

    @key_value_memory_dict.setter
    def key_value_memory_dict(self, value: Dict[int, Any]):
        self.parent._bmojo_f_memory_dict = value


class GatedDeltaNetCache(_HybridCachePart):
    """Cache for Gated DeltaNet layer."""

    part = "gated_deltanet"

    def __init__(
        self,
        parent_cache,
        max_sequence_length: Optional[int] = None,
        max_batch_size: Optional[int] = None,
    ):
        super().__init__(parent_cache)

        if max_sequence_length is None:
            self.max_sequence_length = parent_cache.config.max_position_embeddings
        else:
            self.max_sequence_length = max_sequence_length

        if max_batch_size is None:
            self.max_batch_size = getattr(parent_cache.config, "batch_size", 1)
        else:
            self.max_batch_size = max_batch_size

    @property
    def key_value_memory_dict(self) -> Dict[int, Any]:
        return self.parent._gated_deltanet_memory_dict

    @key_value_memory_dict.setter
    def key_value_memory_dict(self, value: Dict[int, Any]):
        self.parent._gated_deltanet_memory_dict = value

    def __getitem__(self, layer_idx: int):
        """Get cache state for a specific layer"""
        return self.key_value_memory_dict.get(layer_idx, None)

    def update(
        self,
        recurrent_state: Tuple,
        conv_state: Optional[Tuple],
        layer_idx: int,
        offset: int,
    ):
        """Update cache state for a specific layer
        
        Args:
            recurrent_state: The recurrent hidden state
            conv_state: Tuple of (conv_state_q, conv_state_k, conv_state_v) if using short conv, else None
            layer_idx: Index of the layer
            offset: The number of new tokens, which the cache adds to the layer's count (see get_seq_length)
        """
        self.record_tokens(layer_idx, offset)
        if conv_state is not None:
            self.key_value_memory_dict[layer_idx] = {
                "recurrent_state": recurrent_state,
                "conv_state": conv_state,
            }
        else:
            self.key_value_memory_dict[layer_idx] = {"recurrent_state": recurrent_state}


class GatedKalmaNetCache(_HybridCachePart):
    """Cache for Gated KalmaNet layer."""

    part = "gated_kalmanet"

    def __init__(
        self,
        parent_cache,
        max_sequence_length: Optional[int] = None,
        max_batch_size: Optional[int] = None,
    ):
        super().__init__(parent_cache)

        if max_sequence_length is None:
            self.max_sequence_length = parent_cache.config.max_position_embeddings
        else:
            self.max_sequence_length = max_sequence_length

        if max_batch_size is None:
            self.max_batch_size = getattr(parent_cache.config, "batch_size", 1)
        else:
            self.max_batch_size = max_batch_size

    @property
    def key_value_memory_dict(self) -> Dict[int, Any]:
        return self.parent._gated_kalmanet_memory_dict

    @key_value_memory_dict.setter
    def key_value_memory_dict(self, value: Dict[int, Any]):
        self.parent._gated_kalmanet_memory_dict = value

    def __getitem__(self, layer_idx: int):
        """Get cache state for a specific layer"""
        return self.key_value_memory_dict.get(layer_idx, None)

    def update(
        self, recurrent_state: Tuple, conv_state: Tuple, layer_idx: int, offset: int
    ):
        """Update cache state for a specific layer

        Args:
            recurrent_state: Tuple of (h_kk, h_kv) - the recurrent hidden states
            conv_state: Tuple of (conv_state_q, conv_state_k, conv_state_v) - convolution states
            layer_idx: Index of the layer
            offset: The number of new tokens, which the cache adds to the layer's count (see get_seq_length)
        """
        self.record_tokens(layer_idx, offset)
        self.key_value_memory_dict[layer_idx] = {
            "recurrent_state": recurrent_state,
            "conv_state": conv_state,
        }


class HybridCache(Cache):
    """
    Unified cache for Hybrid models with heterogeneous layer types.
    
    Manages separate memory dictionaries for each layer type (Attention, Mamba2,
    B'MOJO-F, GDN, GKA) and provides specialized interfaces for each.

    Each layer adds its new tokens to its own count when it updates its state, and get_seq_length returns that
    count, which is also the position of the next token. The count therefore needs no updates from outside the
    cache, and it is also correct when the inner model runs without the ForCausalLM around it. Cache composition
    sets the counts of the caches it builds with set_seq_length, since it can keep fewer tokens in the attention
    cache than it has read.

    Attributes:
        config: Model configuration
        bmojo_f: BMojoCache interface for B'MOJO-F layers
        mamba2: Mamba2Cache interface for Mamba2 layers
        gated_deltanet: GatedDeltaNetCache interface for GDN layers
        gated_kalmanet: GatedKalmaNetCache interface for GKA layers
    """

    def __init__(self, config):
        # Initialize with an empty layers list to satisfy the new Cache requirements
        super().__init__(layers=[])
        
        self.config = config

        # The number of tokens each layer has processed, for each layer type
        self._seen_tokens: Dict[str, Dict[int, int]] = {part: {} for part in _COUNTED_PARTS}

        # Separate dictionaries for each layer type
        self._attention_memory_dict = {}
        self._bmojo_f_memory_dict = {}
        self._mamba_memory_dict = {}
        self._gated_deltanet_memory_dict = {}
        self._gated_kalmanet_memory_dict = {}

        # Create specialized interfaces for each SSM
        self.bmojo_f = BMojoCache(self)
        self.mamba2 = Mamba2Cache(self)
        self.gated_deltanet = GatedDeltaNetCache(self)
        self.gated_kalmanet = GatedKalmaNetCache(self)

    @property
    def key_value_memory_dict(self) -> Dict[int, Any]:
        """Standard attention cache storage"""
        return self._attention_memory_dict

    @key_value_memory_dict.setter
    def key_value_memory_dict(self, value: Dict[int, Any]):
        self._attention_memory_dict = value

    def _record_tokens(self, part: str, layer_idx: int, count: int):
        counts = self._seen_tokens[part]
        counts[layer_idx] = counts.get(layer_idx, 0) + count

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict] = None,  # Kept for compatibility purposes
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Standard Cache interface for attention layers"""
        self._record_tokens("attention", layer_idx, key_states.shape[-2])
        if layer_idx not in self._attention_memory_dict:
            self._attention_memory_dict[layer_idx] = {
                "keys": key_states,
                "values": value_states,
            }
        else:
            past_key_value = self._attention_memory_dict[layer_idx]
            if isinstance(past_key_value["keys"], torch.Tensor):
                key_states = torch.cat([past_key_value["keys"], key_states], dim=-2)
                value_states = torch.cat(
                    [past_key_value["values"], value_states], dim=-2
                )

            self._attention_memory_dict[layer_idx]["keys"] = key_states
            self._attention_memory_dict[layer_idx]["values"] = value_states

        return key_states, value_states

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """
        Returns the number of tokens that the layer at layer_idx has processed, which is also the position of the
        next token, and 0 before its first update. The count includes every token that the layer has read, whatever
        its type, so it can be larger than the attention cache when cache composition keeps fewer tokens there (see
        cached_attention_length).
        """
        layer_idx = 0 if layer_idx is None else layer_idx
        for part in _COUNTED_PARTS:
            if layer_idx in self._seen_tokens[part]:
                return self._seen_tokens[part][layer_idx]
        return 0

    def set_seq_length(self, length: int):
        """
        Sets the count of every layer that holds a state to length. Cache composition uses it for the caches it
        builds, which count every token that composition read, even where the attention cache keeps fewer of them
        (see get_seq_length).
        """
        stores = {
            "attention": self._attention_memory_dict,
            "bmojo_f": self._bmojo_f_memory_dict,
            "mamba2": self._mamba_memory_dict,
            "gated_deltanet": self._gated_deltanet_memory_dict,
            "gated_kalmanet": self._gated_kalmanet_memory_dict,
        }
        for part, store in stores.items():
            self._seen_tokens[part] = {layer_idx: length for layer_idx in set(store) | set(self._seen_tokens[part])}

    def get_mask_sizes(self, query_length: int, layer_idx: int = 0) -> Tuple[int, int]:
        """
        Returns the key length and offset that the attention masks of transformers' create_causal_mask cover,
        which are the tokens already in the attention cache plus the query_length new ones, starting at 0.

        The base Cache reads these sizes from self.layers, which HybridCache leaves empty, so without this
        override the masks would cover only the new tokens. The padding of the cached tokens would then go
        unmasked during decoding, and a step of several tokens after a prefill would fall back to SDPA's
        top-left causal mask. Every attention layer caches the same number of tokens, so the first attention
        entry gives the cached length. layer_idx is not used, because the layer that create_causal_mask asks
        about may be a GDN or other non-attention layer.
        """
        return self.cached_attention_length() + query_length, 0

    def get_query_offset(self, layer_idx: int = 0) -> int:
        """
        Returns the position of the first new query among the keys that create_causal_mask covers, which is the
        number of tokens already in the attention cache (see get_mask_sizes, whose keys start at 0).
        """
        return self.cached_attention_length()

    def cached_attention_length(self) -> int:
        """Returns the number of tokens in the attention key and value cache, 0 before the first update."""
        for entry in self._attention_memory_dict.values():
            keys = entry.get("keys") if isinstance(entry, dict) else None
            if isinstance(keys, torch.Tensor):
                return keys.shape[-2]
        return 0

    def get_max_length(self) -> int:
        """Get maximum allowed sequence length"""
        return self.config.max_position_embeddings

    def clear(self):
        """Clear all cached tensors and reset state."""
        def _clear_tensor_dict(d: Dict):
            for layer_idx in list(d.keys()):
                entry = d[layer_idx]
                if isinstance(entry, dict):
                    for key in list(entry.keys()):
                        val = entry[key]
                        if torch.is_tensor(val):
                            val.detach_()
                        elif isinstance(val, tuple):
                            for item in val:
                                if torch.is_tensor(item):
                                    item.detach_()
                elif isinstance(entry, tuple):
                    for item in entry:
                        if torch.is_tensor(item):
                            item.detach_()
            d.clear()

        _clear_tensor_dict(self._attention_memory_dict)
        _clear_tensor_dict(self._bmojo_f_memory_dict)
        _clear_tensor_dict(self._mamba_memory_dict)
        _clear_tensor_dict(self._gated_deltanet_memory_dict)
        _clear_tensor_dict(self._gated_kalmanet_memory_dict)

        # Clear Mamba2 log_aprods
        for key in list(self.mamba2.log_aprods.keys()):
            if torch.is_tensor(self.mamba2.log_aprods[key]):
                self.mamba2.log_aprods[key].detach_()
        self.mamba2.log_aprods.clear()

        self._seen_tokens = {part: {} for part in _COUNTED_PARTS}

    def copy(self) -> "HybridCache":
        """Create a deep copy of this cache."""
        import copy as copy_module

        def _clone_entry(entry):
            if torch.is_tensor(entry):
                return entry.clone()
            elif isinstance(entry, dict):
                return {k: _clone_entry(v) for k, v in entry.items()}
            elif isinstance(entry, tuple):
                return tuple(_clone_entry(item) for item in entry)
            else:
                return copy_module.deepcopy(entry)

        new_cache = HybridCache(copy_module.deepcopy(self.config))

        # Copy all memory dicts
        for layer_idx, entry in self._attention_memory_dict.items():
            new_cache._attention_memory_dict[layer_idx] = _clone_entry(entry)
        for layer_idx, entry in self._bmojo_f_memory_dict.items():
            new_cache._bmojo_f_memory_dict[layer_idx] = _clone_entry(entry)
        for layer_idx, entry in self._mamba_memory_dict.items():
            new_cache._mamba_memory_dict[layer_idx] = _clone_entry(entry)
        for layer_idx, entry in self._gated_deltanet_memory_dict.items():
            new_cache._gated_deltanet_memory_dict[layer_idx] = _clone_entry(entry)
        for layer_idx, entry in self._gated_kalmanet_memory_dict.items():
            new_cache._gated_kalmanet_memory_dict[layer_idx] = _clone_entry(entry)

        # Copy Mamba2 log_aprods
        new_cache.mamba2.log_aprods = {
            k: v.clone() if torch.is_tensor(v) else copy_module.deepcopy(v)
            for k, v in self.mamba2.log_aprods.items()
        }
        new_cache.mamba2.cache_aprods = self.mamba2.cache_aprods

        new_cache._seen_tokens = {part: dict(counts) for part, counts in self._seen_tokens.items()}

        return new_cache
