"""Cache implementations for Hybrid models with multiple layer types.

This module provides cache classes for storing and retrieving hidden states
during autoregressive generation for various sequence mixing layers.

Each layer type has its own cache class (Mamba2Cache, BMojoCache, etc.) that
manages layer-specific states. The HybridCache class provides a unified interface
that coordinates all cache types within a single model.
"""

import gc
from typing import Any, Dict, Optional, Tuple

import torch
from transformers.cache_utils import Cache


class Mamba2Cache:
    """Cache for Mamba2 layer.
    
    Attributes:
        cache_aprods: If True, store cumulative A products (log_aprods) for PICASO composition.
        log_aprods: Dict mapping layer_idx to log(A_prod) tensors for state composition.
    """

    def __init__(
        self,
        parent_cache,
        max_sequence_length: Optional[int] = None,
        max_batch_size: Optional[int] = None,
    ):
        self.parent = parent_cache

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

    @property
    def seqlen_offset(self) -> int:
        """
        Expose parent's global seqlen_offset for Mamba2 to read
        """
        return self.parent.seqlen_offset


class BMojoCache:
    """Cache for B'MOJO-F layer."""

    def __init__(self, parent_cache):
        self.parent = parent_cache
        self.max_sequence_length = parent_cache.config.max_position_embeddings
        self.max_batch_size = getattr(parent_cache.config, "batch_size", 1)

    @property
    def key_value_memory_dict(self) -> Dict[int, Any]:
        return self.parent._bmojo_f_memory_dict

    @key_value_memory_dict.setter
    def key_value_memory_dict(self, value: Dict[int, Any]):
        self.parent._bmojo_f_memory_dict = value

    @property
    def seqlen_offset(self) -> int:
        """Expose parent's global seqlen_offset for BMojo attention"""
        return self.parent.seqlen_offset

    @property
    def batch_size_offset(self) -> int:
        """Expose batch_size_offset for batched inference"""
        # If parent doesn't have it, default to 0
        return getattr(self.parent, "batch_size_offset", 0)


class GatedDeltaNetCache:
    """Cache for Gated DeltaNet layer."""

    def __init__(
        self,
        parent_cache,
        max_sequence_length: Optional[int] = None,
        max_batch_size: Optional[int] = None,
    ):
        self.parent = parent_cache

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
            offset: Sequence length offset (not used here, offset updated globally in model forward). Kept for BC.
        """
        if conv_state is not None:
            self.key_value_memory_dict[layer_idx] = {
                "recurrent_state": recurrent_state,
                "conv_state": conv_state,
            }
        else:
            self.key_value_memory_dict[layer_idx] = {"recurrent_state": recurrent_state}


class GatedKalmaNetCache:
    """Cache for Gated KalmaNet layer."""

    def __init__(
        self,
        parent_cache,
        max_sequence_length: Optional[int] = None,
        max_batch_size: Optional[int] = None,
    ):
        self.parent = parent_cache

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
            offset: Sequence length offset (not used here, offset updated globally in model forward). Kept for BC.
        """
        self.key_value_memory_dict[layer_idx] = {
            "recurrent_state": recurrent_state,
            "conv_state": conv_state,
        }


class HybridCache(Cache):
    """
    Unified cache for Hybrid models with heterogeneous layer types.
    
    Manages separate memory dictionaries for each layer type (Attention, Mamba2,
    B'MOJO-F, GDN, GKA) and provides specialized interfaces for each.
    
    The cache maintains a global sequence offset that is shared across all layers
    and updated after each forward pass via update_offset().
    
    Attributes:
        config: Model configuration
        seqlen_offset: Global sequence position tracker
        bmojo_f: BMojoCache interface for B'MOJO-F layers
        mamba2: Mamba2Cache interface for Mamba2 layers
        gated_deltanet: GatedDeltaNetCache interface for GDN layers
        gated_kalmanet: GatedKalmaNetCache interface for GKA layers
    """

    def __init__(self, config):
        # Initialize with an empty layers list to satisfy the new Cache requirements
        super().__init__(layers=[])
        
        self.config = config

        # Global sequence length offset. Shared across all layers
        self.seqlen_offset = 0

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

    def update_offset(self, seq_len: int):
        """
        Update global sequence length offset after processing tokens.
        
        Args:
            seq_len: Number of new tokens processed
        """
        self.seqlen_offset += seq_len

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict] = None,  # Kept for compatibility purposes
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Standard Cache interface for attention layers"""
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
        """Returns the sequence length of the cached states
        
        For attention layers, returns the actual cached sequence length from the key tensor.
        For all other layer types (GDN, GKA, Mamba2, BMOJO), returns the global offset.
        
        Args:
            layer_idx: Layer index to check (only used for attention layers)
            
        Returns:
            Sequence length as an integer
        """
        # Check if this is a standard attention layer with cached keys
        if layer_idx in self._attention_memory_dict:
            cache_entry = self._attention_memory_dict[layer_idx]
            if isinstance(cache_entry, dict) and "keys" in cache_entry:
                keys = cache_entry["keys"]
                if isinstance(keys, torch.Tensor):
                    return keys.size(-2)

        # For all non-attention layers, return the global offset
        # This includes: GDN, GKA, Mamba2, and B'MOJO
        return self.seqlen_offset

    def get_max_length(self) -> int:
        """Get maximum allowed sequence length"""
        return self.config.max_position_embeddings

    def check_seqlen_match(self, layer_idx: Optional[int] = None) -> bool:
        """Check whether KV cache sequence length matches the global sequence offset.
        
        Args:
            layer_idx: Specific attention layer to check. If None, uses first available.
            
        Returns:
            True if KV cache length matches seqlen_offset, False otherwise.
        """
        if layer_idx is None:
            if not self._attention_memory_dict:
                kv_seqlen = 0
            else:
                first_layer = min(self._attention_memory_dict.keys())
                kv_seqlen = self.get_seq_length(first_layer)
        else:
            kv_seqlen = self.get_seq_length(layer_idx)
        
        return kv_seqlen == self.seqlen_offset

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

        # Reset offsets
        self.seqlen_offset = 0

    def __del__(self):
        try:
            self.clear()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

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

        # Copy offset
        new_cache.seqlen_offset = self.seqlen_offset

        return new_cache
