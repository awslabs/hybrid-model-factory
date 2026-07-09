"""L2A-specific KV cache for inference.

The L2A architecture has two attention modules per layer:
  - SWA (Local Attention): sliding window, bounded KV cache
  - Global Attention: full-context KV cache

During decode, every token's global K/V must be stored regardless of the router's
decision, because future tokens that invoke Global Attention need to attend over
the full history.

This cache extends the HybridCache from HMF, reusing:
  - _attention_memory_dict + update() for the Global Attention full-context cache
  - update_offset / seqlen_offset / get_seq_length for offset tracking
  - clear / __del__ / copy for lifecycle management

And adds:
  - _swa_memory_dict + update_swa() for the SWA cache
"""

from typing import Any, Dict, Optional, Tuple

import torch

from hmf.model.hybrid_zoo.models.cache import HybridCache


class L2ACache(HybridCache):
    """
    KV cache for L2A models with dual attention (SWA + Global) per layer.

    Inherits from HybridCache:
      - Uses _attention_memory_dict (via update()) for Global Attention KV (full context)
      - Uses seqlen_offset / update_offset / get_seq_length for position tracking
      - Inherits clear / __del__ / copy for cleanup

    Adds:
      - _swa_memory_dict for SWA (local) attention KV, trimmed to sliding_window

    Args:
        config: Model configuration (needs sliding_window, max_position_embeddings)
    """

    def __init__(self, config):
        super().__init__(config)

        self.sliding_window = getattr(config, "sliding_window", 4096)

        # SWA-specific KV store (bounded by sliding window)
        # Each entry: {layer_idx: {"keys": Tensor, "values": Tensor}}
        self._swa_memory_dict: Dict[int, Dict[str, torch.Tensor]] = {}

    # SWA cache
    def update_swa(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update KV cache for the SWA (local) attention branch.

        Concatenates new KV states with existing cache and trims to
        sliding_window length to bound memory usage.

        Args:
            key_states: [batch, num_kv_heads, seq_len, head_dim]
            value_states: [batch, num_kv_heads, seq_len, head_dim]
            layer_idx: Layer index
            cache_kwargs: Optional dict (kept for API compatibility)

        Returns:
            Tuple of (key_states, value_states) with cached history concatenated.
        """
        if layer_idx not in self._swa_memory_dict:
            self._swa_memory_dict[layer_idx] = {
                "keys": key_states,
                "values": value_states,
            }
        else:
            past = self._swa_memory_dict[layer_idx]
            key_states = torch.cat([past["keys"], key_states], dim=-2)
            value_states = torch.cat([past["values"], value_states], dim=-2)

            # Trim to sliding window size to bound memory
            if key_states.shape[-2] > self.sliding_window:
                key_states = key_states[:, :, -self.sliding_window:, :]
                value_states = value_states[:, :, -self.sliding_window:, :]

            self._swa_memory_dict[layer_idx] = {
                "keys": key_states,
                "values": value_states,
            }

        return key_states, value_states

    # Global Attention cache; delegates to HybridCache._attention_memory_dict
    def update_global(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update KV cache for the Global Attention branch.

        Delegates to HybridCache.update() which stores in _attention_memory_dict
        with full context (no windowing). Future tokens that invoke Global Attention
        need the complete history.

        Args:
            key_states: [batch, num_kv_heads, seq_len, head_dim]
            value_states: [batch, num_kv_heads, seq_len, head_dim]
            layer_idx: Layer index
            cache_kwargs: Optional dict (kept for API compatibility)

        Returns:
            Tuple of (key_states, value_states) with cached history concatenated.
        """
        return super().update(key_states, value_states, layer_idx, cache_kwargs)

    # Override clear to also clean SWA dict
    def clear(self):
        """Clear all cached tensors and reset state."""
        # Clear SWA memory
        for layer_idx in list(self._swa_memory_dict.keys()):
            entry = self._swa_memory_dict[layer_idx]
            if isinstance(entry, dict):
                for key in list(entry.keys()):
                    val = entry[key]
                    if torch.is_tensor(val):
                        val.detach_()
        self._swa_memory_dict.clear()

        # Delegate the rest to HybridCache (clears attention, mamba, etc.)
        super().clear()

    # Override copy to also clone SWA dict
    def copy(self) -> "L2ACache":
        """Create a deep copy of this cache."""
        import copy as copy_module

        new_cache = L2ACache(copy_module.deepcopy(self.config))

        # Copy parent state
        for layer_idx, entry in self._attention_memory_dict.items():
            new_cache._attention_memory_dict[layer_idx] = {
                k: v.clone() if torch.is_tensor(v) else v
                for k, v in entry.items()
            }

        # Copy SWA state
        for layer_idx, entry in self._swa_memory_dict.items():
            new_cache._swa_memory_dict[layer_idx] = {
                k: v.clone() if torch.is_tensor(v) else v
                for k, v in entry.items()
            }

        # Copy offset
        new_cache.seqlen_offset = self.seqlen_offset

        return new_cache
