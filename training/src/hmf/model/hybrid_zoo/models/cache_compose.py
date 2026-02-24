"""Cache composition utilities for Hybrid models.

This module provides functionality for composing SSM states from multiple
context chunks during prefill, enabling efficient long-context processing.

Composition strategies:
- soup: Simple averaging of SSM states across chunks
- fuse: Weighted combination using cumulative A products (Mamba2/DeltaNet)
        or summation of information matrices/vectors (GKA). Based on PICASO: https://arxiv.org/abs/2502.17605
- kv_only: Zero out SSM states, keep only KV cache

KV cache strategies:
- full: Retain full KV cache for all tokens
- concat: Concatenate KV from chunks with prefix masking
- sw: Sliding window (keep prefix + last chunk)
"""

from typing import Optional, Dict, Any, Tuple
import gc
import math

import torch
from transformers.utils import logging

from .cache import HybridCache

logger = logging.get_logger(__name__)


def wrap_for_composition(model, tokenizer):
    """Wrap model with cache composition capabilities."""
    if hasattr(model, 'has_cache_compose'):
        return model
    
    class CacheComposerModel(CacheComposerMixin, type(model)):
        pass
    
    model.__class__ = CacheComposerModel
    
    # Set newline_token_id if tokenizer provided
    model.config.newline_token_id = tokenizer.encode('\n', add_special_tokens=False)[0]
    
    return model


class CacheComposerMixin:
    """Mixin class that adds cache composition capabilities to a model.
    
    This mixin overrides prepare_inputs_for_generation to intercept the
    generation loop and perform chunked prefill with state composition.
    """
    
    has_cache_compose = True
    
    def _validate_model_kwargs(self, model_kwargs: Dict[str, Any]):
        """Handle custom kwargs for cache composition."""
        custom_kwargs = [
            "compose_type", "num_chunks", "query_len", 
            "min_prepend", 'prefix_input_ids', 'suffix_input_ids',
            "sequential_positions"
        ]
        filtered_kwargs = {k: v for k, v in model_kwargs.items() 
                          if k not in custom_kwargs}
        super()._validate_model_kwargs(filtered_kwargs)

    def prepare_inputs_for_generation(
        self,
        input_ids,
        next_sequence_length=None,
        past_key_values=None,
        use_cache=True,
        **kwargs,
    ):
        """Override to perform chunked prefill composition.
        
        In transformers v5.x, the generation loop splits into a _prefill() stage
        (called once with next_sequence_length=None) and a decode loop (called with
        next_sequence_length=1). Our composition runs during _prefill, populating
        the cache with all context tokens. After composition, we must tell the base
        prepare_inputs_for_generation to only forward the last token by setting
        next_sequence_length=1, since the cache already contains everything else.
        """
        # Initialize HybridCache if needed
        if use_cache and past_key_values is not None and not isinstance(past_key_values, HybridCache):
            del past_key_values
            gc.collect()
            torch.cuda.empty_cache()
            past_key_values = HybridCache(self.config)

        kwargs['use_cache'] = use_cache
        
        # Perform chunked prefill composition
        composed = kwargs.get("compose_type") is not None
        input_ids, past_key_values, kwargs = self.compose_cache(input_ids, past_key_values, kwargs)

        # After composition the cache is fully populated — only the last token
        # needs to go through the model forward. In v5.x _prefill() passes
        # next_sequence_length=None for the first call, but we need to override
        # that so the base class slices input_ids down to 1 token.
        if composed and past_key_values is not None and past_key_values.get_seq_length() > 0:
            next_sequence_length = 1

        return super().prepare_inputs_for_generation(
            input_ids,
            next_sequence_length=next_sequence_length,
            past_key_values=past_key_values,
            **kwargs,
        )

    def compose_cache(
        self, 
        input_ids: torch.Tensor, 
        past_key_values: Optional[HybridCache], 
        kwargs: Dict[str, Any]
    ) -> Tuple[torch.Tensor, Optional[HybridCache], Dict[str, Any]]:
        """Handle chunking and processing of context during prefill.
        
        Returns:
            Tuple of (input_ids, past_key_values, kwargs)
        """
        compose_type = kwargs.get("compose_type")
        
        # Early return: no composition needed
        if compose_type is None:
            return input_ids, past_key_values, kwargs
        
        kwargs['use_cache'] = True
        
        # Early return: cache already populated
        if past_key_values is not None and past_key_values.get_seq_length() > 0:
            input_ids = self._prepare_final_inputs(input_ids, past_key_values, kwargs)
            return input_ids, past_key_values, kwargs
        
        # === Main composition prefill logic ===
        config = self._parse_composition_config(kwargs)
        context_ids, query_ids = self._split_context_and_query(input_ids, config['query_len'])
        
        logger.info(
            f'Chunking with compose_type={compose_type}, '
            f'num_chunks={config["num_chunks"]}, '
            f'query_len={config["query_len"]}, '
            f'min_prepend={config["min_prepend"]}, '
            f'sequential_positions={config["sequential_positions"]}'
        )
        
        # Compose cache based on strategy
        if config['kv_strategy'] == 'full':
            self.prefix_ids, past_key_values = self._compose_fullkv_hybrid(
                context_ids, config['num_chunks'], strategy=config['ssm_strategy']
            )
        else:
            self.prefix_ids, past_key_values = self._compose_batch_hybrid(
                context_ids, config['num_chunks'], strategy=config['ssm_strategy'],
                min_prepend=config['min_prepend'],
                prefix_input_ids=config['prefix_input_ids'],
                sequential_positions=config['sequential_positions']
            )
            if config['kv_strategy'] == 'sw':
                past_key_values = self._apply_sliding_window_mask(
                    past_key_values, context_ids, config['num_chunks'],
                    config['min_prepend'], config['prefix_input_ids']
                )
        
        # Prefill tokens from the query part of the prompt
        self._store_composition_metadata(context_ids, config['num_chunks'])
        past_key_values = self.prefill_query(query_ids, config['query_len'], past_key_values)
        
        logger.info(f"After prefill_query: seqlen_offset={past_key_values.seqlen_offset}, "
                   f"kv_seq_len={past_key_values.get_seq_length()}")
        
        # Prepare final inputs
        input_ids = self._prepare_final_inputs(input_ids, past_key_values, kwargs)
        
        return input_ids, past_key_values, kwargs

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _parse_composition_config(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Extract chunking parameters and parse composition strategy."""
        compose_type = kwargs.get("compose_type")
        compose_type_map = {
            'full_kv_only': ('full', 'kv_only'),
            'full_kv_fuse_ssm': ('full', 'fuse'),
            'full_kv_soup_ssm': ('full', 'soup'),
            'concat_kv_only': ('concat', 'kv_only'),
            'concat_kv_fuse_ssm': ('concat', 'fuse'),
            'concat_kv_soup_ssm': ('concat', 'soup'),
            'sw_fuse_ssm': ('sw', 'fuse'),
            'sw_soup_ssm': ('sw', 'soup'),
            'sw_kv_only': ('sw', 'kv_only'),
        }
        if compose_type not in compose_type_map:
            raise ValueError(
                f"Invalid compose_type: '{compose_type}'. "
                f"Choose from: {list(compose_type_map.keys())}"
            )
        
        kv_strategy, ssm_strategy = compose_type_map[compose_type]
        
        suffix_input_ids = kwargs.get("suffix_input_ids")
        query_len = (suffix_input_ids.size(1) if suffix_input_ids is not None 
                     else kwargs.get("query_len", 1))
        
        return {
            'kv_strategy': kv_strategy,
            'ssm_strategy': ssm_strategy,
            'num_chunks': kwargs.get("num_chunks", 2),
            'query_len': query_len,
            'min_prepend': kwargs.get("min_prepend", 1),
            'prefix_input_ids': kwargs.get("prefix_input_ids"),
            'sequential_positions': kwargs.get("sequential_positions", False),
        }

    def _split_context_and_query(
        self, 
        input_ids: torch.Tensor, 
        query_len: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Split input_ids into context and query portions."""
        context_ids = input_ids[:, :-query_len]
        query_ids = input_ids[:, -query_len:-1]
        return context_ids, query_ids

    def _store_composition_metadata(
        self, 
        context_ids: torch.Tensor, 
        num_chunks: int
    ) -> None:
        """Store metadata about the composition for later use."""
        self.len_prefix = self.prefix_ids.size(1) if self.prefix_ids is not None else 0
        self.len_context = context_ids.size(1) if num_chunks > 1 else 0

    def _prepare_final_inputs(
        self, 
        input_ids: torch.Tensor, 
        past_key_values: HybridCache,
        kwargs: Dict[str, Any]
    ) -> torch.Tensor:
        """Prepend prefix, create attention mask, compute cache position."""
        attention_mask = torch.ones_like(input_ids)
        final_input_ids = input_ids
        
        if (
            hasattr(self, "prefix_ids") and 
            self.prefix_ids is not None and 
            self.prefix_ids.numel() > 0
        ):
            final_input_ids = torch.cat([self.prefix_ids, input_ids], dim=1)
            attention_mask = self._create_attention_mask(final_input_ids, past_key_values)
        
        kwargs['attention_mask'] = attention_mask
        kwargs['cache_position'] = torch.tensor(
            [final_input_ids.shape[1] - 1], 
            device=final_input_ids.device,
            dtype=torch.long
        )
        
        if past_key_values is not None and not past_key_values.check_seqlen_match():
            kwargs['position_ids'] = torch.full(
                (final_input_ids.shape[0], 1), 
                past_key_values.seqlen_offset, 
                device=final_input_ids.device, 
                dtype=torch.long
            )
        
        return final_input_ids

    def _create_attention_mask(
        self, 
        input_ids: torch.Tensor, 
        past_key_values: HybridCache
    ) -> torch.Tensor:
        """Create attention mask, handling sliding window case."""
        attention_mask = torch.ones_like(input_ids)
        
        if not self._is_sliding_window(past_key_values):
            return attention_mask
        
        gap_size = past_key_values._sw_gap_size
        prefix_size = past_key_values._sw_prefix_size
        attention_mask[:, prefix_size:prefix_size + gap_size] = 0
        
        logger.debug(
            f'Created sliding window attention mask: '
            f'prefix={prefix_size}, gap={gap_size}, total_len={input_ids.shape[1]}'
        )
        
        return attention_mask

    def _is_sliding_window(self, past_key_values: HybridCache) -> bool:
        """Check if cache is configured for sliding window."""
        return (
            hasattr(past_key_values, '_sw_gap_size') and 
            past_key_values._sw_gap_size > 0
        )


    # =========================================================================
    # Cache Composition Methods
    # =========================================================================

    def _compose_fullkv_hybrid(self, input_ids, num_chunks, strategy='kv_only'):
        """Compose cache retaining full KV cache for all tokens.
        
        Args:
            input_ids: Input token IDs
            num_chunks: Number of chunks to split input into
            strategy: How to combine SSM states - 'fuse', 'soup', or 'kv_only'
        
        Returns:
            None, cache: No prefix IDs, full cache with combined states
        """
        strategy_map = {
            'fuse': (picaso_combine_mamba, picaso_combine_gated_deltanet, fuse_combine_gated_kalmanet),
            'soup': (soup_combine_mamba, soup_combine_gated_deltanet, soup_combine_gated_kalmanet),
            'kv_only': (zero_mamba_states, zero_gated_deltanet_states, zero_gated_kalmanet_states),
        }
        
        if strategy not in strategy_map:
            raise ValueError(f"Unknown strategy '{strategy}'. Choose from: {list(strategy_map.keys())}")
        
        mamba_combine_fn, deltanet_combine_fn, kalmanet_combine_fn = strategy_map[strategy]
        use_chunked_cache = strategy in ['fuse', 'soup']
        
        # Split context into chunks
        tokenized_inputs = self.chunk_input_ids(input_ids, num_chunks)
        
        # Get full KV cache from concatenated inputs
        concat_tokenize_inputs = torch.cat(tokenized_inputs, dim=1)
        output = self.model(input_ids=concat_tokenize_inputs, use_cache=True)
        concat_cache = output.past_key_values
        seq_len = output[0].shape[1]
        concat_cache.update_offset(seq_len)
        
        # Process chunks for SSM combination if needed
        chunked_cache = None
        if use_chunked_cache:
            position_ids = self.create_position_ids_for_chunks(tokenized_inputs, 0, is_sequential=False)
            chunked_cache, chunk_seq_len = self._process_chunked_inputs(tokenized_inputs, position_ids)
            chunked_cache.update_offset(chunk_seq_len)
            
            if strategy == 'fuse':
                if chunked_cache._mamba_memory_dict:
                    assert chunked_cache.mamba2.cache_aprods
                if chunked_cache._gated_deltanet_memory_dict:
                    assert any('projection_matrix' in sub_dict 
                              for sub_dict in chunked_cache._gated_deltanet_memory_dict.values())
        
        # Update cache layer by layer
        for layer_idx, _ in enumerate(self.model.layers):
            if layer_idx in concat_cache.key_value_memory_dict:
                pass  # Keep full KV cache as-is
            elif layer_idx in concat_cache.mamba2.key_value_memory_dict:
                if use_chunked_cache:
                    mamba_combine_fn(chunked_cache, layer_idx, concat_cache)
                else:
                    mamba_combine_fn(concat_cache, layer_idx)
            elif layer_idx in concat_cache.gated_deltanet.key_value_memory_dict:
                if use_chunked_cache:
                    deltanet_combine_fn(chunked_cache, layer_idx, concat_cache)
                else:
                    deltanet_combine_fn(concat_cache, layer_idx)
            elif layer_idx in concat_cache._gated_kalmanet_memory_dict:
                if use_chunked_cache:
                    kalmanet_combine_fn(chunked_cache, layer_idx, concat_cache)
                else:
                    kalmanet_combine_fn(concat_cache, layer_idx)
            else:
                raise KeyError(f"Layer {layer_idx} not in cache")
        
        if chunked_cache is not None:
            del chunked_cache
            gc.collect()
            torch.cuda.empty_cache()
        
        return None, concat_cache

    def _compose_batch_hybrid(
        self, input_ids, num_chunks, strategy='fuse', min_prepend=1, 
        keep_first_prefix=True, prefix_input_ids=None, sequential_positions=False
    ):
        """Compose hybrid cache from batched chunks.
        
        Args:
            input_ids: Input token IDs
            num_chunks: Number of chunks to split input into
            strategy: How to combine SSM states - 'fuse', 'soup', or 'kv_only'
            min_prepend: Minimum tokens to prepend from previous chunk
            keep_first_prefix: Whether to keep the first prefix in output
            prefix_input_ids: Optional prefix tokens to use
            sequential_positions: Whether to use sequential position IDs across chunks
        
        Returns:
            kept_prefix_ids: Prefix tokens (if keep_first_prefix=True)
            new_cache: Composed hybrid cache
        """
        strategy_map = {
            'fuse': (picaso_combine_mamba, picaso_combine_gated_deltanet, fuse_combine_gated_kalmanet),
            'soup': (soup_combine_mamba, soup_combine_gated_deltanet, soup_combine_gated_kalmanet),
            'kv_only': (zero_mamba_states, zero_gated_deltanet_states, zero_gated_kalmanet_states),
        }
        if strategy not in strategy_map:
            raise ValueError(f"Unknown strategy '{strategy}'. Choose from: {list(strategy_map.keys())}")
        mamba_combine_fn, deltanet_combine_fn, kalmanet_combine_fn = strategy_map[strategy]
        
        prefix_token_id = getattr(self.config, 'newline_token_id', 0)
        if prefix_token_id == 0:
            logger.warning_once(
                f"Compose batch {strategy} requires a prefix token id but none was passed, defaulting to 0"
            )
        
        tokenized_inputs, prefix_masks, first_prefix_ids = self.chunk_and_prepend_input_ids(
            input_ids, num_chunks,
            prefix_token_id=prefix_token_id,
            prefix_input_ids=prefix_input_ids,
            min_prepend=min_prepend,
            keep_first_prefix=keep_first_prefix,
        )
        logger.debug(f'Shape chunked inputs: {tokenized_inputs.shape}')

        position_ids = self.create_position_ids_for_chunks(
            tokenized_inputs, first_prefix_ids.size(1), is_sequential=sequential_positions
        )
        logger.debug(position_ids)
        
        chunked_cache, seq_len = self._process_chunked_inputs(tokenized_inputs, position_ids)
        chunked_cache.update_offset(seq_len)
        
        if strategy == 'fuse':
            if chunked_cache._mamba_memory_dict:
                assert chunked_cache.mamba2.cache_aprods
            if chunked_cache._gated_deltanet_memory_dict:
                assert any('projection_matrix' in sub_dict 
                          for sub_dict in chunked_cache._gated_deltanet_memory_dict.values())

        new_cache = HybridCache(self.model.config)
        effective_seq_len = prefix_masks.sum().item() if sequential_positions else seq_len
        new_cache.update_offset(effective_seq_len)
        logger.debug(f'Original Input seqlen: {input_ids.shape[1]}')
        logger.debug(f'Post-padding seqlen: {effective_seq_len}')

        for layer_idx, decoder_layer in enumerate(self.model.layers):
            if layer_idx in chunked_cache.key_value_memory_dict:
                concat_kv_with_mask(chunked_cache, prefix_masks, new_cache, layer_idx)
            elif layer_idx in chunked_cache.mamba2.key_value_memory_dict:
                mamba_combine_fn(chunked_cache, layer_idx, new_cache)
            elif layer_idx in chunked_cache._gated_deltanet_memory_dict:
                deltanet_combine_fn(chunked_cache, layer_idx, new_cache)
            elif layer_idx in chunked_cache._gated_kalmanet_memory_dict:
                kalmanet_combine_fn(chunked_cache, layer_idx, new_cache)
            else:
                raise KeyError(f"Layer {layer_idx} not in cache")
        
        kept_prefix_ids = (first_prefix_ids if keep_first_prefix 
                          else torch.empty((1, 0), dtype=input_ids.dtype, device=input_ids.device))
        
        del chunked_cache
        gc.collect()
        torch.cuda.empty_cache()
        
        return kept_prefix_ids, new_cache

    # =========================================================================
    # Chunking & Prefill Utilities
    # =========================================================================

    def create_position_ids_for_chunks(self, tokenized_inputs, prefix_len, is_sequential=True):
        """Create position_ids for chunks.
        
        If is_sequential=True: Continuous sequential position_ids across all chunks.
        If is_sequential=False: Each chunk gets independent position IDs.
        """
        num_chunks, seq_len = tokenized_inputs.shape
        device = tokenized_inputs.device
        
        if is_sequential:
            position_ids = torch.zeros((num_chunks, seq_len), dtype=torch.long, device=device)
            prefix_positions = torch.arange(prefix_len, dtype=torch.long, device=device)
            position_ids[:, :prefix_len] = prefix_positions.unsqueeze(0)
            
            non_prefix_len = seq_len - prefix_len
            total_non_prefix_tokens = num_chunks * non_prefix_len
            non_prefix_positions = torch.arange(
                prefix_len, prefix_len + total_non_prefix_tokens,
                dtype=torch.long, device=device
            ).reshape(num_chunks, non_prefix_len)
            position_ids[:, prefix_len:] = non_prefix_positions
        else:
            position_ids = torch.arange(
                seq_len, dtype=torch.long, device=device
            ).unsqueeze(0).expand(num_chunks, seq_len)
        
        return position_ids

    def chunk_input_ids(self, input_ids, num_chunks):
        """Split input_ids into approximately equal chunks."""
        seq_length = input_ids.shape[1]
        base_chunk_size = seq_length // num_chunks
        remainder = seq_length % num_chunks
        
        chunks = []
        start_idx = 0
        for i in range(num_chunks):
            current_chunk_size = base_chunk_size + (1 if i < remainder else 0)
            end_idx = start_idx + current_chunk_size
            chunks.append(input_ids[:, start_idx:end_idx])
            start_idx = end_idx
        
        return chunks

    def chunk_and_prepend_input_ids(
        self, input_ids, num_chunks, 
        prefix_token_id=None, prefix_input_ids=None, min_prepend=1, 
        keep_first_prefix=False
    ):
        """Chunk input_ids into equal-length pieces by prepending prefix tokens."""
        assert input_ids.shape[0] == 1, f"Expected batch size 1, got {input_ids.shape[0]}"
        
        prefix_token_id = prefix_token_id or getattr(self.config, 'pad_token_id', 0)
        input_ids = input_ids.squeeze(0)
        
        if prefix_input_ids is not None:
            prefix_input_ids = (prefix_input_ids.squeeze(0) if prefix_input_ids.dim() == 2 
                               else prefix_input_ids).to(input_ids.device)
            base_prefix_len = len(prefix_input_ids)
        else:
            base_prefix_len = min_prepend
        
        chunk_size = math.ceil(len(input_ids) / num_chunks)
        chunks = [input_ids[i*chunk_size:min((i+1)*chunk_size, len(input_ids))] 
                  for i in range(num_chunks)]
        max_len = max(len(c) for c in chunks) + base_prefix_len
        
        chunked_inputs, prefix_masks = [], []
        first_prefix_ids, first_n_prefix = None, None
        
        for i, chunk in enumerate(chunks):
            n_prefix = max_len - len(chunk)
            if i == 0:
                first_n_prefix = n_prefix
            
            if prefix_input_ids is not None:
                extra = n_prefix - base_prefix_len
                prefix = (torch.cat([
                    torch.full((extra,), prefix_token_id, dtype=input_ids.dtype, device=input_ids.device),
                    prefix_input_ids
                ]) if extra > 0 else prefix_input_ids)
            else:
                prefix = torch.full((n_prefix,), prefix_token_id, dtype=input_ids.dtype, device=input_ids.device)
            
            if i == 0 and keep_first_prefix:
                first_prefix_ids = prefix.unsqueeze(0)
            
            chunked_inputs.append(torch.cat([prefix, chunk]))
            mask_value = keep_first_prefix and i == 0
            mask = torch.cat([
                torch.full((n_prefix,), mask_value, dtype=torch.bool, device=input_ids.device),
                torch.ones(len(chunk), dtype=torch.bool, device=input_ids.device)
            ])
            prefix_masks.append(mask)
        
        chunked_inputs = torch.stack(chunked_inputs)
        prefix_masks = torch.stack(prefix_masks)
        
        assert len(set(c.shape[0] for c in chunked_inputs)) == 1, "Chunks have different lengths"
        expected = len(input_ids) + (first_n_prefix if keep_first_prefix else 0)
        assert prefix_masks.sum().item() == expected, \
            f"Token count mismatch: {expected} != {prefix_masks.sum().item()}"
        
        return chunked_inputs, prefix_masks, first_prefix_ids

    def _process_chunked_inputs(self, tokenized_inputs, position_ids):
        """Process chunks with automatic fallback to sequential on OOM."""
        try:
            logger.debug("Attempting batch processing of chunks...")
            chunked_out = self.model(
                input_ids=tokenized_inputs,
                position_ids=position_ids,
                use_cache=True
            )
            return chunked_out.past_key_values, chunked_out[0].shape[1]
            
        except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
            if "out of memory" not in str(e).lower():
                raise
                
            logger.warning("Batch processing failed due to OOM. Falling back to sequential...")
            torch.cuda.empty_cache()
            gc.collect()

            if torch.cuda.is_available():
                mem_allocated = torch.cuda.memory_allocated() / 1024**3
                mem_reserved = torch.cuda.memory_reserved() / 1024**3
                mem_free = (torch.cuda.get_device_properties(0).total_memory 
                           - torch.cuda.memory_allocated()) / 1024**3
                logger.debug(
                    f"Memory after cleanup - Allocated: {mem_allocated:.2f} GB, "
                    f"Reserved: {mem_reserved:.2f} GB, Free: {mem_free:.2f} GB"
                )
            
            chunked_cache, seq_len = None, None
            for i in range(tokenized_inputs.shape[0]):
                logger.debug(f'chunk {i} processing...')
                chunk_out = self.model(
                    input_ids=tokenized_inputs[i:i+1],
                    position_ids=position_ids[i:i+1] if position_ids is not None else None,
                    use_cache=True
                )
                
                if chunked_cache is None:
                    chunked_cache, seq_len = chunk_out.past_key_values, chunk_out[0].shape[1]
                else:
                    self._merge_chunked_cache(chunked_cache, chunk_out.past_key_values)
                
                del chunk_out
                torch.cuda.empty_cache()
            
            logger.debug(f"Sequential processing completed for {tokenized_inputs.shape[0]} chunks")
            return chunked_cache, seq_len

    def _merge_chunked_cache(self, target_cache, source_cache):
        """Merge a single hybrid chunk's cache into the accumulated cache."""
        # Merge attention KV caches
        for layer_idx in source_cache.key_value_memory_dict:
            target_cache.key_value_memory_dict[layer_idx]["keys"] = torch.cat([
                target_cache.key_value_memory_dict[layer_idx]["keys"],
                source_cache.key_value_memory_dict[layer_idx]["keys"]
            ], dim=0)
            target_cache.key_value_memory_dict[layer_idx]["values"] = torch.cat([
                target_cache.key_value_memory_dict[layer_idx]["values"],
                source_cache.key_value_memory_dict[layer_idx]["values"]
            ], dim=0)
        
        # Merge Mamba2 caches
        for layer_idx in source_cache.mamba2.key_value_memory_dict:
            src_conv, src_ssm = source_cache.mamba2.key_value_memory_dict[layer_idx]
            tgt_conv, tgt_ssm = target_cache.mamba2.key_value_memory_dict[layer_idx]
            target_cache.mamba2.key_value_memory_dict[layer_idx] = (
                torch.cat([tgt_conv, src_conv], dim=0),
                torch.cat([tgt_ssm, src_ssm], dim=0)
            )
            target_cache.mamba2.log_aprods[layer_idx] = torch.cat([
                target_cache.mamba2.log_aprods[layer_idx],
                source_cache.mamba2.log_aprods[layer_idx]
            ], dim=0)

        # Merge GDN caches
        for layer_idx in source_cache.gated_deltanet.key_value_memory_dict:
            src_dict = source_cache.gated_deltanet.key_value_memory_dict[layer_idx]
            tgt_dict = target_cache.gated_deltanet.key_value_memory_dict[layer_idx]
            
            new_dict = {
                "recurrent_state": torch.cat([
                    tgt_dict["recurrent_state"], src_dict["recurrent_state"]
                ], dim=0)
            }
            if "projection_matrix" in src_dict:
                new_dict["projection_matrix"] = torch.cat([
                    tgt_dict["projection_matrix"], src_dict["projection_matrix"]
                ], dim=0)
            if "conv_state" in src_dict:
                new_dict["conv_state"] = tuple(
                    torch.cat([tgt_dict["conv_state"][i], src_dict["conv_state"][i]], dim=0) 
                    for i in range(len(src_dict["conv_state"]))
                )
            
            target_cache.gated_deltanet.key_value_memory_dict[layer_idx] = new_dict

    def prefill_query(self, tokens_to_process, query_len, past_key_values):
        """Process query tokens one at a time."""
        if query_len > 1:
            logger.debug(f'Processing {tokens_to_process.shape[1]} query tokens')
            for i in range(tokens_to_process.shape[1]):
                current_token = tokens_to_process[:, i:i+1]
                current_position = past_key_values.seqlen_offset
                batch_size = current_token.shape[0]
                position_ids = torch.full(
                    (batch_size, 1), current_position, 
                    device=current_token.device, dtype=torch.long
                )
                temp_inputs = {
                    "input_ids": current_token,
                    "past_key_values": past_key_values,
                    "position_ids": position_ids,
                    "use_cache": True,
                    "return_dict": True,
                }
                
                outputs = self.model(**temp_inputs)
                past_key_values = outputs.past_key_values
                out_len = outputs[0].shape[1]
                past_key_values.update_offset(out_len)

        return past_key_values

    def _apply_sliding_window_mask(
        self,
        past_key_values: HybridCache,
        context_ids: torch.Tensor,
        num_chunks: int,
        min_prepend: int,
        prefix_input_ids: Optional[torch.Tensor]
    ) -> HybridCache:
        """Apply sliding window attention mask metadata to cache."""
        prefix_token_id = getattr(self.config, 'newline_token_id', 0)
        tokenized_inputs, prefix_masks, first_prefix_ids = self.chunk_and_prepend_input_ids(
            context_ids, num_chunks, prefix_token_id, prefix_input_ids, 
            min_prepend, keep_first_prefix=True
        )
        
        n_first_prefix = first_prefix_ids.shape[1] if first_prefix_ids is not None else 0
        n_last_chunk = prefix_masks[-1].sum().item()
        total_content_tokens = prefix_masks.sum().item() - n_first_prefix
        n_gap = total_content_tokens - n_last_chunk
        
        past_key_values._sw_gap_size = n_gap
        past_key_values._sw_prefix_size = n_first_prefix
        
        logger.debug(
            f'SW: prefix={n_first_prefix}, gap={n_gap}, '
            f'last_chunk={n_last_chunk}, total_cache={past_key_values.get_seq_length()}'
        )
        
        return past_key_values



# =============================================================================
# State Combination Functions
# =============================================================================

def zero_mamba_states(cache, layer_idx, new_cache=None):
    """Zero out mamba2 states and copy to new cache with batch_size=1."""
    conv, ssm = cache._mamba_memory_dict[layer_idx]
    (new_cache or cache).mamba2.key_value_memory_dict[layer_idx] = (
        torch.zeros_like(conv[:1]), torch.zeros_like(ssm[:1])
    )


def zero_gated_deltanet_states(cache, layer_idx, new_cache=None):
    """Zero out gated_deltanet states and copy to new cache with batch_size=1."""
    old = cache._gated_deltanet_memory_dict[layer_idx]
    conv = old.get("conv_state")
    new_dict = {
        "recurrent_state": torch.zeros_like(old["recurrent_state"][:1]),
        "conv_state": tuple(torch.zeros_like(s[:1]) for s in conv) if conv else None,
    }
    (new_cache or cache)._gated_deltanet_memory_dict[layer_idx] = new_dict


def zero_gated_kalmanet_states(cache, layer_idx, new_cache=None):
    """Zero out GKA states and copy to new cache with batch_size=1."""
    old = cache._gated_kalmanet_memory_dict[layer_idx]
    new_dict = {
        "recurrent_state": (tuple(torch.zeros_like(s[:1]) for s in old["recurrent_state"]) 
                           if old.get("recurrent_state") else None),
        "conv_state": (tuple(torch.zeros_like(s[:1]) for s in old["conv_state"]) 
                      if old.get("conv_state") else None),
    }
    (new_cache or cache)._gated_kalmanet_memory_dict[layer_idx] = new_dict


def concat_kv_with_mask(cache, prefix_masks, new_cache, layer_idx):
    """Concatenate KV cache after masking out prefixes."""
    old_past_key = cache.key_value_memory_dict[layer_idx]["keys"]  
    old_past_value = cache.key_value_memory_dict[layer_idx]["values"]
    
    flat_mask = prefix_masks.reshape(-1).to(old_past_key.device)
    past_key = old_past_key.transpose(1, 2).flatten(0, 1)[flat_mask].unsqueeze(0).transpose(1, 2)
    past_value = old_past_value.transpose(1, 2).flatten(0, 1)[flat_mask].unsqueeze(0).transpose(1, 2)
    
    # Debug logging
    if layer_idx == 0:
        logger.debug(f"concat_kv_with_mask layer {layer_idx}: old_key shape={old_past_key.shape}, "
                    f"flat_mask sum={flat_mask.sum().item()}, new_key shape={past_key.shape}")
    
    new_cache.update(past_key, past_value, layer_idx)


def soup_combine_mamba(cache, layer_idx, new_cache):
    """Combine mamba2 states using simple averaging (model soup)."""
    conv_states = cache.mamba2.key_value_memory_dict[layer_idx][0]
    new_conv_states = torch.mean(conv_states, dim=0, keepdim=True)
    
    ssm_states = cache.mamba2.key_value_memory_dict[layer_idx][1]
    new_ssm_states = torch.mean(ssm_states, dim=0, keepdim=True)
    
    new_cache.mamba2.key_value_memory_dict[layer_idx] = (new_conv_states, new_ssm_states)


def soup_combine_gated_deltanet(cache, layer_idx, new_cache):
    """Combine gated_deltanet states using simple averaging (model soup)."""
    conv_states = cache._gated_deltanet_memory_dict[layer_idx]["conv_state"]
    new_conv_states = (tuple(torch.mean(state, dim=0, keepdim=True) for state in conv_states) 
                      if conv_states is not None else None)
    
    recurrent_states = cache._gated_deltanet_memory_dict[layer_idx]["recurrent_state"]
    new_recurrent_states = torch.mean(recurrent_states, dim=0, keepdim=True)
    
    new_cache.gated_deltanet.update(
        recurrent_state=new_recurrent_states, 
        layer_idx=layer_idx, 
        conv_state=new_conv_states,
        offset=0
    )


def soup_combine_gated_kalmanet(cache, layer_idx, new_cache):
    """Combine GKA states using simple averaging (model soup)."""
    conv_states = cache._gated_kalmanet_memory_dict[layer_idx]["conv_state"]
    new_conv_states = (tuple(torch.mean(state, dim=0, keepdim=True) for state in conv_states) 
                      if conv_states is not None else None)
    
    recurrent_states = cache._gated_kalmanet_memory_dict[layer_idx]["recurrent_state"]
    new_recurrent_states = (tuple(torch.mean(state, dim=0, keepdim=True) for state in recurrent_states) 
                           if recurrent_states is not None else None)
    
    new_cache.gated_kalmanet.update(
        recurrent_state=new_recurrent_states, 
        layer_idx=layer_idx, 
        conv_state=new_conv_states,
        offset=0
    )


def picaso_combine_mamba(cache, layer_idx, new_cache):
    """Combine mamba2 states using PICASO weighted combination.
    
    Requires that the Mamba2 layer has cached log_aprods during forward pass.
    This is controlled by cache.mamba2.cache_aprods = True.
    """
    conv_states = cache.mamba2.key_value_memory_dict[layer_idx][0]
    new_conv_states = torch.mean(conv_states, dim=0, keepdim=True)
    
    ssm_states = cache.mamba2.key_value_memory_dict[layer_idx][1]
    
    if layer_idx not in cache.mamba2.log_aprods:
        raise ValueError(
            f"PICASO composition requires log_aprods for Mamba2 layer {layer_idx}, "
            f"but none were cached. Ensure cache.mamba2.cache_aprods=True and that "
            f"the Mamba2 layer implementation stores log_aprods during forward pass. "
            f"Available log_aprods layers: {list(cache.mamba2.log_aprods.keys())}"
        )
    
    log_aprods = cache.mamba2.log_aprods[layer_idx]
    logger.debug(f"Layer {layer_idx}: log_aprods shape={log_aprods.shape}, values={log_aprods[:, :3]}")
    log_chunk_coeffs = get_coef_for_picaso_log(log_aprods)
    chunk_coeffs = torch.exp(log_chunk_coeffs)
    logger.debug(f"Layer {layer_idx}: chunk_coeffs shape={chunk_coeffs.shape}, values={chunk_coeffs[:, :3]}")
    new_ssm_states = (chunk_coeffs[:, :, None, None] * ssm_states).sum(dim=0, keepdim=True)
    
    new_cache.mamba2.key_value_memory_dict[layer_idx] = (new_conv_states, new_ssm_states)


def picaso_combine_gated_deltanet(cache, layer_idx, new_cache):
    """Combine gated_deltanet states using PICASO weighted combination.
    
    Requires that the GDN layer has cached projection_matrix during forward pass.
    """
    conv_states = cache._gated_deltanet_memory_dict[layer_idx]["conv_state"]
    new_conv_states = (tuple(torch.mean(state, dim=0, keepdim=True) for state in conv_states) 
                      if conv_states is not None else None)
    
    recurrent_states = cache._gated_deltanet_memory_dict[layer_idx]["recurrent_state"]
    
    if "projection_matrix" not in cache._gated_deltanet_memory_dict[layer_idx]:
        raise ValueError(
            f"PICASO composition requires projection_matrix for GDN layer {layer_idx}, "
            f"but none was cached. Ensure the GDN layer implementation stores "
            f"projection_matrix during forward pass."
        )
    
    proj_matrices = cache._gated_deltanet_memory_dict[layer_idx]["projection_matrix"]
    chunk_coeffs = get_matrix_coef_for_picaso(proj_matrices)
    new_recurrent_states = (chunk_coeffs @ recurrent_states).sum(dim=0, keepdim=True)
    
    new_cache.gated_deltanet.update(
        recurrent_state=new_recurrent_states, 
        layer_idx=layer_idx, 
        conv_state=new_conv_states,
        offset=0
    )


def fuse_combine_gated_kalmanet(cache, layer_idx, new_cache):
    """Combine GKA states using sum (fuse) combination."""
    conv_states = cache._gated_kalmanet_memory_dict[layer_idx]["conv_state"]
    new_conv_states = (tuple(torch.mean(state, dim=0, keepdim=True) for state in conv_states) 
                      if conv_states is not None else None)
    
    recurrent_states = cache._gated_kalmanet_memory_dict[layer_idx]["recurrent_state"]
    new_recurrent_states = (tuple(torch.sum(state, dim=0, keepdim=True) for state in recurrent_states) 
                           if recurrent_states is not None else None)
    
    new_cache.gated_kalmanet.update(
        recurrent_state=new_recurrent_states, 
        layer_idx=layer_idx, 
        conv_state=new_conv_states,
        offset=0
    )


# =============================================================================
# PICASO Coefficient Math
# 
# Implementation of PICASO (Permutation-Invariant Context Composition with 
# State Space Models) from https://arxiv.org/abs/2502.17605
# =============================================================================

def get_coef_for_picaso(As):
    """Compute PICASO coefficients for scalar A values."""
    if isinstance(As, list):
        As = torch.stack(As, dim=0)
    
    N = As.shape[0]
    final_coeffs = []
    
    for idx in range(N):
        As_stack = torch.cat([As[:idx], As[idx+1:]], dim=0)
        coefs = torch.tensor(
            [1 / math.comb(N - 1, i) for i in range(N)], 
            dtype=As_stack.dtype, device=As_stack.device
        )
        
        e_array = torch.zeros((N, N, *As_stack.shape[1:]), dtype=As_stack.dtype, device=As_stack.device)
        e_array[0] = 1

        for i in range(1, N):
            for j in range(i, N):
                if i <= j:
                    e_array[i][j] = e_array[i][j - 1] + As_stack[j - 1] * e_array[i - 1][j - 1]

        new_coef_shape = [1 for _ in range(len(As_stack.shape))]
        new_coef_shape[0] = -1
        
        coeff = (coefs.reshape(*new_coef_shape) * e_array[:, -1]).sum(dim=0) / N
        final_coeffs.append(coeff)
    
    return torch.stack(final_coeffs, dim=0)


def get_matrix_coef_for_picaso(As):
    """Compute PICASO coefficients for matrix A values."""
    if isinstance(As, list):
        As = torch.stack(As, dim=0)
    
    N = As.shape[0]
    final_coeffs = []
    
    for idx in range(N):
        As_stack = torch.cat([As[:idx], As[idx+1:]], dim=0)
        coefs = torch.tensor(
            [1 / math.comb(N - 1, i) for i in range(N)], 
            dtype=As_stack.dtype, device=As_stack.device
        )
        
        e_array = torch.zeros((N, N, *As_stack.shape[1:]), dtype=As_stack.dtype, device=As_stack.device)
        
        if As_stack.ndim > 2:
            *batch_dims, d1, d2 = As_stack.shape[1:]
            eye_matrix = torch.eye(d1, d2, dtype=As_stack.dtype, device=As_stack.device)
            shape = [1] + list(batch_dims) + [d1, d2]
            e_array[0] = eye_matrix.view(*([1] * (1 + len(batch_dims))), d1, d2).expand(*shape)
        else:
            e_array[0] = 1

        for i in range(1, N):
            for j in range(i, N):
                if i <= j:
                    if As_stack.ndim > 2:
                        e_array[i][j] = e_array[i][j - 1] + As_stack[j - 1] @ e_array[i - 1][j - 1]
                    else:
                        e_array[i][j] = e_array[i][j - 1] + As_stack[j - 1] * e_array[i - 1][j - 1]

        new_coef_shape = [1 for _ in range(len(As_stack.shape))]
        new_coef_shape[0] = -1
        
        coeff = (coefs.reshape(*new_coef_shape) * e_array[:, -1]).sum(dim=0) / N
        final_coeffs.append(coeff)
    
    return torch.stack(final_coeffs, dim=0)


def get_coef_for_picaso_log(log_As):
    """Compute log(PICASO coefficients) using logsumexp for numerical stability.
    
    Args:
        log_As: Input values ALREADY IN LOG SPACE (e.g., log_aprods from Mamba2).
                Shape: (N, num_heads) where N is number of chunks.
    
    Returns:
        log_coeffs: Log of PICASO coefficients, shape (N, num_heads).
    """
    if isinstance(log_As, list):
        log_As = torch.stack(log_As, dim=0)
    
    N = log_As.shape[0]
    final_log_coeffs = []
    
    for idx in range(N):
        # Exclude chunk idx
        log_As_stack = torch.cat([log_As[:idx], log_As[idx+1:]], dim=0)
        # Input is already in log space - no need to take log again
        
        log_coefs = torch.tensor(
            [-math.log(math.comb(N - 1, i)) for i in range(N)],
            dtype=torch.float32, device=log_As_stack.device
        )
        
        log_e_array = torch.full(
            (N, N, *log_As_stack.shape[1:]), float('-inf'), 
            dtype=torch.float32, device=log_As_stack.device
        )
        log_e_array[0] = 0
        
        for i in range(1, N):
            for j in range(i, N):
                if i <= j:
                    log_term1 = log_e_array[i][j - 1]
                    log_term2 = log_As_stack[j - 1] + log_e_array[i - 1][j - 1]
                    stacked = torch.stack([log_term1, log_term2], dim=0)
                    log_e_array[i][j] = torch.logsumexp(stacked, dim=0)
        
        new_coef_shape = [1 for _ in range(len(log_As_stack.shape))]
        new_coef_shape[0] = -1
        
        log_coef_reshaped = log_coefs.reshape(*new_coef_shape)
        log_terms = log_coef_reshaped + log_e_array[:, -1]
        log_coeff = torch.logsumexp(log_terms, dim=0) - math.log(N)
        
        final_log_coeffs.append(log_coeff)
    
    return torch.stack(final_log_coeffs, dim=0)
