"""
Runs a decision model through CUDA graphs, which cut the time of one decision at a time to the GPU's own work.

At batch 1, most of a decision's time is the CPU launching the model's kernels one after another. A 30-layer Qwen3.5
mixture of experts, for example, launches 3,000 kernels per pass and takes 67 ms for 24 ms of GPU work. A CUDA graph
records the pass once, for one shape of inputs, and replays all its kernels in one launch. The model must run without
making the CPU wait for the GPU, which it does when it is given each prompt's length (seq_lens, see
modeling_decision.py), and that needs a stock backbone with flash attention.

A graph replays only the shape it recorded, so the runner pads each batch up to a bucket, recorded the first time
it is used (or ahead, with warm_up):

    - the number of rows up to a power of two, at most max_batch (padding rows are one token long, and dropped);
    - the prompts' padded length up to a multiple of length_multiple (the model gives the same answers whatever the
      padding, since its attention reads each prompt's boundaries and its recurrent layers read left to right);
    - the number of options up to one of option_buckets (padded options are masked by the head).

A batch outside every bucket (more rows, a longer prompt, more options) runs the same pass without a graph. Graphs
help small batches only, since a large batch keeps the GPU busy anyway and padding it would cost real work.

All graphs share one memory pool, which is safe because every replay's outputs are copied out before the next
replay. A runner answers one batch at a time, as the decision server sends them, and is not thread-safe.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from ...extras import logging


logger = logging.get_logger(__name__)


@dataclass
class _Graph:
    """A recorded pass, with the buffers it reads its inputs from and writes its outputs to."""

    graph: torch.cuda.CUDAGraph
    input_ids: torch.Tensor
    opt_idx: torch.Tensor
    answer_idx: torch.Tensor
    seq_lens: torch.Tensor
    log_probs: torch.Tensor


class DecisionGraphRunner:
    """
    Answers batches of encoded prompts with a decision model's log-probabilities, replaying a CUDA graph per bucket
    (see the module's docstring).

    Args:
        model: A DecisionModel on a GPU, in eval mode, with a stock backbone and flash attention.
        pad_token_id: The token that pads prompts.
        max_batch: The most rows a graph takes. Defaults to 8.
        length_multiple: Prompts are padded to a multiple of this many tokens. Defaults to 128.
        max_length: The longest padded prompt a graph takes. Defaults to 4096.
        option_buckets: The numbers of options graphs take, smallest first. Defaults to (8, 16, 32).
    """

    def __init__(
        self,
        model: torch.nn.Module,
        pad_token_id: int,
        max_batch: int = 8,
        length_multiple: int = 128,
        max_length: int = 4096,
        option_buckets: Sequence[int] = (8, 16, 32),
    ):
        if not torch.cuda.is_available() or next(model.parameters()).device.type != "cuda":
            raise ValueError("CUDA graphs need the model on a GPU.")
        model._check_varlen()  # Raises unless the model can run without making the CPU wait
        self.model = model
        self.device = next(model.parameters()).device
        self.pad_token_id = pad_token_id
        self.max_batch = max_batch
        self.length_multiple = length_multiple
        self.max_length = max_length
        self.option_buckets = tuple(sorted(option_buckets))
        self.pool = torch.cuda.graph_pool_handle()
        # One side stream for every recording's warm-up passes, so that their memory returns to one cache
        self.stream = torch.cuda.Stream(device=self.device)
        self.graphs: Dict[Tuple[int, int, int], _Graph] = {}

    def bucket(self, batch: int, length: int, num_options: int) -> Optional[Tuple[int, int, int]]:
        """The (rows, length, options) bucket of a batch, or None if it fits none."""
        rows = 1 << max(0, batch - 1).bit_length()
        padded = -(-length // self.length_multiple) * self.length_multiple
        options = next((k for k in self.option_buckets if k >= num_options), None)
        if rows > self.max_batch or padded > self.max_length or options is None:
            return None
        return rows, padded, options

    def warm_up(self, lengths: Sequence[int], batches: Sequence[int] = (1,), num_options: int = 8) -> None:
        """Records the graphs of these prompt lengths and batch sizes ahead, so that no request waits for one."""
        for batch in batches:
            for length in lengths:
                key = self.bucket(batch, length, num_options)
                if key is not None and key not in self.graphs:
                    self._record(key)

    @torch.no_grad()
    def __call__(
        self, input_ids: Sequence[Sequence[int]], opt_idx: Sequence[Sequence[int]], answer_idx: Sequence[int]
    ) -> List[List[float]]:
        """
        Each prompt's log-probabilities over its own options.

        Args:
            input_ids: Each prompt's token ids, unpadded.
            opt_idx: Each prompt's option positions (see DecisionModel.forward).
            answer_idx: Each prompt's answer position.

        Returns:
            One list per prompt, as long as its options.
        """
        batch = len(input_ids)
        length = max(len(ids) for ids in input_ids)
        num_options = max(len(idx) for idx in opt_idx)
        key = self.bucket(batch, length, num_options)
        shape = key or (batch, -(-length // self.length_multiple) * self.length_multiple, num_options)
        inputs = self._inputs(shape, input_ids, opt_idx, answer_idx)
        if key is None:
            log_probs = self.model(attention_mask=None, **{k: v.to(self.device) for k, v in inputs.items()}).log_probs
        else:
            entry = self.graphs.get(key) or self._record(key)
            for name, value in inputs.items():
                getattr(entry, name).copy_(value)
            entry.graph.replay()
            log_probs = entry.log_probs
        log_probs = log_probs[:batch].float().cpu()
        return [log_probs[j, : len(opt_idx[j])].tolist() for j in range(batch)]

    def _inputs(
        self,
        shape: Tuple[int, int, int],
        input_ids: Sequence[Sequence[int]],
        opt_idx: Sequence[Sequence[int]],
        answer_idx: Sequence[int],
    ) -> Dict[str, torch.Tensor]:
        """A batch's inputs on the CPU, padded to shape, with padding rows one padding token long."""
        rows, length, options = shape
        inputs = {
            "input_ids": torch.full((rows, length), self.pad_token_id, dtype=torch.long),
            "opt_idx": torch.full((rows, options), -1, dtype=torch.long),
            "answer_idx": torch.zeros(rows, dtype=torch.long),
            "seq_lens": torch.ones(rows, dtype=torch.int32),
        }
        inputs["opt_idx"][:, 0] = 0
        for j, ids in enumerate(input_ids):
            inputs["input_ids"][j, : len(ids)] = torch.tensor(ids, dtype=torch.long)
            inputs["opt_idx"][j, 0] = -1
            inputs["opt_idx"][j, : len(opt_idx[j])] = torch.tensor(opt_idx[j], dtype=torch.long)
            inputs["answer_idx"][j] = answer_idx[j]
            inputs["seq_lens"][j] = len(ids)
        return inputs

    @torch.no_grad()
    def _record(self, key: Tuple[int, int, int]) -> _Graph:
        """
        Records the graph of a bucket, on padding rows, after warm-up passes that compile its kernels. It runs
        without gradients, so that neither the warm-up passes nor the graph keep activations for a backward pass.
        """
        static = {k: v.to(self.device) for k, v in self._inputs(key, [], [], []).items()}
        self.stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.stream):
            for _ in range(3):
                self.model(attention_mask=None, **static)
        torch.cuda.current_stream().wait_stream(self.stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=self.pool):
            log_probs = self.model(attention_mask=None, **static).log_probs
        torch.cuda.synchronize()
        torch.cuda.empty_cache()  # The warm-up passes' memory, which replays never need
        entry = _Graph(graph=graph, log_probs=log_probs, **static)
        self.graphs[key] = entry
        logger.info_rank0(
            f"Recorded a CUDA graph for {key[0]} rows of {key[1]} tokens and {key[2]} options "
            f"({torch.cuda.memory_allocated(self.device) / 2**30:.1f} GiB in use)."
        )
        return entry
