from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.init as init


class LinearLowRank(nn.Module):
    """
    Low-rank linear layer that factorizes a linear projection into two smaller ones:
        y = linearB(act_fn(linearA(x)))

    where linearA projects from in_features to rank, and linearB projects from rank to out_features.

    linearB is zero-initialized so that the module outputs zeros at init time. This makes it safe
    to use as a residual branch (e.g. added on top of a repeat/expand) without perturbing the
    pretrained signal at the start of training.

    Note: linearB is marked with ``_is_hf_initialized = True`` to prevent transformers'
    ``post_init`` from overwriting the zero initialization with a random normal.

    Args:
        in_features: Input dimension.
        out_features: Output dimension.
        rank: Bottleneck dimension between the two linear layers.
        act_fn: Optional activation applied between linearA and linearB. Defaults to Identity.
        bias: Whether to include bias terms in both linear layers. Defaults to False.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        act_fn: Optional[Any] = None,
        bias: bool = False,
    ):
        super().__init__()
        self.linearA = nn.Linear(in_features, rank, bias=bias)
        self.linearB = nn.Linear(rank, out_features, bias=bias)

        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.bias = bias
        self.act_fn = act_fn if act_fn is not None else nn.Identity()

        # Initialize B weights and bias to zero
        init.zeros_(self.linearB.weight)
        if self.linearB.bias is not None:
            init.zeros_(self.linearB.bias)
        self.linearB._is_hf_initialized = (
            True
        )  # prevent transformers' post_init from re-initializing

    def forward(self, x: torch.Tensor):
        return self.linearB(self.act_fn(self.linearA(x)))
