"""
Tests the splitting of attention heads that are larger than GKA's and GDN's kernels support (layers/head_split.py),
and the reordering of the attention weights that priming applies for the split heads (split_attention_heads in
priming/hybrid_layer_init.py). With the reordered weights, each split head's q·k must be one piece of an attention
head's q·k, every piece used once, and the output projection of the values must be unchanged, under grouped-query
attention. The tests run on the CPU.
"""

from types import SimpleNamespace

import pytest
import torch

from hmf.model.hybrid_zoo.layers.gated_deltanet.gdn_config import GDN_MAX_HEAD_DIM, construct_gdn_config
from hmf.model.hybrid_zoo.layers.gated_kalmanet.gka_config import GKA_MAX_HEAD_DIM, construct_gka_config
from hmf.model.hybrid_zoo.layers.head_split import split_heads
from hmf.priming.hybrid_layer_init import AttentionWeights, split_attention_heads


def _geometry(config):
    return config.head_dim, config.num_q_heads, config.num_k_heads, config.num_v_heads


def _attention_config(head_dim, num_heads, num_kv_heads):
    """The fields of a transformer's config that GKA's and GDN's settings read."""
    return SimpleNamespace(hidden_size=1536, num_attention_heads=num_heads, num_key_value_heads=num_kv_heads,
                           head_dim=head_dim)


def test_heads_within_the_limit_stay_as_they_are():
    assert split_heads(128, 32, 8, 128) == (128, 32, 8)


def test_larger_heads_split_into_heads_of_the_limit():
    assert split_heads(512, 8, 1, 128) == (128, 32, 4)


def test_heads_that_are_not_a_multiple_of_the_limit_raise():
    with pytest.raises(ValueError, match="not a multiple"):
        split_heads(192, 8, 1, 128)


def test_gemma4_full_attention_heads_split_for_gka_and_gdn():
    # Gemma 4 E2B's full attention layers have 8 query and 1 key/value heads of 512
    config = _attention_config(512, 8, 1)
    assert GKA_MAX_HEAD_DIM == 128 and _geometry(construct_gka_config(config)) == (128, 32, 4, 4)
    assert GDN_MAX_HEAD_DIM == 256 and _geometry(construct_gdn_config(config)) == (256, 16, 2, 2)


def test_qwen3_heads_are_unchanged():
    config = _attention_config(128, 32, 8)
    assert _geometry(construct_gka_config(config)) == (128, 32, 8, 8)
    assert _geometry(construct_gdn_config(config)) == (128, 32, 8, 8)


@pytest.mark.parametrize("num_heads, num_kv_heads, split", [(6, 2, 2), (8, 1, 4), (4, 4, 2), (6, 2, 4)])
def test_split_heads_pair_each_query_piece_with_its_own_key_piece(num_heads, num_kv_heads, split):
    torch.manual_seed(0)
    hidden, d = 16, 4
    head_dim = d * split
    group = num_heads // num_kv_heads
    attn = AttentionWeights(
        q_weight=torch.randn(num_heads * head_dim, hidden),
        k_weight=torch.randn(num_kv_heads * head_dim, hidden),
        v_weight=torch.randn(num_kv_heads * head_dim, hidden),
        o_weight=torch.randn(hidden, num_heads * head_dim),
        q_bias=torch.randn(num_heads * head_dim),
        head_dim=head_dim,
    )
    out = split_attention_heads(attn, d)
    x, y = torch.randn(hidden), torch.randn(hidden)

    # Attention: query head h reads key/value head h // group
    q = (attn.q_weight @ x + attn.q_bias).view(num_heads, head_dim)
    k = (attn.k_weight @ y).view(num_kv_heads, head_dim)
    v = (attn.v_weight @ y).view(num_kv_heads, head_dim)
    kv_of = torch.arange(num_heads) // group

    # The split layer: head j reads key/value head j // group ("(h g)" expansion), all of size d
    q2 = (out.q_weight @ x + out.q_bias).view(num_heads * split, d)
    k2 = (out.k_weight @ y).view(num_kv_heads * split, d)
    v2 = (out.v_weight @ y).view(num_kv_heads * split, d)
    kv2_of = torch.arange(num_heads * split) // group

    scores = (q2 * k2[kv2_of]).sum(-1)
    pieces = (q.view(num_heads, split, d) * k[kv_of].view(num_heads, split, d)).sum(-1)
    torch.testing.assert_close(scores.sort().values, pieces.flatten().sort().values)
    torch.testing.assert_close(attn.o_weight @ v[kv_of].flatten(), out.o_weight @ v2[kv2_of].flatten())
