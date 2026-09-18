"""Tree attention: the prefix and the tree, merged by log-sum-exp.

Section 5 says a foreground batch attends to the prefix with the regular
kernel and to the tree under an exact ancestor mask, merged by log-sum-exp.
That merge has to be exact, or a streamed round would not compute what a
one-shot round does, so it is checked against attention over the plain
concatenation.
"""

from __future__ import annotations

import math

import torch

from slipstream.models.layers import attend_split


def reference(query, keys, values, mask=None):
    scale = 1.0 / math.sqrt(query.shape[-1])
    scores = torch.einsum("nhd,khd->nhk", query, keys) * scale
    if mask is not None:
        scores = scores.masked_fill(~mask.unsqueeze(1), float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    return torch.einsum("nhk,khd->nhd", weights, values)


def test_merge_matches_attention_over_the_concatenation():
    torch.manual_seed(0)
    n, prefix_len, tree_len, heads, dim = 4, 7, 5, 3, 8
    query = torch.randn(n, heads, dim)
    prefix_key, prefix_value = torch.randn(prefix_len, heads, dim), torch.randn(prefix_len, heads, dim)
    tree_key, tree_value = torch.randn(tree_len, heads, dim), torch.randn(tree_len, heads, dim)
    tree_mask = torch.rand(n, tree_len) > 0.4
    tree_mask[:, 0] = True  # every token sees the round's root

    merged = attend_split(query, prefix_key, prefix_value, tree_key, tree_value, tree_mask)

    keys = torch.cat([prefix_key, tree_key])
    values = torch.cat([prefix_value, tree_value])
    mask = torch.cat([torch.ones(n, prefix_len, dtype=torch.bool), tree_mask], dim=1)
    expected = reference(query, keys, values, mask)
    assert torch.allclose(merged, expected, atol=1e-5)


def test_tree_only_and_prefix_only_are_both_plain_attention():
    torch.manual_seed(1)
    query = torch.randn(3, 2, 8)
    keys, values = torch.randn(6, 2, 8), torch.randn(6, 2, 8)
    assert torch.allclose(
        attend_split(query, keys, values, None, None, None),
        reference(query, keys, values),
        atol=1e-5,
    )
    mask = torch.ones(3, 6, dtype=torch.bool)
    assert torch.allclose(
        attend_split(query, None, None, keys, values, mask),
        reference(query, keys, values, mask),
        atol=1e-5,
    )


def test_a_fully_masked_row_contributes_nothing():
    torch.manual_seed(2)
    query = torch.randn(2, 2, 4)
    prefix_key, prefix_value = torch.randn(3, 2, 4), torch.randn(3, 2, 4)
    tree_key, tree_value = torch.randn(2, 2, 4), torch.randn(2, 2, 4)
    mask = torch.zeros(2, 2, dtype=torch.bool)
    merged = attend_split(query, prefix_key, prefix_value, tree_key, tree_value, mask)
    assert torch.allclose(merged, reference(query, prefix_key, prefix_value), atol=1e-5)
