"""
Tests for dit.algorithms.submodular.
"""

import numpy as np
import pytest

from dit.algorithms.submodular import greedy_base_vertex, minimum_norm_base


def _cut_function(cut: frozenset[int]):
    """Submodular cut function on {0,1,2,3}."""

    def f(subset: frozenset[int]) -> float:
        return float(len(subset & cut))

    return f


def test_greedy_base_vertex_cut():
    ground = (0, 1, 2, 3)
    f = _cut_function(frozenset({0, 1}))
    weights = {0: 1.0, 1: 2.0, 2: 3.0, 3: 4.0}
    x = greedy_base_vertex(f, ground, weights)
    assert x[3] == pytest.approx(0.0)
    assert x[2] == pytest.approx(0.0)
    assert x[1] == pytest.approx(1.0)
    assert x[0] == pytest.approx(1.0)
    assert sum(x.values()) == pytest.approx(f(frozenset(ground)))


def test_minimum_norm_base_cut():
    ground = (0, 1, 2)
    f = _cut_function(frozenset({0, 1}))
    x = minimum_norm_base(f, ground)
    norm = np.sqrt(sum(v * v for v in x.values()))
    for scale in (-2.0, -1.0, 0.0, 1.0, 2.0):
        w = {i: scale for i in ground}
        q = greedy_base_vertex(f, ground, w)
        q_norm = np.sqrt(sum(v * v for v in q.values()))
        assert norm <= q_norm + 1e-8
