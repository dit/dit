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
        w = dict.fromkeys(ground, scale)
        q = greedy_base_vertex(f, ground, w)
        q_norm = np.sqrt(sum(v * v for v in q.values()))
        assert norm <= q_norm + 1e-8


def test_minimum_norm_base_singleton():
    ground = (3,)
    f = _cut_function(frozenset({0, 1}))
    x = minimum_norm_base(f, ground)
    assert x == {3: pytest.approx(0.0)}


def test_minimum_norm_base_pair():
    ground = (0, 2)
    f = _cut_function(frozenset({0, 1}))
    x = minimum_norm_base(f, ground)
    norm = np.sqrt(sum(v * v for v in x.values()))
    for scale in (-2.0, -1.0, 0.0, 1.0, 2.0):
        w = dict.fromkeys(ground, scale)
        q = greedy_base_vertex(f, ground, w)
        q_norm = np.sqrt(sum(v * v for v in q.values()))
        assert norm <= q_norm + 1e-8


def test_minimum_norm_base_stagnation():
    """
    A function whose base polytope has vertices only about 1e-10 apart (it arises in
    caekl_mutual_information_psp on a distribution with a tiny probability). Wolfe's algorithm
    reaches the minimum-norm point, then used to add and drop the same nearly coincident vertex
    until it ran out of iterations.
    """
    values = {
        (): 0.0,
        (1,): 0.0,
        (2,): 0.0,
        (3,): 0.0,
        (4,): 0.0,
        (1, 2): 0.0,
        (1, 3): 0.0,
        (1, 4): 0.0,
        (2, 3): -3.8997471918378324e-11,
        (2, 4): -0.9182958317881356,
        (3, 4): -1.0566414410106972e-10,
        (1, 2, 3): -3.8997471918378324e-11,
        (1, 2, 4): -0.9182958317881356,
        (1, 3, 4): -1.0566414410106972e-10,
        (2, 3, 4): -0.9182958341379315,
        (1, 2, 3, 4): -0.9182958341379315,
    }

    def f(subset: frozenset[int]) -> float:
        return values[tuple(sorted(subset))]

    ground = (1, 2, 3, 4)
    x = minimum_norm_base(f, ground)
    # x lies in the base polytope ...
    assert sum(x.values()) == pytest.approx(f(frozenset(ground)), abs=1e-12)
    for subset in values:
        assert sum(x[i] for i in subset) <= f(frozenset(subset)) + 1e-12
    # ... and satisfies Wolfe's optimality condition x.x <= x.q for every vertex q, up to rounding
    q = greedy_base_vertex(f, ground, x)
    assert sum(v * v for v in x.values()) <= sum(x[i] * q[i] for i in ground) + 1e-8
