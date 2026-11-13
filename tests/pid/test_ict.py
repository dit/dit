"""
Tests for dit.pid.measures.ict.
"""

import numpy as np
import pytest

from dit import Distribution
from dit.multivariate import coinformation
from dit.pid.distributions import bivariates, trivariates
from dit.pid.measures.ict import PID_CT, i_triangle, path_redundancy


def test_pid_ct1():
    """
    Test ict on a generic distribution.
    """
    d = bivariates["reduced or"]
    pid = PID_CT(d, ((0,), (1,)), (2,))
    assert pid[((0,), (1,))] == pytest.approx(0.02712, abs=1e-4)
    assert pid[((0,),)] == pytest.approx(0.28416, abs=1e-4)
    assert pid[((1,),)] == pytest.approx(0.28416, abs=1e-4)
    assert pid[((0, 1),)] == pytest.approx(0.40456, abs=1e-4)


def _markov_chain():
    """A genuine X -> Y -> Z Markov chain, so only one path is direct."""
    px = np.array([0.5, 0.5])
    py_x = np.array([[0.8, 0.2], [0.3, 0.7]])
    pz_y = np.array([[0.9, 0.1], [0.2, 0.8]])
    outcomes, pmf = [], []
    for x in range(2):
        for y in range(2):
            for z in range(2):
                p = px[x] * py_x[x, y] * pz_y[y, z]
                if p > 0:
                    outcomes.append((x, y, z))
                    pmf.append(p)
    return Distribution(outcomes, pmf)


def test_i_triangle_direct_yz():
    """X -> Y -> Z: only Y -> Z is direct, exercising the direct_yz branch."""
    d = _markov_chain()
    assert i_triangle(d, [0], [1], [2]) == pytest.approx(0.093281, abs=1e-5)


def test_i_triangle_direct_xz():
    """Swapping the sources exercises the symmetric direct_xz branch."""
    d = _markov_chain()
    assert i_triangle(d, [1], [0], [2]) == pytest.approx(0.093281, abs=1e-5)


def test_pid_ct_markov_chain():
    """PID_CT on the Markov chain: redundancy equals the path information."""
    pid = PID_CT(_markov_chain(), ((0,), (1,)), (2,))
    assert pid[((0,), (1,))] == pytest.approx(0.093281, abs=1e-5)


def test_path_redundancy_single_source():
    """
    The path from a single source straight to the target carries their mutual information.
    """
    d = bivariates["reduced or"]
    assert path_redundancy(d, [[0]], [2]) == pytest.approx(coinformation(d, [[0], [2]]))
    assert path_redundancy(d, [[0, 1]], [2]) == pytest.approx(coinformation(d, [[0, 1], [2]]))


def test_pid_ct_two_sources_unchanged():
    """
    For two sources, the redundancy is that of i_triangle, the least of the two paths
    through the other source whose last link exists.
    """
    for name in ("and", "reduced or", "rdn xor", "prob. 1", "boom"):
        d = bivariates[name]
        assert PID_CT(d)[((0,), (1,))] == pytest.approx(i_triangle(d, [0], [1], [2]), abs=1e-12)


def test_pid_ct_three_sources():
    """
    PID_CT computes every node of a lattice of three sources from the paths through its
    elements, so the decomposition is complete and consistent.
    """
    pid = PID_CT(trivariates["redundant"])
    assert pid.complete
    assert pid.consistent
    assert pid.get_red(((0,), (1,), (2,))) == pytest.approx(1.0)
    assert pid.get_pi(((0,), (1,), (2,))) == pytest.approx(1.0)


def test_pid_ct_max_min_not_monotone():
    """
    On 'max min', enlarging a source lowers the redundancy, which the monotonicity
    argument of Sigtermans, for adding a source, does not cover.
    """
    pid = PID_CT(trivariates["max min"])
    assert pid.get_red(((0,), (1,))) == pytest.approx(0.027119, abs=1e-6)
    assert pid.get_red(((0,), (1, 2))) == pytest.approx(0.0, abs=1e-9)


def test_pid_ct_indirect_link():
    """
    With a copy of the target as a second source, the link of the first source to the
    target is reproduced through the copy, so only the path through the copy counts, and
    the redundancy is the mutual information of the first source and the target.
    """
    d = bivariates["and"]
    merged = {}
    for outcome, p in d.zipped():
        key = (outcome[0], outcome[2], outcome[2])
        merged[key] = merged.get(key, 0.0) + p
    copy = Distribution(list(merged), list(merged.values()))
    assert path_redundancy(copy, [[0], [1]], [2]) == pytest.approx(coinformation(d, [[0], [2]]))
