"""
Tests for dit.algorithms.lattice
"""

from __future__ import division

import pytest

import numpy as np

from dit import Distribution, ScalarDistribution
from dit.algorithms.lattice import (dist_from_induced_sigalg, insert_join,
                                    join, join_sigalg, meet, meet_sigalg,
                                    sigma_algebra_sort)
from dit.utils import powerset

def test_sigalg_sort():
    """ Test sigma_algebra_sort """
    sigalg = frozenset([
        frozenset([]),
        frozenset([1]),
        frozenset([2]),
        frozenset([1, 2])
    ])
    sigalg_ = [(), (1,), (2,), (1, 2)]
    assert sigalg_ == sigma_algebra_sort(sigalg)

def test_join_sigalg():
    """ Test join_sigalg """
    outcomes = ['00', '01', '10', '11']
    pmf = [1/4]*4
    d = Distribution(outcomes, pmf)
    sigalg = frozenset([frozenset(_) for _ in powerset(outcomes)])
    joined = join_sigalg(d, [[0], [1]])
    assert sigalg == joined

def test_meet_sigalg():
    """ Test meet_sigalg """
    outcomes = ['00', '01', '10', '11']
    pmf = [1/4]*4
    d = Distribution(outcomes, pmf)
    sigalg = frozenset([frozenset([]), frozenset(outcomes)])
    meeted = meet_sigalg(d, [[0], [1]])
    assert sigalg == meeted

def test_dist_from_induced():
    """ Test dist_from_induced_sigalg """
    outcomes = [(0,), (1,), (2,)]
    pmf = np.array([1/3] * 3)
    d = ScalarDistribution(outcomes, pmf)

    sigalg = frozenset(map(frozenset, d.event_space()))
    d2 = dist_from_induced_sigalg(d, sigalg)
    assert np.allclose(pmf, d2.pmf)

    sigalg = [(), ((0,),), ((1,), (2,)), ((0,), (1,), (2,))]
    sigalg = frozenset(map(frozenset, sigalg))
    d2 = dist_from_induced_sigalg(d, sigalg, int_outcomes=True)
    pmf = np.array([1/3, 2/3])
    assert np.allclose(pmf, d2.pmf)

    d2 = dist_from_induced_sigalg(d, sigalg, int_outcomes=False)
    outcomes = (((0,),), ((1,), (2,)))
    assert outcomes == d2.outcomes

def test_join():
    """ Test join """
    outcomes = ['00', '01', '10', '11']
    pmf = [1/4]*4
    d = Distribution(outcomes, pmf)
    d2 = join(d, [[0], [1]])
    assert d2.outcomes == (0, 1, 2, 3)
    assert np.allclose(d2.pmf, d.pmf)

def test_meet():
    """ Test meet """
    outcomes = ['00', '01', '10', '11']
    pmf = [1/4]*4
    d = Distribution(outcomes, pmf)
    d2 = meet(d, [[0], [1]])
    assert d2.outcomes == (0,)
    assert np.allclose(d2.pmf, [1])

def test_insert_join():
    """ Test insert_join """
    outcomes = ['00', '01', '10', '11']
    pmf = [1/4]*4
    d = Distribution(outcomes, pmf)
    with pytest.raises(IndexError):
        insert_join(d, 5, [[0], [1]])

    for idx in range(d.outcome_length()):
        d2 = insert_join(d, idx, [[0], [1]])
        m = d2.marginal([idx])
        assert np.allclose(d2.pmf, m.pmf)
