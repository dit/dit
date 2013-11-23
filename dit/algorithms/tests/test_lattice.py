from __future__ import division

from iterutils import powerset

from nose.tools import assert_equal, assert_raises

import numpy as np
import numpy.testing as npt

from dit import Distribution, ScalarDistribution
from dit.algorithms.lattice import (dist_from_induced_sigalg, insert_join,
                                    insert_meet, join, join_sigalg, meet,
                                    meet_sigalg, sigma_algebra_sort)

def test_sigalg_sort():
    sigalg = frozenset([
        frozenset([]),
        frozenset([1]),
        frozenset([2]),
        frozenset([1,2])
    ])
    sigalg_ = [(), (1,), (2,), (1,2)]
    assert_equal( sigalg_, sigma_algebra_sort(sigalg) )

def test_join_sigalg():
    outcomes = ['00', '01', '10', '11']
    pmf = [1/4]*4
    d = Distribution(outcomes, pmf)
    sigalg = frozenset([ frozenset(_) for _ in powerset(outcomes) ])
    join = join_sigalg(d, [[0],[1]])
    assert_equal(sigalg, join)

def test_meet_sigalg():
    outcomes = ['00', '01', '10', '11']
    pmf = [1/4]*4
    d = Distribution(outcomes, pmf)
    sigalg = frozenset([ frozenset([]), frozenset(outcomes) ])
    meet = meet_sigalg(d, [[0],[1]])
    assert_equal(sigalg, meet)

def test_dist_from_induced():
    outcomes = [(0,), (1,), (2,)]
    pmf = np.array([1/3] * 3)
    d = ScalarDistribution(outcomes, pmf)

    sigalg = frozenset(map(frozenset, d.event_space()))
    d2 = dist_from_induced_sigalg(d, sigalg)
    npt.assert_allclose(pmf, d2.pmf)

    sigalg = [(), ((0,),), ((1,),(2,)), ((0,),(1,),(2,))]
    sigalg = frozenset(map(frozenset, sigalg))
    d2 = dist_from_induced_sigalg(d, sigalg, int_outcomes=True)
    pmf = np.array([1/3, 2/3])
    npt.assert_allclose(pmf, d2.pmf)

    d2 = dist_from_induced_sigalg(d, sigalg, int_outcomes=False)
    outcomes = ( ((0,),), ((1,),(2,)) )
    assert_equal(outcomes, d2.outcomes)

def test_join():
    outcomes = ['00', '01', '10', '11']
    pmf = [1/4]*4
    d = Distribution(outcomes, pmf)
    d2 = join(d, [[0],[1]])
    assert_equal(d2.outcomes, (0,1,2,3))
    npt.assert_allclose(d2.pmf, d.pmf)

def test_meet():
    outcomes = ['00', '01', '10', '11']
    pmf = [1/4]*4
    d = Distribution(outcomes, pmf)
    d2 = meet(d, [[0],[1]])
    assert_equal(d2.outcomes, (0,))
    npt.assert_allclose(d2.pmf, [1])

def test_insert_join():
    outcomes = ['00', '01', '10', '11']
    pmf = [1/4]*4
    d = Distribution(outcomes, pmf)
    assert_raises(IndexError, insert_join, d, 5, [[0],[1]])

    for idx in range(d.outcome_length()):
        d2 = insert_join(d, idx, [[0],[1]])
        m = d2.marginal([idx])
        npt.assert_allclose(d2.pmf, m.pmf)
