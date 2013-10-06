from __future__ import division

from iterutils import powerset

from nose.tools import *

from dit import Distribution
from dit.algorithms.lattice import *

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