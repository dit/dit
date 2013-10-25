from __future__ import division

from nose.tools import assert_almost_equal, assert_equal, assert_raises, \
                       assert_true
from numpy.testing import assert_array_almost_equal

from dit import ScalarDistribution
from dit.exceptions import ditException, InvalidDistribution

def test_init1():
    dist = {0: 1/2, 1: 1/2}
    d = ScalarDistribution(dist)
    assert_equal(d.outcomes, (0, 1))
    assert_array_almost_equal(d.pmf, [1/2, 1/2])

def test_init2():
    dist = {0: 1/2, 1: 1/2}
    pmf = [1/2, 1/2]
    assert_raises(ditException, ScalarDistribution, dist, pmf)

def test_init3():
    pmf = ['a', 'b', 'c']
    assert_raises(ditException, ScalarDistribution, pmf)

def test_init4():
    outcomes = float
    pmf = int
    assert_raises(TypeError, ScalarDistribution, outcomes, pmf)

def test_init5():
    outcomes = [0, 1, 2]
    pmf = [1/2, 1/2]
    assert_raises(InvalidDistribution, ScalarDistribution, outcomes, pmf)

def test_init6():
    outcomes = set([0, 1])
    pmf = [1/2, 1/2]
    assert_raises(ditException, ScalarDistribution, outcomes, pmf)

def test_init7():
    outcomes = [0, 1, 2]
    pmf = [1/2, 1/2, 0]
    d = ScalarDistribution(outcomes, pmf, trim=True)
    assert_equal(len(d.outcomes), 2)

def test_init8():
    outcomes = [0, 1, 2]
    pmf = [1/3]*3
    d1 = ScalarDistribution(outcomes, pmf)
    d2 = ScalarDistribution.from_distribution(d1)
    assert_true(d1.is_approx_equal(d2))

def test_init9():
    outcomes = [0, 1, 2]
    pmf = [1/3]*3
    d1 = ScalarDistribution(outcomes, pmf)
    d2 = ScalarDistribution.from_distribution(d1, base=10)
    d1.set_base(10)
    assert_true(d1.is_approx_equal(d2))

def test_add_mul():
    d1 = ScalarDistribution([1/3, 2/3])
    d2 = ScalarDistribution([2/3, 1/3])
    d3 = ScalarDistribution([1/2, 1/2])
    d4 = 0.5*(d1 + d2)
    assert_true(d3.is_approx_equal(d4))

def test_del():
    d = ScalarDistribution([1/2, 1/2])
    del d[1]
    d.normalize()
    assert_almost_equal(d[0], 1)