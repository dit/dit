"""
Tests for dit.npscalardist.
"""

from __future__ import division

from nose.tools import assert_almost_equal, assert_equal, assert_raises, \
                       assert_true, assert_false
from numpy.testing import assert_array_almost_equal

from dit import Distribution, ScalarDistribution
from dit.exceptions import ditException, InvalidDistribution, InvalidOutcome

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

def test_init10():
    outcomes = []
    pmf = []
    assert_raises(InvalidDistribution, ScalarDistribution, outcomes, pmf)

def test_init11():
    outcomes = ['0', '1']
    pmf = [1/2, 1/2]
    d = Distribution(outcomes, pmf)
    sd = ScalarDistribution.from_distribution(d)
    # Different sample space representations
    assert_false(d.is_approx_equal(sd))

def test_init12():
    outcomes = ['0', '1']
    pmf = [1/2, 1/2]
    d = Distribution(outcomes, pmf)
    sd = ScalarDistribution.from_distribution(d, base=10)
    d.set_base(10)
    # Different sample space representations
    assert_false(d.is_approx_equal(sd))

def test_add_mul():
    d1 = ScalarDistribution([1/3, 2/3])
    d2 = ScalarDistribution([2/3, 1/3])
    d3 = ScalarDistribution([1/2, 1/2])
    d4 = 0.5*(d1 + d2)
    assert_true(d3.is_approx_equal(d4))

def test_del1():
    d = ScalarDistribution([1/2, 1/2])
    del d[1]
    d.normalize()
    assert_almost_equal(d[0], 1)

def test_del2():
    d = ScalarDistribution([1/2, 1/2])
    d.make_dense()
    del d[1]
    d.normalize()
    assert_almost_equal(d[0], 1)

def test_del3():
    d = ScalarDistribution([1/2, 1/2])
    assert_raises(InvalidOutcome, d.__delitem__, 2)

def test_setitem1():
    pmf = [1/2, 1/2]
    d = ScalarDistribution(pmf)
    d[0] = 1
    d[1] = 0
    d.make_sparse()
    assert_equal(d.outcomes, (0,))

def test_setitem2():
    pmf = [1/2, 1/2]
    d = ScalarDistribution(pmf)
    assert_raises(InvalidOutcome, d.__setitem__, 2, 1/2)

def test_setitem3():
    outcomes = (0, 1)
    pmf = [1/2, 1/2]
    d = ScalarDistribution(outcomes, pmf, sample_space=(0, 1, 2))
    d[2] = 1/2
    d.normalize()
    assert_array_almost_equal(d.pmf, [1/3]*3)

def test_has_outcome1():
    d = ScalarDistribution([1, 0])
    assert_true(d.has_outcome(1))

def test_has_outcome2():
    d = ScalarDistribution([1, 0])
    assert_false(d.has_outcome(1, null=False))

def test_has_outcome3():
    d = ScalarDistribution([1, 0])
    assert_false(d.has_outcome(2, null=False))

def test_is_approx_equal1():
    d1 = ScalarDistribution([1/2, 1/2, 0])
    d1.make_dense()
    d2 = ScalarDistribution([1/2, 1/2, 0])
    d2.make_dense()
    assert_true(d1.is_approx_equal(d2))

def test_is_approx_equal2():
    d1 = ScalarDistribution([1/2, 1/2, 0])
    d1.make_dense()
    d2 = ScalarDistribution([1/2, 0, 1/2])
    d2.make_dense()
    assert_false(d1.is_approx_equal(d2))

def test_is_approx_equal3():
    d1 = ScalarDistribution([1/2, 1/2], sample_space=(0, 1, 2))
    d2 = ScalarDistribution([1/2, 1/2])
    assert_false(d1.is_approx_equal(d2))
