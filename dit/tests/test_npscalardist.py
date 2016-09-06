"""
Tests for dit.npscalardist.
"""

from __future__ import division

import pytest

import numpy as np

from itertools import product

from dit import Distribution, ScalarDistribution
from dit.distconst import uniform_scalar_distribution
from dit.exceptions import ditException, InvalidDistribution, InvalidOutcome

def test_init1():
    dist = {0: 1/2, 1: 1/2}
    d = ScalarDistribution(dist)
    assert d.outcomes == (0, 1)
    assert np.allclose(d.pmf, [1/2, 1/2])

def test_init2():
    dist = {0: 1/2, 1: 1/2}
    pmf = [1/2, 1/2]
    with pytest.raises(ditException):
        ScalarDistribution(dist, pmf)

def test_init3():
    pmf = ['a', 'b', 'c']
    with pytest.raises(ditException):
        ScalarDistribution(pmf)

def test_init4():
    outcomes = float
    pmf = int
    with pytest.raises(TypeError):
        ScalarDistribution(outcomes, pmf)

def test_init5():
    outcomes = [0, 1, 2]
    pmf = [1/2, 1/2]
    with pytest.raises(InvalidDistribution):
        ScalarDistribution(outcomes, pmf)

def test_init6():
    outcomes = set([0, 1])
    pmf = [1/2, 1/2]
    with pytest.raises(ditException):
        ScalarDistribution(outcomes, pmf)

def test_init7():
    outcomes = [0, 1, 2]
    pmf = [1/2, 1/2, 0]
    d = ScalarDistribution(outcomes, pmf, trim=True)
    assert len(d.outcomes) == 2

def test_init8():
    outcomes = [0, 1, 2]
    pmf = [1/3]*3
    d1 = ScalarDistribution(outcomes, pmf)
    d2 = ScalarDistribution.from_distribution(d1)
    assert d1.is_approx_equal(d2)

def test_init9():
    outcomes = [0, 1, 2]
    pmf = [1/3]*3
    d1 = ScalarDistribution(outcomes, pmf)
    d2 = ScalarDistribution.from_distribution(d1, base=10)
    d1.set_base(10)
    assert d1.is_approx_equal(d2)

def test_init10():
    outcomes = []
    pmf = []
    with pytest.raises(InvalidDistribution):
        ScalarDistribution(outcomes, pmf)

def test_init11():
    outcomes = ['0', '1']
    pmf = [1/2, 1/2]
    d = Distribution(outcomes, pmf)
    sd = ScalarDistribution.from_distribution(d)
    # Different sample space representations
    assert not d.is_approx_equal(sd)

def test_init12():
    outcomes = ['0', '1']
    pmf = [1/2, 1/2]
    d = Distribution(outcomes, pmf)
    sd = ScalarDistribution.from_distribution(d, base=10)
    d.set_base(10)
    # Different sample space representations
    assert not d.is_approx_equal(sd)

def test_del1():
    d = ScalarDistribution([1/2, 1/2])
    del d[1]
    d.normalize()
    assert d[0] == pytest.approx(1)

def test_del2():
    d = ScalarDistribution([1/2, 1/2])
    d.make_dense()
    del d[1]
    d.normalize()
    assert d[0] == pytest.approx(1)

def test_del3():
    d = ScalarDistribution([1/2, 1/2])
    with pytest.raises(InvalidOutcome):
        d.__delitem__(2)

def test_setitem1():
    pmf = [1/2, 1/2]
    d = ScalarDistribution(pmf)
    d[0] = 1
    d[1] = 0
    d.make_sparse()
    assert d.outcomes == (0,)

def test_setitem2():
    pmf = [1/2, 1/2]
    d = ScalarDistribution(pmf)
    with pytest.raises(InvalidOutcome):
        d.__setitem__(2, 1/2)

def test_setitem3():
    outcomes = (0, 1)
    pmf = [1/2, 1/2]
    d = ScalarDistribution(outcomes, pmf, sample_space=(0, 1, 2))
    d[2] = 1/2
    d.normalize()
    assert np.allclose(d.pmf, [1/3]*3)

def test_has_outcome1():
    d = ScalarDistribution([1, 0])
    assert d.has_outcome(1)

def test_has_outcome2():
    d = ScalarDistribution([1, 0])
    assert not d.has_outcome(1, null=False)

def test_has_outcome3():
    d = ScalarDistribution([1, 0])
    assert not d.has_outcome(2, null=False)

def test_is_approx_equal1():
    d1 = ScalarDistribution([1/2, 1/2, 0])
    d1.make_dense()
    d2 = ScalarDistribution([1/2, 1/2, 0])
    d2.make_dense()
    assert d1.is_approx_equal(d2)

def test_is_approx_equal2():
    d1 = ScalarDistribution([1/2, 1/2, 0])
    d1.make_dense()
    d2 = ScalarDistribution([1/2, 0, 1/2])
    d2.make_dense()
    assert not d1.is_approx_equal(d2)

def test_is_approx_equal3():
    d1 = ScalarDistribution([1/2, 1/2], sample_space=(0, 1, 2))
    d2 = ScalarDistribution([1/2, 1/2])
    assert not d1.is_approx_equal(d2)

def test_add():
    d1 = uniform_scalar_distribution(range(3))
    d2 = uniform_scalar_distribution(range(1, 4))
    d3 = ScalarDistribution([0, 1, 2, 3, 4], [1/9, 2/9, 3/9, 2/9, 1/9])
    assert (d1+1).is_approx_equal(d2)
    assert (1+d1).is_approx_equal(d2)
    assert (d2).is_approx_equal(d1+1)
    assert (d2).is_approx_equal(1+d1)
    assert (d1+d1).is_approx_equal(d3)

def test_sub():
    d1 = uniform_scalar_distribution(range(3))
    d2 = uniform_scalar_distribution(range(1, 4))
    d3 = ScalarDistribution([-2, -1, 0, 1, 2], [1/9, 2/9, 3/9, 2/9, 1/9])
    assert (d2-1).is_approx_equal(d1)
    assert (3-d2).is_approx_equal(d1)
    assert (d1).is_approx_equal(d2-1)
    assert (d1).is_approx_equal(3-d2)
    assert (d1-d1).is_approx_equal(d3)

def test_mul():
    d1 = uniform_scalar_distribution(range(1, 3))
    d2 = ScalarDistribution([1, 2, 4], [0.25, 0.5, 0.25])
    d3 = ScalarDistribution([2, 4], [0.5, 0.5])
    assert (d1*d1).is_approx_equal(d2)
    assert (d1*2).is_approx_equal(d3)
    assert (2*d1).is_approx_equal(d3)

def test_div():
    d1 = uniform_scalar_distribution([2,4,6])
    d2 = uniform_scalar_distribution([1,2,3,4,6])
    d3 = uniform_scalar_distribution([1,2,3])
    d4 = uniform_scalar_distribution([12,6,4,3,2])
    d5 = uniform_scalar_distribution([1,2])
    d6 = ScalarDistribution([1,2,3,4,6], [1/6,1/3,1/6,1/6,1/6])
    assert (d1/2).is_approx_equal(d3)
    assert (12/d2).is_approx_equal(d4)
    assert (d1/d5).is_approx_equal(d6)

def test_floordiv():
    d1 = uniform_scalar_distribution(range(1, 7))
    d2 = ScalarDistribution([0,1,2,3], [1/6,1/3,1/3,1/6])
    d3 = ScalarDistribution([1,2,3,6], [1/2,1/6,1/6,1/6])
    d4 = uniform_scalar_distribution(range(1, 3))
    d5 = ScalarDistribution([0,1,2,3,4,5,6],[1/12,1/4,1/4,1/6,1/12,1/12,1/12])
    assert (d1//2).is_approx_equal(d2)
    assert (6//d1).is_approx_equal(d3)
    assert (d1//d4).is_approx_equal(d5)

def test_mod():
    d1 = uniform_scalar_distribution(range(1,7))
    d2 = uniform_scalar_distribution(range(2))
    d3 = uniform_scalar_distribution(range(1,3))
    d4 = ScalarDistribution([0,1],[3/4,1/4])
    assert (d1%2).is_approx_equal(d2)
    assert (5%d3).is_approx_equal(d2)
    assert (d1%d3).is_approx_equal(d4)

def test_lt():
    d1 = uniform_scalar_distribution(range(1,7))
    d2 = uniform_scalar_distribution(range(2))
    d3 = ScalarDistribution([True, False], [11/12, 1/12])
    assert (d1 < 4).is_approx_equal(d2)
    assert (d2 < d1).is_approx_equal(d3)

def test_le():
    d1 = uniform_scalar_distribution(range(1,7))
    d2 = uniform_scalar_distribution(range(2))
    d3 = ScalarDistribution([True], [1])
    assert (d1 <= 3).is_approx_equal(d2)
    assert (d2 <= d1).is_approx_equal(d3)

def test_eq():
    d1 = uniform_scalar_distribution(range(1,7))
    d2 = ScalarDistribution([True, False], [1/6, 5/6])
    assert (d1 == 6).is_approx_equal(d2)
    assert (d1 == d1).is_approx_equal(d2)

def test_ne():
    d1 = uniform_scalar_distribution(range(1,7))
    d2 = ScalarDistribution([True, False], [5/6, 1/6])
    assert (d1 != 6).is_approx_equal(d2)
    assert (d1 != d1).is_approx_equal(d2)

def test_gt():
    d1 = uniform_scalar_distribution(range(1,7))
    d2 = uniform_scalar_distribution(range(2))
    d3 = ScalarDistribution([True, False], [11/12, 1/12])
    assert (d1 > 3).is_approx_equal(d2)
    assert (d1 > d2).is_approx_equal(d3)

def test_ge():
    d1 = uniform_scalar_distribution(range(1,7))
    d2 = uniform_scalar_distribution(range(2))
    d3 = ScalarDistribution([True], [1])
    assert (d1 >= 4).is_approx_equal(d2)
    assert (d1 >= d2).is_approx_equal(d3)

def test_matmul():
    d1 = uniform_scalar_distribution(range(1,7))
    d2 = Distribution(list(product(d1.outcomes, repeat=2)), [1/36]*36)
    assert (d1.__matmul__(d1)).is_approx_equal(d2)

def test_cmp_fail():
    d1 = uniform_scalar_distribution(range(1,7))
    with pytest.raises(NotImplementedError):
        (lambda x: x + '0')(d1)

def test_matmul_fail():
    d1 = uniform_scalar_distribution(range(1,7))
    with pytest.raises(NotImplementedError):
        (lambda x: x.__matmul__('0'))(d1)
