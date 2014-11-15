"""
Tests for dit.other.extropy.
"""

from __future__ import division

from nose.tools import assert_almost_equal, assert_true

import numpy as np

from dit import Distribution as D, ScalarDistribution as SD
from dit.shannon import entropy as H
from dit.other import extropy as J

def test_J1():
    """ Test that H = J for two probabilities """
    assert_almost_equal(H(0.25), J(0.25))

def test_J2():
    """ Test a simple base case using ScalarDistribution """
    d = SD([1/2]*2)
    assert_almost_equal(J(d), 1)

def test_J3():
    """ Test a simple base case using Distribution """
    d = D(['00', '11'], [1/2, 1/2])
    assert_almost_equal(J(d), 1)
    assert_almost_equal(J(d, [0, 1]), 1)

def test_J4():
    """ Test a property of J from result 2 of the paper """
    for i in range(2, 10):
        d = SD([1/i]*i)
        yield assert_almost_equal, J(d), (i-1)*(np.log2(i) - np.log2(i-1))

def test_J5():
    """ Test a property of J from result 2 of the paper with a log base """
    for i in range(2, 10):
        d = SD([1/i]*i)
        d.set_base('e')
        yield assert_almost_equal, J(d), (i-1)*(np.log(i)-np.log(i-1))

def test_J6():
    """ Test a property of J from result 2 of the paper with base 10 """
    for i in range(2, 10):
        d = SD([1/i]*i)
        d.set_base(10)
        yield assert_almost_equal, J(d), (i-1)*(np.log10(i)-np.log10(i-1))

def test_J7():
    """ Test a property of J from result 1 of the paper """
    for i in range(3, 10):
        d = SD([1/i]*i)
        yield assert_true, J(d) < H(d)

def test_J8():
    """ Test a property of J from result 1 of the paper using log bases """
    for i in range(3, 10):
        d = SD([1/i]*i)
        d.set_base(i)
        yield assert_true, J(d) < H(d)

def test_J9():
    """ Test a property of J from result 4 of the paper """
    d = SD([1/2, 1/4, 1/8, 1/16, 1/16])
    j = J(d)
    h = H(d)
    s = sum(H(p) for p in d.pmf)
    assert_almost_equal(j, s-h)

def test_J10():
    """ Test a property of J from result 4 of the paper """
    d = SD([1/2, 1/4, 1/8, 1/16, 1/16])
    j = J(d)
    h = H(d)
    s = sum(J(p) for p in d.pmf)
    assert_almost_equal(h, s-j)

def test_J11():
    """ Test a property of J from result 5 of the paper """
    pmf = np.array([1/2, 1/4, 1/8, 1/16, 1/16])
    N = len(pmf)
    d = SD(pmf)
    q = SD(1/(N-1)*(1-pmf))
    j2 = (N-1)*(H(q)-np.log2(N-1))
    assert_almost_equal(J(d), j2)
