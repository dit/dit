from __future__ import division

from nose.tools import *

from dit import Distribution as D, ScalarDistribution as SD
from dit.algorithms import entropy as H, extropy as J

def test_J1():
    assert_almost_equal(H(0.25), J(0.25))

def test_J2():
    d = SD([1/2]*2)
    assert_almost_equal(J(d), 1)

def test_J3():
    d = D(['00', '11'], [1/2, 1/2])
    assert_almost_equal(J(d), 1)
    assert_almost_equal(J(d, [0,1]), 1)

def test_J4():
    d = SD([ 1/2**i for i in range(1, 10) ] + [1/2**9])
    for i in range(2, 10):
        d.set_base(i)
        assert_less(J(d), H(d))
