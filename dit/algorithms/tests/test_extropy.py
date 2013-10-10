from __future__ import division

from nose.tools import *

import numpy as np

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
    for i in range(2, 10):
        d = SD([1/i]*i)
        assert_almost_equal(J(d), (i-1)*(np.log2(i) - np.log2(i-1)))

# nose on travisCI with python 2.6 doesn't have assert_less
def test_J5():
    for i in range(3, 10):
        d = SD([1/i]*i)
        assert(J(d) < H(d))

def test_J6():
    for i in range(3, 10):
        d = SD([1/i]*i)
        d.set_base(i)
        assert(J(d) < H(d))