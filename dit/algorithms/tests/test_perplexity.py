from __future__ import division

from nose.tools import *

from dit import (ScalarDistribution as SD,
                 Distribution as D)
from dit.algorithms import perplexity as P
from six.moves import range


def test_p1():
    for i in range(2, 10):
        assert_almost_equal(P(SD([1/i]*i)), i)

def test_p2():
    for i in range(2, 10):
        d = SD([1/i]*i)
        d.set_base(i)
        assert_almost_equal(P(d), i)

def test_p3():
    for i in range(2, 10):
        d = D(range(i), [1/i]*i)
        assert_almost_equal(P(d), i)

def test_p4():
    for i in range(2, 10):
        d = D(range(i), [1/i]*i)
        d.set_base(i)
        assert_almost_equal(P(d), i)