from __future__ import division

from nose.tools import assert_almost_equal

from dit import (ScalarDistribution as SD,
                 Distribution as D)
from dit.algorithms import perplexity as P
from six.moves import range


def test_p1():
    for i in range(2, 10):
        yield assert_almost_equal, P(SD([1/i]*i)), i

def test_p2():
    for i in range(2, 10):
        d = SD([1/i]*i)
        d.set_base(i)
        yield assert_almost_equal, P(d), i

def test_p3():
    for i in range(2, 10):
        d = D([str(_) for _ in range(i)], [1/i]*i)
        yield assert_almost_equal, P(d), i

def test_p4():
    for i in range(2, 10):
        d = D([str(_) for _ in range(i)], [1/i]*i)
        d.set_base(i)
        yield assert_almost_equal, P(d), i

def test_p5():
    d = D(['00', '01', '10', '11'], [1/4]*4)
    assert_almost_equal(P(d), 4)
    assert_almost_equal(P(d, [0]), 2)
    assert_almost_equal(P(d, [1]), 2)
    assert_almost_equal(P(d, [0], [1]), 2)
    assert_almost_equal(P(d, [1], [0]), 2)

def test_p6():
    d = D(['00', '11'], [1/2]*2)
    assert_almost_equal(P(d), 2)
    assert_almost_equal(P(d, [0], [1]), 1)
    assert_almost_equal(P(d, [1], [0]), 1)
