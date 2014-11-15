"""
Tests for dit.other.perplexity.
"""

from __future__ import division

from nose.tools import assert_almost_equal

from dit import (ScalarDistribution as SD,
                 Distribution as D)
from dit.other import perplexity as P
from six.moves import range # pylint: disable=redefined-builtin


def test_p1():
    """ Test some simple base cases using SD """
    for i in range(2, 10):
        yield assert_almost_equal, P(SD([1/i]*i)), i

def test_p2():
    """ Test some simple base cases using SD with varying bases """
    for i in range(2, 10):
        d = SD([1/i]*i)
        d.set_base(i)
        yield assert_almost_equal, P(d), i

def test_p3():
    """ Test some simple base cases using D """
    for i in range(2, 10):
        d = D([str(_) for _ in range(i)], [1/i]*i)
        yield assert_almost_equal, P(d), i

def test_p4():
    """ Test some simple base cases using D with varying bases """
    for i in range(2, 10):
        d = D([str(_) for _ in range(i)], [1/i]*i)
        d.set_base(i)
        yield assert_almost_equal, P(d), i

def test_p5():
    """ Test some joint, marginal, and conditional perplexities """
    d = D(['00', '01', '10', '11'], [1/4]*4)
    assert_almost_equal(P(d), 4)
    assert_almost_equal(P(d, [0]), 2)
    assert_almost_equal(P(d, [1]), 2)
    assert_almost_equal(P(d, [0], [1]), 2)
    assert_almost_equal(P(d, [1], [0]), 2)

def test_p6():
    """ Test some joint and conditional perplexities """
    d = D(['00', '11'], [1/2]*2)
    assert_almost_equal(P(d), 2)
    assert_almost_equal(P(d, [0], [1]), 1)
    assert_almost_equal(P(d, [1], [0]), 1)
