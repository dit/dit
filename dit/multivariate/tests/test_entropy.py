"""
Tests for dit.multivariate.entropy.
"""

from __future__ import division

from nose.tools import assert_almost_equal

import numpy as np

from dit import Distribution as D, ScalarDistribution as SD
from dit.multivariate import entropy as H

def test_H1():
    """ Test H of a fair coin """
    d = D(['H', 'T'], [1/2, 1/2])
    assert_almost_equal(H(d), 1)

def test_H2():
    """ Test the various entropies of two independent events """
    d = D(['00', '01', '10', '11'], [1/4]*4)
    assert_almost_equal(H(d), 2)
    assert_almost_equal(H(d, [0]), 1)
    assert_almost_equal(H(d, [1]), 1)
    assert_almost_equal(H(d, [0], [1]), 1)
    assert_almost_equal(H(d, [1], [0]), 1)
    assert_almost_equal(H(d, [0], [0]), 0)
    assert_almost_equal(H(d, [0], [0, 1]), 0)

def test_H3():
    """ Test the various entropies of two independent events with names """
    d = D(['00', '01', '10', '11'], [1/4]*4)
    d.set_rv_names('XY')
    assert_almost_equal(H(d), 2)
    assert_almost_equal(H(d, ['X']), 1)
    assert_almost_equal(H(d, ['Y']), 1)
    assert_almost_equal(H(d, ['X'], ['Y']), 1)
    assert_almost_equal(H(d, ['Y'], ['X']), 1)
    assert_almost_equal(H(d, ['X'], ['X']), 0)
    assert_almost_equal(H(d, ['X'], ['X', 'Y']), 0)

def test_H4():
    """ Test H for uniform distributions """
    for i in range(2, 10):
        d = D([ str(_) for _ in range(i) ], [1/i]*i)
        yield assert_almost_equal, H(d), np.log2(i)

def test_H5():
    """ Test H for uniform distributions in various bases """
    for i in range(2, 10):
        d = D([ str(_) for _ in range(i) ], [1/i]*i)
        d.set_base(i)
        yield assert_almost_equal, H(d), 1

def test_H6():
    """ Test H for uniform distributions using ScalarDistributions """
    for i in range(2, 10):
        d = SD([1/i]*i)
        yield assert_almost_equal, H(d), np.log2(i)

def test_H7():
    """ Test H for uniform distributions using SDs in various bases """
    for i in range(2, 10):
        d = SD([1/i]*i)
        d.set_base(i)
        yield assert_almost_equal, H(d), 1
