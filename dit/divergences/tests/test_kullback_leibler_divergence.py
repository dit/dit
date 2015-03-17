"""
Tests for dit.divergences.kullback_leiber_divergence.
"""

from __future__ import division

from nose.tools import assert_almost_equal, assert_raises

import numpy as np

from dit import Distribution
from dit.divergences import kullback_leibler_divergence
from dit.exceptions import ditException

def get_dists():
    """
    Construct several example distributions.
    """
    d1 = Distribution(['0', '1'], [1/2, 1/2])
    d2 = Distribution(['0', '2'], [1/2, 1/2])
    d3 = Distribution(['0', '1', '2'], [1/3, 1/3, 1/3])
    d4 = Distribution(['00', '11'], [2/5, 3/5])
    d5 = Distribution(['00', '11'], [1/2, 1/2])
    return d1, d2, d3, d4, d5

def test_dkl_1():
    """
    Test against several known values.
    """
    d1, d2, d3, d4, d5 = get_dists()
    tests = [([d1, d3], 0.5849625007211563),
             ([d1, d4], 0.0294468445267841),
             ([d1, d3, [0]], 0.5849625007211563),
             ([d1, d4, [0]], 0.0294468445267841),
             ([d4, d1, [0]], 0.029049405545331419),
             ([d4, d5], 0.029049405545331419),
             ([d5, d4], 0.0294468445267841),
             ([d4, d5, [0], [1]], 0),
             ([d4, d5, [1], [0]], 0),
             ([d1, d2], np.inf),
             ([d2, d1], np.inf),
             ([d3, d1], np.inf)]
    for args, val in tests:
        yield assert_almost_equal, kullback_leibler_divergence(*args), val

def test_dkl_2():
    """
    Test that DKL(d, d) = 0.
    """
    ds = get_dists()
    for d in ds:
        yield assert_almost_equal, kullback_leibler_divergence(d, d), 0

def test_dkl_3():
    """
    Test that when p has outcomes that q doesn't have, that we raise an exception.
    """
    d1, d2, d3, d4, d5 = get_dists()
    tests = [[d4, d1, None, None],
             [d4, d2, None, None],
             [d4, d3, None, None],
             [d1, d2, [0, 1], None],
             [d3, d4, [1], None],
             [d5, d1, [0], [1]],
             [d4, d3, [1], [0]]]
    for first, second, rvs, crvs in tests:
        yield assert_raises, ditException, kullback_leibler_divergence, first, second, rvs, crvs
