"""
Tests for dit.other.renyi_entropy.
"""

from __future__ import division

from nose.tools import assert_almost_equal, assert_raises

import numpy as np

from dit import Distribution
from dit.example_dists import uniform
from dit.other import renyi_entropy

def test_renyi_entropy_1():
    """
    Test that the Renyi entropy of a uniform distirbution is indipendent of the
    order.
    """
    d = uniform(8)
    for alpha in [0, 1/2, 1, 2, 5, np.inf]:
        yield assert_almost_equal, renyi_entropy(d, alpha), 3

def test_renyi_entropy_2():
    """
    Test the Renyi entropy of joint distributions.
    """
    d = Distribution(['00', '11', '22', '33'], [1/4]*4)
    for alpha in [0, 1/2, 1, 2, 5, np.inf]:
        yield assert_almost_equal, renyi_entropy(d, alpha), 2
        yield assert_almost_equal, renyi_entropy(d, alpha, [0]), 2
        yield assert_almost_equal, renyi_entropy(d, alpha, [1]), 2

def test_renyi_entropy_3():
    """
    Test that negative orders raise ValueErrors.
    """
    d = uniform(8)
    for alpha in [-np.inf, -5, -1, -1/2, -0.0000001]:
        yield assert_raises, ValueError, renyi_entropy, d, alpha
