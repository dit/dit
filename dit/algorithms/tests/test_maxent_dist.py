"""
Tests for dit.algorithms.matextropyfw.maxent_dist.
"""

from __future__ import division

import pytest

from dit import Distribution
from dit.distconst import uniform
from dit.algorithms import maxent_dist


pytest.importorskip('cvxopt')

@pytest.mark.parametrize('vars', [
    [[0], [1], [2]],
    [[0, 1], [2]],
    [[0, 2], [1]],
    [[0], [1, 2]],
    [[0, 1], [1, 2]],
    [[0, 1], [0, 2]],
    [[0, 2], [1, 2]],
    [[0, 1], [0, 2], [1, 2]]
])
def test_maxent_1(vars):
    """
    Test xor only fixing individual marginals.
    """
    d1 = uniform(['000', '011', '101', '110'])
    d2 = uniform(['000', '001', '010', '011', '100', '101', '110', '111'])
    d1_maxent = maxent_dist(d1, vars)
    assert d2.is_approx_equal(d1_maxent, rtol=1e-3, atol=1e-3)


def test_maxent_2():
    """
    Text a distribution with differing alphabets.
    """
    d1 = uniform(['00', '10', '21', '31'])
    d2 = uniform(['00', '01', '10', '11', '20', '21', '30', '31'])
    d1_maxent = maxent_dist(d1, [[0], [1]])
    assert d2.is_approx_equal(d1_maxent, rtol=1e-3, atol=1e-3)
