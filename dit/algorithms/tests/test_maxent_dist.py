"""
Tests for dit.algorithms.matextropyfw.maxent_dist.
"""

from __future__ import division

import pytest

from itertools import product

from dit.multivariate import entropy as H
from dit.distconst import uniform
from dit.algorithms import maxent_dist


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

def test_maxent_3():
    """
    Test the RdnUnqXor distribution.
    """
    X00, X01, X02, Y01, Y02 = 'rR', 'aA', [0, 1], 'bB', [0, 1]
    inputs = product(X00, X01, X02, Y01, Y02)
    events = [(x00+x01+str(x02), x00+y01+str(y02), x00+x01+y01+str(x02^y02)) for x00, x01, x02, y01, y02 in inputs]
    RdnUnqXor = uniform(events)
    d = maxent_dist(RdnUnqXor, [[0,1],[0,2],[1,2]])
    assert H(d) == pytest.approx(6)
