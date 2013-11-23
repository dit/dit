"""
Tests for dit.multivariate.interaction_information.
"""

from __future__ import division

from nose.tools import assert_almost_equal

from dit import Distribution as D
from dit.multivariate import interaction_information, coinformation
from dit.example_dists import Xor

def test_ii1():
    """ Test II for giant bit distributions """
    for i in range(2, 6):
        outcomes = ['0'*i, '1'*i]
        pmf = [1/2, 1/2]
        d = D(outcomes, pmf)
        yield assert_almost_equal, interaction_information(d), (-1)**i

def test_ii2():
    """ Test II = -1^n * I for giant bit distributions """
    for i in range(2, 6):
        outcomes = ['0'*i, '1'*i]
        pmf = [1/2, 1/2]
        d = D(outcomes, pmf)
        ci = coinformation(d)
        ii = interaction_information(d)
        yield assert_almost_equal, ii, (-1)**i * ci

def test_ii3():
    """ Test II and conditional II for xor """
    d = Xor()
    ii1 = interaction_information(d, [[0], [1], [2]], [2])
    ii2 = interaction_information(d, [[0], [1]], [2])
    assert_almost_equal(ii1, 0)
    assert_almost_equal(ii2, 1)
