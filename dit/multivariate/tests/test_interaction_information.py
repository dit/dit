"""
Tests for dit.multivariate.interaction_information.
"""

from __future__ import division

import pytest

from dit import Distribution as D
from dit.multivariate import interaction_information, coinformation
from dit.example_dists import Xor

@pytest.mark.parametrize('i', range(2, 6))
def test_ii1(i):
    """ Test II for giant bit distributions """
    outcomes = ['0'*i, '1'*i]
    pmf = [1/2, 1/2]
    d = D(outcomes, pmf)
    assert interaction_information(d) == pytest.approx((-1)**i)

@pytest.mark.parametrize('i', range(2, 6))
def test_ii2(i):
    """ Test II = -1^n * I for giant bit distributions """
    outcomes = ['0'*i, '1'*i]
    pmf = [1/2, 1/2]
    d = D(outcomes, pmf)
    ci = coinformation(d)
    ii = interaction_information(d)
    assert ii == pytest.approx((-1)**i * ci)

def test_ii3():
    """ Test II and conditional II for xor """
    d = Xor()
    ii1 = interaction_information(d, [[0], [1], [2]], [2])
    ii2 = interaction_information(d, [[0], [1]], [2])
    assert ii1 == pytest.approx(0)
    assert ii2 == pytest.approx(1)
