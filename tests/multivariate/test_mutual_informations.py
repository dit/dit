"""
Tests for the various mutual informations.
"""

import pytest

from dit import random_distribution
from dit.multivariate import caekl_mutual_information as J
from dit.multivariate import coinformation as I
from dit.multivariate import dual_total_correlation as B
from dit.multivariate import interaction_information as II
from dit.multivariate import total_correlation as T


@pytest.mark.parametrize('d', [random_distribution(2, 4) for _ in range(10)])
def test_mis1(d):
    """
    Test that all the mutual informations match for bivariate distributions.
    """
    i = I(d)
    t = T(d)
    b = B(d)
    j = J(d)
    ii = II(d)
    assert i == pytest.approx(t)
    assert t == pytest.approx(b)
    assert b == pytest.approx(j)
    assert j == pytest.approx(ii)
