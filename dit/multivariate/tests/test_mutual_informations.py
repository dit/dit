"""
Tests for the various mutual informations.
"""

import pytest

from dit import random_distribution
from dit.multivariate import (coinformation as I,
                              total_correlation as T,
                              dual_total_correlation as B,
                              caekl_mutual_information as J,
                              interaction_information as II,
                             )

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
