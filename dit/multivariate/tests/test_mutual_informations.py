"""
Tests for the various mutual informations.
"""

from nose.tools import assert_almost_equal

from dit import random_distribution
from dit.multivariate import (coinformation as I,
                              total_correlation as T,
                              dual_total_correlation as B,
                              caekl_mutual_information as J,
                              interaction_information as II,
                             )

def test_mis1():
    """
    Test that all the mutual informations match for bivariate distributions.
    """
    for d in [random_distribution(2, 4) for _ in range(10)]:
        i = I(d)
        t = T(d)
        b = B(d)
        j = J(d)
        ii = II(d)
        yield assert_almost_equal, i, t
        yield assert_almost_equal, t, b
        yield assert_almost_equal, b, j
        yield assert_almost_equal, j, ii
