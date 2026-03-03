"""
Tests for dit.inference.pycounts
"""

import numpy as np

from dit import Distribution
from dit.inference import distribution_from_data


def test_dfd():
    """
    Test distribution_from_data.
    """
    data = [0, 0, 0, 1, 1, 1]
    d1 = Distribution([(0,), (1,)], [1 / 2, 1 / 2])
    d2 = Distribution([(0, 0), (0, 1), (1, 1)], [2 / 5, 1 / 5, 2 / 5])
    d1_ = distribution_from_data(data, 1, base="linear")
    d2_ = distribution_from_data(data, 2, base="linear")
    assert d1.is_approx_equal(d1_)
    assert d2.is_approx_equal(d2_)


def test_dfd_three_variables():
    """
    With shape (N, 3) and L=1, should get a 3-variable distribution, not just first column.
    """
    data = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0], [0, 0, 0]])
    d = distribution_from_data(data, L=1, base="linear")
    # Should have 3 variables (outcome_length), not 1
    assert d.outcome_length() == 3
    # Outcomes should be 3-tuples
    assert all(len(o) == 3 for o in d.outcomes)
