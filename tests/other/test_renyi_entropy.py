"""
Tests for dit.other.renyi_entropy.
"""

import numpy as np
import pytest

from dit import Distribution
from dit.example_dists import uniform
from dit.other import renyi_entropy
from dit.other.renyi_entropy import (
    sibson_mutual_information,
    sibson_mutual_information_pmf,
)


@pytest.mark.parametrize("alpha", [0, 1 / 2, 1, 2, 5, np.inf])
def test_renyi_entropy_1(alpha):
    """
    Test that the Renyi entropy of a uniform distirbution is indipendent of the
    order.
    """
    d = uniform(8)
    assert renyi_entropy(d, alpha) == pytest.approx(3)


@pytest.mark.parametrize("alpha", [0, 1 / 2, 1, 2, 5, np.inf])
def test_renyi_entropy_2(alpha):
    """
    Test the Renyi entropy of joint distributions.
    """
    d = Distribution(["00", "11", "22", "33"], [1 / 4] * 4)
    assert renyi_entropy(d, alpha) == pytest.approx(2)
    assert renyi_entropy(d, alpha, [0]) == pytest.approx(2)
    assert renyi_entropy(d, alpha, [1]) == pytest.approx(2)


@pytest.mark.parametrize("alpha", [-np.inf, -5, -1, -1 / 2, -0.0000001])
def test_renyi_entropy_3(alpha):
    """
    Test that negative orders raise ValueErrors.
    """
    d = uniform(8)
    with pytest.raises(ValueError, match="`order` must be a non-negative real number"):
        renyi_entropy(d, alpha)


@pytest.mark.parametrize("alpha", [0, 1 / 2, 1, 2, 5, np.inf])
def test_sibson_mutual_information_1(alpha):
    """
    For a uniform distribution every order yields the same value.
    """
    d = uniform(8)
    assert sibson_mutual_information(d, alpha) == pytest.approx(3)


@pytest.mark.parametrize("alpha", [0, 1 / 2, 1, 2, 5, np.inf])
def test_sibson_mutual_information_2(alpha):
    """
    Test the rvs / marginalization path on a joint distribution.
    """
    d = Distribution(["00", "11", "22", "33"], [1 / 4] * 4)
    assert sibson_mutual_information(d, alpha) == pytest.approx(2)
    assert sibson_mutual_information(d, alpha, [0]) == pytest.approx(2)


@pytest.mark.parametrize("alpha", [-np.inf, -5, -1, -1 / 2, -0.0000001])
def test_sibson_mutual_information_3(alpha):
    """
    Negative orders raise ValueErrors.
    """
    d = uniform(8)
    with pytest.raises(ValueError, match="`order` must be a non-negative real number"):
        sibson_mutual_information(d, alpha)


@pytest.mark.parametrize(("alpha", "expected"), [(2, 2.0), (5, 2.0), (np.inf, 2.0)])
def test_sibson_mutual_information_pmf_1(alpha, expected):
    """
    The pmf form on a uniform 2x2 joint yields log2 of the support size.
    """
    p_xy = np.full((2, 2), 1 / 4)
    assert sibson_mutual_information_pmf(p_xy, alpha) == pytest.approx(expected)


@pytest.mark.parametrize("alpha", [0, -1, -np.inf])
def test_sibson_mutual_information_pmf_2(alpha):
    """
    Non-positive orders raise ValueErrors.
    """
    p_xy = np.full((2, 2), 1 / 4)
    with pytest.raises(ValueError, match="`order` must be a non-zero positive real number"):
        sibson_mutual_information_pmf(p_xy, alpha)
