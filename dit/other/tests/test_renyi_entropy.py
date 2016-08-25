"""
Tests for dit.other.renyi_entropy.
"""

from __future__ import division

import pytest

import numpy as np

from dit import Distribution
from dit.example_dists import uniform
from dit.other import renyi_entropy

@pytest.mark.parametrize('alpha', [0, 1/2, 1, 2, 5, np.inf])
def test_renyi_entropy_1(alpha):
    """
    Test that the Renyi entropy of a uniform distirbution is indipendent of the
    order.
    """
    d = uniform(8)
    assert renyi_entropy(d, alpha) == pytest.approx(3)

@pytest.mark.parametrize('alpha', [0, 1/2, 1, 2, 5, np.inf])
def test_renyi_entropy_2(alpha):
    """
    Test the Renyi entropy of joint distributions.
    """
    d = Distribution(['00', '11', '22', '33'], [1/4]*4)
    assert renyi_entropy(d, alpha) == pytest.approx(2)
    assert renyi_entropy(d, alpha, [0]) == pytest.approx(2)
    assert renyi_entropy(d, alpha, [1]) == pytest.approx(2)

@pytest.mark.parametrize('alpha', [-np.inf, -5, -1, -1/2, -0.0000001])
def test_renyi_entropy_3(alpha):
    """
    Test that negative orders raise ValueErrors.
    """
    d = uniform(8)
    with pytest.raises(ValueError):
        renyi_entropy(d, alpha)
