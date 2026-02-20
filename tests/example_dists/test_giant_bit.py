"""
Tests for dit.example_dists.giant_bit
"""

import numpy as np
import pytest

from dit.example_dists.giant_bit import giant_bit
from dit.multivariate import coinformation as I
from dit.multivariate import entropy as H


@pytest.mark.parametrize("n", range(1, 5))
@pytest.mark.parametrize("k", range(1, 5))
def test_giant_bit1(n, k):
    """
    tests for the giant bit entropy
    """
    d = giant_bit(n, k)
    assert H(d) == pytest.approx(np.log2(k))


@pytest.mark.parametrize("n", range(1, 5))
@pytest.mark.parametrize("k", range(1, 5))
def test_giant_bit2(n, k):
    """
    tests for the giant bit coinformation
    """
    d = giant_bit(n, k)
    assert I(d) == pytest.approx(np.log2(k))
