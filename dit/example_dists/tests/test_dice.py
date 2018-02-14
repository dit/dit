"""
Tests for dit.example_dists.dice
"""

import pytest

import numpy as np

from dit.example_dists.dice import iid_sum, Wolfs_dice
from dit.multivariate import entropy as H, total_correlation as I


def test_iidsum():
    """
    Test against known value.
    """
    d = iid_sum(2, 6)
    cmi = I(d, [[0], [1]], [2])
    assert cmi == pytest.approx(1.8955230821535425)


def test_wolf():
    """
    Test against value from publication.
    """
    d = Wolfs_dice()
    entropy = H(d, 'W')/np.log2(np.e)
    assert entropy == pytest.approx(1.784990, abs=1e-6)
