"""
Tests for dit.multivariate.secret_key_agreement.two_way_skar.py
"""

import numpy as np
import pytest

from dit import Distribution
from dit.example_dists.intrinsic import intrinsic_1
from dit.multivariate.secret_key_agreement import two_way_skar
from tests._backends import backends


@pytest.mark.parametrize('backend', backends)
def test_two_way_skar1(backend):
    """
    Test simple example 1.
    """
    skar = two_way_skar(intrinsic_1, [[0], [1]], [2], backend=backend)
    assert skar == pytest.approx(0.0)


@pytest.mark.parametrize('backend', backends)
def test_two_way_skar2(backend):
    """
    Test an unknown example (reduced or).
    """
    d = Distribution(['000', '011', '101'], [1 / 2, 1 / 4, 1 / 4])
    skar = two_way_skar(d, [[0], [2]], [1], backend=backend)
    assert np.isnan(skar)
