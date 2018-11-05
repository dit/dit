"""
Tests for dit.multivariate.secret_key_agreement.two_way_skar.py
"""
from __future__ import division

import pytest

import numpy as np

from dit import Distribution
from dit.example_dists.intrinsic import intrinsic_1
from dit.multivariate.secret_key_agreement import two_way_skar


def test_two_way_skar1():
    """
    Test simple example 1.
    """
    skar = two_way_skar(intrinsic_1, [[0], [1]], [2])
    assert skar == pytest.approx(0.0)

def test_two_way_skar2():
    """
    Test an unknown example (reduced or).
    """
    d = Distribution(['000', '011', '101'], [1/2, 1/4, 1/4])
    skar = two_way_skar(d, [[0], [2]], [1])
    assert np.isnan(skar)
