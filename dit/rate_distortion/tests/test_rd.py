"""
Tests for dit.rate_distortion.curves
"""

from __future__ import division

import pytest

import numpy as np

from dit import Distribution
from dit.rate_distortion import RateDistortionHamming


def test_rd():
    """
    """
    dist = Distribution(['0', '1'], [1/2, 1/2])
    rd = RateDistortionHamming.functional()
    r, d = rd(dist, beta=0.0)
    assert d == pytest.approx(0.5)