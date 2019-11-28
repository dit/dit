"""
Tests for dit.rate_distortion.rate_distortion
"""

from __future__ import division

import pytest

import numpy as np

from dit import Distribution
from dit.rate_distortion.rate_distortion import RateDistortionHamming


def test_rd():
    """
    Test specific RD optimizer.
    """
    dist = Distribution(['0', '1'], [1/2, 1/2])
    rd = RateDistortionHamming.functional()
    r, d = rd(dist, beta=0.0)
    assert d == pytest.approx(0.5, abs=1e-5)