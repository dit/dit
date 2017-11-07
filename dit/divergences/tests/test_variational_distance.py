"""
Tests for dit.divergences.variational_distance
"""

from __future__ import division

import pytest

from dit import Distribution
from dit.divergences import variational_distance


def test_vd1():
    """
    Test against known value
    """
    d1 = Distribution(['0', '1'], [1/2, 1/2])
    d2 = Distribution(['0', '1'], [1/4, 3/4])
    v = variational_distance(d1, d2)
    assert v == pytest.approx(0.25)


def test_vd2():
    """
    Test against known value
    """
    d1 = Distribution(['0', '1'], [1/2, 1/2])
    d2 = Distribution(['1', '2'], [1/4, 3/4])
    v = variational_distance(d1, d2)
    assert v == pytest.approx(0.75)
