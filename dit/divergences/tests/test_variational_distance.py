"""
Tests for dit.divergences.variational_distance
"""

from __future__ import division

import pytest

from dit import Distribution
from dit.divergences import (bhattacharyya_coefficient,
                             variational_distance,
                             )


d1 = Distribution(['0', '1'], [1/2, 1/2])
d2 = Distribution(['0', '1'], [1/4, 3/4])
d3 = Distribution(['1', '2'], [1/4, 3/4])


def test_vd1():
    """
    Test against known value.
    """
    vd = variational_distance(d1, d2)
    assert vd == pytest.approx(0.25)


def test_vd2():
    """
    Test against known value.
    """
    vd = variational_distance(d1, d3)
    assert vd == pytest.approx(0.75)


def test_bc():
    """
    Test against known value.
    """
    bc = bhattacharyya_coefficient(d1, d2)
    assert bc == pytest.approx(0.9659258262890682)
