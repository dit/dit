"""
Tests for dit.profiles.entropy_triangle. Known examples taken from http://arxiv.org/abs/1409.4708 .
"""

from __future__ import division

import pytest

from dit import Distribution
from dit.profiles import EntropyTriangle, EntropyTriangle2

ex1 = Distribution(['000', '001', '010', '011', '100', '101', '110', '111'], [1/8]*8)
ex2 = Distribution(['000', '111'], [1/2]*2)
ex3 = Distribution(['000', '001', '110', '111'], [1/4]*4)
ex4 = Distribution(['000', '011', '101', '110'], [1/4]*4)

@pytest.mark.parametrize(('d', 'val'), [
    (ex1, (0, 0, 1)),
    (ex2, (0, 1, 0)),
    (ex3, (0, 2/3, 1/3)),
    (ex4, (0, 1, 0)),
])
def test_et_1(d, val):
    """
    Test EntropyTriangle against known values.
    """
    assert EntropyTriangle(d).points[0] == val

@pytest.mark.parametrize('val', [(0, 0, 1), (0, 1, 0), (0, 2/3, 1/3), (0, 1, 0)])
def test_et_2(val):
    """
    Test EntropyTriangle against known values.
    """
    et = EntropyTriangle([ex1, ex2, ex3, ex4])
    assert val in et.points

@pytest.mark.parametrize(('d', 'val'), [
    (ex1, (1, 0, 0)),
    (ex2, (0, 2/3, 1/3)),
    (ex3, (1/3, 1/3, 1/3)),
    (ex4, (0, 1/3, 2/3)),
])
def test_et2_1(d, val):
    """
    Test EntropyTriangle2 against known values.
    """
    assert EntropyTriangle2(d).points[0] == val

@pytest.mark.parametrize('val', [(1, 0, 0), (0, 2/3, 1/3), (1/3, 1/3, 1/3), (0, 1/3, 2/3)])
def test_et_2(val):
    """
    Test EntropyTriangle against known values.
    """
    et = EntropyTriangle2([ex1, ex2, ex3, ex4])
    assert val in et.points
