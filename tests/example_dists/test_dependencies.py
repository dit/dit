"""
Tests of dit.example_dists.dependencies
"""

from __future__ import division

import pytest

from dit.example_dists.dependencies import mixed, stacked
from dit.multivariate import coinformation, intrinsic_mutual_information


def test_mixed1():
    """
    Test against known values.
    """
    i = coinformation(mixed)
    assert i == pytest.approx(0.0)


def test_mixed2():
    """
    Test against known values.
    """
    i = coinformation(mixed, [[0], [1]], [2])
    assert i == pytest.approx(2.0)


def test_mixed3():
    """
    Test against known values.
    """
    i = intrinsic_mutual_information(mixed, [[0], [1]], [2])
    assert i == pytest.approx(1.0)


def test_stacked1():
    """
    Test against known values.
    """
    i = coinformation(stacked)
    assert i == pytest.approx(1.5849625007211565)


def test_stacked2():
    """
    Test against known values.
    """
    i = coinformation(stacked, [[0], [1]], [2])
    assert i == pytest.approx(2/3)


def test_stacked3():
    """
    Test against known values.
    """
    i = intrinsic_mutual_information(stacked, [[0], [1]], [2])
    assert i == pytest.approx(1/3)
