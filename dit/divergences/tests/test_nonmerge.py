"""
Tests for dit.divergences._kl_nonmerge.
"""
from __future__ import division

import pytest

from dit import Distribution
from dit.divergences import pmf
from dit.divergences._kl_nonmerge import cross_entropy, relative_entropy

ds = [[1/2, 1/2], [3/5, 2/5]]


def test_cross_entropy():
    ce1 = pmf.cross_entropy(ds[0], ds[1])
    ce2 = pmf.cross_entropy(ds[1], ds[0])
    assert ce1 == pytest.approx(1.0294468445267841)
    assert ce2 == pytest.approx(1.0)


def test_entropy():
    e1 = pmf.cross_entropy(ds[0])
    e2 = pmf.cross_entropy(ds[1])
    assert e1 == pytest.approx(1.0)
    assert e2 == pytest.approx(0.97095059445466858)


def test_dkl():
    dkl1 = pmf.relative_entropy(ds[0], ds[1])
    dkl2 = pmf.relative_entropy(ds[1], ds[0])
    assert dkl1 == pytest.approx(0.029446844526784144)
    assert dkl2 == pytest.approx(0.029049405545331419)


def test_cross_entropy2():
    """
    Test against known value.
    """
    d1 = Distribution(['0', '1'], [1/2, 1/2])
    d2 = Distribution(['1', '2'], [1/2, 1/2])
    ce = cross_entropy(d1, d2)
    assert ce == pytest.approx(1.0)


def test_cross_entropy3():
    """
    Test against known value.
    """
    d1 = Distribution(['0', '1'], [1/2, 1/2])
    d2 = Distribution(['1', '2'], [1/2, 1/2])
    ce = cross_entropy(d1, d2, pmf_only=False)
    assert ce == pytest.approx(1.0)


def test_relative_entropy2():
    """
    Test against known value.
    """
    d1 = Distribution(['0', '1'], [1/2, 1/2])
    d2 = Distribution(['1', '2'], [1/2, 1/2])
    ce = relative_entropy(d1, d2)
    assert ce == pytest.approx(0.0)
