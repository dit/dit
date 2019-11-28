"""
Tests for dit.inference.binning.
"""

import numpy as np

from dit.inference import binned


def test_maxent_binning1():
    """
    Test maxent binning.
    """
    data = np.random.random(10)
    bd = binned(data)
    assert sum(bd == 0) == 5


def test_maxent_binning2():
    """
    Test maxent binning.
    """
    data = np.random.random((10, 2))
    bd = binned(data)
    assert np.all(bd.sum(axis=0) == [5, 5])


def test_uniform_binning():
    """
    Test maxent binning.
    """
    data = np.random.random(10)
    bins = np.array([0]*8 + [1]*2)
    bd = binned(data+(2*bins), style='uniform')
    assert np.all(bd == bins)
