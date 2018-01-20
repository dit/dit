"""
Tests for dit.inference.time_series.
"""
from __future__ import division

import pytest

from random import choice

from dit import Distribution
from dit.inference import dist_from_timeseries


def golden_mean():
    """
    Generator of the golden mean process
    """
    val = choice([0, 1])
    while True:
        val = 0 if val == 1 else choice([0, 1])
        yield val


@pytest.mark.flaky(reruns=5)
def test_dfts():
    """
    Test inferring a distribution from a time-series.
    """
    gm = golden_mean()
    ts = [next(gm) for _ in range(1000000)]
    d1 = dist_from_timeseries(ts)
    d2 = Distribution([((0,), 0), ((0,), 1), ((1,), 0)], [1/3, 1/3, 1/3])
    assert d1.is_approx_equal(d2, atol=1e-3)


@pytest.mark.flaky(reruns=5)
def test_dfts():
    """
    Test inferring a distribution from a time-series.
    """
    gm = golden_mean()
    ts = [next(gm) for _ in range(1000000)]
    d1 = dist_from_timeseries(ts, base=3)
    d2 = Distribution([((0,), 0), ((0,), 1), ((1,), 0)], [-1, -1, -1], base=3)
    assert d1.is_approx_equal(d2, atol=1e-3)
