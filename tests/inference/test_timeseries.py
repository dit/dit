"""
Tests for dit.inference.time_series.
"""

from random import choice

import numpy as np
import pytest

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
def test_dfts1():
    """
    Test inferring a distribution from a time-series.
    """
    gm = golden_mean()
    ts = [next(gm) for _ in range(1000000)]
    d1 = dist_from_timeseries(ts)
    d2 = Distribution([(0, 0), (0, 1), (1, 0)], [1 / 3, 1 / 3, 1 / 3])
    assert d1.is_approx_equal(d2, atol=1e-3)


@pytest.mark.flaky(reruns=5)
def test_dfts2():
    """
    Test inferring a distribution from a time-series.
    """
    gm = golden_mean()
    ts = [next(gm) for _ in range(1000000)]
    d1 = dist_from_timeseries(ts, base=None)
    d2 = Distribution([(0, 0), (0, 1), (1, 0)], [np.log2(1 / 3)] * 3, base=2)
    assert d1.is_approx_equal(d2, atol=1e-2)


@pytest.mark.flaky(reruns=5)
def test_dfts3():
    """
    Test inferring a distribution from a time-series.
    """
    gm = golden_mean()
    ts = [next(gm) for _ in range(1000000)]
    d1 = dist_from_timeseries(ts, history_length=0)
    d2 = Distribution([(0,), (1,)], [2 / 3, 1 / 3])
    assert d1.is_approx_equal(d2, atol=1e-3)


@pytest.mark.flaky(reruns=5)
def test_dfts4():
    """
    Test inferring a distribution from a time-series.
    """
    gm = golden_mean()
    ts = np.array([next(gm) for _ in range(1000000)]).reshape(1000000, 1)
    d1 = dist_from_timeseries(ts)
    d2 = Distribution([(0, 0), (0, 1), (1, 0)], [1 / 3, 1 / 3, 1 / 3])
    assert d1.is_approx_equal(d2, atol=1e-3)


def test_dfts_multivariate_history1():
    """
    Multivariate series with history_length=1 triggers the variable-grouped
    reorder branch; outcomes are (past1, past2, present1, present2).
    """
    obs = [(0, 1), (1, 0)] * 50
    d = dist_from_timeseries(obs, history_length=1)
    assert d.outcome_length() == 4
    assert set(d.outcomes) == {(0, 1, 1, 0), (1, 0, 0, 1)}


def test_dfts_multivariate_history2():
    """
    Multivariate series with history_length=2 regroups each length-3 window
    from time-interleaved (v1_t0, v2_t0, ...) to variable-grouped
    (p1_t0, p1_t1, p2_t0, p2_t1, present1, present2).
    """
    obs = [(0, 0), (0, 1), (1, 0)] * 40
    d = dist_from_timeseries(obs, history_length=2)
    assert d.outcome_length() == 6
    assert set(d.outcomes) == {
        (0, 0, 0, 1, 1, 0),
        (0, 1, 1, 0, 0, 0),
        (1, 0, 0, 0, 0, 1),
    }
