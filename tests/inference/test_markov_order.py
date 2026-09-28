"""
Tests for dit.inference.markov_order.
"""

import numpy as np
import pytest
from scipy.stats import chi2_contingency

from dit.inference import dist_from_timeseries, markov_order_test, select_markov_order
from dit.inference.markov_order import _statistics


def _chain(order, M, N, rng):
    """
    A strongly `order`-th order Markov chain, as constructed by Pethel & Hahs.
    """
    P = (rng.random((M**order, M)) + 1) ** 10
    P /= P.sum(axis=1, keepdims=True)
    x = list(rng.integers(0, M, order))
    for _ in range(N - order):
        state = 0
        for v in x[len(x) - order :]:
            state = state * M + v
        x.append(rng.choice(M, p=P[state]))
    return np.array(x)


def _even_process(N, rng):
    x, state = [], 0
    for _ in range(N):
        if state == 0 and rng.random() < 0.5:
            x.append(0)
        else:
            x.append(1)
            state = 1 - state
    return np.array(x)


def test_chi2_matches_contingency():
    """
    For order 0 the statistic is Pearson's test of independence of consecutive symbols.
    """
    x = _chain(1, 3, 500, np.random.default_rng(0))
    table = np.zeros((3, 3))
    for a, b in zip(x[:-1], x[1:], strict=False):
        table[a, b] += 1
    stat, _, dof, _ = chi2_contingency(table, correction=False)
    _, _, chi_sq, our_dof, _ = _statistics([x], 0, 3)
    assert chi_sq == pytest.approx(stat)
    assert our_dof == dof


@pytest.mark.parametrize("statistic", ["entropy_rate", "chi2"])
def test_rejects_too_small_order(statistic):
    x = _chain(2, 3, 600, np.random.default_rng(1))
    assert markov_order_test(x, 1, statistic=statistic, n_surrogates=200, prng=0).pvalue < 0.05
    assert markov_order_test(x, 2, statistic=statistic, n_surrogates=200, prng=0).pvalue > 0.05


def test_exact_pvalue_bounds():
    x = _chain(1, 2, 100, np.random.default_rng(2))
    result = markov_order_test(x, 1, n_surrogates=99, prng=0)
    assert 0.01 <= result.pvalue <= 1.0
    assert result.null.shape == (99,)


def test_exact_size():
    """
    Under the null, the exact test rejects at about the nominal rate.
    """
    rng = np.random.default_rng(3)
    trials = 200
    rejections = sum(
        markov_order_test(_chain(1, 3, 60, rng), 1, n_surrogates=99, prng=t).pvalue <= 0.05 for t in range(trials)
    )
    assert rejections / trials < 0.1


@pytest.mark.parametrize("method", ["exact", "chi2", "aic", "bic"])
def test_select_markov_order(method):
    x = _chain(2, 3, 1000, np.random.default_rng(4))
    assert select_markov_order(x, 4, method=method, n_surrogates=200, prng=0) == 2


def test_even_process_order_grows():
    """
    The even process has infinite Markov order; more data supports longer histories.
    """
    rng = np.random.default_rng(5)
    short = select_markov_order(_even_process(200, rng), 8, method="bic")
    long = select_markov_order(_even_process(20000, rng), 8, method="bic")
    assert long > short


def test_dist_from_timeseries_auto():
    x = _chain(2, 2, 2000, np.random.default_rng(6))
    d = dist_from_timeseries(x, history_length="auto", prng=0)
    assert d.outcome_length() == 3
    d = dist_from_timeseries(x, history_length="bic")
    assert d.outcome_length() == 3


def test_invalid_arguments():
    with pytest.raises(ValueError):
        markov_order_test([0, 1, 0], 1, statistic="nope")
    with pytest.raises(ValueError):
        select_markov_order([0, 1, 0, 1], 1, method="nope")
