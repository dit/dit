"""
Tests for dit.algorithms.stats
"""

from math import ceil, floor

import numpy as np
import pytest

from dit import Distribution as D
from dit.algorithms import (
    cdf,
    central_moment,
    correlation,
    covariance,
    expectation,
    iqr,
    kurtosis,
    maximum,
    mean,
    median,
    minimum,
    mode,
    percentile,
    quantile,
    range_,
    skewness,
    standard_deviation,
    standard_moment,
    variance,
)
from dit.example_dists import binomial


# ── expectation ──────────────────────────────────────────────────────────


@pytest.mark.parametrize("n", range(2, 10))
@pytest.mark.parametrize("p", np.linspace(0, 1, 11))
def test_expectation_identity_is_mean(n, p):
    d = binomial(n, p)
    assert expectation(d) == pytest.approx(mean(d))


def test_expectation_with_func():
    d = D([(0,), (1,), (2,)], [0.2, 0.3, 0.5])
    result = expectation(d, lambda x: x**2)
    expected = 0.2 * 0 + 0.3 * 1 + 0.5 * 4
    assert result == pytest.approx(expected)


def test_expectation_joint():
    d = D([(0, 0), (1, 0), (2, 1), (3, 1)], [1 / 8, 1 / 8, 3 / 8, 3 / 8])
    assert np.allclose(expectation(d), [2, 3 / 4])


# ── mean ─────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("n", range(2, 10))
@pytest.mark.parametrize("p", np.linspace(0, 1, 11))
def test_mean1(n, p):
    d = binomial(n, p)
    assert mean(d) == pytest.approx(n * p)


def test_mean2():
    d = D([(0, 0), (1, 0), (2, 1), (3, 1)], [1 / 8, 1 / 8, 3 / 8, 3 / 8])
    assert np.allclose(mean(d), [2, 3 / 4])


# ── variance / standard_deviation ────────────────────────────────────────


@pytest.mark.parametrize("n", range(2, 10))
@pytest.mark.parametrize("p", np.linspace(0, 1, 11))
def test_variance1(n, p):
    d = binomial(n, p)
    assert variance(d) == pytest.approx(n * p * (1 - p))


@pytest.mark.parametrize("n", range(2, 10))
@pytest.mark.parametrize("p", np.linspace(0, 1, 11))
def test_standard_deviation1(n, p):
    d = binomial(n, p)
    assert standard_deviation(d) == pytest.approx(np.sqrt(n * p * (1 - p)))


# ── standard_moment / central_moment ─────────────────────────────────────


@pytest.mark.parametrize("n", range(3, 10))
@pytest.mark.parametrize("p", np.linspace(0.1, 0.9, 9))
def test_standard_moment1(n, p):
    d = binomial(n, p)
    for i, m in {1: 0, 2: 1, 3: (1 - 2 * p) / np.sqrt(n * p * (1 - p))}.items():
        assert standard_moment(d, i) == pytest.approx(m, abs=1e-5)


# ── skewness ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize("n", range(3, 10))
@pytest.mark.parametrize("p", np.linspace(0.1, 0.9, 9))
def test_skewness_binomial(n, p):
    d = binomial(n, p)
    expected = (1 - 2 * p) / np.sqrt(n * p * (1 - p))
    assert skewness(d) == pytest.approx(expected, abs=1e-5)


def test_skewness_symmetric():
    d = D([(0,), (1,), (2,)], [0.25, 0.50, 0.25])
    assert skewness(d) == pytest.approx(0.0)


# ── kurtosis ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize("n", range(3, 10))
@pytest.mark.parametrize("p", np.linspace(0.1, 0.9, 9))
def test_kurtosis_binomial(n, p):
    d = binomial(n, p)
    expected = (1 - 6 * p * (1 - p)) / (n * p * (1 - p))
    assert kurtosis(d) == pytest.approx(expected, abs=1e-5)


def test_kurtosis_non_excess():
    d = binomial(5, 0.5)
    assert kurtosis(d, excess=False) == pytest.approx(kurtosis(d) + 3)


# ── covariance ───────────────────────────────────────────────────────────


def test_covariance_independent():
    d = D([(0, 0), (0, 1), (1, 0), (1, 1)], [0.25] * 4)
    cov = covariance(d)
    assert cov.shape == (2, 2)
    assert cov[0, 1] == pytest.approx(0.0)
    assert cov[1, 0] == pytest.approx(0.0)
    assert cov[0, 0] == pytest.approx(0.25)
    assert cov[1, 1] == pytest.approx(0.25)


def test_covariance_correlated():
    d = D([(0, 0), (1, 1)], [0.5, 0.5])
    cov = covariance(d)
    assert cov[0, 0] == pytest.approx(0.25)
    assert cov[0, 1] == pytest.approx(0.25)
    assert cov[1, 0] == pytest.approx(0.25)


def test_covariance_1d():
    d = binomial(5, 0.5)
    cov = covariance(d)
    assert cov.shape == (1, 1)
    assert cov[0, 0] == pytest.approx(variance(d))


# ── correlation ──────────────────────────────────────────────────────────


def test_correlation_perfect():
    d = D([(0, 0), (1, 1)], [0.5, 0.5])
    cor = correlation(d)
    assert cor[0, 1] == pytest.approx(1.0)


def test_correlation_independent():
    d = D([(0, 0), (0, 1), (1, 0), (1, 1)], [0.25] * 4)
    cor = correlation(d)
    assert cor[0, 1] == pytest.approx(0.0)


def test_correlation_diagonal_ones():
    d = D([(0, 0), (1, 2), (2, 1)], [1 / 3] * 3)
    cor = correlation(d)
    np.testing.assert_allclose(np.diag(cor), [1.0, 1.0])


# ── quantile / percentile ───────────────────────────────────────────────


@pytest.mark.parametrize("n", range(2, 10))
@pytest.mark.parametrize("p", np.linspace(0, 1, 11))
def test_median_binomial(n, p):
    d = binomial(n, p)
    assert median(d) in [floor(n * p), n * p, ceil(n * p)]


def test_median_joint():
    d = D([(0, 0), (1, 0), (2, 1), (3, 1)], [1 / 8, 1 / 8, 3 / 8, 3 / 8])
    assert np.allclose(median(d), [2, 1])


def test_quantile_extremes():
    d = D([(1,), (2,), (3,), (4,)], [0.25] * 4)
    assert quantile(d, 0.0) == pytest.approx(1.0)
    assert quantile(d, 1.0) == pytest.approx(4.0)


def test_percentile_50():
    d = D([(1,), (2,), (3,)], [0.25, 0.5, 0.25])
    assert percentile(d, 50) == pytest.approx(median(d))


# ── iqr ──────────────────────────────────────────────────────────────────


def test_iqr_uniform():
    d = D([(i,) for i in range(1, 5)], [0.25] * 4)
    result = iqr(d)
    assert result == pytest.approx(quantile(d, 0.75) - quantile(d, 0.25))


def test_iqr_binomial():
    d = binomial(10, 0.5)
    q1 = quantile(d, 0.25)
    q3 = quantile(d, 0.75)
    assert iqr(d) == pytest.approx(q3 - q1)


# ── minimum / maximum / range_ ──────────────────────────────────────────


def test_minimum_binomial():
    d = binomial(5, 0.5)
    assert minimum(d) == 0


def test_maximum_binomial():
    d = binomial(5, 0.5)
    assert maximum(d) == 5


def test_range_binomial():
    d = binomial(5, 0.5)
    assert range_(d) == 5


def test_min_max_joint():
    d = D([(0, 10), (5, 20), (3, 15)], [1 / 3] * 3)
    assert np.array_equal(minimum(d), [0, 10])
    assert np.array_equal(maximum(d), [5, 20])
    assert np.array_equal(range_(d), [5, 10])


# ── cdf ──────────────────────────────────────────────────────────────────


def test_cdf_uniform():
    d = D([(1,), (2,), (3,), (4,)], [0.25] * 4)
    vals, cumprobs = cdf(d)
    np.testing.assert_allclose(vals, [1, 2, 3, 4])
    np.testing.assert_allclose(cumprobs, [0.25, 0.5, 0.75, 1.0])


def test_cdf_sums_to_one():
    d = binomial(5, 0.5)
    vals, cumprobs = cdf(d)
    assert cumprobs[-1] == pytest.approx(1.0)
    assert all(cumprobs[i] <= cumprobs[i + 1] for i in range(len(cumprobs) - 1))


def test_cdf_joint():
    d = D([(0, 0), (0, 1), (1, 0), (1, 1)], [0.25] * 4)
    result = cdf(d)
    assert len(result) == 2
    vals0, cum0 = result[0]
    np.testing.assert_allclose(vals0, [0, 1])
    np.testing.assert_allclose(cum0, [0.5, 1.0])


# ── mode ─────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("n", range(2, 10))
@pytest.mark.parametrize("p", np.linspace(0, 1, 11))
def test_mode1(n, p):
    d = binomial(n, p)
    assert mode(d)[0][0] in [floor((n + 1) * p), floor((n + 1) * p) - 1]


def test_mode2():
    d = D([(0, 0), (1, 0), (2, 1), (3, 1)], [1 / 8, 1 / 8, 3 / 8, 3 / 8])
    modes = [np.array([2, 3]), np.array([1])]
    for m1, m2 in zip(mode(d), modes, strict=True):
        assert np.allclose(m1, m2)
