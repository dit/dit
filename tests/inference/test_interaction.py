"""
Tests for dit.inference.interaction: the likelihood-ratio test of the highest-order
connected information.
"""

import warnings
from itertools import combinations

import numpy as np
import pytest
from scipy.stats import chi2_contingency

from dit import Distribution
from dit.algorithms.distribution_optimizers import maxent_dist
from dit.inference import ConnectedInformationTest, UndersamplingWarning, connected_information_test
from dit.multivariate import entropy


def _logit(n, c, rng):
    """Binary inputs and a logistic target, logit -1 + x0/2 + x1/2 + c x0 x1 (paper, App.)."""
    x = rng.integers(0, 2, size=(n, 2))
    eta = -1 + 0.5 * x[:, 0] + 0.5 * x[:, 1] + c * x[:, 0] * x[:, 1]
    y = (rng.random(n) < 1 / (1 + np.exp(-eta))).astype(int)
    return np.column_stack([x, y])


def _connected_information(data):
    """H(P2) - H(P) of the empirical distribution, computed independently, in bits."""
    cells, counts = np.unique(data, axis=0, return_counts=True)
    empirical = Distribution([tuple(map(int, cell)) for cell in cells], counts / counts.sum())
    k = data.shape[1]
    null = maxent_dist(empirical, [list(s) for s in combinations(range(k), k - 1)])
    return entropy(null) - entropy(empirical)


def test_parity():
    """The parity of two bits is a pure three-way interaction: one bit."""
    rng = np.random.default_rng(0)
    x = rng.integers(0, 2, size=(2000, 2))
    result = connected_information_test(np.column_stack([x, x[:, 0] ^ x[:, 1]]))
    assert isinstance(result, ConnectedInformationTest)
    assert result.order == 3
    assert result.statistic == "G2"
    assert result.dof == 1
    assert result.connected_information == pytest.approx(1.0, abs=0.01)
    assert result.pvalue < 1e-100
    assert result.null is None


@pytest.mark.parametrize("seed", range(3))
def test_identity(seed):
    """G2 = 2 N ln 2 I3, with I3 = H(P2) - H(P) of the empirical distribution."""
    data = _logit(1000, 1.0, np.random.default_rng(seed))
    result = connected_information_test(data)
    assert result.value == pytest.approx(2 * 1000 * np.log(2) * result.connected_information)
    assert result.connected_information == pytest.approx(_connected_information(data), abs=1e-8)
    assert result.n_samples == 1000


def test_two_variables_is_independence_test():
    """With two variables the test is the G-test of independence."""
    rng = np.random.default_rng(1)
    x = rng.integers(0, 3, 500)
    y = (x + rng.integers(0, 2, 500)) % 4
    result = connected_information_test(np.column_stack([x, y]))
    table = np.zeros((3, 4))
    np.add.at(table, (x, y), 1)
    reference = chi2_contingency(table, correction=False, lambda_="log-likelihood")
    assert result.value == pytest.approx(reference.statistic)
    assert result.dof == reference.dof
    assert result.pvalue == pytest.approx(reference.pvalue)


def test_additive_null_level():
    """Additive logit models have no three-way interaction: over 400 replicates at N = 2000,
    as in the paper's Appendix, the test rejects at about its level."""
    rng = np.random.default_rng(3)
    rejections = [connected_information_test(_logit(2000, 0.0, rng)).pvalue < 0.05 for _ in range(400)]
    assert 0.02 <= np.mean(rejections) <= 0.08


@pytest.mark.parametrize("c", [1, 2, 3])
def test_power(c):
    """Interactions with c in {1, 2, 3} (I3 of at least 0.0097 bits) are detected at N = 2000."""
    rng = np.random.default_rng(10 + c)
    assert all(connected_information_test(_logit(2000, c, rng)).pvalue < 0.05 for _ in range(5))


def test_bootstrap():
    """The bootstrap is reproducible and agrees with the asymptotic test on a large sample."""
    data = _logit(1000, 0.0, np.random.default_rng(4))
    first = connected_information_test(data, method="bootstrap", n_surrogates=200, rng=5)
    second = connected_information_test(data, method="bootstrap", n_surrogates=200, rng=5)
    assert first.pvalue == second.pvalue
    assert np.array_equal(first.null, second.null)
    assert first.null.shape == (200,)
    assert first.pvalue == pytest.approx(connected_information_test(data).pvalue, abs=0.1)


def test_bootstrap_detects_interaction():
    rng = np.random.default_rng(6)
    x = rng.integers(0, 2, size=(200, 2))
    data = np.column_stack([x, x[:, 0] ^ x[:, 1]])
    result = connected_information_test(data, method="bootstrap", n_surrogates=49, rng=7)
    assert result.pvalue == pytest.approx(1 / 50)


def test_sparse_warning():
    """A sparse table warns for the asymptotic test, not for the bootstrap."""
    data = np.random.default_rng(8).integers(0, 4, size=(30, 3))
    with pytest.warns(UndersamplingWarning, match="bootstrap"):
        result = connected_information_test(data)
    assert result.min_expected < 1
    with warnings.catch_warnings():
        warnings.simplefilter("error", UndersamplingWarning)
        connected_information_test(data, method="bootstrap", n_surrogates=10, rng=0)


def test_groups():
    """Groups of columns form the variables; any symbols are allowed."""
    rng = np.random.default_rng(9)
    x = rng.integers(0, 2, size=(800, 3))
    data = np.column_stack([x, x[:, 0] ^ x[:, 1] ^ x[:, 2]]).astype(str)
    grouped = connected_information_test(data, rvs=[[0, 1], [2], [3]])
    assert grouped.order == 3
    assert grouped.dof == 3
    assert grouped.connected_information == pytest.approx(1.0, abs=0.02)
    assert connected_information_test(data).order == 4


def test_constant_variable():
    """A constant variable leaves no degrees of freedom."""
    rng = np.random.default_rng(10)
    data = np.column_stack([rng.integers(0, 2, size=(100, 2)), np.zeros(100, dtype=int)])
    result = connected_information_test(data)
    assert result.dof == 0
    assert result.pvalue == 1.0
    assert result.value == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize(
    "data, rvs, method, match",
    [
        (np.zeros((10, 3)), None, "exact", "Unknown method"),
        (np.zeros(10), None, "asymptotic", "rows"),
        (np.zeros((10, 1)), None, "asymptotic", "two disjoint"),
        (np.zeros((10, 3)), [[0, 1], [1, 2]], "asymptotic", "two disjoint"),
    ],
)
def test_errors(data, rvs, method, match):
    with pytest.raises(ValueError, match=match):
        connected_information_test(data, rvs=rvs, method=method)
