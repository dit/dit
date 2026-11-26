"""
Tests for dit.rate_distortion.functional_rate_distortion.
"""

import importlib
from itertools import combinations
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.optimize import linprog

from dit.exceptions import OptimizationException, ditException
from dit.other import functional_information_pmf
from dit.rate_distortion import functional_rate_distortion, hamming, maximum_correlation, residual_entropy
from dit.rate_distortion.distortions import hamming_distortion


def _hb(p):
    return float(-(p * np.log2(p) + (1 - p) * np.log2(1 - p))) if 0 < p < 1 else 0.0


def _erasure(n):
    """Reproduce x exactly (distortion 0) or erase it (distortion 1); any other reproduction is forbidden."""
    delta = np.full((n, n + 1), np.inf)
    delta[np.arange(n), np.arange(n)] = 0
    delta[:, n] = 1
    return delta


def _compact(p_x, delta, D):
    """
    R_F(D) by a second, independent linear program: the Lovász extension of a submodular F is also the least
    sum_A lambda_A F(A) over conic combinations sum_A lambda_A 1_A of the column, one variable per subset A.
    """
    n, m = delta.shape
    subsets = [list(s) for r in range(1, n + 1) for s in combinations(range(n), r)]
    f = np.array([-p_x[s].sum() * np.log2(p_x[s].sum()) for s in subsets])
    nk, ns = n * m, len(subsets)
    a_eq, b_eq = [], []
    for x in range(n):
        row = np.zeros(nk + ns * m)
        row[x * m : (x + 1) * m] = 1
        a_eq.append(row)
        b_eq.append(1)
    for y in range(m):
        for x in range(n):
            row = np.zeros(nk + ns * m)
            row[x * m + y] = -1
            row[nk + y * ns : nk + (y + 1) * ns] = [x in s for s in subsets]
            a_eq.append(row)
            b_eq.append(0)
    allowed = np.isfinite(delta)
    a_ub = [np.r_[(p_x[:, None] * np.where(allowed, delta, 0)).ravel(), np.zeros(ns * m)]]
    bounds = [(0, 1 if ok else 0) for ok in allowed.ravel()] + [(0, None)] * (ns * m)
    res = linprog(np.r_[np.zeros(nk), np.tile(f, m)], A_ub=a_ub, b_ub=[D], A_eq=a_eq, b_eq=b_eq, bounds=bounds)
    return res.fun


@pytest.mark.parametrize("p", [0.25, 0.5, 0.9])
@pytest.mark.parametrize("D", [0.0, 0.03, 0.1, 0.2, 0.3])
def test_bernoulli_hamming(p, D):
    """R_F(D) = h(p) (1 - D / p_min) below p_min (eq. 70 of Flamich et al.), above R(D) = h(p) - h(D)."""
    p_min = min(p, 1 - p)
    (rate, distortion), _ = functional_rate_distortion([1 - p, p], D)
    assert rate == pytest.approx(_hb(p) * max(0, 1 - D / p_min), abs=1e-9)
    assert rate >= _hb(p) - _hb(min(D, p_min)) - 1e-9
    assert distortion <= D + 1e-9


@pytest.mark.parametrize("n", [3, 4])
@pytest.mark.parametrize("D", [0.0, 0.3, 0.7, 1.0, 1.5])
def test_uniform_erasure(n, D):
    """The optimal channel erases independently of X, so it is singular and R_F(D) = R(D) = (1 - D) log2 n (eq. 74)."""
    (rate, _), _ = functional_rate_distortion(np.ones(n) / n, D, _erasure(n))
    assert rate == pytest.approx(max(0, 1 - D) * np.log2(n), abs=1e-9)


def test_matches_compact_program():
    rng = np.random.default_rng(0)
    for _ in range(25):
        n, m = rng.integers(2, 5, size=2)
        p_x = rng.dirichlet(np.ones(n))
        delta = rng.uniform(0, 2, (n, m))
        delta[rng.uniform(size=(n, m)) < 0.15] = np.inf
        delta[np.arange(n), rng.integers(m, size=n)] = rng.uniform(0, 1, n)
        D = p_x @ np.where(np.isfinite(delta), delta, np.inf).min(axis=1) + rng.uniform(0, 0.6)
        (rate, distortion), q_xy = functional_rate_distortion(p_x, D, delta)
        assert rate == pytest.approx(_compact(p_x, delta, D), abs=1e-9)
        assert rate == pytest.approx(functional_information_pmf(q_xy), abs=1e-12)
        assert q_xy.sum(axis=1) == pytest.approx(p_x)
        assert distortion <= D + 1e-9


def test_nonincreasing_and_convex():
    rng = np.random.default_rng(1)
    p_x = rng.dirichlet(np.ones(4))
    delta = rng.uniform(0, 1, (4, 4))
    np.fill_diagonal(delta, 0)
    rates = [functional_rate_distortion(p_x, D, delta)[0].rate for D in np.linspace(0, 0.5, 11)]
    assert np.all(np.diff(rates) <= 1e-9)
    assert np.all(np.diff(rates, 2) >= -1e-9)


def test_impossible_inputs():
    """Outcomes of X with zero probability are ignored, and get a zero row."""
    (rate, _), q_xy = functional_rate_distortion([0.5, 0.0, 0.5], 0.1, 1 - np.eye(3))
    assert rate == pytest.approx(0.8)
    assert q_xy[1] == pytest.approx(0)


def test_distortion_arguments():
    expected = functional_rate_distortion([0.3, 0.7], 0.1)[0].rate
    for distortion in (hamming, hamming_distortion, 1 - np.eye(2)):
        assert functional_rate_distortion([0.3, 0.7], 0.1, distortion)[0].rate == pytest.approx(expected)
    for distortion in (residual_entropy, maximum_correlation, -np.eye(2), np.ones((3, 2)), np.ones(2)):
        with pytest.raises(ditException, match="distortion"):
            functional_rate_distortion([0.3, 0.7], 0.1, distortion)


def test_infeasible_distortion():
    with pytest.raises(ditException, match="least is 0.1"):
        functional_rate_distortion([0.5, 0.5], 0.05, np.array([[0.2, 1.0], [0.0, 1.0]]))
    with pytest.raises(ditException, match="least is inf"):
        functional_rate_distortion([0.5, 0.5], 0.5, np.array([[np.inf, np.inf], [0.0, 1.0]]))


def test_optimization_failures(monkeypatch):
    with pytest.raises(OptimizationException, match="did not converge"):
        functional_rate_distortion([0.3, 0.7], 0.1, maxiter=1)
    module = importlib.import_module("dit.rate_distortion.functional_rate_distortion")
    monkeypatch.setattr(module, "linprog", lambda *args, **kwargs: SimpleNamespace(status=2, message="infeasible"))
    with pytest.raises(OptimizationException, match="infeasible"):
        functional_rate_distortion([0.3, 0.7], 0.1)
