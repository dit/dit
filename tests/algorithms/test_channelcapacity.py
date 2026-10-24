"""
Tests for dit.algorithms.channelcapacity
"""

import warnings

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy.optimize import minimize

from dit import Distribution, joint_from_factors
from dit.algorithms.channelcapacity import (
    _blahut_arimoto,
    _row_plogp,
    channel_capacity,
    channel_capacity_cost,
    channel_capacity_joint,
    sibson_capacity,
)
from dit.cdisthelpers import cdist_array
from dit.exceptions import ditException
from dit.other.sibson_mutual_information import sibson_mutual_information_pmf


def capacity_bounds(P, pmf):
    """
    Blahut's lower and upper capacity bounds (bits) at the input ``pmf``.
    """
    q = pmf @ P
    with np.errstate(divide="ignore", invalid="ignore"):
        D = np.where(P > 0, P * np.log2(P / q), 0).sum(axis=1)
    return pmf @ D, D.max()


def binary_entropy(p):
    return -p * np.log2(p) - (1 - p) * np.log2(1 - p)


def BEC_joint(epsilon):
    """
    The joint distribution for the binary erase channel at channel capacity.

    Parameters
    ----------
    epsilon : float
        The noise level at which the input is erased.

    """
    pX = Distribution(["0", "1"], [1 / 2, 1 / 2])
    pYgX0 = Distribution(["0", "1", "e"], [1 - epsilon, 0, epsilon])
    pYgX1 = Distribution(["0", "1", "e"], [0, 1 - epsilon, epsilon])
    pYgX = [pYgX0, pYgX1]
    pXY = joint_from_factors(pX, pYgX, strict=False)
    return pXY


def test_channel_capacity_no_rvnames():
    epsilon = 0.3
    pXY = BEC_joint(epsilon)
    pX, pYgX = pXY.condition_on([0])
    cc, pXopt = channel_capacity(pYgX, pX)

    # Verify channel capacity.
    assert np.allclose(cc, 1 - epsilon)

    # Verify maximizing distribution.
    assert pX.is_approx_equal(pXopt)

    # Verify joint distribution at channel capacity.
    pXYopt = joint_from_factors(pXopt, pYgX)
    assert pXY.is_approx_equal(pXYopt)


def test_channel_capacity_rvnames():
    epsilon = 0.01
    pXY = BEC_joint(epsilon)
    pXY.set_rv_names("XY")
    pX, pYgX = pXY.condition_on(["X"])
    cc, pXopt = channel_capacity(pYgX, pX)

    # Verify channel capacity.
    assert np.allclose(cc, 1 - epsilon)

    # Verify maximizing distribution.
    assert pX.is_approx_equal(pXopt)

    # Verify joint distribution at channel capacity.
    pXYopt = joint_from_factors(pXopt, pYgX)
    assert pXY.is_approx_equal(pXYopt)


def test_channel_capacity_array1():
    epsilon = 0.3
    pXY = BEC_joint(epsilon)
    pX, pYgX = pXY.condition_on([0])
    cc, pXopt_pmf = channel_capacity(pYgX)

    # Verify channel capacity.
    assert np.allclose(cc, 1 - epsilon)

    # Verify maximizing distribution.
    assert np.allclose(pX.pmf, pXopt_pmf)


def test_channel_capacity_array2():
    epsilon = 0.3
    pXY = BEC_joint(epsilon)
    pX, pYgX = pXY.condition_on([0])
    pYgX = cdist_array(pYgX, base="linear", mode="dense")
    cc, pXopt_pmf = channel_capacity(pYgX, atol=1e-9, rtol=1e-9)

    # Verify channel capacity.
    assert np.allclose(cc, 1 - epsilon)

    # Verify maximizing distribution.
    assert np.allclose(pX.pmf, pXopt_pmf)


def test_bad_marginal():
    epsilon = 0.01
    pXY = BEC_joint(epsilon)
    pXY.set_rv_names("XY")
    pX, pYgX = pXY.condition_on(["X"])
    pX["0"] = 0
    # Now make its length disagree with the number of cdists.
    pX.make_sparse()
    with pytest.raises(ditException):
        channel_capacity(pYgX, pX)


def test_channel_capacity_joint1():
    """
    Test against a known value.
    """
    gm = Distribution(["00", "01", "10"], [1 / 3] * 3)
    cc = channel_capacity_joint(gm, [0], [1])
    assert cc == pytest.approx(0.3219280796196524)


def test_channel_capacity_joint2():
    """
    Test against a known value.
    """
    gm = Distribution(["00", "01", "10"], [1 / 3] * 3)
    m = Distribution(["0", "1"], [2 / 5, 3 / 5])
    _, marg = channel_capacity_joint(gm, [0], [1], marginal=True)
    assert marg.is_approx_equal(m, atol=1e-3)


# ── Distribution tests ─────────────────────────────────────────────────


def BEC_joint_xr(epsilon):
    """Build the BEC joint distribution as an Distribution."""
    pXY = BEC_joint(epsilon)
    pXY.set_rv_names("XY")
    return pXY


def test_channel_capacity_distribution():
    epsilon = 0.3
    pXY = BEC_joint_xr(epsilon)
    pX, pYgX = pXY.condition_on(["X"])
    cc, pXopt_pmf = channel_capacity(pYgX)

    assert np.allclose(cc, 1 - epsilon)
    assert np.allclose(pXopt_pmf, [0.5, 0.5])


def test_channel_capacity_distribution_with_marginal():
    epsilon = 0.3
    pXY = BEC_joint_xr(epsilon)
    pX, pYgX = pXY.condition_on(["X"])
    pX = pXY.marginal("X")
    cc, pXopt = channel_capacity(pYgX, pX)

    assert np.allclose(cc, 1 - epsilon)
    assert isinstance(pXopt, Distribution)
    assert np.allclose(pXopt.data.values, [0.5, 0.5])


def test_channel_capacity_native_conditional():
    """A native conditional Distribution (given_vars set) is reshaped and solved."""
    epsilon = 0.3
    pXY = BEC_joint_xr(epsilon)
    pYgX = pXY.condition_on("X")
    assert pYgX.is_conditional()

    cc, pXopt_pmf = channel_capacity(pYgX)
    assert np.allclose(cc, 1 - epsilon)
    assert np.allclose(pXopt_pmf, [0.5, 0.5])


def test_channel_capacity_native_conditional_with_marginal():
    """A native conditional plus a marginal Distribution returns a Distribution."""
    epsilon = 0.3
    pXY = BEC_joint_xr(epsilon)
    pYgX = pXY.condition_on("X")
    pX = pXY.marginal("X")

    cc, pXopt = channel_capacity(pYgX, pX)
    assert np.allclose(cc, 1 - epsilon)
    assert isinstance(pXopt, Distribution)
    assert np.allclose(pXopt.data.values, [0.5, 0.5])


def test_channel_capacity_distribution_not_conditional():
    pXY = BEC_joint_xr(0.3)
    with pytest.raises(ditException):
        channel_capacity(pXY)


def test_channel_capacity_joint_distribution():
    gm = Distribution(["00", "01", "10"], [1 / 3] * 3)
    gm.set_rv_names("XY")
    cc = channel_capacity_joint(gm, ["X"], ["Y"])
    assert cc == pytest.approx(0.3219280796196524)


def test_channel_capacity_joint_distribution_marginal():
    gm = Distribution(["00", "01", "10"], [1 / 3] * 3)
    gm.set_rv_names("XY")
    cc, marg = channel_capacity_joint(gm, ["X"], ["Y"], marginal=True)
    assert cc == pytest.approx(0.3219280796196524)
    assert isinstance(marg, Distribution)
    assert np.allclose(marg.data.values, [2 / 5, 3 / 5], atol=1e-3)


# ── Certified convergence ──────────────────────────────────────────────


STOPS_EARLY = np.array(
    [
        [0.076, 0.001, 0.078, 0.811, 0.031, 0.003],
        [0.647, 0.006, 0.023, 0.302, 0.000, 0.021],
        [0.508, 0.112, 0.043, 0.294, 0.044, 0.000],
        [0.059, 0.031, 0.426, 0.043, 0.065, 0.376],
    ]
)
STOPS_EARLY = STOPS_EARLY / STOPS_EARLY.sum(axis=1, keepdims=True)


@pytest.mark.parametrize("method", ["ba", "accelerated"])
@pytest.mark.parametrize("eliminate", [False, True])
def test_does_not_stop_early(method, eliminate):
    """
    Comparing successive iterates stopped ~1e-5 bits short on this channel.
    """
    cc, pmf = channel_capacity(STOPS_EARLY, atol=1e-10, rtol=0, method=method, eliminate=eliminate)
    assert cc == pytest.approx(0.742122012970905, abs=1e-10)
    lower, upper = capacity_bounds(STOPS_EARLY, pmf)
    assert upper - lower <= 1e-10


def test_max_iters_warns():
    with pytest.warns(RuntimeWarning, match="did not reach tolerance"):
        cc, _ = channel_capacity(STOPS_EARLY, atol=1e-12, rtol=0, max_iters=3, method="ba")
    assert cc < 0.742122012970905


def test_unknown_method():
    with pytest.raises(ditException):
        channel_capacity(STOPS_EARLY, method="newton")


def test_weakly_symmetric_closed_form():
    """
    Cover & Thomas Theorem 7.2.1: C = log|Y| - H(row).
    """
    P = np.array([[1 / 3, 1 / 6, 1 / 2], [1 / 3, 1 / 2, 1 / 6]])
    cc, pmf = channel_capacity(P)
    h = -np.sum(P[0] * np.log2(P[0]))
    assert cc == pytest.approx(np.log2(3) - h)
    assert np.allclose(pmf, [0.5, 0.5])


def test_square_invertible_matches_iteration():
    P = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.2, 0.2, 0.6]])
    cc, pmf = channel_capacity(P, atol=1e-12, rtol=0)
    plogp = _row_plogp(P)
    out = _blahut_arimoto(P, plogp, np.zeros(3), lambda _: 1e-14, 10**6, True)
    assert cc == pytest.approx(out[4] / np.log(2), abs=1e-12)
    lower, upper = capacity_bounds(P, pmf)
    assert upper - lower <= 1e-12


def test_square_with_unused_input_falls_back():
    """
    The middle input is a mixture of the others, so p = q P^{-1} is negative.
    """
    P = np.array([[0.9, 0.05, 0.05], [0.45, 0.1, 0.45], [0.05, 0.05, 0.9]])
    cc, pmf = channel_capacity(P, atol=1e-10, rtol=0)
    lower, upper = capacity_bounds(P, pmf)
    assert upper - lower <= 1e-10
    assert pmf.min() >= 0


def test_duplicate_rows_and_zero_columns():
    P = np.array([[0.8, 0.0, 0.2], [0.8, 0.0, 0.2], [0.1, 0.0, 0.9], [0.5, 0.0, 0.5]])
    cc, pmf = channel_capacity(P, atol=1e-10, rtol=0)
    reduced, reduced_pmf = channel_capacity(P[[0, 2, 3]][:, [0, 2]], atol=1e-10, rtol=0)
    assert cc == pytest.approx(reduced, abs=1e-10)
    assert pmf.shape == (4,)
    assert pmf[1] == 0
    assert np.allclose(pmf[[0, 2, 3]], reduced_pmf, atol=1e-6)


def test_sparse_support_elimination():
    """
    Many inputs are mixtures of three extreme rows and get no probability.
    """
    rng = np.random.default_rng(0)
    base = np.array([[0.9, 0.05, 0.05], [0.05, 0.9, 0.05], [0.05, 0.05, 0.9]])
    P = np.vstack([base, rng.dirichlet(np.ones(3), size=60) @ base])
    full, _ = channel_capacity(P, atol=1e-10, rtol=0)
    elim, pmf = channel_capacity(P, atol=1e-10, rtol=0, eliminate=True)
    assert elim == pytest.approx(full, abs=1e-10)
    assert elim == pytest.approx(channel_capacity(base)[0], abs=1e-8)
    assert pmf[3:].sum() < 1e-6


def channels(max_inputs=6, max_outputs=6):
    @st.composite
    def strategy(draw):
        n = draw(st.integers(2, max_inputs))
        m = draw(st.integers(2, max_outputs))
        flat = draw(st.lists(st.sampled_from([0.0, 0.01, 0.1, 0.3, 0.5, 1.0]), min_size=n * m, max_size=n * m))
        P = np.array(flat).reshape(n, m)
        P[np.arange(n), np.arange(n) % m] += 1e-3
        return P / P.sum(axis=1, keepdims=True)

    return strategy()


@settings(max_examples=60, deadline=None)
@given(P=channels())
def test_capacity_properties(P):
    n, m = P.shape
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        cc, pmf = channel_capacity(P, atol=1e-9, rtol=0)
    assert -1e-12 <= cc <= min(np.log2(n), np.log2(m)) + 1e-9
    lower, upper = capacity_bounds(P, pmf)
    assert upper - lower <= 1e-9 + 1e-12
    assert cc == pytest.approx(lower, abs=1e-9)

    rng = np.random.default_rng(0)
    permuted = P[rng.permutation(n)][:, rng.permutation(m)]
    assert channel_capacity(permuted, atol=1e-9, rtol=0)[0] == pytest.approx(cc, abs=2e-9)

    for method in ("ba", "accelerated"):
        for eliminate in (False, True):
            other, _ = channel_capacity(P, atol=1e-9, rtol=0, method=method, eliminate=eliminate)
            assert other == pytest.approx(cc, abs=2e-9)


# ── Cost-constrained capacity ──────────────────────────────────────────


@pytest.mark.parametrize("budget", [0.05, 0.2, 0.4])
def test_bsc_hamming_cost(budget):
    """
    C(B) = H(B * p) - H(p) for a BSC with Hamming-weight cost and B < 1/2.
    """
    p = 0.1
    P = np.array([[1 - p, p], [p, 1 - p]])
    cc, pmf, s = channel_capacity_cost(P, [0, 1], budget)
    expected = binary_entropy(budget * (1 - p) + (1 - budget) * p) - binary_entropy(p)
    assert cc == pytest.approx(expected, abs=1e-8)
    assert pmf @ [0, 1] == pytest.approx(budget, abs=1e-6)
    assert s > 0


def test_cost_budget_not_binding():
    P = np.array([[0.9, 0.1], [0.1, 0.9]])
    cc, pmf, s = channel_capacity_cost(P, [0, 1], 0.7)
    assert cc == pytest.approx(channel_capacity(P)[0])
    assert s == 0


def test_cost_budget_only_cheapest():
    P = np.array([[0.9, 0.1, 0.0], [0.1, 0.9, 0.0], [0.0, 0.0, 1.0]])
    cc, pmf, s = channel_capacity_cost(P, [0, 0, 1], 0.0)
    assert cc == pytest.approx(channel_capacity(P[:2])[0])
    assert pmf[2] == 0
    assert s == np.inf


def test_cost_capacity_is_nondecreasing():
    rng = np.random.default_rng(1)
    P = rng.dirichlet(np.ones(5), size=4)
    cost = np.array([0.0, 1.0, 2.0, 3.0])
    values = [channel_capacity_cost(P, cost, b)[0] for b in np.linspace(0, 3, 7)]
    assert np.all(np.diff(values) >= -1e-9)
    assert values[-1] == pytest.approx(channel_capacity(P)[0], abs=1e-8)


def test_cost_errors():
    P = np.array([[0.9, 0.1], [0.1, 0.9]])
    with pytest.raises(ditException):
        channel_capacity_cost(P, [1, 2], 0.5)
    with pytest.raises(ditException):
        channel_capacity_cost(P, [1, 2, 3], 1.5)


# ── Sibson capacity ────────────────────────────────────────────────────


def test_sibson_order_one_is_shannon():
    assert sibson_capacity(STOPS_EARLY, 1)[0] == pytest.approx(channel_capacity(STOPS_EARLY)[0])


def test_sibson_order_infinity():
    cc, _ = sibson_capacity(STOPS_EARLY, np.inf)
    assert cc == pytest.approx(np.log2(STOPS_EARLY.max(axis=0).sum()))


@pytest.mark.parametrize("order", [0.5, 2.0])
def test_sibson_matches_brute_force(order):
    P = STOPS_EARLY
    cc, pmf = sibson_capacity(P, order, atol=1e-11, rtol=0)
    assert sibson_mutual_information_pmf(pmf[:, np.newaxis] * P, order) == pytest.approx(cc, abs=1e-9)

    def objective(z):
        w = np.exp(z - z.max())
        return -sibson_mutual_information_pmf((w / w.sum())[:, np.newaxis] * P, order)

    best = -minimize(
        objective, np.zeros(len(P)), method="Nelder-Mead", options={"xatol": 1e-10, "fatol": 1e-13, "maxiter": 20_000}
    ).fun
    assert cc >= best - 1e-9
    assert cc == pytest.approx(best, abs=1e-6)


def test_sibson_increases_with_order():
    values = [sibson_capacity(STOPS_EARLY, a)[0] for a in (0.25, 0.5, 1.0, 2.0, 4.0, np.inf)]
    assert np.all(np.diff(values) >= -1e-9)


def test_sibson_bad_order():
    with pytest.raises(ditException):
        sibson_capacity(STOPS_EARLY, 0)
