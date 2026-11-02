"""
Tests for dit.algorithms.channelcapacity
"""

import re
import warnings

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy.optimize import minimize

from dit import Distribution, joint_from_factors
from dit.algorithms.channelcapacity import (
    _active_set,
    _blahut_arimoto,
    _row_plogp,
    channel_capacity,
    channel_capacity_cost,
    channel_capacity_joint,
    commitment_capacity,
    maximal_equivocation,
    nonredundant_inputs,
    sibson_capacity,
)
from dit.cdisthelpers import cdist_array
from dit.example_channels import (
    binary_erasure_channel,
    binary_symmetric_channel,
    noisy_typewriter,
    q_ary_symmetric_channel,
    z_channel,
)
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
        cc, _ = channel_capacity(STOPS_EARLY, atol=1e-12, rtol=0, maxiter=3, method="ba")
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


# ── Iteration limits, step control and cost offsets ────────────────────


def reported_interval(warning):
    """
    The [lower, upper] capacity interval (bits) stated by a non-convergence warning.
    """
    lower, upper = re.search(r"\[(.+), (.+)\] bits", str(warning.message)).groups()
    return float(lower), float(upper)


def sibson_terms(P, pmf, order):
    """
    Sibson's I_alpha at ``pmf`` and the divergences D_alpha(P_x || q) from Sibson's optimal
    output q for ``pmf``, in bits. Their maximum bounds the capacity of order alpha.
    """
    g = (pmf @ P**order) ** (1 / order)
    q = g / g.sum()
    return order / (order - 1) * np.log2(g.sum()), np.log2(P**order @ q ** (1 - order)) / (order - 1)


def test_cost_unknown_method():
    P = np.array([[0.9, 0.1], [0.1, 0.9]])
    with pytest.raises(ditException, match="Unknown method 'newton'; expected 'ba' or 'accelerated'"):
        channel_capacity_cost(P, [0, 1], 0.5, method="newton")


def test_cost_max_iters_warns():
    """
    With a budget that does not bind, the unconstrained iteration decides. Stopped early, it
    warns with an interval containing the capacity, and returns the interval's lower end.
    """
    with pytest.warns(RuntimeWarning, match="did not reach tolerance in 2 iterations") as record:
        cc, pmf, s = channel_capacity_cost(STOPS_EARLY, [0, 1, 2, 3], 10, atol=1e-12, rtol=0, maxiter=2)
    lower, upper = reported_interval(record.pop(RuntimeWarning))
    assert s == 0
    assert cc == pytest.approx(lower, abs=1e-15)
    assert lower < 0.742122012970905 < upper
    assert capacity_bounds(STOPS_EARLY, pmf)[0] == pytest.approx(cc, abs=1e-12)


def test_sibson_max_iters_warns():
    """
    Stopped early, Sibson's iteration warns with an interval containing C_alpha, and returns
    I_alpha of the input it reached.
    """
    true, _ = sibson_capacity(STOPS_EARLY, 2.0, atol=1e-11, rtol=0)
    with pytest.warns(RuntimeWarning, match="sibson_capacity did not reach tolerance in 2 iterations") as record:
        cc, pmf = sibson_capacity(STOPS_EARLY, 2.0, atol=1e-12, rtol=0, maxiter=2)
    lower, upper = reported_interval(record.pop(RuntimeWarning))
    assert cc == pytest.approx(lower, abs=1e-15)
    assert lower < true < upper
    assert sibson_mutual_information_pmf(pmf[:, np.newaxis] * STOPS_EARLY, 2.0) == pytest.approx(cc, abs=1e-12)


def test_sibson_halves_overshooting_step():
    """
    At order 0.05 the full step from the uniform input lowers I_alpha on this sparse channel,
    so it is halved until I_alpha increases. The iteration still converges, as certified by
    the Rényi radius bound.
    """
    P = np.array([[0.0, 0.7, 0.08, 0.22], [0.07, 0.0, 0.9, 0.03], [0.01, 0.99, 0.0, 0.0]])
    order = 0.05
    uniform = np.full(3, 1 / 3)
    start, D = sibson_terms(P, uniform, order)
    full_step = uniform * np.exp2(D - D.max())
    assert sibson_terms(P, full_step / full_step.sum(), order)[0] < start - 0.1

    with pytest.warns(RuntimeWarning, match="did not reach tolerance in 1 iterations"):
        first, _ = sibson_capacity(P, order, maxiter=1)
    assert first > start

    cc, pmf = sibson_capacity(P, order, atol=1e-10, rtol=0)
    lower, divergences = sibson_terms(P, pmf, order)
    assert cc == pytest.approx(lower, abs=1e-12)
    assert divergences.max() - lower <= 1e-10 + 1e-12


def test_active_set_with_cost_offset():
    """
    With the per-input offset -s cost(x), the active-set iteration maximizes the Lagrangian
    I(r) - s E_r[cost] as the plain iteration does, and certifies it over all inputs.
    """
    base = np.array([[0.9, 0.05, 0.05], [0.05, 0.9, 0.05], [0.05, 0.05, 0.9]])
    weights = np.array([[0.5, 0.5, 0.0], [0.2, 0.3, 0.5], [1 / 3, 1 / 3, 1 / 3], [0.7, 0.1, 0.2], [0.1, 0.8, 0.1]])
    P = np.vstack([base, weights @ base])
    cost = np.array([0.0, 1.0, 2.0] + [0.5] * 5)
    s = 0.3
    offset = -s * np.log(2) * cost
    plogp = _row_plogp(P)

    def tol(lower):
        return 1e-12

    r, lower, upper, _ = _active_set(P, plogp, tol, 100_000, True, offset)
    _, _, _, _, plain_lower, _, _ = _blahut_arimoto(P, plogp, np.zeros(len(P)), tol, 100_000, True, offset)
    assert upper - lower <= 1e-12
    assert lower == pytest.approx(plain_lower, abs=1e-11)
    assert r.sum() == pytest.approx(1.0)
    mutual_information, _ = capacity_bounds(P, r)
    assert lower / np.log(2) == pytest.approx(mutual_information - s * (r @ cost), abs=1e-11)


# ── Maximal equivocation and commitment capacity ───────────────────────


def equivocation(P, pmf):
    """
    ``H[X | Y]`` in bits for the input ``pmf`` to the channel ``P``.
    """
    joint = pmf[:, np.newaxis] * P
    q = joint.sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(-np.where(joint > 0, joint * np.log2(joint / q), 0).sum())


@pytest.mark.parametrize("p", [0.05, 0.11, 0.3, 0.7, 0.95])
def test_commitment_capacity_bsc(p):
    """
    Example 1 of Winter, Nascimento and Imai: the commitment capacity of the binary
    symmetric channel is H_b(p), so the noisier the channel the better.
    """
    cc, pmf = commitment_capacity(binary_symmetric_channel(p), atol=1e-12, rtol=0)
    assert cc == pytest.approx(binary_entropy(p), abs=1e-12)
    assert np.allclose(pmf, [0.5, 0.5])


@pytest.mark.parametrize("p", [0, 0.5, 1])
def test_commitment_capacity_bsc_trivial(p):
    """
    At p = 0 and p = 1 the output reveals the input; at p = 1/2 the two inputs coincide.
    """
    assert commitment_capacity(binary_symmetric_channel(p))[0] == 0


def test_commitment_capacity_discontinuous():
    """
    Inputs within ``redundancy_tol`` of each other count as one, so the commitment capacity
    drops to zero next to p = 1/2, where the maximal equivocation is one bit.
    """
    bsc = binary_symmetric_channel(0.5 - 1e-11)
    assert commitment_capacity(bsc)[0] == 0
    assert commitment_capacity(bsc, redundancy_tol=1e-12)[0] == pytest.approx(1)
    assert maximal_equivocation(binary_symmetric_channel(0.5))[0] == pytest.approx(1)


def test_trivial_channel():
    """
    Example 2: the first input is the average of the other two, which form a noiseless
    channel. Using it, the equivocation reaches one bit at (1/2, 1/4, 1/4), the wiretap
    capacity the paper states, but the commitment capacity is zero.
    """
    P = np.array([[1 / 2, 1 / 2], [1, 0], [0, 1]])
    value, pmf = maximal_equivocation(P, atol=1e-12, rtol=0)
    assert value == pytest.approx(1, abs=1e-12)
    assert np.allclose(pmf, [1 / 2, 1 / 4, 1 / 4])
    assert list(nonredundant_inputs(P)) == [1, 2]
    cc, pmf = commitment_capacity(P)
    assert cc == 0
    assert pmf[0] == 0


def test_z_channel():
    """
    Example 3: transmission and commitment favour different inputs. Commitment gives the
    noisy input probability a = 1 - 1/sqrt(5), and C + C_com exceeds the one bit of input
    entropy.
    """
    channel = z_channel(0.5)
    cc, pmf = commitment_capacity(channel, atol=1e-12, rtol=0)
    a = 1 - 1 / np.sqrt(5)
    assert cc == pytest.approx(binary_entropy(a) + a - binary_entropy(a / 2), abs=1e-12)
    assert np.allclose(pmf, [1 - a, a])
    capacity = channel_capacity(channel)[0]
    assert capacity == pytest.approx(np.log2(5 / 4))
    assert capacity + cc > 1


@pytest.mark.parametrize("n", [3, 4, 26])
def test_noisy_typewriter(n):
    """
    The single-use scheme of Remark 6 runs over the noisy typewriter on four letters. For
    any n >= 3 the uniform input leaves one bit of equivocation.
    """
    cc, pmf = commitment_capacity(noisy_typewriter(n), atol=1e-12, rtol=0)
    assert cc == pytest.approx(1, abs=1e-12)
    assert np.allclose(pmf, 1 / n)


def test_noisy_typewriter_two_letters():
    """
    On two letters both inputs give the uniform output.
    """
    assert commitment_capacity(noisy_typewriter(2))[0] == 0


@pytest.mark.parametrize("epsilon", [0, 0.2, 0.9, 1])
def test_commitment_capacity_erasure(epsilon):
    expected = epsilon if epsilon < 1 else 0
    cc, _ = commitment_capacity(binary_erasure_channel(epsilon), atol=1e-12, rtol=0)
    assert cc == pytest.approx(expected, abs=1e-12)


def test_commitment_capacity_q_ary_symmetric():
    """
    The uniform input maximizes the equivocation of a symmetric channel.
    """
    cc, _ = commitment_capacity(q_ary_symmetric_channel(3, 0.2), atol=1e-12, rtol=0)
    assert cc == pytest.approx(binary_entropy(0.2) + 0.2 * np.log2(2), abs=1e-12)


def test_redundant_inputs_removed():
    """
    Copies and mixtures of other inputs are removed, the first copy is kept, and the
    removed inputs get no probability. Outputs that never occur are ignored.
    """
    bsc = np.array([[0.9, 0.0, 0.1], [0.1, 0.0, 0.9]])
    P = np.vstack([bsc[0], bsc[0], [0.5, 0.0, 0.5], bsc[1], [0.3, 0.0, 0.7], bsc[1]])
    assert list(nonredundant_inputs(P)) == [0, 3]
    cc, pmf = commitment_capacity(P, atol=1e-12, rtol=0)
    assert cc == pytest.approx(binary_entropy(0.1), abs=1e-12)
    assert np.allclose(pmf, [0.5, 0, 0, 0.5, 0, 0])


def test_disjoint_blocks():
    """
    The output reveals which block of inputs was used, so the best block takes all the
    probability. An input whose outputs no other input produces adds no equivocation.
    """
    P = np.zeros((5, 5))
    P[0, 0] = 1
    P[1:3, 1:3] = [[0.7, 0.3], [0.3, 0.7]]
    P[3:5, 3:5] = [[0.89, 0.11], [0.11, 0.89]]
    for function in (maximal_equivocation, commitment_capacity):
        value, pmf = function(P, atol=1e-12, rtol=0)
        assert value == pytest.approx(binary_entropy(0.3), abs=1e-12)
        assert np.allclose(pmf, [0, 0.5, 0.5, 0, 0])


def test_commitment_capacity_single_input():
    """
    Empty rows are not inputs, and a single input leaves nothing to hide.
    """
    P = np.array([[0.0, 0.0], [0.3, 0.7]])
    assert list(nonredundant_inputs(P)) == [1]
    for function in (maximal_equivocation, commitment_capacity):
        value, pmf = function(P)
        assert value == 0
        assert list(pmf) == [0, 1]


def test_commitment_capacity_input_forms():
    """
    An array, a list of conditional distributions and a conditional Distribution agree.
    """
    cdists = [Distribution(["0", "1"], [1, 0]), Distribution(["0", "1"], [1 / 2, 1 / 2])]
    values = [
        commitment_capacity(channel, atol=1e-12, rtol=0)[0]
        for channel in (np.array([[1, 0], [1 / 2, 1 / 2]]), cdists, z_channel(0.5))
    ]
    assert values == pytest.approx([values[0]] * 3, abs=1e-12)
    assert list(nonredundant_inputs(cdists)) == [0, 1]


def test_equivocation_matches_brute_force():
    P = STOPS_EARLY
    value, pmf = maximal_equivocation(P, atol=1e-11, rtol=0)
    assert equivocation(P, pmf) == pytest.approx(value, abs=1e-9)
    assert commitment_capacity(P, atol=1e-11, rtol=0)[0] == pytest.approx(value, abs=1e-9)

    def objective(z):
        w = np.exp(z - z.max())
        return -equivocation(P, w / w.sum())

    best = -minimize(
        objective, np.zeros(len(P)), method="Nelder-Mead", options={"xatol": 1e-10, "fatol": 1e-13, "maxiter": 20_000}
    ).fun
    assert value >= best - 1e-9
    assert value == pytest.approx(best, abs=1e-6)


@settings(max_examples=60, deadline=None)
@given(P=channels())
def test_equivocation_properties(P):
    n = len(P)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        value, pmf = maximal_equivocation(P, atol=1e-9, rtol=0)
        cc, cc_pmf = commitment_capacity(P, atol=1e-9, rtol=0)
    assert equivocation(P, pmf) == pytest.approx(value, abs=1e-9)
    assert equivocation(P, cc_pmf) == pytest.approx(cc, abs=1e-9)
    assert -1e-12 <= cc <= value + 1e-9
    assert value <= np.log2(n) + 1e-9

    keep = nonredundant_inputs(P)
    assert np.all(cc_pmf[np.setdiff1d(np.arange(n), keep)] == 0)
    rng = np.random.default_rng(0)
    for r in rng.dirichlet(np.ones(n), size=20):
        assert equivocation(P, r) <= value + 1e-9
        restricted = np.zeros(n)
        restricted[keep] = r[keep] / r[keep].sum()
        assert equivocation(P, restricted) <= cc + 1e-9

    permutation = rng.permutation(n)
    assert maximal_equivocation(P[permutation], atol=1e-9, rtol=0)[0] == pytest.approx(value, abs=2e-9)
    assert commitment_capacity(P[permutation], atol=1e-9, rtol=0)[0] == pytest.approx(cc, abs=2e-9)


@pytest.mark.parametrize("function", [maximal_equivocation, commitment_capacity])
def test_equivocation_maxiter_warns(function):
    """
    Stopped early, the iteration warns with an interval containing the value, and returns
    the equivocation of the input it reached.
    """
    channel = z_channel(0.5)
    true, _ = function(channel, atol=1e-12, rtol=0)
    with pytest.warns(RuntimeWarning, match=f"{function.__name__} did not reach tolerance in 1 iterations") as record:
        value, pmf = function(channel, atol=1e-12, rtol=0, maxiter=1)
    lower, upper = reported_interval(record.pop(RuntimeWarning))
    assert value == pytest.approx(lower, abs=1e-15)
    assert lower < true < upper
    assert equivocation(np.array([[1, 0], [1 / 2, 1 / 2]]), pmf) == pytest.approx(value, abs=1e-12)
