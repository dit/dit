"""
Tests for dit.multivariate.delta_gamma.
"""

import pytest
from hypothesis import given

from dit import Distribution
from dit.example_dists import giant_bit, n_mod_m
from dit.multivariate import (
    coinformation,
    delta_k,
    dual_total_correlation,
    gamma_k,
    o_information,
    s_information,
    total_correlation,
)
from dit.utils.testing import distributions

d1 = giant_bit(5, 2)
d2 = n_mod_m(5, 2)


def _loo_tc(dist):
    """
    Sum of the total correlations of every leave-one-out marginal,
    :math:`\\sum_i \\mathcal{T}(X^{-i})`.
    """
    n = dist.outcome_length()
    return sum(total_correlation(dist.marginal([j for j in range(n) if j != i])) for i in range(n))


def _loo_cond_dtc(dist):
    """
    Sum of the conditional dual total correlations of every leave-one-out
    marginal, :math:`\\sum_i \\mathcal{D}(X^{-i} \\mid X_i)`.
    """
    n = dist.outcome_length()
    return sum(dual_total_correlation(dist, rvs=[[j] for j in range(n) if j != i], crvs=[i]) for i in range(n))


def _independent_joint(dist_a, dist_b):
    """
    Build the joint distribution of two independent systems by taking the
    product of their outcomes and probabilities.
    """
    outcomes = []
    pmf = []
    for out_a, p_a in dist_a.zipped():
        for out_b, p_b in dist_b.zipped():
            outcomes.append(tuple(out_a) + tuple(out_b))
            pmf.append(p_a * p_b)
    return Distribution(outcomes, pmf)


@pytest.mark.parametrize(
    ["dist", "k", "value"],
    [
        (d1, 0, 5),
        (d1, 1, 1),
        (d1, 2, -3),
        (d2, 0, 5),
        (d2, 1, 4),
        (d2, 2, 3),
    ],
)
def test_delta_k(dist, k, value):
    """
    Test the Delta^k measure against known values.
    """
    assert delta_k(dist, k) == pytest.approx(value)


@pytest.mark.parametrize(
    ["dist", "k", "value"],
    [
        (d1, 0, 5),
        (d1, 1, 4),
        (d1, 2, 3),
        (d2, 0, 5),
        (d2, 1, 1),
        (d2, 2, -3),
    ],
)
def test_gamma_k(dist, k, value):
    """
    Test the Gamma^k measure against known values.
    """
    assert gamma_k(dist, k) == pytest.approx(value)


@pytest.mark.parametrize("dist", [d1, d2])
def test_delta_k_special_cases(dist):
    """
    Delta^1 is the dual total correlation, and Delta^2 is the negative
    O-information.
    """
    assert delta_k(dist, 1) == pytest.approx(dual_total_correlation(dist))
    assert delta_k(dist, 2) == pytest.approx(-o_information(dist))


@pytest.mark.parametrize("dist", [d1, d2])
def test_gamma_k_special_cases(dist):
    """
    Gamma^1 is the total correlation, and Gamma^2 is the O-information.
    """
    assert gamma_k(dist, 1) == pytest.approx(total_correlation(dist))
    assert gamma_k(dist, 2) == pytest.approx(o_information(dist))


def test_conditional():
    """
    Test that conditioning is threaded through to the underlying measures.
    """
    rvs = [[0], [1], [2], [3]]
    crvs = [4]
    t = total_correlation(d2, rvs=rvs, crvs=crvs)
    b = dual_total_correlation(d2, rvs=rvs, crvs=crvs)
    assert delta_k(d2, 2, rvs=rvs, crvs=crvs) == pytest.approx(b - t)
    assert gamma_k(d2, 2, rvs=rvs, crvs=crvs) == pytest.approx(t - b)


# ---------------------------------------------------------------------------
# Property tests (hypothesis) for the claims of Varley 2026, arXiv:2601.08030.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("k", [0, 1, 2, 3])
@given(dist=distributions(alphabets=(2,) * 4))
def test_delta_k_general_form(dist, k):
    """
    Delta^k equals its defining leave-one-out form (Eq. 15):
    Delta^k(X) = (N - k) T(X) - sum_i T(X^{-i}).
    """
    n = dist.outcome_length()
    expected = (n - k) * total_correlation(dist) - _loo_tc(dist)
    assert delta_k(dist, k) == pytest.approx(expected, abs=1e-4)


@pytest.mark.parametrize("k", [0, 1, 2, 3])
@given(dist=distributions(alphabets=(2,) * 4))
def test_gamma_k_conditional_form(dist, k):
    """
    Gamma^k equals its intuitive conditional form (Eq. 22):
    Gamma^k(X) = (N - k) D(X) - sum_i D(X^{-i} | X_i).
    """
    n = dist.outcome_length()
    expected = (n - k) * dual_total_correlation(dist) - _loo_cond_dtc(dist)
    assert gamma_k(dist, k) == pytest.approx(expected, abs=1e-4)


@given(dist=distributions(alphabets=(2,) * 3))
def test_delta_k_special_cases_property(dist):
    """
    Delta^0 = S-information, Delta^1 = D, Delta^2 = -O, over random dists.
    """
    assert delta_k(dist, 0) == pytest.approx(s_information(dist), abs=1e-4)
    assert delta_k(dist, 1) == pytest.approx(dual_total_correlation(dist), abs=1e-4)
    assert delta_k(dist, 2) == pytest.approx(-o_information(dist), abs=1e-4)


@given(dist=distributions(alphabets=(2,) * 3))
def test_gamma_k_special_cases_property(dist):
    """
    Gamma^0 = S-information, Gamma^1 = T, Gamma^2 = O, over random dists.
    """
    assert gamma_k(dist, 0) == pytest.approx(s_information(dist), abs=1e-4)
    assert gamma_k(dist, 1) == pytest.approx(total_correlation(dist), abs=1e-4)
    assert gamma_k(dist, 2) == pytest.approx(o_information(dist), abs=1e-4)


@pytest.mark.parametrize("k", [1, 2, 3])
@given(dist=distributions(alphabets=(2,) * 3))
def test_delta_k_affine_slope(dist, k):
    """
    Delta^k is affine in k with slope -T: Delta^k - Delta^{k-1} = -T.
    """
    diff = delta_k(dist, k) - delta_k(dist, k - 1)
    assert diff == pytest.approx(-total_correlation(dist), abs=1e-4)


@pytest.mark.parametrize("k", [1, 2, 3])
@given(dist=distributions(alphabets=(2,) * 3))
def test_gamma_k_affine_slope(dist, k):
    """
    Gamma^k is affine in k with slope -D: Gamma^k - Gamma^{k-1} = -D.
    """
    diff = gamma_k(dist, k) - gamma_k(dist, k - 1)
    assert diff == pytest.approx(-dual_total_correlation(dist), abs=1e-4)


@pytest.mark.parametrize("k", [0, 1, 2, 3])
@given(dist=distributions(alphabets=(2,) * 3))
def test_delta_gamma_conjugate_relations(dist, k):
    """
    Delta^k + Gamma^k = (2 - k) S and Delta^k - Gamma^k = -k O, which follow
    from Delta^k = S - kT, Gamma^k = S - kD, and O = T - D.
    """
    s = s_information(dist)
    o = o_information(dist)
    assert delta_k(dist, k) + gamma_k(dist, k) == pytest.approx((2 - k) * s, abs=1e-4)
    assert delta_k(dist, k) - gamma_k(dist, k) == pytest.approx(-k * o, abs=1e-4)


@pytest.mark.parametrize("k", [0, 1, 2])
@given(dist=distributions(alphabets=(2,) * 3))
def test_delta_k_monotone_in_k(dist, k):
    """
    Delta^k is non-increasing in k, since Delta^k - Delta^{k+1} = T >= 0.
    """
    assert delta_k(dist, k) >= delta_k(dist, k + 1) - 1e-9


@pytest.mark.parametrize("k", [0, 1, 2])
@given(dist=distributions(alphabets=(2,) * 3))
def test_gamma_k_monotone_in_k(dist, k):
    """
    Gamma^k is non-increasing in k, since Gamma^k - Gamma^{k+1} = D >= 0.
    """
    assert gamma_k(dist, k) >= gamma_k(dist, k + 1) - 1e-9


@given(dist=distributions(alphabets=(2,) * 3))
def test_removal_versus_revelation(dist):
    """
    Removing X_i destroys as much total correlation as revealing X_i destroys
    dual total correlation, and both equal I(X_i; X^{-i}) (Eq. 24):
    T(X) - T(X^{-i}) = D(X) - D(X^{-i} | X_i) = I(X_i; X^{-i}).
    """
    n = dist.outcome_length()
    for i in range(n):
        others = [j for j in range(n) if j != i]
        removal = total_correlation(dist) - total_correlation(dist.marginal(others))
        revelation = dual_total_correlation(dist) - dual_total_correlation(dist, rvs=[[j] for j in others], crvs=[i])
        mi = coinformation(dist, rvs=[[i], others])
        assert removal == pytest.approx(mi, abs=1e-4)
        assert revelation == pytest.approx(mi, abs=1e-4)


@pytest.mark.parametrize("k", [0, 1, 2, 3])
@given(
    dist_a=distributions(alphabets=(2,) * 2),
    dist_b=distributions(alphabets=(2,) * 2),
)
def test_delta_k_additive_over_independent_subsets(dist_a, dist_b, k):
    """
    Delta^k is additive over independent subsystems (Appendix):
    Delta^k(A (x) B) = Delta^k(A) + Delta^k(B).
    """
    joint = _independent_joint(dist_a, dist_b)
    expected = delta_k(dist_a, k) + delta_k(dist_b, k)
    assert delta_k(joint, k) == pytest.approx(expected, abs=1e-4)


@pytest.mark.parametrize("k", [0, 1, 2, 3])
@given(
    dist_a=distributions(alphabets=(2,) * 2),
    dist_b=distributions(alphabets=(2,) * 2),
)
def test_gamma_k_additive_over_independent_subsets(dist_a, dist_b, k):
    """
    Gamma^k is additive over independent subsystems:
    Gamma^k(A (x) B) = Gamma^k(A) + Gamma^k(B).
    """
    joint = _independent_joint(dist_a, dist_b)
    expected = gamma_k(dist_a, k) + gamma_k(dist_b, k)
    assert gamma_k(joint, k) == pytest.approx(expected, abs=1e-4)


@pytest.mark.parametrize("n", [3, 4, 5])
def test_delta_k_zero_for_pure_synergy(n):
    """
    n_mod_m(n, 2) is a pure n-th-order synergy, so Delta^n = 0.
    """
    dist = n_mod_m(n, 2)
    assert delta_k(dist, n) == pytest.approx(0, abs=1e-9)


@pytest.mark.parametrize("n", [3, 4, 5])
def test_gamma_k_zero_for_pure_redundancy(n):
    """
    giant_bit(n, 2) is a pure n-th-order redundancy, so Gamma^n = 0.
    """
    dist = giant_bit(n, 2)
    assert gamma_k(dist, n) == pytest.approx(0, abs=1e-9)
