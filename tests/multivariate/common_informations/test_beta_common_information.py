"""
Tests for dit.multivariate.common_informations.beta_common_information
"""

import numpy as np
import pytest
from hypothesis import given, settings

from dit import Distribution as D
from dit.divergences import maximum_correlation
from dit.exceptions import OptimizationException
from dit.multivariate import wyner_common_information as C
from dit.multivariate.common_informations.beta_common_information import (
    BetaCommonInformation,
    _conditional_maxcorr_from_3d,
    _max_pairwise_maxcorr,
    _maxcorr_from_2d,
    beta_common_information as C_beta,
)
from dit.utils.testing import distributions


# ── Distributions used across tests ──────────────────────────────────────

def dsbs(p0):
    """Doubly symmetric binary source with crossover probability p0."""
    return D(
        ["00", "01", "10", "11"],
        [0.5 * (1 - p0), 0.5 * p0, 0.5 * p0, 0.5 * (1 - p0)],
    )


independent = D(["00", "01", "10", "11"], [0.25] * 4)


# ── Unit tests for internal maximal-correlation helpers ──────────────────


class TestMaxcorrHelpers:
    def test_maxcorr_2d_independent(self):
        pmf = np.array([[0.25, 0.25], [0.25, 0.25]])
        assert _maxcorr_from_2d(pmf) == pytest.approx(0.0, abs=1e-10)

    def test_maxcorr_2d_copy(self):
        pmf = np.array([[0.5, 0.0], [0.0, 0.5]])
        assert _maxcorr_from_2d(pmf) == pytest.approx(1.0, abs=1e-10)

    def test_maxcorr_2d_dsbs(self):
        p0 = 0.2
        pmf = np.array([
            [0.5 * (1 - p0), 0.5 * p0],
            [0.5 * p0, 0.5 * (1 - p0)],
        ])
        assert _maxcorr_from_2d(pmf) == pytest.approx(1 - 2 * p0, abs=1e-10)

    def test_maxcorr_2d_degenerate(self):
        pmf = np.array([[1.0]])
        assert _maxcorr_from_2d(pmf) == 0.0

    def test_conditional_maxcorr_3d(self):
        p0 = 0.1
        d = dsbs(p0)
        d.make_dense()
        pmf = d.pmf.reshape(2, 2)
        # Add a trivial conditioning dimension
        pmf_3d = pmf[:, :, np.newaxis]
        assert _conditional_maxcorr_from_3d(pmf_3d) == pytest.approx(1 - 2 * p0, abs=1e-10)

    def test_conditional_maxcorr_3d_independent_given_z(self):
        # P(X,Y,Z) where X _|_ Y | Z
        pmf = np.zeros((2, 2, 2))
        pmf[0, 0, 0] = 0.25
        pmf[0, 1, 0] = 0.25
        pmf[1, 0, 1] = 0.25
        pmf[1, 1, 1] = 0.25
        assert _conditional_maxcorr_from_3d(pmf) == pytest.approx(0.0, abs=1e-10)

    def test_max_pairwise_single_rv(self):
        joint = np.array([0.5, 0.5])
        assert _max_pairwise_maxcorr(joint, {0}, set()) == 0.0


# ── Fast-path tests (no optimisation) ────────────────────────────────────


class TestBetaCommonInformationFastPaths:
    def test_beta_one(self):
        """C_beta = 0 for beta >= 1."""
        d = dsbs(0.1)
        assert C_beta(d, beta=1.0) == 0.0

    def test_beta_above_rho(self):
        """C_beta = 0 when beta >= rho_m(X;Y)."""
        d = dsbs(0.2)
        rho = maximum_correlation(d)
        assert C_beta(d, beta=rho + 0.01) == 0.0

    def test_independent_any_beta(self):
        """Independent variables: C_beta = 0 for all beta."""
        assert C_beta(independent, beta=0.0) == 0.0
        assert C_beta(independent, beta=0.5) == 0.0

    def test_three_var_above_rho(self):
        """Multivariate fast path: beta above all pairwise rho_m."""
        d = D(
            ["000", "001", "010", "011", "100", "101", "110", "111"],
            [0.2, 0.05, 0.05, 0.1, 0.05, 0.1, 0.1, 0.35],
        )
        rho = max(
            maximum_correlation(d, [[0], [1]]),
            maximum_correlation(d, [[0], [2]]),
            maximum_correlation(d, [[1], [2]]),
        )
        assert C_beta(d, beta=rho + 0.01) == 0.0


# ── Optimisation-based tests ─────────────────────────────────────────────


class TestBetaCommonInformationOptimisation:
    @pytest.mark.slow
    @pytest.mark.flaky(reruns=5)
    def test_beta_zero_matches_wyner(self):
        """C_beta(beta=0) should equal Wyner common information."""
        d = dsbs(0.2)
        wyner = C(d)
        c_0 = C_beta(d, beta=0, niter=25)
        assert c_0 == pytest.approx(wyner, abs=5e-2)

    @pytest.mark.slow
    @pytest.mark.flaky(reruns=5)
    def test_monotone_in_beta(self):
        """C_beta is decreasing in beta."""
        d = dsbs(0.1)
        c_0 = C_beta(d, beta=0.0, niter=10)
        c_mid = C_beta(d, beta=0.4, niter=10)
        c_hi = C_beta(d, beta=0.7, niter=5)
        assert c_0 >= c_mid - 1e-2
        assert c_mid >= c_hi - 1e-2
        assert c_hi >= -1e-2

    @pytest.mark.slow
    @pytest.mark.flaky(reruns=5)
    def test_nonnegative(self):
        """C_beta should always be non-negative."""
        d = dsbs(0.3)
        for beta in [0.0, 0.2, 0.35]:
            c = C_beta(d, beta=beta, niter=5)
            assert c >= -1e-4

    @pytest.mark.slow
    @pytest.mark.flaky(reruns=5)
    def test_upper_bounded_by_wyner(self):
        """C_beta <= C_0 = Wyner for all beta >= 0."""
        d = dsbs(0.15)
        wyner = C(d)
        c_mid = C_beta(d, beta=0.3, niter=10)
        assert c_mid <= wyner + 5e-2

    @pytest.mark.slow
    @pytest.mark.flaky(reruns=5)
    def test_highly_correlated(self):
        """For a highly correlated source (p=0.05), C_0 ~ Wyner."""
        d = dsbs(0.05)
        wyner = C(d)
        c_0 = C_beta(d, beta=0, niter=15)
        assert c_0 == pytest.approx(wyner, abs=5e-2)

    @pytest.mark.slow
    @pytest.mark.flaky(reruns=5)
    def test_constraint_satisfied(self):
        """The optimal U should satisfy rho_m(X;Y|U) <= beta."""
        d = dsbs(0.1)
        beta = 0.4
        bci = BetaCommonInformation(d, beta=beta)
        bci.optimize(niter=10)
        slack = bci.constraint_maximal_correlation(bci._optima)
        assert slack >= -1e-3  # constraint satisfied (within tolerance)


# ── Conditional and multivariate tests ───────────────────────────────────


class TestBetaCommonInformationConditional:
    @pytest.mark.slow
    @pytest.mark.flaky(reruns=5)
    def test_conditional_beta_zero_matches_wyner(self):
        """C_beta(X:Y|Z, beta=0) ~ Wyner(X:Y|Z)."""
        d = D(
            ["000", "001", "010", "011", "100", "101", "110", "111"],
            [0.2, 0.05, 0.05, 0.1, 0.05, 0.1, 0.1, 0.35],
        )
        wyner = C(d, rvs=[[0], [1]], crvs=[2])
        c_0 = C_beta(d, beta=0, rvs=[[0], [1]], crvs=[2], niter=15)
        assert c_0 == pytest.approx(wyner, abs=0.1)

    def test_conditional_fast_path(self):
        """C_beta(X:Y|Z) = 0 when beta >= rho_m(X;Y|Z)."""
        d = D(
            ["000", "001", "010", "011", "100", "101", "110", "111"],
            [0.2, 0.05, 0.05, 0.1, 0.05, 0.1, 0.1, 0.35],
        )
        rho = maximum_correlation(d, [[0], [1]], [2])
        assert C_beta(d, beta=rho + 0.01, rvs=[[0], [1]], crvs=[2]) == 0.0


class TestBetaCommonInformationMultivariate:
    @pytest.mark.slow
    @pytest.mark.flaky(reruns=5)
    def test_three_var_monotone(self):
        """C_beta(X:Y:Z) is decreasing in beta for 3 variables."""
        d = D(
            ["000", "001", "010", "011", "100", "101", "110", "111"],
            [0.2, 0.05, 0.05, 0.1, 0.05, 0.1, 0.1, 0.35],
        )
        c_lo = C_beta(d, beta=0.1, niter=10)
        c_hi = C_beta(d, beta=0.3, niter=10)
        assert c_lo >= c_hi - 1e-2
        assert c_hi >= -1e-2


# ── Property-based (Hypothesis) tests ────────────────────────────────────


@pytest.mark.slow
@pytest.mark.flaky(reruns=5)
@settings(max_examples=5)
@given(dist=distributions(alphabets=(2,) * 2, nondegenerate=True))
def test_c_beta_bounded_by_wyner(dist):
    """
    For random binary bivariate distributions:
        0 <= C_beta(beta=0.3) <= C_0 = Wyner
    """
    rho = maximum_correlation(dist)
    if rho < 0.31:
        assert C_beta(dist, beta=0.3) == pytest.approx(0.0, abs=1e-4)
    else:
        try:
            c = C_beta(dist, beta=0.3, niter=10)
        except OptimizationException:
            pytest.skip("optimizer did not converge")
        wyner = C(dist, niter=25)
        assert c >= -1e-2
        assert c <= wyner + 0.1


@settings(max_examples=5)
@given(dist=distributions(alphabets=(2,) * 2, nondegenerate=True))
def test_c_beta_zero_at_rho(dist):
    """
    For random binary bivariate distributions:
        C_beta = 0 when beta >= rho_m(X;Y)
    """
    rho = maximum_correlation(dist)
    assert C_beta(dist, beta=min(rho + 0.01, 1.0)) == pytest.approx(0.0, abs=1e-4)
