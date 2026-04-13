"""
Tests for dit.multivariate.common_informations.stochastic_gk_common_information.

For finite discrete distributions the stochastic GK common information equals
the classical GK common information, because the bipartite support graph's
connected-component structure forces all conditionals within a component to
agree. This lets us reuse known GK values as ground truth.
"""

import pytest
from hypothesis import given, settings

from dit import Distribution
from dit.multivariate import gk_common_information as K
from dit.multivariate.common_informations.stochastic_gk_common_information import (
    StochasticGKCommonInformation,
    stochastic_gk_common_information as SGK,
)
from dit.shannon import mutual_information as I
from dit.utils.testing import distributions


# ---------------------------------------------------------------------------
# Analytic tests against known GK values
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.flaky(reruns=5)
def test_sgk_independent():
    """SGK of independent variables is 0 (full support forces all conditionals equal)."""
    d = Distribution(["00", "01", "10", "11"], [1 / 4] * 4)
    assert SGK(d) == pytest.approx(0.0, abs=1e-3)


@pytest.mark.slow
@pytest.mark.flaky(reruns=5)
def test_sgk_mixed():
    """SGK is 0 when support connectivity merges all values."""
    d = Distribution(["00", "01", "11"], [1 / 3] * 3)
    assert SGK(d) == pytest.approx(0.0, abs=1e-3)


@pytest.mark.slow
@pytest.mark.flaky(reruns=5)
def test_sgk_block_structure():
    """SGK with disconnected support components equals GK = 1.5 bits."""
    d = Distribution(
        ["00", "01", "10", "11", "22", "33"],
        [1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 4, 1 / 4],
    )
    assert SGK(d) == pytest.approx(1.5, abs=1e-3)


# ---------------------------------------------------------------------------
# rvs / crvs / rv_names interface
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.flaky(reruns=5)
def test_sgk_explicit_rvs():
    """Passing explicit rvs gives the same result."""
    d = Distribution(
        ["00", "01", "10", "11", "22", "33"],
        [1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 4, 1 / 4],
    )
    assert SGK(d, [[0], [1]]) == pytest.approx(1.5, abs=1e-3)


@pytest.mark.slow
@pytest.mark.flaky(reruns=5)
def test_sgk_rv_names():
    """String-based rv names should work."""
    d = Distribution(
        ["00", "01", "10", "11", "22", "33"],
        [1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 4, 1 / 4],
    )
    d.set_rv_names("XY")
    assert SGK(d, [["X"], ["Y"]]) == pytest.approx(1.5, abs=1e-3)


@pytest.mark.slow
@pytest.mark.flaky(reruns=5)
def test_sgk_three_vars():
    """
    Trivariate distribution from the GK test suite:
    K([[0],[1]]) = 1.5, K(all) = 1.0
    """
    d = Distribution(
        ["000", "010", "100", "110", "221", "331"],
        [1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 4, 1 / 4],
    )
    assert SGK(d, [[0], [1]]) == pytest.approx(1.5, abs=1e-3)
    assert SGK(d) == pytest.approx(1.0, abs=1e-3)


@pytest.mark.slow
@pytest.mark.flaky(reruns=5)
def test_sgk_with_crvs():
    """Conditioning reduces the common information."""
    d = Distribution(
        ["000", "010", "100", "110", "221", "331"],
        [1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 4, 1 / 4],
    )
    d.set_rv_names("XYZ")
    assert SGK(d, ["X", "Y"], ["Z"]) == pytest.approx(0.5, abs=1e-3)


# ---------------------------------------------------------------------------
# Class-level API
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.flaky(reruns=5)
def test_sgk_class_api():
    """Direct use of the optimizer class should produce the same result."""
    d = Distribution(
        ["00", "01", "10", "11", "22", "33"],
        [1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 4, 1 / 4],
    )
    sgk = StochasticGKCommonInformation(d)
    sgk.optimize()
    val = -sgk.objective(sgk._optima)
    assert val == pytest.approx(1.5, abs=1e-3)


# ---------------------------------------------------------------------------
# Ordering: GK <= SGK <= I(X;Y)
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.flaky(reruns=5)
@settings(max_examples=5)
@given(dist=distributions(alphabets=(2,) * 2))
def test_sgk_ordering(dist):
    """
    For any bivariate distribution: K(X;Y) <= SGK(X;Y) <= I(X;Y).
    """
    epsilon = 1e-2
    k = K(dist)
    sgk = SGK(dist)
    mi = I(dist, [0], [1])
    assert k <= sgk + epsilon
    assert sgk <= mi + epsilon
