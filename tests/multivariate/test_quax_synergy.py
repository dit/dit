"""
Tests for dit.multivariate.quax_synergy.
"""

import numpy as np
import pytest

from dit import Distribution
from dit.example_dists import And, Xor
from dit.multivariate import max_synergistic_entropy, quax_synergy


# ---------------------------------------------------------------------------
# max_synergistic_entropy (analytical, no optimisation)
# ---------------------------------------------------------------------------

def test_mse_two_independent_bits():
    """H(X1,X2) - max(H(X1), H(X2)) = 2 - 1 = 1 for two fair bits."""
    d = Distribution(["00", "01", "10", "11"], [1 / 4] * 4)
    assert max_synergistic_entropy(d) == pytest.approx(1.0)


def test_mse_single_variable():
    """With a single variable, synergistic entropy is 0."""
    d = Distribution(["0", "1"], [1 / 2] * 2)
    assert max_synergistic_entropy(d) == pytest.approx(0.0)


def test_mse_identical_variables():
    """Two perfectly correlated variables: H(X,X) = H(X), so bound is 0."""
    d = Distribution(["00", "11"], [1 / 2] * 2)
    assert max_synergistic_entropy(d) == pytest.approx(0.0)


def test_mse_three_bits():
    """H(X1,X2,X3) - max_i H(Xi) = 3 - 1 = 2 for three fair bits."""
    outcomes = [f"{a}{b}{c}" for a in "01" for b in "01" for c in "01"]
    d = Distribution(outcomes, [1 / 8] * 8)
    assert max_synergistic_entropy(d) == pytest.approx(2.0)


def test_mse_with_names():
    """max_synergistic_entropy works with named random variables."""
    d = Distribution(["00", "01", "10", "11"], [1 / 4] * 4)
    d.set_rv_names("XY")
    assert max_synergistic_entropy(d, [["X"], ["Y"]]) == pytest.approx(1.0)


def test_mse_unequal_marginals():
    """When marginals differ, uses the largest."""
    d = Distribution(["00", "01", "10", "11"], [1 / 4] * 4)
    assert max_synergistic_entropy(d, [[0], [1]]) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# quax_synergy (optimisation-based)
# ---------------------------------------------------------------------------

def test_synergy_single_source():
    """With a single source, synergy is always 0 (Eq. 10)."""
    d = Distribution(["00", "01", "10", "11"], [1 / 4] * 4)
    val = quax_synergy(d, sources=[[0]], target=[1])
    assert val == pytest.approx(0.0, abs=1e-3)


@pytest.mark.slow
@pytest.mark.flaky(reruns=5)
def test_synergy_xor():
    """XOR of two independent bits: I_syn should be close to 1.0 (Section 5.1)."""
    d = Xor()
    val = quax_synergy(d, sources=[[0], [1]], target=[2], niter=10)
    assert val == pytest.approx(1.0, abs=0.1)


@pytest.mark.slow
@pytest.mark.flaky(reruns=5)
def test_synergy_and():
    """AND gate: I_syn ~ 0.311 bits (Section 5.3)."""
    d = And()
    expected = -3 / 4 * np.log2(3 / 4)
    val = quax_synergy(d, sources=[[0], [1]], target=[2], niter=10)
    assert val == pytest.approx(expected, abs=0.1)


@pytest.mark.slow
@pytest.mark.flaky(reruns=5)
def test_synergy_nonneg():
    """Synergistic information is non-negative (Eq. 6)."""
    d = Xor()
    val = quax_synergy(d, sources=[[0], [1]], target=[2], niter=5)
    assert val >= -1e-6


@pytest.mark.slow
@pytest.mark.flaky(reruns=5)
def test_synergy_bounded_by_mi():
    """I_syn <= I(X:Y) (Eq. 7)."""
    from dit.shannon import mutual_information as I

    d = Xor()
    isyn = quax_synergy(d, sources=[[0], [1]], target=[2], niter=5)
    total_mi = I(d, [0, 1], [2])
    assert isyn <= total_mi + 1e-6


@pytest.mark.slow
@pytest.mark.flaky(reruns=5)
def test_synergy_bounded_by_mse():
    """I_syn(X -> Y) <= max_synergistic_entropy(X)."""
    d = Xor()
    isyn = quax_synergy(d, sources=[[0], [1]], target=[2], niter=5)
    mse = max_synergistic_entropy(d, [[0], [1]])
    assert isyn <= mse + 1e-6
