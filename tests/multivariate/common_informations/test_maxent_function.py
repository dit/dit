"""
Tests for dit.multivariate.common_informations.maxent_function.
"""

import numpy as np
import pytest

from dit import Distribution
from dit.exceptions import ditException
from dit.multivariate import gk_common_information as K
from dit.multivariate import maxent_function, plot_maxent_function


def _h2(*probs):
    p = np.asarray(probs, dtype=float)
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


@pytest.fixture
def block_diag():
    """
    Salamatian et al. 2016, §II-B example.  P/(1/8) = block-diag of two
    2x2 blocks of ones, so the bipartite graph has two disjoint
    components and Gács-Körner = 1 bit.
    """
    outcomes = ["00", "01", "10", "11", "22", "23", "32", "33"]
    pmf = [1 / 8] * 8
    return Distribution(outcomes, pmf)


@pytest.fixture
def leak_distribution():
    """
    Salamatian et al. 2016, §III-B example with delta = 1: the block
    diagonal P with the (1,1) atom moved to (1,2), connecting the two
    components.  Gács-Körner = 0, but the (+1,+1,-1,-1) partition is
    still nearly optimal once a helper is allowed.
    """
    outcomes = ["00", "01", "10", "12", "22", "23", "32", "33"]
    pmf = [1 / 8] * 8
    return Distribution(outcomes, pmf)


def test_block_diag_exact_recovers_gk(block_diag):
    """Exact M_0 = K = 1 on disjoint components."""
    assert maxent_function(block_diag, epsilon=0.0, method="exact") == pytest.approx(1.0)
    assert K(block_diag) == pytest.approx(1.0)


def test_block_diag_spectral_recovers_gk(block_diag):
    """Spectral M_0 = 1 on disjoint components."""
    assert maxent_function(block_diag, epsilon=0.0, method="spectral") == pytest.approx(1.0)


def test_leak_at_eps_zero_is_zero(leak_distribution):
    """K = 0 forces both methods to the trivial value at epsilon = 0."""
    assert K(leak_distribution) == pytest.approx(0.0)
    assert maxent_function(leak_distribution, epsilon=0.0, method="exact") == pytest.approx(0.0)
    assert maxent_function(leak_distribution, epsilon=0.0, method="spectral") == pytest.approx(0.0)


def test_leak_with_helper_reaches_one_bit(leak_distribution):
    """
    With delta = 1 the (+1,+1,-1,-1) partition has H(phi_X) = 1 bit and
    H(phi_X | phi_Y) = (5/8) * h(1/5).  At that helper rate both methods
    should attain the full bit.
    """
    helper = (5 / 8) * _h2(1 / 5, 4 / 5)
    assert maxent_function(leak_distribution, epsilon=helper, method="exact") == pytest.approx(1.0)
    assert maxent_function(leak_distribution, epsilon=helper, method="spectral") == pytest.approx(1.0)


def test_eps_below_zero_returns_zero(block_diag):
    """Negative epsilon admits no partition at all."""
    assert maxent_function(block_diag, epsilon=-0.1, method="exact") == pytest.approx(0.0)
    assert maxent_function(block_diag, epsilon=-0.1, method="spectral") == pytest.approx(0.0)


def test_large_eps_bounded_by_one(block_diag, leak_distribution):
    """Binary functions have entropy at most 1 bit regardless of epsilon."""
    for dist in (block_diag, leak_distribution):
        for method in ("exact", "spectral"):
            assert maxent_function(dist, epsilon=10.0, method=method) <= 1.0 + 1e-9


def test_independent_uniforms_gives_zero_at_eps_zero():
    """For X _||_ Y, H(phi_X | phi_Y) = H(phi_X) > 0 for any non-trivial phi."""
    outcomes = ["00", "01", "10", "11"]
    pmf = [1 / 4] * 4
    d = Distribution(outcomes, pmf)
    assert maxent_function(d, epsilon=0.0, method="exact") == pytest.approx(0.0)
    assert maxent_function(d, epsilon=0.0, method="spectral") == pytest.approx(0.0)


def test_giant_bit_full_correlation():
    """For perfectly correlated bits, M_0 = K = 1 bit."""
    d = Distribution(["00", "11"], [1 / 2] * 2)
    assert maxent_function(d, epsilon=0.0, method="exact") == pytest.approx(1.0)
    assert maxent_function(d, epsilon=0.0, method="spectral") == pytest.approx(1.0)


def test_spectral_lower_bound_by_exact(leak_distribution):
    """
    The spectral algorithm is a feasibility-preserving heuristic on the
    same search space as the exact brute force, so it can never exceed
    the exact value at the same epsilon.
    """
    for eps in (0.0, 0.2, 0.5, 0.8):
        spec = maxent_function(leak_distribution, epsilon=eps, method="spectral")
        exact = maxent_function(leak_distribution, epsilon=eps, method="exact")
        assert spec <= exact + 1e-9


def test_method_monotone_in_eps(leak_distribution):
    """M_eps is monotone non-decreasing in epsilon."""
    eps_grid = [0.0, 0.1, 0.3, 0.6, 1.0, 5.0]
    for method in ("exact", "spectral"):
        values = [maxent_function(leak_distribution, epsilon=e, method=method) for e in eps_grid]
        for v_prev, v_next in zip(values[:-1], values[1:], strict=True):
            assert v_next + 1e-9 >= v_prev


def test_rv_names_accepted(block_diag):
    """Random-variable names should be accepted wherever indices are."""
    block_diag.set_rv_names("XY")
    assert maxent_function(block_diag, epsilon=0.0, rvs=[["X"], ["Y"]], method="exact") == pytest.approx(1.0)
    assert maxent_function(block_diag, epsilon=0.0, rvs=[["X"], ["Y"]], method="spectral") == pytest.approx(1.0)


def test_crvs_rejected(block_diag):
    """Conditioning is not implemented; must raise."""
    with pytest.raises(ditException):
        maxent_function(block_diag, epsilon=0.5, rvs=[[0], [1]], crvs=[0])


def test_requires_two_variables():
    """Strictly bivariate; an n != 2 rvs list must raise."""
    d = Distribution(["000", "111"], [1 / 2] * 2)
    with pytest.raises(ditException):
        maxent_function(d, epsilon=0.5)
    with pytest.raises(ditException):
        maxent_function(d, epsilon=0.5, rvs=[[0]])
    with pytest.raises(ditException):
        maxent_function(d, epsilon=0.5, rvs=[[0], [1], [2]])


def test_unknown_method_raises(block_diag):
    """Misspelled method should fail loudly."""
    with pytest.raises(ditException):
        maxent_function(block_diag, epsilon=0.5, method="lagrangian")


def test_exact_alphabet_cap_raises():
    """Exact mode refuses absurdly large alphabets."""
    n = 14  # 2^13 * 2^13 = 2^26, > 2^20
    outcomes = [f"{i:02d}{j:02d}" for i in range(n) for j in range(n)]
    pmf = [1 / (n * n)] * (n * n)
    d = Distribution(outcomes, pmf)
    with pytest.raises(ditException):
        maxent_function(d, epsilon=0.0, method="exact")


def test_plot_returns_axes_with_curve(block_diag):
    """Smoke test: plot helper returns an axis with a single line."""
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")

    ax = plot_maxent_function(block_diag)
    lines = ax.get_lines()
    assert len(lines) == 1
    xs, ys = lines[0].get_data()
    assert len(xs) == len(ys) >= 4  # at least 4 distinct thresholds for n_X = 4
    assert max(ys) == pytest.approx(1.0)
    assert all(0.0 - 1e-9 <= y <= 1.0 + 1e-9 for y in ys)


def test_plot_accepts_kwargs_and_axis(block_diag):
    """plot_maxent_function should respect a user-supplied axis and kwargs."""
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    out = plot_maxent_function(block_diag, ax=ax, color="red", linestyle="--", marker="x")
    assert out is ax
    line = ax.get_lines()[0]
    assert line.get_color() == "red"
    assert line.get_linestyle() == "--"
    assert line.get_marker() == "x"
    plt.close(fig)
