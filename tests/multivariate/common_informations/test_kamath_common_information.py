"""
Tests for dit.multivariate.kamath_common_information.
"""

import pytest
from scipy.stats import entropy as _h

from dit import Distribution
from dit.multivariate import directed_kamath_common_information as G
from dit.multivariate import gk_common_information as K
from dit.multivariate import kamath_common_information as U


def _h2(*probs):
    """Shannon entropy (base 2) of the given probabilities."""
    return _h(list(probs), base=2)


@pytest.fixture
def kamath_example():
    """
    The 3x4 joint distribution from Kamath & Anantharam (2010), Section III-A.

    Joint matrix (rows = X in {a,b,c}, cols = Y in {α,β,γ,δ}):

        Y =   α      β      γ      δ
        a    4/37   0      0      0
        b    0      9/37   2/37   3/37
        c    0     12/37   3/37   4/37
    """
    outcomes = [
        "aα",
        "bβ",
        "bγ",
        "bδ",
        "cβ",
        "cγ",
        "cδ",
    ]
    pmf = [
        4 / 37,
        9 / 37,
        2 / 37,
        3 / 37,
        12 / 37,
        3 / 37,
        4 / 37,
    ]
    return Distribution(outcomes, pmf)


def test_paper_example_G_y_to_x(kamath_example):
    """G(Y -> X) on the paper's example: Phi^X_Y collapses β,δ to one atom."""
    expected = _h2(4 / 37, 28 / 37, 5 / 37)
    assert G(kamath_example, rvs=[1], about=[0]) == pytest.approx(expected)


def test_paper_example_G_x_to_y(kamath_example):
    """G(X -> Y) on the paper's example: Phi^Y_X is bijective with X."""
    expected = _h2(4 / 37, 14 / 37, 19 / 37)
    assert G(kamath_example, rvs=[0], about=[1]) == pytest.approx(expected)


def test_paper_example_U(kamath_example):
    """U(X; Y) = max{G(Y->X), G(X->Y)} on the paper's example."""
    expected = max(_h2(4 / 37, 28 / 37, 5 / 37), _h2(4 / 37, 14 / 37, 19 / 37))
    assert U(kamath_example) == pytest.approx(expected)
    assert U(kamath_example, [[0], [1]]) == pytest.approx(expected)


def test_U_giant_bit():
    """For perfectly correlated bits, U = I(X;Y) = 1."""
    d = Distribution(["00", "11"], [1 / 2] * 2)
    assert U(d) == pytest.approx(1.0)
    assert U(d, [[0], [1]]) == pytest.approx(1.0)


def test_U_independent():
    """For independent uniforms, every Y gives the same P(X|Y); U = 0."""
    d = Distribution(["00", "01", "10", "11"], [1 / 4] * 4)
    assert U(d) == pytest.approx(0.0)
    assert U(d, [[0], [1]]) == pytest.approx(0.0)


def test_U_mixed():
    """K's canonical 6-outcome example: U coincides with K = 1.5 here."""
    outcomes = ["00", "01", "10", "11", "22", "33"]
    pmf = [1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 4, 1 / 4]
    d = Distribution(outcomes, pmf)
    assert U(d) == pytest.approx(1.5)
    assert K(d) == pytest.approx(1.5)
    assert U(d) >= K(d)


def test_U_xor_three_variables():
    """
    For XOR on three bits, every single variable is a deterministic function
    of the other two, so the MSS of X_i about the rest is X_i itself; thus
    U = max_i H(X_i) = 1, while K = 0.
    """
    d = Distribution(["000", "011", "101", "110"], [1 / 4] * 4)
    assert K(d) == pytest.approx(0.0)
    assert U(d) == pytest.approx(1.0)


def test_U_conditional():
    """Conditional U threads `crvs` through to the conditional entropy."""
    outcomes = ["000", "010", "100", "110", "221", "331"]
    pmf = [1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 4, 1 / 4]
    d = Distribution(outcomes, pmf)
    assert U(d, [[0], [1]]) == pytest.approx(1.5)
    assert U(d, [[0], [1]], [2]) == pytest.approx(0.5)


def test_U_rv_names(kamath_example):
    """Random-variable names should be accepted everywhere indices are."""
    expected = max(_h2(4 / 37, 28 / 37, 5 / 37), _h2(4 / 37, 14 / 37, 19 / 37))
    kamath_example.set_rv_names("XY")
    assert U(kamath_example, [["X"], ["Y"]]) == pytest.approx(expected)
    assert G(kamath_example, rvs=["Y"], about=["X"]) == pytest.approx(_h2(4 / 37, 28 / 37, 5 / 37))
    assert G(kamath_example, rvs=["X"], about=["Y"]) == pytest.approx(_h2(4 / 37, 14 / 37, 19 / 37))


def test_U_bounds_K():
    """K(d) <= U(d) for any joint distribution (a sanity bound)."""
    outcomes = ["00", "01", "10", "11", "22", "33"]
    pmf = [1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 4, 1 / 4]
    d = Distribution(outcomes, pmf)
    assert K(d) <= U(d) + 1e-9
