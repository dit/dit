"""
Tests for dit.other.sibson_mutual_information.
"""

import numpy as np
import pytest
from scipy.optimize import minimize

from dit import Distribution
from dit.example_dists import Xor, uniform
from dit.other import (
    maximal_leakage,
    renyi_entropy,
    sibson_conditional_mutual_information_y_given_z,
    sibson_conditional_mutual_information_z,
    sibson_mutual_information,
    sibson_mutual_information_pmf,
)
from dit.shannon import mutual_information


@pytest.mark.parametrize("alpha", [1 / 2, 1, 2, 5, np.inf])
def test_sibson_independent(alpha):
    """Independent X, Y give zero Sibson MI for all orders."""
    d = Distribution(["00", "01", "10", "11"], [1 / 4] * 4)
    assert sibson_mutual_information(d, [0], [1], alpha) == pytest.approx(0.0)


@pytest.mark.parametrize("alpha", [1 / 2, 1, 2, 5, np.inf])
def test_sibson_deterministic_xy(alpha):
    """When Y = X, I_alpha = H_{1/alpha}(X)."""
    d = Distribution(["00", "11"], [1 / 2, 1 / 2])
    expected = renyi_entropy(d.marginal([0]), 1 / alpha if alpha != np.inf else 0)
    if alpha == np.inf:
        expected = np.log2(2)
    assert sibson_mutual_information(d, [0], [1], alpha) == pytest.approx(expected)


def test_sibson_alpha_one_matches_shannon():
    d = Xor()
    assert sibson_mutual_information(d, [0], [1], 1) == pytest.approx(mutual_information(d, [0], [1]))


def test_maximal_leakage_deterministic():
    d = Distribution(["00", "11"], [0.5, 0.5])
    assert maximal_leakage(d, [0], [1]) == pytest.approx(1.0)
    assert sibson_mutual_information(d, [0], [1], np.inf) == pytest.approx(1.0)


def test_sibson_asymmetric():
    """Directed channel: I(X;Y) != I(Y;X) in general."""
    d = Distribution(["00", "01", "10", "11"], [0.5, 0.25, 0.0, 0.25])
    a = sibson_mutual_information(d, [0], [1], 2)
    b = sibson_mutual_information(d, [1], [0], 2)
    assert a != pytest.approx(b)


def test_sibson_pmf_matches_distribution():
    p_xy = np.array([[0.5, 0.0], [0.0, 0.5]])
    d = Distribution(["00", "11"], [0.5, 0.5])
    assert sibson_mutual_information_pmf(p_xy, 2) == pytest.approx(sibson_mutual_information(d, [0], [1], 2))


@pytest.mark.parametrize("alpha", [-1, 0, -0.001])
def test_sibson_invalid_order(alpha):
    d = uniform(4)
    with pytest.raises(ValueError, match="`order` must be a positive real number"):
        sibson_mutual_information(d, [0], [1], alpha)


def test_sibson_min_q_crosscheck():
    """Closed form matches min_Q D_alpha(P_XY || P_X Q) at alpha=2."""
    p_xy = np.array([[0.4, 0.1], [0.1, 0.4]])
    alpha = 2.0
    closed = sibson_mutual_information_pmf(p_xy, alpha)

    p_x = p_xy.sum(axis=1)

    def objective(q_y):
        q_y = np.clip(q_y, 1e-15, 1.0)
        q_y = q_y / q_y.sum()
        s = 0.0
        for x in range(2):
            for y in range(2):
                q_xy = p_x[x] * q_y[y]
                if q_xy > 0 and p_xy[x, y] > 0:
                    s += p_xy[x, y] ** alpha * q_xy ** (1 - alpha)
        return (1 / (alpha - 1)) * np.log2(s)

    result = minimize(
        objective,
        x0=np.array([0.5, 0.5]),
        method="SLSQP",
        bounds=[(1e-15, 1.0)] * 2,
        constraints={"type": "eq", "fun": lambda q: q.sum() - 1},
    )
    assert closed == pytest.approx(result.fun, rel=1e-4)


def test_conditional_z_constant_z_matches_unconditional():
    """I^Y|Z_alpha reduces to I_alpha(X;Y) when Z is constant."""
    d = Distribution(["000", "110"], [0.5, 0.5])
    for alpha in [2, 3, np.inf]:
        unc = sibson_mutual_information(d, [0], [1], alpha)
        cond = sibson_conditional_mutual_information_y_given_z(d, [0], [1], [2], alpha)
        assert cond == pytest.approx(unc)


def test_conditional_z_symmetric():
    d = Distribution(["000", "011", "100", "111"], [0.25] * 4)
    a = sibson_conditional_mutual_information_z(d, [0], [1], [2], 2)
    b = sibson_conditional_mutual_information_z(d, [1], [0], [2], 2)
    assert a == pytest.approx(b)


def test_conditional_xor_given_z():
    """3-var XOR: Shannon I(X;Y|Z) = 1; I^Z_alpha is positive at alpha > 1."""
    d = Xor()
    cmi = sibson_conditional_mutual_information_z(d, [0], [1], [2], 1)
    assert cmi == pytest.approx(1.0)
    assert sibson_conditional_mutual_information_z(d, [0], [1], [2], 2) > 0


@pytest.mark.parametrize("alpha", [0, -1])
def test_sibson_pmf_invalid_order(alpha):
    p_xy = np.full((2, 2), 0.25)
    with pytest.raises(ValueError, match="`order` must be a positive real number"):
        sibson_mutual_information_pmf(p_xy, alpha)
