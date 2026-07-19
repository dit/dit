"""
Tests for dit.multivariate.exact_common_information
"""

import pytest

from dit import Distribution as D
from dit.multivariate import exact_common_information as G
from dit.multivariate.common_informations.exact_common_information import ExactCommonInformation
from dit.shannon import entropy
from tests._backends import backends

outcomes = ["0000", "0001", "0110", "0111", "1010", "1011", "1100", "1101"]
pmf = [1 / 8] * 8
xor = D(outcomes, pmf)

sbec = lambda p: D(["00", "0e", "1e", "11"], [(1 - p) / 2, p / 2, p / 2, (1 - p) / 2])
G_sbec = lambda p: min(1, entropy(p) + 1 - p)


@pytest.mark.flaky(reruns=5)
@pytest.mark.parametrize("backend", backends)
@pytest.mark.parametrize(
    ("rvs", "crvs", "val"),
    [
        (None, None, 2.0),
        ([[0], [1], [2]], None, 2.0),
        ([[0], [1]], [2, 3], 1.0),
        ([[0], [1]], [2], 1.0),
        ([[0], [1]], None, 0.0),
    ],
)
def test_eci1(rvs, crvs, val, backend):
    """
    Test against known values.
    """
    assert G(xor, rvs, crvs, backend=backend) == pytest.approx(val, abs=1e-3)


@pytest.fixture
def x0():
    return {"x0": None}


@pytest.mark.flaky(reruns=5)
@pytest.mark.parametrize("i", range(1, 10))
def test_eci2(i, x0):
    """
    Test the binary symmetric erasure channel.
    """
    p = i / 10
    eci = ExactCommonInformation(sbec(p))
    eci.optimize(x0=x0["x0"])
    x0["x0"] = eci._optima
    assert eci.objective(eci._optima) == pytest.approx(G_sbec(p), abs=1e-3)


def test_eci_not_subadditive_under_product():
    """
    Kumar et al.: G(d ⊗ d) can be strictly less than 2 G(d) for independent pairs.
    """
    d = D(["00", "01", "10"], [1 / 3] * 3)
    assert G(d @ d, [[0, 2], [1, 3]]) < 2 * G(d, [[0], [1]])


def test_eci_large_alphabet_bound_no_overflow():
    """
    Regression: the cardinality bound must not overflow int64 for large alphabets.

    ``compute_bound`` evaluates ``2 ** k`` with ``k`` a product of alphabet
    sizes. As an ``np.int64`` this wraps negative once ``k >= 63``, which made
    the bound negative and later raised "negative dimensions are not allowed"
    while allocating the auxiliary variable. Here the two smallest alphabets
    multiply to ``8 * 8 = 64``, so constructing the optimizer must still succeed
    and yield a positive bound.
    """
    giant8 = D([str(i) * 3 for i in range(8)], [1 / 8] * 8)
    eci = ExactCommonInformation(giant8, bound=2)  # must not raise
    assert eci._optvec_size > 0
