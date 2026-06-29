"""
Test various information theory inequalities.
"""

import numpy as np
import pytest
from hypothesis import given, settings

from dit import Distribution
from dit.divergences import (
    chernoff_information,
    hellinger_distance,
    hypercontractivity_coefficient,
    maximum_correlation,
    relative_entropy,
    variational_distance,
)
from dit.helpers import normalize_pmfs
from dit.multivariate import (
    entropy as H,
)
from dit.multivariate import (
    exact_common_information as G,
)
from dit.multivariate import (
    gk_common_information as K,
)
from dit.multivariate import (
    total_correlation as I,
)
from dit.multivariate import (
    wyner_common_information as C,
)
from dit.utils.testing import distributions, markov_chains
from tests._backends import backends

epsilon = 1e-4

# Dougherty–Freiling–Zeger unified form (dougherty2011four, eq. 37–42):
#   2 I(A:B) <= a I(A:B|C) + b I(A:C|B) + c I(B:C|A)
#               + d I(A:B|D) + e I(A:D|B) + f I(B:D|A) + g I(C:D)
# Variables A,B,C,D map to rvs 0,1,2,3.
DFZ_INEQUALITIES = [
    ("dfz_37", (5, 3, 1, 2, 0, 0, 2)),
    ("dfz_38", (4, 2, 1, 3, 1, 0, 2)),
    ("dfz_39", (4, 4, 1, 2, 1, 1, 2)),
    ("dfz_40", (3, 3, 3, 2, 0, 0, 2)),
    ("dfz_41", (3, 4, 2, 3, 1, 0, 2)),
    ("dfz_42", (3, 2, 2, 2, 1, 1, 2)),
]

FOUR_VAR_ALPHABETS = ((2, 4),) * 4


def _dfz_rhs(dist, coeffs):
    """Right-hand side of a DFZ non-Shannon inequality."""
    a, b, c, d, e, f, g = coeffs
    rhs = 0.0
    if a:
        rhs += a * I(dist, [[0], [1]], [2])
    if b:
        rhs += b * I(dist, [[0], [2]], [1])
    if c:
        rhs += c * I(dist, [[1], [2]], [0])
    if d:
        rhs += d * I(dist, [[0], [1]], [3])
    if e:
        rhs += e * I(dist, [[0], [3]], [1])
    if f:
        rhs += f * I(dist, [[1], [3]], [0])
    if g:
        rhs += g * I(dist, [[2], [3]])
    return rhs


@given(dist=distributions())
def test_entropy_upper_bound(dist):
    """
    H(X) <= log(|X|)
    """
    h = H(dist)
    logX = np.log2(len(dist.outcomes))
    assert h <= logX + epsilon


@given(dist1=distributions(alphabets=(10,)), dist2=distributions(alphabets=(10,)))
def test_pinskers_inequality(dist1, dist2):
    """
    DKL(p||q) >= V(p||q)**2 / (2log(2))
    """
    dkl = relative_entropy(dist1, dist2)
    vd = variational_distance(dist1, dist2)
    assert dkl >= vd**2 / (2 * np.log(2)) - epsilon


@given(dist=distributions(alphabets=(10, 10), nondegenerate=True))
def test_fanos_inequality(dist):
    """
    H(X|Y) <= hb(P_e) + P_e log(|X| - 1)
    """
    ce = H(dist, [0], [1])

    dist1 = dist.marginal([0])
    X = len(dist1.alphabet[0])

    P_correct = sum(p for o, p in dist.zipped() if o[0] == o[1])
    P_e = 1 - P_correct

    hb = H(Distribution([P_e, 1 - P_e]))

    assert ce <= hb + P_e * np.log2(X - 1) + epsilon


@given(dist=distributions(alphabets=((2, 4),) * 4))
def test_entropy_subadditivity(dist):
    """
    H(X1, ...) <= sum(H(X_i))
    """
    h = H(dist)
    h_sum = sum(H(dist.marginal(rv)) for rv in dist.rvs)
    assert h <= h_sum + epsilon


@given(dist1=distributions(alphabets=(10,)), dist2=distributions(alphabets=(10,)))
def test_gibbs_inequality(dist1, dist2):
    """
    DKL(p||q) >= 0
    """
    dkl = relative_entropy(dist1, dist2)
    assert dkl >= 0 - epsilon


@given(dist=distributions(alphabets=((2, 4),) * 2))
def test_conditional_entropy(dist):
    """
    H(X|Y) <= H(X)
    """
    ch = H(dist, [0], [1])
    h = H(dist, [0])
    assert ch <= h + epsilon


@given(dist=distributions(alphabets=((2, 4),) * 3))
def test_shannon_inequality(dist):
    """
    I(X:Y|Z) >= 0
    """
    i = I(dist, [[0], [1]], [2])
    assert i >= 0 - epsilon


@given(dist=distributions(alphabets=FOUR_VAR_ALPHABETS))
def test_zhang_yeung_inequality(dist):
    """
    2I(C:D) <= I(A:B)+I(A:CD)+3I(C:D|A)+I(C:D|B)
    """
    I_a_b = I(dist, [[0], [1]])
    I_c_d = I(dist, [[2], [3]])
    I_a_cd = I(dist, [[0], [2, 3]])
    I_c_d_g_a = I(dist, [[2], [3]], [0])
    I_c_d_g_b = I(dist, [[2], [3]], [1])
    assert 2 * I_c_d <= I_a_b + I_a_cd + 3 * I_c_d_g_a + I_c_d_g_b + epsilon


@pytest.mark.parametrize("name,coeffs", DFZ_INEQUALITIES, ids=[n for n, _ in DFZ_INEQUALITIES])
@given(dist=distributions(alphabets=FOUR_VAR_ALPHABETS))
def test_dfz_non_shannon_inequality(dist, name, coeffs):
    """
    DFZ unified non-Shannon inequality (Dougherty–Freiling–Zeger 2011).

    2 I(A:B) <= a I(A:B|C) + b I(A:C|B) + c I(B:C|A)
                + d I(A:B|D) + e I(A:D|B) + f I(B:D|A) + g I(C:D)
    """
    lhs = 2 * I(dist, [[0], [1]])
    assert lhs <= _dfz_rhs(dist, coeffs) + epsilon


@given(dist=markov_chains(alphabets=((2, 4),) * 3))
def test_data_processing_inequality(dist):
    """
    given X - Y - Z:
        I(X:Z) <= I(X:Y)
    """
    i_xy = I(dist, [[0], [1]])
    i_xz = I(dist, [[0], [2]])
    assert i_xz <= i_xy + epsilon


@given(dist=markov_chains(alphabets=((2, 4),) * 3))
def test_data_processing_inequality_mc(dist):
    """
    given X - Y - Z:
        rho(X:Z) <= rho(X:Y)
    """
    rho_xy = maximum_correlation(dist, [[0], [1]])
    rho_xz = maximum_correlation(dist, [[0], [2]])
    assert rho_xz <= rho_xy + 10 * epsilon


@given(dist=markov_chains(alphabets=((2, 4),) * 3))
def test_data_processing_inequality_gk(dist):
    """
    given X - Y - Z:
        K(X:Z) <= K(X:Y)
    """
    k_xy = K(dist, [[0], [1]])
    k_xz = K(dist, [[0], [2]])
    assert k_xz <= k_xy + epsilon


@pytest.mark.flaky(reruns=5)
@pytest.mark.parametrize("backend", backends)
@given(dist=markov_chains(alphabets=(2,) * 3))
@settings(max_examples=5)
def test_data_processing_inequality_wyner(dist, backend):
    """
    given X - Y - Z:
        C(X:Z) <= C(X:Y)
    """
    c_xy = C(dist, [[0], [1]], niter=150, backend=backend)
    c_xz = C(dist, [[0], [2]], niter=150, backend=backend)
    assert c_xz <= c_xy + 150 * epsilon


@pytest.mark.flaky(reruns=10)
@pytest.mark.parametrize("backend", backends)
@given(dist=markov_chains(alphabets=(2,) * 3))
@settings(max_examples=3)
def test_data_processing_inequality_exact(dist, backend):
    """
    given X - Y - Z:
        G(X:Z) <= G(X:Y)
    """
    niter = 300
    g_xy = G(dist, [[0], [1]], niter=niter, backend=backend)
    g_xz = G(dist, [[0], [2]], niter=niter, backend=backend)
    assert g_xz <= g_xy + niter * epsilon


@given(dist=distributions(alphabets=((2, 4),) * 2))
def test_max_correlation_mutual_information(dist):
    """
    (p_min * rho(X:Y))^2 <= (2 ln 2)I(X:Y)
    """
    p_min = dist.marginal([0]).pmf.min()
    rho = maximum_correlation(dist, [[0], [1]])
    i = I(dist, [[0], [1]])
    assert (p_min * rho) ** 2 <= (2 * np.log(2)) * i + epsilon


@given(dist1=distributions(alphabets=(10,)), dist2=distributions(alphabets=(10,)))
def test_hellinger_variational(dist1, dist2):
    """
    H^2(p||q) <= V(p||q) <= sqrt(2)*H(p||q)
    """
    h = hellinger_distance(dist1, dist2)
    v = variational_distance(dist1, dist2)
    assert h**2 <= v + epsilon
    assert v <= np.sqrt(2) * h + epsilon


@given(dist1=distributions(alphabets=(10,), zeros=False), dist2=distributions(alphabets=(10,), zeros=False))
def test_chernoff_inequalities(dist1, dist2):
    """
    1/8 sum p_i ((q_i - p_i)/max(p_i, q_i))^2 <= 1 - 2^(-C)
                                              <= 1/8 sum p_i ((q_i - p_i)/min(p_i, q_i))^2
    """
    p, q = normalize_pmfs(dist1, dist2)
    pq = np.vstack([p, q])
    c = chernoff_information(dist1, dist2)
    a = (p * ((q - p) / pq.max(axis=0)) ** 2).sum() / 8
    b = (p * ((q - p) / pq.min(axis=0)) ** 2).sum() / 8
    assert a <= 1 - 2 ** (-c) + epsilon
    assert 1 - 2 ** (-c) <= b + epsilon


@given(dist=markov_chains(alphabets=(2,) * 3))
def test_mi_hc(dist):
    """
    given U - X - Y:
        I[U:Y] <= s*(X||Y)*I[U:X]
    """
    a = I(dist, [[0], [2]])
    b = hypercontractivity_coefficient(dist, [[1], [2]], niter=150)
    c = I(dist, [[0], [1]])
    assert a <= b * c + epsilon
