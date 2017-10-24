"""
Test various information theory inequalities.
"""

from hypothesis import given

import numpy as np

from dit.utils.testing import distributions

from dit import ScalarDistribution as SD
from dit.divergences import relative_entropy, variational_distance
from dit.multivariate import entropy as H, total_correlation as I


@given(dist=distributions())
def test_entropy_upper_bound(dist):
    """
    H(X) <= log(|X|)
    """
    h = H(dist)
    logX = np.log2(len(dist.outcomes))
    assert h <= logX + 1e-10

@given(dist1=distributions(size=1, alphabet=10), dist2=distributions(size=1, alphabet=10))
def test_pinskers_inequality(dist1, dist2):
    """
    DKL(p||q) >= V(p||q)**2 / (2log(2))
    """
    dkl = relative_entropy(dist1, dist2)
    vd = variational_distance(dist1, dist2)
    assert dkl >= vd**2 / (2 * np.log(2)) - 1e-10

@given(dist=distributions(size=2, alphabet=10, min_events=3))
def test_fanos_inequality(dist):
    """
    H(X) <= hb(P_e) + P_e log(|X| - 1)
    """
    dist1 = SD.from_distribution(dist.marginal([0]))
    dist2 = SD.from_distribution(dist.marginal([1]))

    ce = H(dist, [0], [1])

    X = len(set().union(dist1.outcomes, dist2.outcomes))

    eq_dist = dist1 == dist2
    P_e = eq_dist[False] if False in eq_dist else 0

    hb = H(SD([P_e, 1-P_e]))

    assert ce <= hb + P_e * np.log2(X - 1) + 1e-10

@given(dist=distributions())
def test_entropy_subadditivity(dist):
    """
    H(X1, ...) <= sum(H(X_i))
    """
    h = H(dist)
    h_sum = sum(H(dist.marginal(rv)) for rv in dist.rvs)
    assert h <= h_sum + 1e-10

@given(dist=distributions(size=3))
def test_shannon_inequality(dist):
    """
    I(X:Y|Z) >= 0
    """
    i = I(dist, [[0], [1]], [2])
    assert i >= -1e-10

@given(dist=distributions(size=4))
def test_zhang_yeung_inequality(dist):
    """
    2I(C:D) <= I(A:B)+I(A:CD)+3I(C:D|A)+I(C:D|B)
    """
    I_a_b = I(dist, [[0], [1]])
    I_c_d = I(dist, [[2], [3]])
    I_a_cd = I(dist, [[0], [2, 3]])
    I_c_d_g_a = I(dist, [[2], [3]], [0])
    I_c_d_g_b = I(dist, [[2], [3]], [1])
    assert 2*I_c_d <= I_a_b + I_a_cd + 3*I_c_d_g_a + I_c_d_g_b + 1e-10
