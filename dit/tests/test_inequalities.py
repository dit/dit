"""
Test various information theory inequalities.
"""

from hypothesis import given

import numpy as np

from dit.utils.testing import distributions, markov_chains

from dit import ScalarDistribution as SD
from dit.algorithms import maximum_correlation
from dit.divergences import hellinger_distance, relative_entropy, variational_distance
from dit.multivariate import entropy as H, total_correlation as I


epsilon = 1e-8

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
    H(X) <= hb(P_e) + P_e log(|X| - 1)
    """
    dist1 = SD.from_distribution(dist.marginal([0]))
    dist2 = SD.from_distribution(dist.marginal([1]))

    ce = H(dist, [0], [1])

    X = len(set().union(dist1.outcomes, dist2.outcomes))

    eq_dist = dist1 == dist2
    P_e = eq_dist[False] if False in eq_dist else 0

    hb = H(SD([P_e, 1-P_e]))

    assert ce <= hb + P_e * np.log2(X - 1) + epsilon


@given(dist=distributions(alphabets=(4, 4, 4, 4)))
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


@given(dist=distributions(alphabets=((2, 4),)*2))
def test_conditional_entropy(dist):
    """
    H(X|Y) <= H(X)
    """
    ch = H(dist, [0], [1])
    h = H(dist, [0])
    assert ch <= h + epsilon


@given(dist=distributions(alphabets=((2, 4),)*3))
def test_shannon_inequality(dist):
    """
    I(X:Y|Z) >= 0
    """
    i = I(dist, [[0], [1]], [2])
    assert i >= 0 - epsilon


@given(dist=distributions(alphabets=((2, 4),)*4))
def test_zhang_yeung_inequality(dist):
    """
    2I(C:D) <= I(A:B)+I(A:CD)+3I(C:D|A)+I(C:D|B)
    """
    I_a_b = I(dist, [[0], [1]])
    I_c_d = I(dist, [[2], [3]])
    I_a_cd = I(dist, [[0], [2, 3]])
    I_c_d_g_a = I(dist, [[2], [3]], [0])
    I_c_d_g_b = I(dist, [[2], [3]], [1])
    assert 2*I_c_d <= I_a_b + I_a_cd + 3*I_c_d_g_a + I_c_d_g_b + epsilon


@given(dist=markov_chains(alphabets=((2, 4),)*3))
def test_data_processing_inequality(dist):
    """
    given X - Y - Z:
        I(X:Z) <= I(X:Y)
    """
    i_xy = I(dist, [[0], [1]])
    i_xz = I(dist, [[0], [2]])
    assert i_xz <= i_xy + epsilon


@given(dist=markov_chains(alphabets=((2, 4),)*3))
def test_data_processing_inequality_mc(dist):
    """
    given X - Y - Z:
        rho(X:Z) <= rho(X:Y)
    """
    rho_xy = maximum_correlation(dist, [[0], [1]])
    rho_xz = maximum_correlation(dist, [[0], [2]])
    assert rho_xz <= rho_xy + epsilon


@given(dist=distributions(alphabets=((2, 4),)*2))
def test_max_correlation_mutual_information(dist):
    """
    (p_min * rho(X:Y))^2 <= (2 ln 2)I(X:Y)
    """
    p_min = dist.marginal([0]).pmf.min()
    rho = maximum_correlation(dist, [[0], [1]])
    i = I(dist, [[0], [1]])
    assert (p_min*rho)**2 <= (2*np.log(2))*i + epsilon


@given(dist1=distributions(alphabets=(10,)), dist2=distributions(alphabets=(10,)))
def test_hellinger_variational(dist1, dist2):
    """
    H^2(p||q) <= V(p||q) <= sqrt(2)*H(p||q)
    """
    h = hellinger_distance(dist1, dist2)
    v = variational_distance(dist1, dist2)
    assert h**2 <= v + epsilon
    assert v <= np.sqrt(2)*h + epsilon
