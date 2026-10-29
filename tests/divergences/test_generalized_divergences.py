"""
Tests for dit.divergences.kullback_leibler_divergence.
"""

from functools import partial
from itertools import combinations, product

import numpy as np
import pytest

from dit import Distribution
from dit.divergences import (
    alpha_divergence,
    f_divergence,
    hellinger_divergence,
    hellinger_sum,
    kullback_leibler_divergence,
    renyi_divergence,
    tsallis_divergence,
)
from dit.exceptions import ditException
from dit.other import renyi_entropy

divergences = [alpha_divergence, renyi_divergence, tsallis_divergence, hellinger_divergence]


d1 = Distribution(["0", "1"], [1 / 2, 1 / 2])
d2 = Distribution(["0", "2"], [1 / 2, 1 / 2])
d3 = Distribution(["0", "1", "2"], [1 / 3, 1 / 3, 1 / 3])
d4 = Distribution(["00", "11"], [2 / 5, 3 / 5])
d5 = Distribution(["00", "11"], [1 / 2, 1 / 2])


def get_dists_2():
    """
    Construct several example distributions.
    """
    d1 = Distribution(["0", "1"], [1 / 2, 1 / 2])
    d2 = Distribution(["0", "1"], [1 / 3, 2 / 3])
    d3 = Distribution(["0", "1"], [2 / 5, 3 / 5])
    return d1, d2, d3


def get_dists_3():
    """
    Construct several example distributions.
    """
    d1 = Distribution(["0", "1", "2"], [1 / 5, 2 / 5, 2 / 5])
    d2 = Distribution(["0", "1", "2"], [1 / 4, 1 / 2, 1 / 4])
    d3 = Distribution(["0", "1", "2"], [1 / 3, 1 / 3, 1 / 3])
    d4 = Distribution(["0", "1", "2"], [1 / 6, 2 / 6, 3 / 6])
    return d1, d2, d3, d4


@pytest.mark.parametrize("dist", [d1, d2, d3, d4, d5])
@pytest.mark.parametrize("alpha", [0, 1, 2, 0.5])
@pytest.mark.parametrize("divergence", divergences)
def test_positive_definite1(dist, alpha, divergence):
    """
    Tests that divergences are zero when the input distributions are the same, and that the Hellinger sum is equal to 1.
    """
    assert divergence(dist, dist, alpha) == pytest.approx(0)
    assert hellinger_sum(dist, dist, alpha) == pytest.approx(1)


@pytest.mark.parametrize("dist", [d4, d5])
@pytest.mark.parametrize("alpha", [0, 1, 2, 0.5])
@pytest.mark.parametrize("divergence", divergences)
def test_positive_definite2(dist, alpha, divergence):
    """
    Tests that divergences are zero when the input distributions are the same, and that the Hellinger sum is equal to 1.
    """
    assert divergence(dist, dist, alpha, rvs=[0, 1]) == pytest.approx(0)
    assert hellinger_sum(dist, dist, alpha, rvs=[0, 1]) == pytest.approx(1)


@pytest.mark.parametrize("alpha", [0.1, 0.5, 1, 1.5])
@pytest.mark.parametrize("dists", [get_dists_2(), get_dists_3()])
@pytest.mark.parametrize("divergence", divergences)
def test_positivity(alpha, dists, divergence):
    """
    Tests that the divergence functions return positive values for non-equal arguments.
    """
    for dist1, dist2 in combinations(dists, 2):
        assert divergence(dist1, dist2, alpha) > 0


@pytest.mark.parametrize("alpha", [-1, 0, 0.5, 1, 2])
@pytest.mark.parametrize("dists", [get_dists_2(), get_dists_3()])
def test_alpha_symmetry(alpha, dists):
    """
    Tests the alpha -> -alpha symmetry for the alpha divergence, and a similar
    symmetry for the Hellinger and Renyi divergences.
    """
    for dist1, dist2 in product(dists, repeat=2):
        assert alpha_divergence(dist1, dist2, alpha) == pytest.approx(alpha_divergence(dist2, dist1, -alpha))
        assert (1 - alpha) * hellinger_divergence(dist1, dist2, alpha) == pytest.approx(
            alpha * hellinger_divergence(dist2, dist1, 1 - alpha)
        )
        assert (1 - alpha) * renyi_divergence(dist1, dist2, alpha) == pytest.approx(
            alpha * renyi_divergence(dist2, dist1, 1 - alpha)
        )


@pytest.mark.parametrize("dists", [get_dists_2(), get_dists_3()])
def test_divergences_to_kl(dists):
    """
    Tests that the generalized divergences properly fall back to KL for the appropriate values of alpha, and not otherwise.
    """
    for dist1, dist2 in combinations(dists, 2):
        kl_nats = kullback_leibler_divergence(dist1, dist2) * np.log(2)
        assert alpha_divergence(dist1, dist2, alpha=-1) == pytest.approx(kl_nats)

        assert alpha_divergence(dist1, dist2, alpha=0) != pytest.approx(kl_nats)
        assert alpha_divergence(dist1, dist2, alpha=2) != pytest.approx(kl_nats)


@pytest.mark.parametrize("dists", [get_dists_2(), get_dists_3()])
@pytest.mark.parametrize("divergence", divergences)
def test_divergences_to_kl2(dists, divergence):
    """
    Tests that the generalized divergences properly fall back to KL for the appropriate values of alpha.
    """
    for dist1, dist2 in combinations(dists, 2):
        if divergence is alpha_divergence:
            kl = kullback_leibler_divergence(dist2, dist1) * np.log(2)
        elif divergence is renyi_divergence:
            kl = kullback_leibler_divergence(dist1, dist2)
        else:
            kl = kullback_leibler_divergence(dist1, dist2) * np.log(2)
        assert divergence(dist1, dist2, alpha=1) == pytest.approx(kl)


@pytest.mark.parametrize("dists", [get_dists_2(), get_dists_3()])
@pytest.mark.parametrize("divergence", divergences)
def test_continuous_at_one(dists, divergence):
    """
    The alpha = 1 special case must agree with the limit of the general formula.
    """
    for dist1, dist2 in combinations(dists, 2):
        at_one = divergence(dist1, dist2, alpha=1)
        near = divergence(dist1, dist2, alpha=1 + 1e-6)
        assert at_one == pytest.approx(near, rel=1e-4)


def test_union_support():
    """
    Outcomes outside the support of `dist1` must still contribute.
    """
    p = Distribution(["0", "1"], [1, 0])
    q = Distribution(["0", "1"], [1 / 2, 1 / 2])
    assert alpha_divergence(p, q, alpha=3) == pytest.approx(alpha_divergence(q, p, alpha=-3))
    assert f_divergence(p, q, lambda t: abs(t - 1) / 2) == pytest.approx(0.5)
    assert f_divergence(p, q, lambda t: (t - 1) ** 2) == pytest.approx(1.0)


@pytest.mark.parametrize(
    "args",
    [
        [d4, d1, None],
        [d4, d2, None],
        [d4, d3, None],
        [d1, d2, [0, 1]],
        [d3, d4, [1]],
        [d5, d1, [0, 1]],
        [d4, d3, [1, 0]],
    ],
)
@pytest.mark.parametrize("divergence", divergences)
@pytest.mark.parametrize("alpha", [0, 1, 2, 0.5])
def test_exceptions(args, divergence, alpha):
    """
    Test that when p has outcomes that q doesn't have, that we raise an exception.
    """
    first, second, rvs = args
    with pytest.raises(ditException):
        divergence(first, second, alpha, rvs)


def test_renyi_values():
    """
    Test specific values of the Renyi divergence.
    """
    d1 = Distribution(["0", "1"], [0, 1])
    d2 = Distribution(["0", "1"], [1 / 2, 1 / 2])
    d3 = Distribution(["0", "1"], [1, 0])

    assert renyi_divergence(d1, d2, 1 / 2) == pytest.approx(np.log2(2))
    assert renyi_divergence(d2, d3, 1 / 2) == pytest.approx(np.log2(2))
    assert renyi_divergence(d1, d3, 1 / 2) == pytest.approx(np.inf)
    assert renyi_divergence(d1, d2, np.inf) == pytest.approx(np.log2(2))
    assert renyi_divergence(d3, d2, np.inf) == pytest.approx(np.log2(2))
    assert renyi_divergence(d2, d3, np.inf) == pytest.approx(np.inf)


def test_renyi_infinity_is_limit():
    """
    The Renyi divergence of order infinity is the max-divergence, the limit of
    the divergences of increasing order.
    """
    p = Distribution(["0", "1", "2"], [1 / 2, 1 / 3, 1 / 6])
    q = Distribution(["0", "1", "2"], [1 / 3, 1 / 3, 1 / 3])
    d_inf = renyi_divergence(p, q, np.inf)
    assert d_inf == pytest.approx(np.log2(3 / 2))
    assert renyi_divergence(p, q, 2) <= renyi_divergence(p, q, 20) <= d_inf
    assert renyi_divergence(p, q, 200) == pytest.approx(d_inf, abs=1e-2)

    joint1 = Distribution(["00", "11"], [2 / 5, 3 / 5])
    joint2 = Distribution(["00", "01", "10", "11"], [1 / 4] * 4)
    assert renyi_divergence(joint1, joint2, np.inf) == pytest.approx(np.log2(12 / 5))
    assert renyi_divergence(joint1, joint2, np.inf, rvs=[0]) == pytest.approx(np.log2(6 / 5))
    with pytest.raises(ditException):
        renyi_divergence(d4, d1, np.inf)


@pytest.mark.parametrize("alpha", [0, 1, 2, 0.5, np.inf])
def test_renyi(alpha):
    """
    Consistency test for Renyi entropy and Renyi divergence
    """
    dist1 = Distribution(["0", "1", "2"], [1 / 4, 1 / 2, 1 / 4])
    uniform = Distribution(["0", "1", "2"], [1 / 3, 1 / 3, 1 / 3])
    h = renyi_entropy(dist1, alpha)
    h_u = renyi_entropy(uniform, alpha)
    div = renyi_divergence(dist1, uniform, alpha)
    assert h == pytest.approx(h_u - div)


@pytest.mark.parametrize("alpha", [-1.0, 0.1, 0.5, 1.0, 1.1])
@pytest.mark.parametrize("dist1", get_dists_3())
@pytest.mark.parametrize("dist2", get_dists_3())
def test_f_divergence(alpha, dist1, dist2):
    """
    Tests various known relations of f-divergences to other divergences.
    """

    def f_alpha(alpha):
        if alpha == 1:

            def f(x):
                return x * np.log2(x)
        elif alpha == -1:

            def f(x):
                return -np.log2(x)
        else:

            def f(x):
                return 4.0 / (1.0 - alpha**2) * (1.0 - np.power(x, (1.0 + alpha) / 2))

        return f

    def f_tsallis(alpha):
        if alpha == 1:

            def f(x):
                return -np.log2(x)
        else:

            def f(x):
                return (np.power(x, 1.0 - alpha) - 1.0) / (alpha - 1.0)

        return f

    test_functions = [
        (f_alpha(alpha), partial(alpha_divergence, alpha=alpha)),
        (f_tsallis(alpha), partial(tsallis_divergence, alpha=alpha)),
    ]

    for f, div_func in test_functions:
        div1 = f_divergence(dist1, dist2, f)
        div2 = div_func(dist1, dist2)
        assert div1 == pytest.approx(div2, abs=1e-1)


@pytest.mark.parametrize("alpha", [-1.0, 0.1, 0.5, 1.0, 1.1])
@pytest.mark.parametrize("dist1", [d4, d5])
@pytest.mark.parametrize("dist2", [d4, d5])
def test_f_divergence2(alpha, dist1, dist2):
    """
    Tests various known relations of f-divergences to other divergences.
    """

    def f_alpha(alpha):
        if alpha == 1:

            def f(x):
                return x * np.log2(x)
        elif alpha == -1:

            def f(x):
                return -np.log2(x)
        else:

            def f(x):
                return 4.0 / (1.0 - alpha**2) * (1.0 - np.power(x, (1.0 + alpha) / 2))

        return f

    def f_tsallis(alpha):
        if alpha == 1:

            def f(x):
                return -np.log2(x)
        else:

            def f(x):
                return (np.power(x, 1.0 - alpha) - 1.0) / (alpha - 1.0)

        return f

    test_functions = [
        (f_alpha(alpha), partial(alpha_divergence, alpha=alpha)),
        (f_tsallis(alpha), partial(tsallis_divergence, alpha=alpha)),
    ]

    for f, div_func in test_functions:
        div1 = f_divergence(dist1=dist1, dist2=dist2, f=f, rvs=[0, 1])
        div2 = div_func(dist1=dist1, dist2=dist2, rvs=[0, 1])
        assert div1 == pytest.approx(div2, abs=1e-1)
