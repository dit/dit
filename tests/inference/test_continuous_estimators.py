"""
Tests for the nearest-neighbor divergence and generalized-entropy estimators,
mixed KSG, and the Gaussian-copula mutual information.
"""

import numpy as np
import pytest

from dit.inference import (
    copula_normalize,
    cross_entropy_knn,
    differential_entropy_knn,
    gaussian_copula_mutual_information,
    jensen_shannon_divergence_knn,
    kullback_leibler_divergence_knn,
    mutual_information_mixed_ksg,
    renyi_entropy_knn,
    tsallis_entropy_knn,
)

LN2 = np.log(2)


def _reference_data():
    """
    The samples used to generate the gcmi and mixed-KSG references.
    """
    rng = np.random.default_rng(0)
    x = rng.normal(size=(500, 2))
    y = x[:, :1] ** 3 + rng.normal(size=(500, 1))
    z = rng.normal(size=(500, 1)) + x[:, 1:]
    xd = rng.integers(0, 4, (800, 1)).astype(float)
    yd = xd + rng.normal(size=(800, 1)) * (xd > 1)
    return x, y, z, xd, yd


def test_gaussian_copula_matches_gcmi():
    """
    Values from Ince's gcmi package (``gcmi_cc`` and ``gccmi_ccc``).
    """
    x, y, z, _, _ = _reference_data()
    assert gaussian_copula_mutual_information(x, y) == pytest.approx(0.7143659566900046)
    assert gaussian_copula_mutual_information(x, y, z) == pytest.approx(0.7154642342770527)


def test_gaussian_copula_bivariate_normal():
    rng = np.random.default_rng(1)
    rho = 0.6
    data = rng.multivariate_normal([0, 0], [[1, rho], [rho, 1]], size=5000)
    truth = -0.5 * np.log2(1 - rho**2)
    mi = gaussian_copula_mutual_information(data[:, 0], data[:, 1])
    assert mi == pytest.approx(truth, abs=0.02)
    # invariant to monotone transformations of the marginals
    assert gaussian_copula_mutual_information(np.exp(data[:, 0]), data[:, 1] ** 3) == pytest.approx(mi)
    assert copula_normalize(data).shape == (5000, 2)
    with pytest.raises(ValueError):
        gaussian_copula_mutual_information(data[:10, 0], data[:, 1])


def test_mixed_ksg_matches_reference():
    """
    Values from Gao et al.'s ``Mixed_KSG`` (converted to bits).
    """
    x, y, _, xd, yd = _reference_data()
    assert mutual_information_mixed_ksg(xd, yd) == pytest.approx(1.5103176288312814)
    assert mutual_information_mixed_ksg(x, y) == pytest.approx(0.6961562253816136)


def test_mixed_ksg_discrete():
    """
    On purely discrete data the estimate approaches the plug-in mutual information.
    """
    rng = np.random.default_rng(2)
    x = rng.integers(0, 3, 3000)
    y = (x + (rng.random(3000) < 0.2)) % 3
    p = np.histogram2d(x, y, bins=[3, 3])[0] / 3000
    px, py = p.sum(axis=1), p.sum(axis=0)
    plugin = np.nansum(p * np.log2(p / np.outer(px, py)))
    assert mutual_information_mixed_ksg(x, y) == pytest.approx(plugin, abs=0.02)
    with pytest.raises(ValueError):
        mutual_information_mixed_ksg(x, y[:-1])
    with pytest.raises(ValueError):
        mutual_information_mixed_ksg(x[:3], y[:3], k=5)


def test_divergences_gaussian():
    rng = np.random.default_rng(3)
    n, s1, s2 = 20000, 1.0, 2.0
    p, q = rng.normal(0, s1, (n, 1)), rng.normal(0, s2, (n, 1))
    kl = (np.log(s2 / s1) + s1**2 / (2 * s2**2) - 0.5) / LN2
    ce = (0.5 * np.log(2 * np.pi * s2**2) + s1**2 / (2 * s2**2)) / LN2
    assert kullback_leibler_divergence_knn(p, q, rng=0) == pytest.approx(kl, abs=0.03)
    assert cross_entropy_knn(p, q, rng=0) == pytest.approx(ce, abs=0.03)
    assert kullback_leibler_divergence_knn(p, q, rng=0) == pytest.approx(
        cross_entropy_knn(p, q, rng=0) - differential_entropy_knn(p, rng=0), abs=1e-3
    )
    xs = np.linspace(-15, 15, 300001)
    dx = xs[1] - xs[0]

    def f(s):
        return np.exp(-(xs**2) / (2 * s**2)) / np.sqrt(2 * np.pi * s**2)

    m = (f(s1) + f(s2)) / 2
    jsd = (
        -np.sum(m * np.log2(m)) * dx
        - 0.25 * np.log2(2 * np.pi * np.e * s1**2)
        - 0.25 * np.log2(2 * np.pi * np.e * s2**2)
    )
    assert jensen_shannon_divergence_knn(p, q, rng=0) == pytest.approx(jsd, abs=0.03)


@pytest.mark.parametrize("order", [2.0, 3.0])
def test_renyi_tsallis_gaussian(order):
    rng = np.random.default_rng(4)
    d = 2
    data = rng.normal(size=(20000, d))
    integral = (2 * np.pi) ** (d * (1 - order) / 2) * order ** (-d / 2)
    assert renyi_entropy_knn(data, order, rng=0) == pytest.approx(np.log2(integral) / (1 - order), abs=0.06)
    assert tsallis_entropy_knn(data, order, rng=0) == pytest.approx((1 - integral) / (order - 1), abs=0.01)


def test_renyi_order_one_and_errors():
    rng = np.random.default_rng(5)
    data = rng.normal(size=(2000, 1))
    assert renyi_entropy_knn(data, 1, rng=0) == pytest.approx(differential_entropy_knn(data, rng=0))
    assert tsallis_entropy_knn(data, 1, rng=0) == pytest.approx(differential_entropy_knn(data, rng=0) * LN2)
    with pytest.raises(ValueError):
        renyi_entropy_knn(data, 6, k=4)
