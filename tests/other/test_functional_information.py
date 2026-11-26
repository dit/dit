"""
Tests for dit.other.functional_information.
"""

import numpy as np
import pytest

import dit
from dit import Distribution
from dit.divergences.pmf import channel_simulation_divergence
from dit.other import (
    functional_information,
    functional_information_pmf,
    renyi_entropy,
    sibson_mutual_information,
    sibson_mutual_information_pmf,
)
from dit.rate_distortion import excess_functional_information
from dit.shannon import entropy, mutual_information


def dsbs(c):
    """A doubly-symmetric binary source with crossover c."""
    return Distribution(["00", "01", "10", "11"], [(1 - c) / 2, c / 2, c / 2, (1 - c) / 2])


def bec(eps):
    """A uniform bit through a binary erasure channel: a singular channel."""
    return Distribution(["00", "0e", "11", "1e"], [(1 - eps) / 2, eps / 2, (1 - eps) / 2, eps / 2])


def _mi(p_xy):
    p_x, p_y = p_xy.sum(axis=1, keepdims=True), p_xy.sum(axis=0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(np.nansum(p_xy * np.log2(p_xy / (p_x * p_y))))


def _random_joints(seed, shapes, count):
    rng = np.random.default_rng(seed)
    for _ in range(count):
        shape = shapes[rng.integers(len(shapes))]
        p = rng.dirichlet(np.full(np.prod(shape), 0.5)).reshape(shape)
        p[p < 0.02] = 0
        yield p / p.sum()


@pytest.mark.parametrize("order", [0.25, 0.5, 0.9, 1])
def test_self_information(order):
    """I_{F, alpha}(X -> X) = H_alpha[X] (Propositions 3 and 10)."""
    d = Distribution(["00", "11", "22"], [0.5, 0.3, 0.2])
    expected = entropy(d, [0]) if order == 1 else renyi_entropy(d.marginal([0]), order)
    assert functional_information(d, [0], [1], order=order) == pytest.approx(expected)


@pytest.mark.parametrize("order", [0.25, 0.5, 1])
def test_independent(order):
    d = Distribution(["00", "01", "10", "11"], [0.18, 0.12, 0.42, 0.28])
    assert functional_information(d, [0], [1], order=order) == pytest.approx(0, abs=1e-12)


@pytest.mark.parametrize("c", [0.05, 0.1, 0.3])
def test_dsbs(c):
    """A binary channel: I_F = 1 - 2c, strictly above the mutual information."""
    d = dsbs(c)
    assert functional_information(d, [0], [1]) == pytest.approx(1 - 2 * c)
    assert functional_information(d, [0], [1]) > mutual_information(d, [0], [1])


@pytest.mark.parametrize("order", [0.3, 0.7, 1])
def test_singular_channel_meets_sibson(order):
    """For a singular channel I_{F, alpha} equals Sibson's I_{1/alpha} (Propositions 3 and 11)."""
    d = bec(0.3)
    expected = mutual_information(d, [0], [1]) if order == 1 else sibson_mutual_information(d, [0], [1], 1 / order)
    assert functional_information(d, [0], [1], order=order) == pytest.approx(expected)


def test_asymmetric():
    """Both directions of a 2 x 3 pair, computed by hand from Lemma 3."""
    p_xy = np.array([[0.3, 0.2, 0.0], [0.0, 0.1, 0.4]])
    assert functional_information_pmf(p_xy) == pytest.approx(0.8)
    assert functional_information_pmf(p_xy.T) == pytest.approx(0.941064029)


def test_mutual_information_sandwich():
    """I <= I_F <= I + log2(I + 1) + 1 (Proposition 3)."""
    for p_xy in _random_joints(0, [(2, 3), (3, 3), (4, 2), (4, 5)], 60):
        mi, fi = _mi(p_xy), functional_information_pmf(p_xy)
        assert mi - 1e-12 <= fi <= mi + np.log2(mi + 1) + 1


@pytest.mark.parametrize("order", [0.2, 0.5, 0.8])
def test_sibson_lower_bound(order):
    """Sibson's I_{1/alpha} <= I_{F, alpha} <= I_F (Proposition 11, monotonicity in the order)."""
    for p_xy in _random_joints(1, [(2, 3), (3, 3), (4, 4)], 40):
        fi_alpha = functional_information_pmf(p_xy, order)
        assert sibson_mutual_information_pmf(p_xy, 1 / order) <= fi_alpha + 1e-12
        assert functional_information_pmf(p_xy) <= fi_alpha + 1e-12


def test_order_one_limit():
    p_xy = np.array([[0.3, 0.1, 0.1], [0.05, 0.25, 0.2]])
    assert functional_information_pmf(p_xy, 1 - 1e-7) == pytest.approx(functional_information_pmf(p_xy), abs=1e-5)


def test_data_processing():
    """I_F(X -> Y) >= I_F(X -> Z) for a Markov chain X -> Y -> Z (Proposition 4)."""
    rng = np.random.default_rng(2)
    for _ in range(40):
        p_xy = rng.dirichlet(np.ones(12)).reshape(3, 4)
        p_z_y = rng.dirichlet(np.ones(3), size=4)
        assert functional_information_pmf(p_xy @ p_z_y) <= functional_information_pmf(p_xy) + 1e-12


def test_conditional_divergence_form():
    """I_F(X -> Y) = sum_y p(y) D_CS(P_{X|Y=y} || P_X) (Definition 3)."""
    p_xy = np.array([[0.2, 0.05, 0.05], [0.1, 0.3, 0.0], [0.0, 0.1, 0.2]])
    p_x, p_y = p_xy.sum(axis=1), p_xy.sum(axis=0)
    expected = sum(p_y[y] * channel_simulation_divergence(p_xy[:, y] / p_y[y], p_x) for y in range(3))
    assert functional_information_pmf(p_xy) == pytest.approx(expected)


def test_brackets_the_optimal_rate():
    """I_F <= inf_{W indep X} H[Y|W] <= I_F + log2(e) (Theorem 1), with equality for binary outputs."""
    for p_xy in _random_joints(3, [(3, 2), (2, 3), (3, 3), (4, 3)], 40):
        fi = functional_information_pmf(p_xy)
        _, inf_h = excess_functional_information(p_xy)
        assert fi - 1e-7 <= inf_h <= fi + np.log2(np.e) + 1e-7
        if min(p_xy.shape) == 2:
            assert inf_h == pytest.approx(fi, abs=1e-7)


def test_ignores_impossible_inputs():
    p_xy = np.array([[0.3, 0.2], [0.0, 0.0], [0.1, 0.4]])
    assert functional_information_pmf(p_xy) == pytest.approx(functional_information_pmf(p_xy[[0, 2]]))


def test_units():
    d = dsbs(0.1)
    d.set_base("e")
    assert functional_information(d, [0], [1]) == pytest.approx(0.8 * np.log(2))


def test_multivariate_arguments():
    """X and Y may each be several variables: a giant bit plus independent noise is the identity on the bit."""
    d = dit.Distribution(["000", "011", "100", "111"], [1 / 4] * 4)
    assert functional_information(d, [0], [1, 2]) == pytest.approx(0)
    assert functional_information(d, [0, 1], [2]) == pytest.approx(1)


@pytest.mark.parametrize("order", [0, -0.5, 1.5, np.inf])
def test_order_validation(order):
    with pytest.raises(ValueError, match="order"):
        functional_information(dsbs(0.1), [0], [1], order=order)
    with pytest.raises(ValueError, match="order"):
        functional_information_pmf(np.eye(2) / 2, order)
