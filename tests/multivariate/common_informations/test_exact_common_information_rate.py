"""
Tests for dit.multivariate.exact_common_information_rate.
"""

from importlib import import_module

import numpy as np
import pytest
from hypothesis import given, settings

from dit import Distribution
from dit.example_dists import erasure_channel
from dit.exceptions import ditException
from dit.multivariate import (
    exact_common_information,
    exact_common_information_rate,
    exact_common_information_rate_bounds,
    wyner_common_information,
)
from dit.multivariate.common_informations.exact_common_information_rate import (
    ExactCommonInformationRate,
    _cost_matrix,
    _outer_sum_rate,
    _pair_distribution,
)
from dit.utils.testing import distributions


def h(p):
    """The binary entropy."""
    return -p * np.log2(p) - (1 - p) * np.log2(1 - p)


def dsbs_matrix(p):
    """The doubly symmetric binary source with crossover `p`, as a matrix."""
    return np.array([[1 - p, p], [p, 1 - p]]) / 2


def dsbs(p):
    """The doubly symmetric binary source with crossover `p`."""
    return Distribution(["00", "01", "10", "11"], dsbs_matrix(p).ravel())


def dsbs_rate(p):
    """
    The exact common information rate of the doubly symmetric binary source, from
    Yu and Tan's Theorem 3 without common randomness.
    """
    a = (1 - np.sqrt(1 - 2 * p)) / 2
    a0, b0 = (1 - p) / 2, p / 2
    return np.log2(1 / a0) + 2 * a * np.log2(a0 / b0) - 2 * h(a)


# The module, which its function of the same name shadows as an attribute.
module = import_module("dit.multivariate.common_informations.exact_common_information_rate")

z_source = Distribution(["00", "01", "10"], [1 / 3] * 3)


@pytest.mark.flaky(reruns=3)
@pytest.mark.parametrize("p", [0.1, 0.3])
def test_dsbs(p):
    """
    The bounds meet at Yu and Tan's closed form, strictly between the Wyner and
    one-shot exact common informations.
    """
    lower, upper = exact_common_information_rate_bounds(dsbs(p))
    assert lower == pytest.approx(dsbs_rate(p), abs=1e-4)
    assert upper == pytest.approx(dsbs_rate(p), abs=1e-4)
    assert wyner_common_information(dsbs(p)) < lower - 0.01
    assert upper < exact_common_information(dsbs(p)) - 0.01


@pytest.mark.flaky(reruns=3)
@pytest.mark.parametrize("p", [0.3, 0.7])
def test_erasure(p):
    """
    Kumar, Li, and El Gamal: the exact common information rate of the erasure
    channel is its Wyner common information.
    """
    assert exact_common_information_rate(erasure_channel(p)) == pytest.approx(1 if p < 1 / 2 else h(p), abs=1e-4)


@pytest.mark.flaky(reruns=3)
def test_z_source():
    """
    The one-shot exact common information of uniform over {00, 01, 10} is
    subadditive, but its rate is the Wyner common information, 2/3.
    """
    rate = exact_common_information_rate(z_source)
    assert rate == pytest.approx(2 / 3, abs=1e-6)
    assert rate >= 2 / 3 - 1e-9


@pytest.mark.flaky(reruns=3)
def test_block_length():
    """
    Two i.i.d. copies never do worse than one, and the result remains an upper
    bound.
    """
    rate = exact_common_information_rate(z_source, block_length=2, bound=5)
    assert 2 / 3 - 1e-9 <= rate <= 2 / 3 + 1e-6


@pytest.mark.slow
def test_block_length_dsbs():
    """
    For the doubly symmetric binary source nothing beats the single-letter bound.
    """
    assert exact_common_information_rate(dsbs(0.2), block_length=2) == pytest.approx(dsbs_rate(0.2), abs=1e-4)


def test_trivial():
    """
    Independent variables need no common randomness, and identical ones need all
    of their entropy.
    """
    independent = Distribution(["00", "01", "10", "11"], [0.12, 0.28, 0.18, 0.42])
    assert exact_common_information_rate_bounds(independent) == pytest.approx((0, 0), abs=1e-9)
    identical = Distribution(["00", "11"], [0.3, 0.7])
    assert exact_common_information_rate_bounds(identical) == pytest.approx((h(0.3), h(0.3)))


@pytest.mark.flaky(reruns=3)
@settings(max_examples=2, deadline=None)
@given(dist=distributions(alphabets=(2, 3), zeros=False))
def test_ordering(dist):
    """
    The bounds lie between the Wyner and one-shot exact common informations.
    """
    lower, upper = exact_common_information_rate_bounds(dist)
    assert wyner_common_information(dist) - 1e-3 <= lower <= upper
    assert upper <= exact_common_information(dist) + 1e-3


@pytest.mark.flaky(reruns=3)
def test_rvs():
    """
    Groups of variables act as single variables, and an independent variable in
    a group changes nothing.
    """
    outcomes = [f"{x}{y}{n}" for x in "01" for y in "01" for n in "01"]
    pmf = [dsbs_matrix(0.2)[int(o[0]), int(o[1])] / 2 for o in outcomes]
    d = Distribution(outcomes, pmf)
    assert exact_common_information_rate(d, rvs=[[0, 2], [1]]) == pytest.approx(dsbs_rate(0.2), abs=1e-4)


def test_base():
    """
    Values are reported in the base of the distribution.
    """
    identical = Distribution(["00", "11"], [0.3, 0.7]).copy(base=np.e)
    assert exact_common_information_rate(identical) == pytest.approx(h(0.3) * np.log(2))
    assert exact_common_information_rate_bounds(identical) == pytest.approx((h(0.3) * np.log(2),) * 2)


def test_errors():
    """
    The bounds are defined for two groups of variables, unconditionally.
    """
    d = Distribution(["000", "111"], [0.5, 0.5])
    with pytest.raises(ditException):
        exact_common_information_rate(d)
    with pytest.raises(ditException):
        exact_common_information_rate(d, rvs=[[0], [1], [2]])
    with pytest.raises(ditException):
        exact_common_information_rate(dsbs(0.2), block_length=0)
    with pytest.raises(ditException):
        ExactCommonInformationRate(d, [[0], [1]], [2])


pairs = {
    "dsbs": dsbs_matrix(0.2),
    "erasure": np.array([[0.35, 0.15, 0.0], [0.0, 0.15, 0.35]]),
    "3x3": np.random.default_rng(0).dirichlet(np.ones(9)).reshape(3, 3),
}


@pytest.mark.parametrize("outer", [False, True])
@pytest.mark.parametrize("name", list(pairs))
def test_gradient(name, outer):
    """
    The analytic gradient of the sum rate matches finite differences.
    """
    opt = ExactCommonInformationRate(_pair_distribution(pairs[name]), [[0], [1]], outer=outer)
    objective = opt._objective().__get__(opt)
    np.random.seed(1)
    for _ in range(3):
        x = opt.construct_random_initial()
        steps = 1e-6 * np.eye(x.size)
        numeric = np.array([(objective(x + step) - objective(x - step)) / 2e-6 for step in steps])
        assert opt._jacobian(x) == pytest.approx(numeric, abs=1e-5 * max(1.0, np.abs(numeric).max()))


def test_diagonal_shortcut(monkeypatch):
    """
    Skipping the lower bound's coupling program when pairing each state with
    itself is optimal changes nothing.
    """
    rng = np.random.default_rng(2)
    for _ in range(50):
        k = int(rng.integers(1, 5))
        weights = rng.dirichlet(np.ones(k))
        px_w, py_w = rng.dirichlet(np.ones(2) / 2, k), rng.dirichlet(np.ones(3) / 2, k)
        joint = np.einsum("w,wx,wy->xyw", weights, px_w, py_w)
        pxy = joint.sum(axis=2)
        args = joint, _cost_matrix(pxy), (pxy <= 0).astype(float)
        value, _ = _outer_sum_rate(*args)
        with monkeypatch.context() as patch:
            patch.setattr(module, "_diagonal_is_optimal", lambda *_: False)
            assert _outer_sum_rate(*args)[0] == pytest.approx(value)
