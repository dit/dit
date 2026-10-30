"""
Tests for dit.rate_distortion.information_bottleneck.InformationBottleneckRate.
"""

import numpy as np
import pytest
from scipy.optimize import brentq

from dit import Distribution
from dit.example_dists import erasure_channel
from dit.exceptions import ditException
from dit.rate_distortion import InformationBottleneckRate, information_bottleneck_rate


def _h(p):
    """
    The binary entropy.
    """
    return 0.0 if p <= 0 or p >= 1 else float(-p * np.log2(p) - (1 - p) * np.log2(1 - p))


def _h_inv(v):
    """
    The inverse of the binary entropy on [0, 1/2].
    """
    return 0.0 if v <= 0 else brentq(lambda p: _h(p) - v, 1e-15, 0.5)


def _dsbs(p):
    """
    A doubly symmetric binary source with crossover p.
    """
    return Distribution(["00", "01", "10", "11"], [(1 - p) / 2, p / 2, p / 2, (1 - p) / 2])


@pytest.mark.parametrize(("p", "rate"), [(0.1, 0.2), (0.1, 0.5), (0.25, 0.8)])
def test_mr_gerbers_lemma(p, rate):
    """
    For a doubly symmetric binary source with crossover p, Mrs. Gerber's lemma
    gives F(R) = 1 - h(p * h^{-1}(1 - R)), where a * b = a (1 - b) + b (1 - a).
    """
    e = _h_inv(1 - rate)
    expected = 1 - _h(p * (1 - e) + e * (1 - p))
    assert information_bottleneck_rate(_dsbs(p), rate, rng=0) == pytest.approx(expected, abs=1e-4)


@pytest.mark.parametrize("rate", [0.3, 0.7, 1.5])
def test_erasure(rate):
    """
    Through an erasure channel with erasure probability e, every T of X has
    I[Y : T] = (1 - e) I[X : T], so F(R) = (1 - e) min(R, H[X]).
    """
    d = erasure_channel(0.3)
    assert information_bottleneck_rate(d, rate, rng=0) == pytest.approx(0.7 * min(rate, 1.0), abs=1e-4)


def test_zero_rate():
    """
    At zero rate T is independent of X.
    """
    assert information_bottleneck_rate(_dsbs(0.1), 0.0, rng=0) == pytest.approx(0.0, abs=1e-6)


def test_full_rate():
    """
    At R = H[X] nothing needs to be discarded, so F(R) = I[X : Y].
    """
    assert information_bottleneck_rate(_dsbs(0.1), 1.0, rng=0) == pytest.approx(1 - _h(0.1), abs=1e-6)


def test_conditional():
    """
    Conditioning on a variable independent of X and Y changes nothing.
    """
    p = 0.1
    d = Distribution(
        [x + y + z for x in "01" for y in "01" for z in "01"],
        [((1 - p) if x == y else p) / 4 for x in "01" for y in "01" for _ in "01"],
    )
    e = _h_inv(0.5)
    expected = 1 - _h(p * (1 - e) + e * (1 - p))
    assert information_bottleneck_rate(d, 0.5, rvs=[[0], [1]], crvs=[2], rng=0) == pytest.approx(expected, abs=1e-4)


def test_bound():
    """
    The size of T defaults to |X||Z| + 1, one more than the Lagrangian needs.
    """
    assert InformationBottleneckRate(_dsbs(0.1), 0.5)._aux_vars[0].bound == 3


def test_bad_rate():
    """
    The rate must be non-negative.
    """
    with pytest.raises(ditException):
        InformationBottleneckRate(_dsbs(0.1), -0.1)


def test_bad_rvs():
    """
    The bottleneck needs exactly two variables.
    """
    with pytest.raises(ditException):
        InformationBottleneckRate(_dsbs(0.1), 0.5, rvs=[[0]])
