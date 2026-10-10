"""
Tests for dit.rate_distortion.information_bottleneck
"""

import pytest

from dit import Distribution
from dit.divergences.pmf import relative_entropy
from dit.exceptions import ditException
from dit.rate_distortion import DeterministicInformationBottleneck, GeneralizedInformationBottleneck
from dit.rate_distortion.information_bottleneck import InformationBottleneck, InformationBottleneckDivergence

dist = Distribution(["00", "02", "12", "21", "22"], [1 / 5] * 5)
dist2 = Distribution(["000", "001", "020", "021", "120", "121", "210", "211", "220", "221"], [1 / 10] * 10)


def test_ib_1():
    """
    Test simple IB.
    """
    ib = InformationBottleneck.functional()
    c, r = ib(dist, beta=0.0)
    assert c == pytest.approx(0.0, abs=1e-4)
    assert r == pytest.approx(0.0, abs=1e-4)


def test_ib_2():
    """
    Test simple IB failure.
    """
    with pytest.raises(ditException):
        InformationBottleneck(dist, rvs=[[0]], beta=0.0)


def test_ib_3():
    """
    Test simple IB failure.
    """
    with pytest.raises(ditException):
        InformationBottleneck(dist, beta=0.0, alpha=99)


def test_ib_4():
    """
    Test simple IB failure.
    """
    with pytest.raises(ditException):
        InformationBottleneck(dist, rvs=[[0]], beta=0.0)


def test_ib_5():
    """
    Test simple IB failure.
    """
    with pytest.raises(ditException):
        InformationBottleneck(dist, beta=-10.0)


def test_ib_first_class_variants():
    """
    Test first-class generalized and deterministic bottleneck classes.
    """
    gib = GeneralizedInformationBottleneck(dist, beta=0.0, alpha=0.5)
    dib = DeterministicInformationBottleneck(dist, beta=0.0)

    assert gib._alpha == 0.5
    assert dib._alpha == 0.0

    with pytest.raises(ditException):
        DeterministicInformationBottleneck(dist, beta=0.0, alpha=0.5)


def test_ibd_1():
    """
    Test with custom distortion.
    """
    ibd = InformationBottleneckDivergence(dist, beta=0.0, divergence=relative_entropy)
    ibd.optimize()
    pmf = ibd.construct_joint(ibd._optima)
    assert float(ibd.complexity(pmf)) == pytest.approx(0.0, abs=1e-4)
    assert float(ibd.relevance(pmf)) == pytest.approx(0.0, abs=1e-4)


def test_ibd_2():
    """
    Test with custom distortion.
    """
    ibd = InformationBottleneckDivergence(dist2, rvs=[[0], [1]], crvs=[2], beta=0.0, divergence=relative_entropy)
    ibd.optimize()
    pmf = ibd.construct_joint(ibd._optima)
    assert float(ibd.complexity(pmf)) == pytest.approx(0.0, abs=1e-4)
    assert float(ibd.relevance(pmf)) == pytest.approx(0.0, abs=1e-4)


def test_ib_high_beta_seeded():
    """
    At beta = 10 random starts often stay near an uninformative encoder; the
    seeded search reaches the best of 60 self-consistent iterations and all
    deterministic encoders, 1.8306 bits.
    """
    pxy = [[0.0031, 0.0, 0.2947], [0.1006, 0.0353, 0.0065], [0.3585, 0.0127, 0.0071], [0.0047, 0.1389, 0.0378]]
    outcomes = [f"{x}{y}" for x in range(4) for y in range(3)]
    pmf = [p for row in pxy for p in row]
    d = Distribution(outcomes, [p / sum(pmf) for p in pmf])
    ib = InformationBottleneck(d, beta=10.0)
    ib.optimize()
    assert ib.objective(ib._optima) == pytest.approx(1.8306, abs=1e-3)
