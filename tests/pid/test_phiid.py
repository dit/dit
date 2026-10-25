"""
Tests for dit.pid.phiid.
"""

from itertools import product

import numpy as np
import pytest
from hypothesis import given, settings

from dit import Distribution as D
from dit.multivariate import coinformation
from dit.pid import PhiID
from dit.utils.testing import distributions

RR = (((0,), (1,)), ((2,), (3,)))
SYN_TO_X = (((0, 1),), ((2,),))
SYN_SYN = (((0, 1),), ((2, 3),))


def _random_system():
    """
    The same random distribution over (X1, X2, Y1, Y2) used to generate the Syntropy references.
    """
    rng = np.random.default_rng(5)
    w = rng.random(16)
    outcomes = ["".join(map(str, s)) for s in product([0, 1], repeat=4)]
    return D(outcomes, w / w.sum())


@pytest.mark.parametrize(
    ("measure", "expected"),
    [
        ("mmi", (0.00031239131495822657, 0.004524459226504981, 0.06991644800962239)),
        ("pm", (0.11734464199972239, 0.005078112284298025, 0.08621437619613972)),
        ("sx", (0.00945156121985762, 0.046951555169688616, -0.012431365619148937)),
    ],
)
def test_matches_syntropy(measure, expected):
    """
    Reference atoms from Syntropy's ``integrated_information_decomposition((0, 1), (2, 3), ...)``.
    """
    phiid = PhiID(_random_system(), measure=measure)
    for node, value in zip((RR, SYN_TO_X, SYN_SYN), expected, strict=True):
        assert phiid[node] == pytest.approx(value, abs=1e-9)


def test_sixteen_atoms():
    phiid = PhiID(_random_system())
    assert len(phiid.nodes) == 16
    assert "{0}{1} -> {2}{3}" in str(phiid)


def test_redundant_copy():
    """
    Copying one shared bit forward is pure persistent redundancy.
    """
    d = D(["0000", "1111"], [1 / 2] * 2)
    for measure in ("mmi", "pm"):
        phiid = PhiID(d, measure=measure)
        assert phiid[RR] == pytest.approx(1.0)
        assert sum(abs(phiid[n]) for n in phiid.nodes if n != RR) == pytest.approx(0.0, abs=1e-9)


def test_pointwise_averages_to_atoms():
    d = _random_system()
    phiid = PhiID(d, measure="sx", pointwise=True)
    for node in phiid.nodes:
        avg = sum(d[o] * v for o, v in phiid.get_pw_pi(node).items())
        assert avg == pytest.approx(phiid[node])
        red = sum(d[o] * v for o, v in phiid.get_pw_red(node).items())
        assert red == pytest.approx(phiid.get_red(node))


def test_explicit_sources_and_targets():
    d = _random_system()
    default = PhiID(d)
    explicit = PhiID(d, sources=[[0], [1]], targets=[[2], [3]])
    assert all(default[n] == pytest.approx(explicit[n]) for n in default.nodes)


def test_errors():
    d = _random_system()
    with pytest.raises(ValueError):
        PhiID(d, measure="nope")
    with pytest.raises(ValueError):
        PhiID(d, measure="mmi", pointwise=True)
    with pytest.raises(ValueError):
        PhiID(d, sources=[[0], [1]])
    with pytest.raises(ValueError):
        PhiID(D(["000", "111"], [1 / 2] * 2))
    with pytest.raises(ValueError):
        PhiID(d).get_red((((0,),), ((0,),)))


@settings(max_examples=10, deadline=None)
@given(dist=distributions(alphabets=(2, 2, 2, 2)))
def test_atoms_sum_to_mutual_information(dist):
    mi = coinformation(dist, [[0, 1], [2, 3]])
    for measure in ("mmi", "pm", "sx"):
        phiid = PhiID(dist, measure=measure)
        assert sum(phiid[n] for n in phiid.nodes) == pytest.approx(mi, abs=1e-6)
