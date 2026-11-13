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


def _parity_system(targets):
    """
    Two uniform bits (a, b) and a fresh uniform bit c, with targets `targets(a, b, c)`.
    """
    outcomes = [(a, b, *targets(a, b, c)) for a, b, c in product([0, 1], repeat=3)]
    return D(outcomes, [1 / 8] * 8)


def test_phi_of_redundant_copy():
    """
    Phi_WMS counts the copied bit's persistent redundancy against the system; Phi_R does not.
    """
    d = D(["0000", "1111"], [1 / 2] * 2)
    for measure in ("mmi", "pm"):
        phiid = PhiID(d, measure=measure)
        assert phiid.phi_wms == pytest.approx(-1.0)
        assert phiid.phi_r == pytest.approx(0.0, abs=1e-9)


def test_causal_decoupling():
    """
    A future whose parity is the past's parity, and is otherwise fresh, is purely decoupled.
    """
    phiid = PhiID(_parity_system(lambda a, b, c: (c, c ^ a ^ b)))
    assert phiid.causal_decoupling() == pytest.approx(1.0)
    assert phiid.downward_causation() == pytest.approx(0.0, abs=1e-9)
    assert phiid.emergence_capacity() == pytest.approx(1.0)
    assert phiid.phi_wms == pytest.approx(1.0)
    assert phiid.phi_r == pytest.approx(1.0)


def test_downward_causation():
    """
    A future bit that is the past's parity is pure downward causation.
    """
    phiid = PhiID(_parity_system(lambda a, b, c: (a ^ b, c)))
    assert phiid.downward_causation() == pytest.approx(1.0)
    assert phiid.causal_decoupling() == pytest.approx(0.0, abs=1e-9)
    assert phiid.emergence_capacity() == pytest.approx(1.0)


def test_emergence_order():
    """
    Synergy between two of three sources is emergent at order 1 but not order 2.
    """
    outcomes = [(a, b, e, c, f, c ^ f ^ a ^ b) for a, b, e, c, f in product([0, 1], repeat=5)]
    phiid = PhiID(D(outcomes, [1 / 32] * 32))
    assert phiid.causal_decoupling(k=1) == pytest.approx(1.0)
    assert phiid.causal_decoupling(k=2) == pytest.approx(0.0, abs=1e-9)
    assert phiid.emergence_capacity(k=3) == 0


@settings(max_examples=10, deadline=None)
@given(dist=distributions(alphabets=(2, 2, 2, 2)))
def test_phi_r_is_transfer_and_synergy(dist):
    """
    Phi_R is the atoms outside both parts' own mutual informations, and Phi_WMS
    is that less the persistent redundancy.
    """
    for measure in ("mmi", "pm", "sx"):
        phiid = PhiID(dist, measure=measure)
        own = [n for n in phiid.nodes if ((0,) in n[0] and (2,) in n[1]) or ((1,) in n[0] and (3,) in n[1])]
        rest = sum(phiid[n] for n in phiid.nodes if n not in own)
        assert phiid.phi_r == pytest.approx(rest, abs=1e-6)
        assert phiid.phi_wms == pytest.approx(rest - phiid[RR], abs=1e-6)
        assert phiid.emergence_capacity() == pytest.approx(
            phiid.causal_decoupling() + phiid.downward_causation(), abs=1e-9
        )


def test_phi_units():
    d = _parity_system(lambda a, b, c: (c, c ^ a ^ b))
    phiid = PhiID(d.copy(base="e"))
    assert phiid.phi_wms == pytest.approx(np.log(2))
    assert phiid.phi_r == pytest.approx(np.log(2))
    assert phiid.causal_decoupling() == pytest.approx(np.log(2))


def test_phi_errors():
    d = _random_system()
    with pytest.raises(ValueError):
        _ = PhiID(d, sources=[[0], [1]], targets=[[2, 3]]).phi_wms
    with pytest.raises(ValueError):
        _ = PhiID(d, sources=[[0, 1]], targets=[[2, 3]]).phi_r
