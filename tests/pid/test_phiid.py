"""
Tests for dit.pid.phiid.
"""

from collections import defaultdict
from itertools import product

import numpy as np
import pytest
from hypothesis import given, settings

from dit import Distribution as D
from dit.multivariate import coinformation
from dit.pid import PID_GK, PhiID
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


def _common_parts_system():
    """
    Two sources sharing a bit c and two targets sharing a bit e, with c and e dependent
    and everything else random, so the Gács-Körner meets of both sides are c and e.
    """
    rng = np.random.default_rng(0)
    bits = list(product([0, 1], repeat=6))
    outcomes = [(2 * c + a, 2 * c + b, 2 * e + f, 2 * e + g) for c, e, a, b, f, g in bits]
    w = rng.random(len(bits)) * [3 if c == e else 1 for c, e, *_ in bits]
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
    for measure in ("mmi", "pm", "gk"):
        phiid = PhiID(d, measure=measure)
        assert phiid[RR] == pytest.approx(1.0)
        assert sum(abs(phiid[n]) for n in phiid.nodes if n != RR) == pytest.approx(0.0, abs=1e-9)


def test_gk_extends_pid_gk():
    """
    With a single group on one side, 'gk' is the I_wedge redundancy of the groups on
    the other side, forward and backward.
    """
    d = _common_parts_system()
    phiid = PhiID(d, measure="gk")
    for target in ([2], [3], [2, 3]):
        pid = PID_GK(d, [[0], [1]], target)
        for alpha in {alpha for alpha, _ in phiid.nodes}:
            assert phiid.get_red((alpha, (tuple(target),))) == pytest.approx(pid.get_red(alpha))
    for source in ([0], [1], [0, 1]):
        pid = PID_GK(d, [[2], [3]], source)
        for beta in {beta for _, beta in phiid.nodes}:
            assert phiid.get_red(((tuple(source),), beta)) == pytest.approx(pid.get_red(beta))
    shared = defaultdict(float)
    for (x, _, y, _), p in zip(d.outcomes, d.pmf, strict=True):
        shared[(x // 2, y // 2)] += p
    mi = coinformation(D(list(shared), list(shared.values())), [[0], [1]])
    assert mi > 0.05
    assert phiid.get_red(RR) == pytest.approx(mi)


def test_gk_independent_copies():
    """
    Two independent bits, each copied forward, are stored separately and carry nothing else.
    """
    phiid = PhiID(D([(a, b, a, b) for a, b in product([0, 1], repeat=2)], [1 / 4] * 4), measure="gk")
    stored = {(((0,),), ((2,),)), (((1,),), ((3,),))}
    for node in phiid.nodes:
        assert phiid[node] == pytest.approx(1.0 if node in stored else 0.0, abs=1e-9)


def test_gk_full_support():
    """
    Variables with full joint support share no common part, so no node with two groups
    on a side has double redundancy.
    """
    phiid = PhiID(_random_system(), measure="gk")
    for alpha, beta in phiid.nodes:
        if len(alpha) > 1 or len(beta) > 1:
            assert phiid.get_red((alpha, beta)) == pytest.approx(0.0, abs=1e-12)


def test_gk_ignores_impossible_outcomes():
    """
    Outcomes of probability zero do not join the components of a meet.
    """
    d = _common_parts_system()
    dense = d.copy()
    dense.make_dense()
    sparse, full = PhiID(d, measure="gk"), PhiID(dense, measure="gk")
    assert all(full[n] == pytest.approx(sparse[n]) for n in sparse.nodes)


def _assert_monotone(phiid):
    for node in phiid.nodes:
        for below in phiid._descendants(node):
            assert phiid.get_red(below) <= phiid.get_red(node) + 1e-9


def test_gk_monotone():
    """
    The 'gk' double redundancy never decreases up the product lattice, by data processing.
    """
    _assert_monotone(PhiID(_common_parts_system(), measure="gk"))


@settings(max_examples=10, deadline=None)
@given(dist=distributions(alphabets=(2, 2, 2, 2)))
def test_gk_monotone_random(dist):
    phiid = PhiID(dist, measure="gk")
    _assert_monotone(phiid)
    assert sum(phiid[n] for n in phiid.nodes) == pytest.approx(coinformation(dist, [[0, 1], [2, 3]]), abs=1e-6)


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
        PhiID(d, measure="gk", pointwise=True)
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
