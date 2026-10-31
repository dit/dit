"""
Tests for dit.pid.measures.idelta.
"""

import pytest

from dit import Distribution
from dit.exceptions import ditException
from dit.pid.distributions import bivariates
from dit.pid.measures.idelta import PID_Delta
from dit.pid.measures.ideltalambda import PID_DeltaLambda


def test_pid_delta_synergy():
    """
    A purely synergistic (XOR) distribution: all information is synergistic.
    """
    d = bivariates["synergy"]
    pid = PID_Delta(d, ((0,), (1,)), (2,))
    assert pid[((0, 1),)] == pytest.approx(1.0, abs=1e-4)
    assert pid[((0,),)] == pytest.approx(0.0, abs=1e-4)
    assert pid[((1,),)] == pytest.approx(0.0, abs=1e-4)
    assert pid[((0,), (1,))] == pytest.approx(0.0, abs=1e-4)


def test_pid_delta_redundant():
    """
    A purely redundant distribution: redundancy is the full mutual information.
    """
    d = bivariates["redundant"]
    pid = PID_Delta(d, ((0,), (1,)), (2,))
    assert pid[((0,), (1,))] == pytest.approx(1.0, abs=1e-4)
    assert pid[((0,),)] == pytest.approx(0.0, abs=1e-4)
    assert pid[((1,),)] == pytest.approx(0.0, abs=1e-4)
    assert pid[((0, 1),)] == pytest.approx(0.0, abs=1e-4)


def test_pid_delta_and():
    """
    The AND gate has a known delta-PID decomposition: redundancy ~0.311,
    synergy 0.5, and no unique information.
    """
    d = bivariates["and"]
    pid = PID_Delta(d, ((0,), (1,)), (2,))
    assert pid[((0,), (1,))] == pytest.approx(0.311268, abs=1e-4)
    assert pid[((0, 1),)] == pytest.approx(0.5, abs=1e-4)
    assert pid[((0,),)] == pytest.approx(0.0, abs=1e-4)
    assert pid[((1,),)] == pytest.approx(0.0, abs=1e-4)


def test_pid_delta_unique_copy():
    """
    T copies X0 and X1 is independent noise: all information is unique to X0.
    """
    from dit import Distribution

    d = Distribution(["000", "010", "101", "111"], [1 / 4] * 4)
    pid = PID_Delta(d, ((0,), (1,)), (2,))
    assert pid[((0,),)] == pytest.approx(1.0, abs=1e-4)
    assert pid[((1,),)] == pytest.approx(0.0, abs=1e-4)
    assert pid[((0,), (1,))] == pytest.approx(0.0, abs=1e-4)


AND3 = Distribution(["0000", "0010", "0100", "0110", "1000", "1010", "1100", "1111"], [1 / 8] * 8)


@pytest.mark.parametrize("pid", [PID_Delta, PID_DeltaLambda])
def test_three_sources(pid):
    """
    These redundancies are only defined for two sources. With three, the redundancies of
    pairs are computed and the rest of the lattice is filled in where it can be, rather
    than unpacking three sources into two.
    """
    with pytest.raises(ditException, match="two sources"):
        pid._measure(AND3, ((0,), (1,), (2,)), (3,))
    decomposition = pid(AND3)
    assert decomposition.get_red(((0,), (1,))) == pytest.approx(0.1379, abs=1e-3)
    assert decomposition.consistent
