"""
Tests for dit.pid.measures.idelta.
"""

import pytest

from dit.pid.distributions import bivariates
from dit.pid.measures.idelta import PID_Delta


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
