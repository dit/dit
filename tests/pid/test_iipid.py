"""
Tests for dit.pid.measures.iipid (I-PID).
"""

import pytest

from dit.pid.distributions import bivariates
from dit.pid.measures.iipid import PID_IPID


@pytest.mark.flaky(reruns=5)
def test_pid_ipid_redundant():
    """
    Test I-PID on the redundant distribution (M=X=Y).
    All information is redundant.
    """
    d = bivariates["redundant"]
    pid = PID_IPID(d, ((0,), (1,)), (2,))
    assert pid[((0,), (1,))] == pytest.approx(1.0, abs=1e-3)
    assert pid[((0,),)] == pytest.approx(0.0, abs=1e-3)
    assert pid[((1,),)] == pytest.approx(0.0, abs=1e-3)
    assert pid[((0, 1),)] == pytest.approx(0.0, abs=1e-3)


@pytest.mark.flaky(reruns=5)
def test_pid_ipid_synergy():
    """
    Test I-PID on the XOR/synergy distribution.
    All information is synergistic.
    """
    d = bivariates["synergy"]
    pid = PID_IPID(d, ((0,), (1,)), (2,))
    assert pid[((0,), (1,))] == pytest.approx(0.0, abs=1e-3)
    assert pid[((0,),)] == pytest.approx(0.0, abs=1e-3)
    assert pid[((1,),)] == pytest.approx(0.0, abs=1e-3)
    assert pid[((0, 1),)] == pytest.approx(1.0, abs=1e-3)


@pytest.mark.flaky(reruns=5)
def test_pid_ipid_cat():
    """
    Test I-PID on the cat distribution (M = concatenation of X and Y).
    Each source has 1 bit of unique information.
    """
    d = bivariates["cat"]
    pid = PID_IPID(d, ((0,), (1,)), (2,))
    assert pid[((0,), (1,))] == pytest.approx(0.0, abs=1e-3)
    assert pid[((0,),)] == pytest.approx(1.0, abs=1e-3)
    assert pid[((1,),)] == pytest.approx(1.0, abs=1e-3)
    assert pid[((0, 1),)] == pytest.approx(0.0, abs=1e-3)


@pytest.mark.flaky(reruns=5)
def test_pid_ipid_unique1():
    """
    Test I-PID on the unique 1 distribution (M = Y, X independent of M).
    All information is unique to Y.
    """
    d = bivariates["unique 1"]
    pid = PID_IPID(d, ((0,), (1,)), (2,))
    assert pid[((0,), (1,))] == pytest.approx(0.0, abs=1e-3)
    assert pid[((0,),)] == pytest.approx(0.0, abs=1e-3)
    assert pid[((1,),)] == pytest.approx(1.0, abs=1e-3)
    assert pid[((0, 1),)] == pytest.approx(0.0, abs=1e-3)


@pytest.mark.flaky(reruns=5)
def test_pid_ipid_sum():
    """
    Test I-PID on the sum distribution (M = X + Y).
    By symmetry, delta_I is zero in both directions,
    so RI = I(M;X) = I(M;Y) ~= 0.5.
    """
    d = bivariates["sum"]
    pid = PID_IPID(d, ((0,), (1,)), (2,))
    assert pid[((0,), (1,))] == pytest.approx(0.5, abs=1e-2)
    assert pid[((0,),)] == pytest.approx(0.0, abs=1e-2)
    assert pid[((1,),)] == pytest.approx(0.0, abs=1e-2)
    assert pid[((0, 1),)] == pytest.approx(1.0, abs=1e-2)


@pytest.mark.flaky(reruns=5)
def test_pid_ipid_and():
    """
    Test I-PID on the AND distribution (M = X AND Y).
    By symmetry, delta_I is zero in both directions.
    """
    d = bivariates["and"]
    pid = PID_IPID(d, ((0,), (1,)), (2,))
    assert pid[((0,),)] == pytest.approx(pid[((1,),)], abs=1e-3)
    assert pid[((0,), (1,))] >= -1e-3
    assert pid[((0, 1),)] >= -1e-3
