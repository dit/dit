"""
Tests for dit.pid.imin.
"""

import pytest

from dit.pid.imin import i_min, PID_WB
from dit.pid.distributions import bivariates, trivariates

def test_imin1():
    """
    Test imin on redundant distribution.
    """
    d = bivariates['redundant']
    red = i_min(d, ((0,), (1,)), (2,))
    assert red == pytest.approx(1)

def test_imin2():
    """
    Test imin on synergistic distribution.
    """
    d = bivariates['synergy']
    red = i_min(d, ((0,), (1,)), (2,))
    assert red == pytest.approx(0)

def test_imin3():
    """
    Test imin on unique distribution.
    """
    d = bivariates['cat']
    red = i_min(d, ((0,), (1,)), (2,))
    assert red == pytest.approx(1)

def test_pid_wb1():
    """
    Test imin on a generic distribution.
    """
    d = bivariates['prob 1']
    pid = PID_WB(d, ((0,), (1,)), (2,))
    assert pid[((0,), (1,))] == pytest.approx(0.019973094021974794)
    assert pid[((0,),)] == pytest.approx(0.15097750043269376)
    assert pid[((1,),)] == pytest.approx(0.0)
    assert pid[((0, 1),)] == pytest.approx(0.0)

def test_pid_wb2():
    """
    Test imin on another generic distribution.
    """
    d = trivariates['sum']
    pid = PID_WB(d, [[0], [1], [2]], [3])
    for atom in pid._lattice:
        if atom == ((0,), (1,), (2,)):
            assert pid[atom] == pytest.approx(0.31127812445913294)
        elif atom == ((0, 1), (0, 2), (1, 2)):
            assert pid[atom] == pytest.approx(0.5)
        elif atom == ((0, 1, 2),):
            assert pid[atom] == pytest.approx(1.0)
        else:
            assert pid[atom] == pytest.approx(0.0)

def test_pid_wb3():
    """
    Test imin on a generic distribution.
    """
    d = bivariates['jeff']
    pid = PID_WB(d)
    assert pid.complete
    assert pid.nonnegative
    assert pid.consistent
