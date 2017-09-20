"""
Tests for dit.pid.ipm.
"""

import pytest

from dit.pid.ipm import i_pm, PID_PM
from dit.pid.distributions import bivariates

def test_ipm1():
    """
    Test imin on redundant distribution.
    """
    d = bivariates['redundant']
    red = i_pm(d, ((0,), (1,)), (2,))
    assert red == pytest.approx(1)

def test_ipm2():
    """
    Test imin on synergistic distribution.
    """
    d = bivariates['synergy']
    red = i_pm(d, ((0,), (1,)), (2,))
    assert red == pytest.approx(0)

def test_ipm3():
    """
    Test ipm on unique distribution.
    """
    d = bivariates['cat']
    red = i_pm(d, ((0,), (1,)), (2,))
    assert red == pytest.approx(1)

def test_pid_pm1():
    """
    Test ipm on a generic distribution.
    """
    d = bivariates['pnt. unq']
    pid = PID_PM(d, ((0,), (1,)), (2,))
    assert pid[((0,), (1,))] == pytest.approx(0.0)
    assert pid[((0,),)] == pytest.approx(0.5)
    assert pid[((1,),)] == pytest.approx(0.5)
    assert pid[((0, 1),)] == pytest.approx(0.0)

def test_pid_pm2():
    """
    Test imin on another generic distribution.
    """
    d = bivariates['unique 1']
    pid = PID_PM(d, [[0], [1]], [2])
    assert pid[((0,), (1,))] == pytest.approx(1.0)
    assert pid[((0,),)] == pytest.approx(-1.0)
    assert pid[((1,),)] == pytest.approx(0.0)
    assert pid[((0, 1),)] == pytest.approx(1.0)
