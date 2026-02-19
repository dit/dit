"""
Tests for dit.pid.iig.
"""

import pytest

from dit.pid.distributions import bivariates
from dit.pid.measures.iig import PID_IG


def test_pid_ig1():
    """
    Test iproj on a generic distribution.
    """
    d = bivariates['and']
    pid = PID_IG(d, ((0,), (1,)), (2,))
    assert pid[((0,), (1,))] == pytest.approx(0.08283, abs=1e-4)
    assert pid[((0,),)] == pytest.approx(0.22845, abs=1e-4)
    assert pid[((1,),)] == pytest.approx(0.22845, abs=1e-4)
    assert pid[((0, 1),)] == pytest.approx(0.27155, abs=1e-4)


def test_pid_proj2():
    """
    Test iproj on another generic distribution.
    """
    d = bivariates['reduced or']
    pid = PID_IG(d, [[0], [1]], [2])
    assert pid[((0,), (1,))] == pytest.approx(-0.03122, abs=1e-4)
    assert pid[((0,),)] == pytest.approx(0.34250, abs=1e-4)
    assert pid[((1,),)] == pytest.approx(0.34250, abs=1e-4)
    assert pid[((0, 1),)] == pytest.approx(0.34622, abs=1e-4)
