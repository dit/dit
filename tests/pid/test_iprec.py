"""
Tests for dit.pid.measures.iprec.
"""

import pytest

from dit import Distribution
from dit.pid.distributions import bivariates
from dit.pid.measures.iprec import PID_Prec


def test_pid_prec1():
    """
    Test iprec on a generic distribution.
    """
    d = bivariates["and"]
    pid = PID_Prec(d, ((0,), (1,)), (2,))
    assert pid[((0,), (1,))] == pytest.approx(0.31127812445913294, abs=1e-3)
    assert pid[((0,),)] == pytest.approx(0.0, abs=1e-3)
    assert pid[((1,),)] == pytest.approx(0.0, abs=1e-3)
    assert pid[((0, 1),)] == pytest.approx(0.5, abs=1e-3)


def test_pid_prec2():
    """
    Test iprec on another generic distribution.
    """
    d = bivariates["sum"]
    pid = PID_Prec(d, [[0], [1]], [2])
    assert pid[((0,), (1,))] == pytest.approx(0.5, abs=1e-3)
    assert pid[((0,),)] == pytest.approx(0.0, abs=1e-3)
    assert pid[((1,),)] == pytest.approx(0.0, abs=1e-3)
    assert pid[((0, 1),)] == pytest.approx(1.0, abs=1e-3)


def test_pid_prec3():
    """
    Test iprec on a generic trivariate source distribution.
    """
    events = ["0000", "0010", "0100", "0110", "1000", "1010", "1100", "1111"]
    d = Distribution(events, [1 / 8] * 8)
    pid = PID_Prec(d, [[0], [1], [2]], [3], compute=False)
    assert pid[((0,), (1,), (2,))] == pytest.approx(0.13795718192252743, abs=1e-3)


def test_pid_prec4():
    """
    Test iprec on unique gate X1,X2 -> X1, which should have redundancy equal to I(X1;X2)
    """
    d = Distribution(["000", "011", "100", "111"], [0.35, 0.15, 0.15, 0.35])
    pid = PID_Prec(
        d,
        [
            [0],
            [1],
        ],
        [2],
        compute=False,
    )
    assert pid[
        (
            (0,),
            (1,),
        )
    ] == pytest.approx(0.119, abs=1e-2)
