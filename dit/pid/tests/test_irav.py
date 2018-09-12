"""
Tests for dit.pid.irav.
"""

import pytest

from dit.pid.irav import PID_RAV
from dit.pid.distributions import bivariates, trivariates

def test_pid_rav1():
    """
    Test irav on a generic distribution.
    """
    d = bivariates['diff']
    pid = PID_RAV(d, ((0,), (1,)), (2,))
    assert pid[((0,), (1,))] == pytest.approx(0.12255624891826589)
    assert pid[((0,),)] == pytest.approx(0.18872187554086706)
    assert pid[((1,),)] == pytest.approx(0.18872187554086706)
    assert pid[((0, 1),)] == pytest.approx(0)

def test_pid_rav2():
    """
    Test irav on a trivariate distribution based on a bivariate one.
    """
    d = trivariates['anddup']
    pid = PID_RAV(d, [[0], [1], [2]], [3])
    d_and = bivariates['and']
    pid_and = PID_RAV(d_and, [[0], [1]], [2])
    assert pid[((0, 1), (1, 2))] == pytest.approx(pid_and[((0,1),)])
    assert pid[((0,), (2,))] == pytest.approx(pid_and[((0,),)])
    assert pid[((1,),)] == pytest.approx(pid_and[((1,),)])
    assert pid[((0,), (1,), (2,))] == pytest.approx(pid_and[((0,),(1,))])
    for atom in pid._lattice:
        if atom not in [((0, 1), (1, 2)), ((0,), (2,)), ((1,),), ((0,), (1,), (2,))]:
            assert pid[atom] == pytest.approx(0.0)


def test_pid_rav3():
    """
    Test irav on a generic trivariate distribution.
    """
    d = trivariates['xor shared']
    pid = PID_RAV(d)
    for atom in pid._lattice:
        if atom == ((0,), (1, 2)):
            assert pid[atom] == pytest.approx(1.0)
        else:
            assert pid[atom] == pytest.approx(0.0)
