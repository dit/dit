"""
Tests for dit.pid.irr.
"""

import pytest

from dit.pid.irr import i_rr, PID_RR
from dit.pid.distributions import bivariates, trivariates


def test_irr1():
    """
    Test irr on redundant distribution.
    """
    d = bivariates['redundant']
    red = i_rr(d, ((0,), (1,)), (2,))
    assert red == pytest.approx(1)


def test_irr2():
    """
    Test irr on synergistic distribution.
    """
    d = bivariates['synergy']
    red = i_rr(d, ((0,), (1,)), (2,))
    assert red == pytest.approx(0)


def test_irr3():
    """
    Test irr on unique distribution.
    """
    d = bivariates['cat']
    red = i_rr(d, ((0,), (1,)), (2,))
    assert red == pytest.approx(0)


def test_pid_rr1():
    """
    Test irr on a generic distribution.
    """
    d = bivariates['diff']
    pid = PID_RR(d, ((0,), (1,)), (2,))
    assert pid[((0,), (1,))] == pytest.approx(0.15107, abs=1e-5)
    assert pid[((0,),)] == pytest.approx(0.16021, abs=1e-5)
    assert pid[((1,),)] == pytest.approx(0.16021, abs=1e-5)
    assert pid[((0, 1),)] == pytest.approx(0.02851, abs=1e-5)


def test_pid_rr2():
    """
    Test irr on another generic distribution.
    """
    d = trivariates['anddup']
    pid = PID_RR(d, [[0], [1], [2]], [3])
    for atom in pid._lattice:
        if atom == ((0, 1), (1, 2)):
            assert pid[atom] == pytest.approx(0.18872, abs=1e-5)
        elif atom in [((0,), (2,)), ((1,),)]:
            assert pid[atom] == pytest.approx(0.31128, abs=1e-5)
        else:
            assert pid[atom] == pytest.approx(0.0, abs=1e-5)
