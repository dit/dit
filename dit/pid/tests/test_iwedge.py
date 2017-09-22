"""
Tests for dit.pid.iwedge.
"""

import pytest

from dit.pid.iwedge import i_wedge, PID_GK
from dit.pid.distributions import bivariates, trivariates

def test_iwedge1():
    """
    Test iwedge on redundant distribution.
    """
    d = bivariates['redundant']
    red = i_wedge(d, ((0,), (1,)), (2,))
    assert red == pytest.approx(1)

def test_iwedge2():
    """
    Test iwedge on synergistic distribution.
    """
    d = bivariates['synergy']
    red = i_wedge(d, ((0,), (1,)), (2,))
    assert red == pytest.approx(0)

def test_iwedge3():
    """
    Test iwedge on unique distribution.
    """
    d = bivariates['cat']
    red = i_wedge(d, ((0,), (1,)), (2,))
    assert red == pytest.approx(0)

def test_pid_gk1():
    """
    Test iwedge on a generic distribution.
    """
    d = bivariates['imp. rdn']
    pid = PID_GK(d, ((0,), (1,)), (2,))
    assert pid[((0,), (1,))] == pytest.approx(0.0)
    assert pid[((0,),)] == pytest.approx(1.0)
    assert pid[((1,),)] == pytest.approx(0.98959)
    assert pid[((0, 1),)] == pytest.approx(-0.98959)

def test_pid_gk2():
    """
    Test iwedge on another generic distribution.
    """
    d = trivariates['sum']
    pid = PID_GK(d, [[0], [1], [2]], [3])
    for atom in pid._lattice:
        if atom in [((0, 1, 2),), ((0,),), ((1,),), ((2,),)]:
            assert pid[atom] == pytest.approx(0.31127812445913294)
        elif atom in [((0, 1),), ((0, 2),), ((1, 2),)]:
            assert pid[atom] == pytest.approx(0.18872187554086706)
        else:
            assert pid[atom] == pytest.approx(0.0)
