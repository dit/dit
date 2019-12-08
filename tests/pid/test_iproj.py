# -*- coding: utf-8 -*-

"""
Tests for dit.pid.iproj.
"""

import pytest

from dit.pid.measures.iproj import PID_Proj
from dit.pid.distributions import bivariates, trivariates


def test_pid_proj1():
    """
    Test iproj on a generic distribution.
    """
    d = bivariates['diff']
    pid = PID_Proj(d, ((0,), (1,)), (2,))
    assert pid[((0,), (1,))] == pytest.approx(0.16503749927884365)
    assert pid[((0,),)] == pytest.approx(0.1462406251802893)
    assert pid[((1,),)] == pytest.approx(0.1462406251802893)
    assert pid[((0, 1),)] == pytest.approx(0.04248125036057776)


def test_pid_proj2():
    """
    Test iproj on another generic distribution.
    """
    d = trivariates['anddup']
    pid = PID_Proj(d, [[0], [1], [2]], [3])
    assert pid[((0, 1), (1, 2))] == pytest.approx(0.5)
    assert pid[((0,), (1,), (2,))] == pytest.approx(0.31127812445913305)
    for atom in pid._lattice:
        if atom not in [((0, 1), (1, 2)), ((0,), (1,), (2,))]:
            assert pid[atom] == pytest.approx(0.0)


def test_pid_proj3():
    """
    Test iproj on another generic distribution.
    """
    d = trivariates['xor shared']
    pid = PID_Proj(d)
    for atom in pid._lattice:
        if atom == ((0,), (1, 2)):
            assert pid[atom] == pytest.approx(1.0)
        else:
            assert pid[atom] == pytest.approx(0.0)
