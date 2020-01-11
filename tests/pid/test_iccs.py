# -*- coding: utf-8 -*-

"""
Tests for dit.pid.iccs.
"""

import pytest

import sys

from dit.pid.measures.iccs import PID_CCS
from dit.pid.distributions import bivariates, trivariates


def test_pid_ccs1():
    """
    Test iccs on a generic distribution.
    """
    d = bivariates['gband']
    pid = PID_CCS(d, ((0,), (1,)), (2,))
    assert pid[((0,), (1,))] == pytest.approx(0.81127812445913283)
    assert pid[((0,),)] == pytest.approx(0.5)
    assert pid[((1,),)] == pytest.approx(0.5)
    assert pid[((0, 1),)] == pytest.approx(0.0)


def test_pid_ccs2():
    """
    Test iccs on another generic distribution.
    """
    d = trivariates['anddup']
    pid = PID_CCS(d, [[0], [1], [2]], [3])
    for atom in pid._lattice:
        if atom == ((0,), (1,), (2,)):
            assert pid[atom] == pytest.approx(0.10375937481971094)
        elif atom in [((0,), (2,)), ((1,),)]:
            assert pid[atom] == pytest.approx(0.20751874963942191)
        elif atom in [((2,), (0, 1)), ((0,), (1, 2)), ((0, 1), (0, 2)), ((0, 1), (1, 2)), ((0, 2), (1, 2)), ((0, 1, 2),)]:
            assert pid[atom] == pytest.approx(0.14624062518028902)
        elif atom in [((2,),), ((0,),), ((0, 1), (0, 2), (1, 2)), ((0, 2),)]:
            assert pid[atom] == pytest.approx(-0.14624062518028902)
        else:
            assert pid[atom] == pytest.approx(0.0)


def test_pid_ccs3():
    """
    Test iccs on another generic distribution.
    """
    d = trivariates['anddup']
    pid = PID_CCS(d)
    for atom in pid._lattice:
        if atom == ((0,), (1,), (2,)):
            assert pid[atom] == pytest.approx(0.10375937481971094)
        elif atom in [((0,), (2,)), ((1,),)]:
            assert pid[atom] == pytest.approx(0.20751874963942191)
        elif atom in [((2,), (0, 1)), ((0,), (1, 2)), ((0, 1), (0, 2)), ((0, 1), (1, 2)), ((0, 2), (1, 2)), ((0, 1, 2),)]:
            assert pid[atom] == pytest.approx(0.14624062518028902)
        elif atom in [((2,),), ((0,),), ((0, 1), (0, 2), (1, 2)), ((0, 2),)]:
            assert pid[atom] == pytest.approx(-0.14624062518028902)
        else:
            assert pid[atom] == pytest.approx(0.0)


def test_pid_ccs4():
    d = bivariates['gband']
    pid = PID_CCS(d)
    string = """\
+--------+--------+--------+
| I_ccs  |  I_r   |   pi   |
+--------+--------+--------+
| {0:1}  | 1.8113 | 0.0000 |
|  {0}   | 1.3113 | 0.5000 |
|  {1}   | 1.3113 | 0.5000 |
| {0}{1} | 0.8113 | 0.8113 |
+--------+--------+--------+"""
    assert str(pid) == string
