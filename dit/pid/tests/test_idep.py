"""
Tests for dit.pid.idep.
"""

import pytest

from dit.pid.ibroja import PID_BROJA
from dit.pid.idep import i_dep, PID_dep
from dit.pid.distributions import bivariates, trivariates

def test_idep1():
    """
    Test idep on redundant distribution.
    """
    d = bivariates['redundant']
    uniques = i_dep(d, ((0,), (1,)), (2,))
    assert uniques[(0,)] == pytest.approx(0)
    assert uniques[(1,)] == pytest.approx(0)

def test_idep2():
    """
    Test idep on synergistic distribution.
    """
    d = bivariates['synergy']
    uniques = i_dep(d, ((0,), (1,)), (2,))
    assert uniques[(0,)] == pytest.approx(0)
    assert uniques[(1,)] == pytest.approx(0)

def test_idep3():
    """
    Test idep on unique distribution.
    """
    d = bivariates['cat']
    uniques = i_dep(d, ((0,), (1,)), (2,))
    assert uniques[(0,)] == pytest.approx(1)
    assert uniques[(1,)] == pytest.approx(1)

def test_pid_dep1():
    """
    Test idep on a generic distribution.
    """
    d = bivariates['reduced or']
    pid = PID_dep(d, ((0,), (1,)), (2,))
    assert pid[((0,), (1,))] == pytest.approx(0.073761301440421256)
    assert pid[((0,),)] == pytest.approx(0.23751682301871169)
    assert pid[((1,),)] == pytest.approx(0.23751682301871169)
    assert pid[((0, 1),)] == pytest.approx(0.45120505252215537)

def test_pid_dep2():
    """
    Test idep on another generic distribution.
    """
    d = trivariates['anddup']
    pid = PID_dep(d, [[0], [1], [2]], [3])
    for atom in pid._lattice:
        if atom == ((0,), (1,), (2,)):
            assert pid[atom] == pytest.approx(0.081704409646414788, abs=1e-4)
        elif atom == ((0, 1), (1, 2)):
            assert pid[atom] == pytest.approx(0.27042624480113808, abs=1e-4)
        elif atom in [((0,), (2,)), ((1,),)]:
            assert pid[atom] == pytest.approx(0.22957374150893717, abs=1e-4)
        else:
            assert pid[atom] == pytest.approx(0.0, abs=1e-6)

def test_pid_dep3():
    """
    Test that idep and ibroja differ on reduced or.
    """
    d = bivariates['reduced or']
    pid1 = PID_BROJA(d)
    pid2 = PID_dep(d)
    assert pid1 != pid2

def test_pid_dep4():
    """
    Test that anddup is complete.
    """
    d = trivariates['anddup']
    pid = PID_dep(d)
    assert pid.complete
    assert pid.nonnegative
    assert pid.consistent
