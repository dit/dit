"""
Tests for dit.pid.idep.
"""

import pytest

from dit.pid.ibroja import PID_BROJA
from dit.pid.idep import PID_dep, PID_RA
from dit.pid.distributions import bivariates, trivariates


def test_pid_dep1():
    """
    Test idep on a generic distribution.
    """
    d = bivariates['reduced or']
    pid = PID_dep(d, ((0,), (1,)), (2,))
    assert pid[((0,), (1,))] == pytest.approx(0.073761301440421256, abs=1e-4)
    assert pid[((0,),)] == pytest.approx(0.23751682301871169, abs=1e-4)
    assert pid[((1,),)] == pytest.approx(0.23751682301871169, abs=1e-4)
    assert pid[((0, 1),)] == pytest.approx(0.45120505252215537, abs=1e-4)


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
            assert pid[atom] == pytest.approx(0.0, abs=1e-4)


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

def test_pid_ra1():
    """
    """
    d = bivariates['and']
    pid = PID_RA(d, ((0,), (1,)), (2,))
    assert pid[((0,), (1,))] == pytest.approx(-0.18872, abs=1e-4)
    assert pid[((0,),)] == pytest.approx(0.5, abs=1e-4)
    assert pid[((1,),)] == pytest.approx(0.5, abs=1e-4)
    assert pid[((0, 1),)] == pytest.approx(0.0, abs=1e-4)
