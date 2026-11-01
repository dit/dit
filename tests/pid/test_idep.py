"""
Tests for dit.pid.idep.
"""

import pytest

from dit.pid.distributions import bivariates, trivariates
from dit.pid.measures.ibroja import PID_BROJA
from dit.pid.measures.idep import PID_RA, PID_dep, PID_dep_a, PID_dep_b, PID_dep_c


def test_pid_dep1():
    """
    Test idep on a generic distribution.
    """
    d = bivariates["reduced or"]
    pid = PID_dep(d, ((0,), (1,)), (2,))
    assert pid[((0,), (1,))] == pytest.approx(0.073761301440421256, abs=1e-4)
    assert pid[((0,),)] == pytest.approx(0.23751682301871169, abs=1e-4)
    assert pid[((1,),)] == pytest.approx(0.23751682301871169, abs=1e-4)
    assert pid[((0, 1),)] == pytest.approx(0.45120505252215537, abs=1e-4)


def test_pid_dep2():
    """
    Test idep on another generic distribution.
    """
    d = trivariates["anddup"]
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
    d = bivariates["reduced or"]
    pid1 = PID_BROJA(d)
    pid2 = PID_dep(d)
    assert pid1 != pid2


def test_pid_dep4():
    """
    Test that anddup is complete.
    """
    d = trivariates["anddup"]
    pid = PID_dep(d)
    assert pid.complete
    assert pid.nonnegative
    assert pid.consistent


def test_pid_ra1():
    """ """
    d = bivariates["and"]
    pid = PID_RA(d, ((0,), (1,)), (2,))
    assert pid[((0,), (1,))] == pytest.approx(-0.18872, abs=1e-4)
    assert pid[((0,),)] == pytest.approx(0.5, abs=1e-4)
    assert pid[((1,),)] == pytest.approx(0.5, abs=1e-4)
    assert pid[((0, 1),)] == pytest.approx(0.0, abs=1e-4)


def test_pid_ra2():
    """ """
    d = trivariates["cat"]
    pid = PID_RA(d, ((0,), (1,), (2,)), (3,))
    for atom in pid._lattice:
        if atom in (((0,),), ((1,),), ((2,),)):
            assert pid[atom] == pytest.approx(1.0, abs=1e-4)
        else:
            assert pid[atom] == pytest.approx(0.0, abs=1e-4)


@pytest.mark.parametrize("name", ["reduced or", "and", "sum", "diff", "boom", "erase"])
def test_pid_dep_a_two_sources(name):
    """
    With two sources the full dependency lattice is the one of PID_dep.
    """
    d = bivariates[name]
    pid_a, pid = PID_dep_a(d), PID_dep(d)
    for node in pid._lattice:
        assert pid_a[node] == pytest.approx(pid[node], abs=1e-6)


def test_pid_dep_b_inconsistent():
    """
    The reduced decomposition I_dep_b is inconsistent, as documented.
    """
    pid = PID_dep_b(bivariates["simple"])
    assert pid.get_red(((0,), (1,))) == pytest.approx(0.0817, abs=1e-4)
    assert not pid.consistent


@pytest.mark.parametrize("name, red", [("diff", 0.122556), ("and", 0.0), ("boom", 0.207519), ("erase", 0.130376)])
def test_pid_dep_c(name, red):
    """
    I_dep_c keeps the dependencies among the sources on every edge it considers.
    """
    pid = PID_dep_c(bivariates[name])
    assert pid.get_red(((0,), (1,))) == pytest.approx(red, abs=1e-4)
    assert pid.consistent
    assert pid.nonnegative


@pytest.mark.parametrize("measure", [PID_dep_a, PID_dep_b, PID_dep_c])
def test_pid_dep_variants_three_sources(measure):
    """
    The variants handle three sources (and a target of several variables).
    """
    pid = measure(trivariates["anddup"])
    assert pid.get_red(((0,),)) == pytest.approx(pid._scaled(0.311278), abs=1e-4)
