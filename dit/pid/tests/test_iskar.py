"""
Tests for dit.pid.idownarrow.
"""

import pytest

from dit.pid.iskar import (i_downarrow,
                           PID_uparrow,
                           PID_double_uparrow,
                           PID_triple_uparrow,
                           PID_downarrow,
                           PID_triple_downarrow
                           )
from dit.pid.distributions import bivariates, trivariates


@pytest.mark.flaky(rerun=5)
def test_idownarrow1():
    """
    Test idownarrow on redundant distribution.
    """
    d = bivariates['redundant']
    uniques = i_downarrow(d, ((0,), (1,)), (2,))
    assert uniques[(0,)] == pytest.approx(0, abs=1e-4)
    assert uniques[(1,)] == pytest.approx(0, abs=1e-4)


@pytest.mark.flaky(rerun=5)
def test_idownarrow2():
    """
    Test idownarrow on synergistic distribution.
    """
    d = bivariates['synergy']
    uniques = i_downarrow(d, ((0,), (1,)), (2,))
    assert uniques[(0,)] == pytest.approx(0, abs=1e-4)
    assert uniques[(1,)] == pytest.approx(0, abs=1e-4)


@pytest.mark.flaky(rerun=5)
def test_idownarrow3():
    """
    Test idownarrow on unique distribution.
    """
    d = bivariates['cat']
    uniques = i_downarrow(d, ((0,), (1,)), (2,))
    assert uniques[(0,)] == pytest.approx(1, abs=1e-4)
    assert uniques[(1,)] == pytest.approx(1, abs=1e-4)


@pytest.mark.flaky(rerun=5)
def test_pid_downarrow1():
    """
    Test idownarrow on a generic distribution.
    """
    d = bivariates['prob 2']
    pid = PID_downarrow(d, ((0,), (1,)), (2,))
    assert pid[((0,), (1,))] == pytest.approx(0.12255624891826589, abs=1e-4)
    assert pid[((0,),)] == pytest.approx(0.18872187554086706, abs=1e-4)
    assert pid[((1,),)] == pytest.approx(0.5, abs=1e-4)
    assert pid[((0, 1),)] == pytest.approx(0.18872187554086706, abs=1e-4)


@pytest.mark.flaky(rerun=5)
def test_pid_downarrow2():
    """
    Test idownarrow on another generic distribution.
    """
    d = trivariates['anddup']
    pid = PID_downarrow(d)
    for atom in pid._lattice:
        if atom == ((0, 1), (1, 2)):
            assert pid[atom] == pytest.approx(0.18872187554086706, abs=1e-4)
        elif atom in [((0,), (2,)), ((1,),)]:
            assert pid[atom] == pytest.approx(0.31127812445913305, abs=1e-4)
        else:
            assert pid[atom] == pytest.approx(0.0, abs=1e-4)


@pytest.mark.flaky(rerun=5)
def test_pid_downarrow3():
    """
    Test that idownarrow is inconsistent on prob 2.
    """
    d = bivariates['prob 2']
    pid = PID_downarrow(d)
    assert not pid.consistent


def test_pid_uparrow1():
    """
    """
    d = bivariates['boom']
    pid = PID_uparrow(d)
    assert pid[((0,), (1,))] == pytest.approx(0.666666666666666667, abs=1e-4)
    assert pid[((0,),)] == pytest.approx(0.0, abs=1e-4)
    assert pid[((1,),)] == pytest.approx(0.0, abs=1e-4)
    assert pid[((0, 1),)] == pytest.approx(0.45914791702724411, abs=1e-4)


@pytest.mark.flaky(rerun=5)
def test_pid_double_uparrow1():
    """
    """
    d = bivariates['boom']
    pid = PID_double_uparrow(d)
    assert pid[((0,), (1,))] == pytest.approx(0.62581458369451781, abs=1e-4)
    assert pid[((0,),)] == pytest.approx(0.040852082972692771, abs=1e-4)
    assert pid[((1,),)] == pytest.approx(0.12581458369389198, abs=1e-4)
    assert pid[((0, 1),)] == pytest.approx(0.3333333333376699, abs=1e-4)


@pytest.mark.flaky(rerun=5)
def test_pid_triple_uparrow1():
    """
    """
    d = bivariates['boom']
    pid = PID_triple_uparrow(d, niter=10, bound_u=3, bound_v=3)
    assert pid[((0,), (1,))] == pytest.approx(0.61301198620181374, abs=1e-4)
    assert pid[((0,),)] == pytest.approx(0.053654682112724617, abs=1e-4)
    assert pid[((1,),)] == pytest.approx(0.12581458368589959, abs=1e-4)
    assert pid[((0, 1),)] == pytest.approx(0.33333333334723803, abs=1e-4)


@pytest.mark.flaky(rerun=5)
def test_pid_downarrow4():
    """
    """
    d = bivariates['boom']
    pid = PID_downarrow(d)
    assert pid[((0,), (1,))] == pytest.approx(0.20751874963942218, abs=1e-4)
    assert pid[((0,),)] == pytest.approx(0.45914791702724433, abs=1e-4)
    assert pid[((1,),)] == pytest.approx(0.33333333333333348, abs=1e-4)
    assert pid[((0, 1),)] == pytest.approx(0.12581458369391196, abs=1e-4)


@pytest.mark.flaky(rerun=5)
def test_pid_triple_downarrow1():
    """
    """
    d = bivariates['boom']
    pid = PID_triple_downarrow(d, bounds=(3,))
    assert pid[((0,), (1,))] == pytest.approx(0.29288371792167206, abs=1e-4)
    assert pid[((0,),)] == pytest.approx(0.3737829487445149, abs=1e-4)
    assert pid[((1,),)] == pytest.approx(0.33333333333333348, abs=1e-4)
    assert pid[((0, 1),)] == pytest.approx(0.12581458369391196, abs=1e-4)
