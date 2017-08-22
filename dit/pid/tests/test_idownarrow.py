"""
Tests for dit.pid.idownarrow.
"""

import pytest

from dit.pid.idownarrow import i_downarrow, PID_downarrow
from dit.pid.distributions import bivariates, trivariates

def test_idownarrow1():
    """
    Test idownarrow on redundant distribution.
    """
    d = bivariates['redundant']
    uniques = i_downarrow(d, ((0,), (1,)), (2,))
    assert uniques[(0,)] == pytest.approx(0)
    assert uniques[(1,)] == pytest.approx(0)

def test_idownarrow2():
    """
    Test idownarrow on synergistic distribution.
    """
    d = bivariates['synergy']
    uniques = i_downarrow(d, ((0,), (1,)), (2,))
    assert uniques[(0,)] == pytest.approx(0)
    assert uniques[(1,)] == pytest.approx(0)

def test_idownarrow3():
    """
    Test idownarrow on unique distribution.
    """
    d = bivariates['cat']
    uniques = i_downarrow(d, ((0,), (1,)), (2,))
    assert uniques[(0,)] == pytest.approx(1)
    assert uniques[(1,)] == pytest.approx(1)

def test_pid_downarrow1():
    """
    Test idownarrow on a generic distribution.
    """
    d = bivariates['prob 2']
    pid = PID_downarrow(d, ((0,), (1,)), (2,))
    assert pid[((0,), (1,))] == pytest.approx(0.12255624891826589)
    assert pid[((0,),)] == pytest.approx(0.18872187554086706)
    assert pid[((1,),)] == pytest.approx(0.5)
    assert pid[((0, 1),)] == pytest.approx(0.18872187554086706)

def test_pid_downarrow2():
    """
    Test idownarrow on another generic distribution.
    """
    d = trivariates['anddup']
    pid = PID_downarrow(d)
    for atom in pid._lattice:
        if atom == ((0, 1), (1, 2)):
            assert pid[atom] == pytest.approx(0.18872187554086706)
        elif atom in [((0,), (2,)), ((1,),)]:
            assert pid[atom] == pytest.approx(0.31127812445913305)
        else:
            assert pid[atom] == pytest.approx(0.0)

def test_pid_downarrow3():
    """
    Test that idownarrow is inconsistent on prob 2.
    """
    d = bivariates['prob 2']
    pid = PID_downarrow(d)
    assert not pid.consistent
