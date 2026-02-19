"""
Tests for dit.pid.isx.
"""

import pytest
import numpy as np

from dit.pid.measures.isx import PID_SX
from dit.pid.distributions import bivariates, trivariates


def test_pid_sx1():
    """
    Test isx on a generic distribution.
    """
    d = bivariates['pnt. unq.']
    pid = PID_SX(d, ((0,), (1,)), (2,))
    assert pid[((0,), (1,))] == pytest.approx(0.0)
    assert pid[((0,),)] == pytest.approx(0.5)
    assert pid[((1,),)] == pytest.approx(0.5)
    assert pid[((0, 1),)] == pytest.approx(0.0)


def test_pid_sx2():
    """
    Test isx on another generic distribution.
    """
    d = bivariates['unique 1']
    pid = PID_SX(d, [[0], [1]], [2])
    assert pid[((0,), (1,))] == pytest.approx(np.log2(4/3))
    assert pid[((0,),)] == pytest.approx(-np.log2(4/3))
    assert pid[((1,),)] == pytest.approx(np.log2(3/2))
    assert pid[((0, 1),)] == pytest.approx(np.log2(4/3))

def test_pid_sx3():
    """
    Test isx on another generic distribution.
    """
    d = bivariates['reduced or']
    pid = PID_SX(d, [[0], [1]], [2])
    assert pid[((0,), (1,))] == pytest.approx(1/2 * np.log2(4/3) - 1/2)
    assert pid[((0,),)] == pytest.approx(1/2 * np.log2(4/3) + 1/4 * np.log2(3))
    assert pid[((1,),)] == pytest.approx(1/2 * np.log2(4/3) + 1/4 * np.log2(3))
    assert pid[((0, 1),)] == pytest.approx(1/2 * np.log2(9/8))

def test_pid_sx_trivariate():
    """
    Test isx on a trivariate distribution.
    """
    d = trivariates['synergy']
    pid = PID_SX(d, ((0,), (1,), (2,)), (3,))

    atoms = [((0,),), ((1,),), ((2,),), ((0, 1),), ((0, 2),), ((1, 2),), ((0, 1, 2),), ((0,), (1,)), ((0,), (2,)), ((0,), (1, 2)), ((1,), (2,)), ((1,), (0, 2)), ((2,), (0, 1)), ((0, 1), (0, 2)), ((0, 1), (1, 2)), ((0, 2), (1, 2)), ((0,), (1,), (2,)), ((0, 1), (0, 2), (1, 2))]
    true_values = np.array([np.log2(5/4), np.log2(5/4), np.log2(5/4), np.log2(9/8), np.log2(9/8), np.log2(9/8), np.log2(32/27), np.log2(7/8), np.log2(7/8), np.log2(32/35), np.log2(7/8), np.log2(32/35), np.log2(32/35), np.log2(16/15), np.log2(16/15), np.log2(16/15), np.log2(8/7), np.log2(875/1024)])
    
    for atom, true_value in zip(atoms, true_values):
        assert pid[atom] == pytest.approx(true_value)

if __name__ == "__main__":
    test_pid_sx3()