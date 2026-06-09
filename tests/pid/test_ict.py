"""
Tests for dit.pid.measures.ict.
"""

import numpy as np
import pytest

from dit import Distribution
from dit.pid.distributions import bivariates
from dit.pid.measures.ict import PID_CT, i_triangle


def test_pid_ct1():
    """
    Test ict on a generic distribution.
    """
    d = bivariates["reduced or"]
    pid = PID_CT(d, ((0,), (1,)), (2,))
    assert pid[((0,), (1,))] == pytest.approx(0.02712, abs=1e-4)
    assert pid[((0,),)] == pytest.approx(0.28416, abs=1e-4)
    assert pid[((1,),)] == pytest.approx(0.28416, abs=1e-4)
    assert pid[((0, 1),)] == pytest.approx(0.40456, abs=1e-4)


def _markov_chain():
    """A genuine X -> Y -> Z Markov chain, so only one path is direct."""
    px = np.array([0.5, 0.5])
    py_x = np.array([[0.8, 0.2], [0.3, 0.7]])
    pz_y = np.array([[0.9, 0.1], [0.2, 0.8]])
    outcomes, pmf = [], []
    for x in range(2):
        for y in range(2):
            for z in range(2):
                p = px[x] * py_x[x, y] * pz_y[y, z]
                if p > 0:
                    outcomes.append((x, y, z))
                    pmf.append(p)
    return Distribution(outcomes, pmf)


def test_i_triangle_direct_yz():
    """X -> Y -> Z: only Y -> Z is direct, exercising the direct_yz branch."""
    d = _markov_chain()
    assert i_triangle(d, [0], [1], [2]) == pytest.approx(0.093281, abs=1e-5)


def test_i_triangle_direct_xz():
    """Swapping the sources exercises the symmetric direct_xz branch."""
    d = _markov_chain()
    assert i_triangle(d, [1], [0], [2]) == pytest.approx(0.093281, abs=1e-5)


def test_pid_ct_markov_chain():
    """PID_CT on the Markov chain: redundancy equals the path information."""
    pid = PID_CT(_markov_chain(), ((0,), (1,)), (2,))
    assert pid[((0,), (1,))] == pytest.approx(0.093281, abs=1e-5)
