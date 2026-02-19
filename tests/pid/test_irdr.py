"""
Tests for dit.pid.irdr.
"""

import pytest

from dit.pid.distributions import bivariates, trivariates
from dit.pid.measures.irdr import PID_RDR


def test_pid_rdr1():
    """
    Test rdr on a generic distribution (same as test_ibroja)
    """
    d = bivariates['diff']
    pid = PID_RDR(d, ((0,), (1,)), (2,))
    assert pid[((0,), (1,))] == pytest.approx(0.12255624891826589, abs=1e-4)
    assert pid[((0,),)] == pytest.approx(0.18872187554086706, abs=1e-4)
    assert pid[((1,),)] == pytest.approx(0.18872187554086706, abs=1e-4)
    assert pid[((0, 1),)] == pytest.approx(0.0, abs=1e-4)
    assert pid.complete
    assert pid.nonnegative
    assert pid.consistent


def test_pid_rdr2():
    """
    Test irdr on reduced or. (same as broja)
    """
    d = bivariates['reduced or']
    pid = PID_RDR(d)
    assert pid[((0,), (1,))] == pytest.approx(0.31127812445913305, abs=1e-4)
    assert pid[((0,),)] == pytest.approx(0.0, abs=1e-4)
    assert pid[((1,),)] == pytest.approx(0.0, abs=1e-4)
    assert pid[((0, 1),)] == pytest.approx(0.6887218755408672, abs=1e-4)
    assert pid.complete
    assert pid.nonnegative
    assert pid.consistent


def test_pid_rdr3():
    """
    Test that xor cat is complete, nonnegative and consistent.
    """
    d = trivariates['xor cat']
    pid = PID_RDR(d)
    assert pid.complete
    assert pid.nonnegative
    assert pid.consistent


def test_pid_rdr4():
    """
    Test that dbl xor is complete, nonnegative and consistent.
    """
    d = trivariates['dbl xor']
    pid = PID_RDR(d)
    assert pid.complete
    assert pid.nonnegative
    assert pid.consistent
