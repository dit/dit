"""
Tests for dit.pid.iskar.
"""

import pytest

from dit.pid.iskar import (PID_SKAR_nw,
                           PID_SKAR_owa,
                           PID_SKAR_owb,
                           PID_SKAR_tw,
                           )
from dit.pid.distributions import bivariates, trivariates


def test_pid_nw1():
    """
    Test idep on a generic distribution.
    """
    d = bivariates['sum']
    pid = PID_SKAR_nw(d, ((0,), (1,)), (2,))
    assert pid[((0,), (1,))] == pytest.approx(0.5, abs=1e-4)
    assert pid[((0,),)] == pytest.approx(0.0, abs=1e-4)
    assert pid[((1,),)] == pytest.approx(0.0, abs=1e-4)
    assert pid[((0, 1),)] == pytest.approx(1.0, abs=1e-4)


def test_pid_owa1():
    """
    Test idep on a generic distribution.
    """
    d = bivariates['pnt. unq.']
    pid = PID_SKAR_owa(d, ((0,), (1,)), (2,))
    assert pid[((0,), (1,))] == pytest.approx(0.0, abs=1e-4)
    assert pid[((0,),)] == pytest.approx(0.5, abs=1e-4)
    assert pid[((1,),)] == pytest.approx(0.5, abs=1e-4)
    assert pid[((0, 1),)] == pytest.approx(0.0, abs=1e-4)


def test_pid_owb1():
    """
    Test idep on a generic distribution.
    """
    d = bivariates['pnt. unq.']
    pid = PID_SKAR_owb(d, ((0,), (1,)), (2,))
    assert pid[((0,), (1,))] == pytest.approx(0.5, abs=1e-4)
    assert pid[((0,),)] == pytest.approx(0.0, abs=1e-4)
    assert pid[((1,),)] == pytest.approx(0.0, abs=1e-4)
    assert pid[((0, 1),)] == pytest.approx(0.5, abs=1e-4)


def test_pid_owb2():
    """
    Test idep on a generic distribution.
    """
    d = bivariates['prob. 2']
    pid = PID_SKAR_owb(d, ((0,), (1,)), (2,))
    assert pid[((0,), (1,))] == pytest.approx(0.31128, abs=1e-4)
    assert pid[((0,),)] == pytest.approx(0.0, abs=1e-4)
    assert pid[((1,),)] == pytest.approx(0.18872, abs=1e-4)
    assert pid[((0, 1),)] == pytest.approx(0.5, abs=1e-4)


def test_pid_tw1():
    """
    Test idep on a generic distribution.
    """
    d = bivariates['erase']
    pid = PID_SKAR_tw(d, ((0,), (1,)), (2,))
    assert pid[((0,), (1,))] == pytest.approx(0.125, abs=1e-4)
    assert pid[((0,),)] == pytest.approx(0.125, abs=1e-4)
    assert pid[((1,),)] == pytest.approx(0.125, abs=1e-4)
    assert pid[((0, 1),)] == pytest.approx(0.0, abs=1e-4)
