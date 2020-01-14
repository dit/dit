"""
Tests for dit.pid.measures.igh.
"""

import pytest

from dit import Distribution
from dit.pid.measures.igh import GHOptimizer, PID_GH
from dit.pid.distributions import bivariates, trivariates


@pytest.mark.flaky(reruns=5)
def test_pid_gh1():
    """
    Test igh on a generic distribution.
    """
    d = bivariates['and']
    pid = PID_GH(d, ((0,), (1,)), (2,))
    assert pid[((0,), (1,))] == pytest.approx(0.1226, abs=1e-4)
    assert pid[((0,),)] == pytest.approx(0.1887, abs=1e-4)
    assert pid[((1,),)] == pytest.approx(0.1887, abs=1e-4)
    assert pid[((0, 1),)] == pytest.approx(0.3113, abs=1e-4)


@pytest.mark.flaky(reruns=5)
def test_pid_gh2():
    """
    Test igh on another generic distribution.
    """
    d = bivariates['sum']
    pid = PID_GH(d, [[0], [1]], [2])
    assert pid[((0,), (1,))] == pytest.approx(0.0, abs=1e-4)
    assert pid[((0,),)] == pytest.approx(0.5, abs=1e-4)
    assert pid[((1,),)] == pytest.approx(0.5, abs=1e-4)
    assert pid[((0, 1),)] == pytest.approx(0.5, abs=1e-4)


@pytest.mark.flaky(reruns=5)
def test_pid_gh3():
    """
    Test igh on another generic distribution.
    """
    d = bivariates['pnt. unq.']
    pid = PID_GH(d, [[0], [1]], [2])
    assert pid[((0,), (1,))] == pytest.approx(0.0, abs=1e-4)
    assert pid[((0,),)] == pytest.approx(0.5, abs=1e-4)
    assert pid[((1,),)] == pytest.approx(0.5, abs=1e-4)
    assert pid[((0, 1),)] == pytest.approx(0.0, abs=1e-4)


@pytest.mark.flaky(reruns=5)
def test_pid_gh4():
    """
    Test igh on a generic trivariate source distribution.
    """
    events = ['0000', '0010', '0100', '0110', '1000', '1010', '1100', '1111']
    d = Distribution(events, [1/8]*8)
    gho = GHOptimizer(d, [[0], [1], [2]], [3])
    res = gho.optimize()
    assert -res.fun == pytest.approx(0.03471177057967193, abs=1e-3)
