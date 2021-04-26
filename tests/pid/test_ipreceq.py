"""
Tests for dit.pid.measures.ipreceq.
"""

import pytest

from dit import Distribution
from dit.pid.measures.ipreceq import KolchinskyOptimizer, PID_Preceq
from dit.pid.distributions import bivariates


@pytest.mark.flaky(reruns=5)
def test_pid_preceq1():
    """
    Test ipreceq on a generic distribution.
    """
    d = bivariates['and']
    pid = PID_Preceq(d, ((0,), (1,)), (2,))
    assert pid[((0,), (1,))] == pytest.approx(0.31127812445913294, abs=1e-3)
    assert pid[((0,),)] == pytest.approx(0.0, abs=1e-3)
    assert pid[((1,),)] == pytest.approx(0.0, abs=1e-3)
    assert pid[((0, 1),)] == pytest.approx(0.5, abs=1e-3)


@pytest.mark.flaky(reruns=5)
def test_pid_preceq2():
    """
    Test ipreceq on another generic distribution.
    """
    d = bivariates['sum']
    pid = PID_Preceq(d, [[0], [1]], [2])
    assert pid[((0,), (1,))] == pytest.approx(0.5, abs=1e-3)
    assert pid[((0,),)] == pytest.approx(0.0, abs=1e-3)
    assert pid[((1,),)] == pytest.approx(0.0, abs=1e-3)
    assert pid[((0, 1),)] == pytest.approx(1.0, abs=1e-3)


@pytest.mark.flaky(reruns=5)
def test_pid_preceq3():
    """
    Test ipreceq on a generic trivariate source distribution.
    """
    events = ['0000', '0010', '0100', '0110', '1000', '1010', '1100', '1111']
    d = Distribution(events, [1 / 8] * 8)
    ko = KolchinskyOptimizer(d, [[0], [1], [2]], [3])
    res = ko.optimize()
    assert -res.fun == pytest.approx(0.13795718192252743, abs=1e-3)
