"""
Tests for dit.pid.measures.igh.
"""

import pytest

from dit import Distribution
from dit.pid.distributions import bivariates
from dit.pid.measures.igh import PID_GH, GHOptimizer
from dit.pid.measures.iprec import PID_Prec
from dit.pid.measures.iwedge import PID_GK

RED = ((0,), (1,))


@pytest.mark.flaky(reruns=5)
def test_pid_gh1():
    """
    Test igh on a generic distribution.
    """
    d = bivariates["and"]
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
    d = bivariates["sum"]
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
    d = bivariates["pnt. unq."]
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
    events = ["0000", "0010", "0100", "0110", "1000", "1010", "1100", "1111"]
    d = Distribution(events, [1 / 8] * 8)
    gho = GHOptimizer(d, [[0], [1], [2]], [3])
    res = gho.optimize()
    assert -res.fun == pytest.approx(0.03471177057967193, abs=1e-3)


def test_pid_gh_rdn_xor():
    """
    The redundant bit is a function of each source, so it is a valid Q carrying 1 bit. The
    optimizer used to return 0.
    """
    assert PID_GH(bivariates["rdn xor"]).get_red(RED) == pytest.approx(1.0, abs=1e-9)


def test_pid_gh_erase():
    """
    The optimal Q has four values, more than the local optimization's vertices with at most
    three (which reach 0.125) can represent.
    """
    assert PID_GH(bivariates["erase"]).get_red(RED) == pytest.approx(0.155639, abs=1e-6)


@pytest.mark.parametrize("name", ["and", "sum", "diff", "boom", "rdn xor", "gband", "erase", "prob. 2"])
def test_pid_gh_bounds(name):
    """A common function is independent of the target given each source, and independence implies the Blackwell order."""
    d = bivariates[name]
    gh = PID_GH(d).get_red(RED)
    assert PID_GK(d).get_red(RED) - 1e-9 <= gh <= PID_Prec(d).get_red(RED) + 1e-9


def test_pid_gh_three_sources():
    """
    The exact redundancy of the three-way AND gate. The local optimization found 0.0347,
    with its Markov constraints violated by about 1e-6.
    """
    events = ["0000", "0010", "0100", "0110", "1000", "1010", "1100", "1111"]
    pid = PID_GH(Distribution(events, [1 / 8] * 8))
    assert pid.get_red(((0,), (1,), (2,))) == pytest.approx(0.034632, abs=1e-6)
    assert pid.complete
    assert pid.consistent


def test_pid_gh_method():
    """The local optimization is used when asked for."""
    d = bivariates["and"]
    assert PID_GH._measure(d, RED, (2,), method="optimizer") == pytest.approx(0.1226, abs=1e-3)
    with pytest.raises(ValueError, match="method"):
        PID_GH._measure(d, RED, (2,), method="vertex")
