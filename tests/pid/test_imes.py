"""
Tests for dit.pid.measures.imes.
"""

import pytest

from dit.pid.distributions import bivariates
from dit.pid.measures.ido import PID_Do
from dit.pid.measures.imes import PID_MES


def test_pid_mes1():
    """
    Test imes on a generic distribution.
    """
    d = bivariates["pnt. unq."]
    pid = PID_MES(d, ((0,), (1,)), (2,))
    assert pid[((0,), (1,))] == pytest.approx(0.25, abs=1e-4)
    assert pid[((0,),)] == pytest.approx(0.25, abs=1e-4)
    assert pid[((1,),)] == pytest.approx(0.25, abs=1e-4)
    assert pid[((0, 1),)] == pytest.approx(0.25, abs=1e-4)


@pytest.mark.parametrize("name", ["and", "reduced or", "sum", "boom", "pnt. unq.", "rand"])
def test_pid_mes_is_do(name):
    """
    For two sources, the maximum entropy distribution p(t) p(x0 | t) p(x1 | t) is the
    intervened distribution of I_do, so the two decompositions are the same.
    """
    d = bivariates[name]
    mes, do = PID_MES(d), PID_Do(d)
    for node in mes._lattice:
        assert mes[node] == pytest.approx(do[node], abs=1e-12)
