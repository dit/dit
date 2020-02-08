"""
Tests for dit.pid.measures.imes.
"""

import pytest

from dit.pid.measures.imes import PID_MES
from dit.pid.distributions import bivariates


@pytest.mark.flaky(reruns=5)
def test_pid_mes1():
    """
    Test imes on a generic distribution.
    """
    d = bivariates['pnt. unq.']
    pid = PID_MES(d, ((0,), (1,)), (2,))
    assert pid[((0,), (1,))] == pytest.approx(0.25, abs=1e-4)
    assert pid[((0,),)] == pytest.approx(0.25, abs=1e-4)
    assert pid[((1,),)] == pytest.approx(0.25, abs=1e-4)
    assert pid[((0, 1),)] == pytest.approx(0.25, abs=1e-4)
