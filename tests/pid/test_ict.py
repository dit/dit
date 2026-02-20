"""
Tests for dit.pid.measures.ict.
"""

import pytest

from dit.pid.distributions import bivariates
from dit.pid.measures.ict import PID_CT


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
