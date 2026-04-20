"""
Tests for dit.pid.measures.ideg (degradation intersection information I_d^∩).
"""

import pytest

from dit.pid.distributions import bivariates
from dit.pid.measures.ideg import PID_Deg
from dit.pid.measures.immi import PID_MMI


@pytest.mark.flaky(reruns=5)
def test_pid_deg_and():
    """
    Test I_d^∩ on the AND distribution.
    Paper Table 2: I_d^∩(AND) = 0.311 bits.
    """
    d = bivariates["and"]
    pid = PID_Deg(d, ((0,), (1,)), (2,))
    assert pid[((0,), (1,))] == pytest.approx(0.3113, abs=1e-3)
    assert pid[((0,),)] == pytest.approx(0.0, abs=1e-3)
    assert pid[((1,),)] == pytest.approx(0.0, abs=1e-3)


@pytest.mark.flaky(reruns=5)
def test_pid_deg_sum():
    """
    Test I_d^∩ on the sum distribution.
    Paper Table 2: I_d^∩(SUM) = 0.5 bits.
    """
    d = bivariates["sum"]
    pid = PID_Deg(d, ((0,), (1,)), (2,))
    assert pid[((0,), (1,))] == pytest.approx(0.5, abs=1e-3)
    assert pid[((0,),)] == pytest.approx(0.0, abs=1e-3)
    assert pid[((1,),)] == pytest.approx(0.0, abs=1e-3)


@pytest.mark.flaky(reruns=5)
def test_pid_deg_unique():
    """
    Test I_d^∩ on the unique-1 distribution (T = Y2, Y1 independent).
    Paper Table 2: I_d^∩ = 0.
    """
    d = bivariates["unique 1"]
    pid = PID_Deg(d, ((0,), (1,)), (2,))
    assert pid[((0,), (1,))] == pytest.approx(0.0, abs=1e-3)
    assert pid[((0,),)] == pytest.approx(0.0, abs=1e-3)
    assert pid[((1,),)] == pytest.approx(1.0, abs=1e-3)


@pytest.mark.flaky(reruns=5)
def test_pid_deg_redundant():
    """All information is redundant when sources are copies of target."""
    d = bivariates["redundant"]
    pid = PID_Deg(d, ((0,), (1,)), (2,))
    assert pid[((0,), (1,))] == pytest.approx(1.0, abs=1e-3)
    assert pid[((0,),)] == pytest.approx(0.0, abs=1e-3)
    assert pid[((1,),)] == pytest.approx(0.0, abs=1e-3)
    assert pid[((0, 1),)] == pytest.approx(0.0, abs=1e-3)


@pytest.mark.flaky(reruns=5)
def test_pid_deg_synergy():
    """All information is synergistic for XOR."""
    d = bivariates["synergy"]
    pid = PID_Deg(d, ((0,), (1,)), (2,))
    assert pid[((0,), (1,))] == pytest.approx(0.0, abs=1e-3)
    assert pid[((0,),)] == pytest.approx(0.0, abs=1e-3)
    assert pid[((1,),)] == pytest.approx(0.0, abs=1e-3)
    assert pid[((0, 1),)] == pytest.approx(1.0, abs=1e-3)


@pytest.mark.flaky(reruns=5)
def test_pid_deg_cat():
    """
    Copy/cat distribution: T = (Y1, Y2) with independent sources.
    I_d^∩ = Gács-Körner common info = 0 (Kolchinsky 2022).
    """
    d = bivariates["cat"]
    pid = PID_Deg(d, ((0,), (1,)), (2,))
    assert pid[((0,), (1,))] == pytest.approx(0.0, abs=1e-3)
    assert pid[((0,),)] == pytest.approx(1.0, abs=1e-3)
    assert pid[((1,),)] == pytest.approx(1.0, abs=1e-3)


@pytest.mark.flaky(reruns=5)
def test_pid_deg_leq_mmi():
    """
    I_d^∩ ≤ I_MMI for all distributions (paper eq. 8-9).
    """
    for name in ["and", "sum", "redundant", "synergy", "unique 1",
                  "diff", "reduced or", "cat"]:
        d = bivariates[name]
        deg = PID_Deg(d, ((0,), (1,)), (2,))
        mmi = PID_MMI(d, ((0,), (1,)), (2,))
        r_deg = deg[((0,), (1,))]
        r_mmi = mmi[((0,), (1,))]
        assert r_deg <= r_mmi + 1e-3, (
            f"I_d^∩ > I_MMI on '{name}': {r_deg:.4f} > {r_mmi:.4f}"
        )


@pytest.mark.flaky(reruns=5)
def test_pid_deg_consistent():
    """PID_Deg is consistent on standard distributions."""
    for name in ["and", "sum", "redundant", "synergy", "unique 1"]:
        d = bivariates[name]
        pid = PID_Deg(d, ((0,), (1,)), (2,))
        assert pid.consistent, f"PID_Deg not consistent on '{name}'"
