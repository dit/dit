"""
Tests for dit.pid.measures.imc (more-capable intersection information I_mc^∩).
"""

import pytest

from dit.pid.distributions import bivariates
from dit.pid.measures.ideg import PID_Deg
from dit.pid.measures.imc import PID_MC
from dit.pid.measures.immi import PID_MMI


@pytest.mark.flaky(reruns=5)
def test_pid_mc_and():
    """
    Test I_mc^∩ on the AND distribution.
    Paper Table 2: I_mc^∩(AND) = 0.311 bits.
    """
    d = bivariates["and"]
    pid = PID_MC(d, ((0,), (1,)), (2,))
    assert pid[((0,), (1,))] == pytest.approx(0.3113, abs=1e-3)
    assert pid[((0,),)] == pytest.approx(0.0, abs=1e-3)
    assert pid[((1,),)] == pytest.approx(0.0, abs=1e-3)


@pytest.mark.flaky(reruns=5)
def test_pid_mc_sum():
    """
    Test I_mc^∩ on the sum distribution.
    Paper Table 2: I_mc^∩(SUM) = 0.5 bits.
    """
    d = bivariates["sum"]
    pid = PID_MC(d, ((0,), (1,)), (2,))
    assert pid[((0,), (1,))] == pytest.approx(0.5, abs=1e-3)
    assert pid[((0,),)] == pytest.approx(0.0, abs=1e-3)
    assert pid[((1,),)] == pytest.approx(0.0, abs=1e-3)


@pytest.mark.flaky(reruns=5)
def test_pid_mc_unique():
    """
    Test I_mc^∩ on the unique-1 distribution (T = Y2, Y1 independent).
    Paper Table 2: I_mc^∩ = 0.
    """
    d = bivariates["unique 1"]
    pid = PID_MC(d, ((0,), (1,)), (2,))
    assert pid[((0,), (1,))] == pytest.approx(0.0, abs=1e-3)
    assert pid[((0,),)] == pytest.approx(0.0, abs=1e-3)
    assert pid[((1,),)] == pytest.approx(1.0, abs=1e-3)


@pytest.mark.flaky(reruns=5)
def test_pid_mc_redundant():
    """All information is redundant when sources are copies of target."""
    d = bivariates["redundant"]
    pid = PID_MC(d, ((0,), (1,)), (2,))
    assert pid[((0,), (1,))] == pytest.approx(1.0, abs=1e-3)
    assert pid[((0,),)] == pytest.approx(0.0, abs=1e-3)
    assert pid[((1,),)] == pytest.approx(0.0, abs=1e-3)
    assert pid[((0, 1),)] == pytest.approx(0.0, abs=1e-3)


@pytest.mark.flaky(reruns=5)
def test_pid_mc_synergy():
    """All information is synergistic for XOR."""
    d = bivariates["synergy"]
    pid = PID_MC(d, ((0,), (1,)), (2,))
    assert pid[((0,), (1,))] == pytest.approx(0.0, abs=1e-3)
    assert pid[((0,),)] == pytest.approx(0.0, abs=1e-3)
    assert pid[((1,),)] == pytest.approx(0.0, abs=1e-3)
    assert pid[((0, 1),)] == pytest.approx(1.0, abs=1e-3)


@pytest.mark.flaky(reruns=5)
def test_pid_mc_cat():
    """
    Copy/cat distribution: T = (Y1, Y2) with independent sources.
    Theorem 3: I_mc^∩ = C(Y1 ∧ Y2) = 0 when I(Y1; Y2) = 0.
    """
    d = bivariates["cat"]
    pid = PID_MC(d, ((0,), (1,)), (2,))
    assert pid[((0,), (1,))] == pytest.approx(0.0, abs=1e-3)
    assert pid[((0,),)] == pytest.approx(1.0, abs=1e-3)
    assert pid[((1,),)] == pytest.approx(1.0, abs=1e-3)


@pytest.mark.flaky(reruns=5)
def test_pid_mc_ordering():
    """
    I_d^∩ ≤ I_mc^∩ ≤ I_MMI for all distributions (paper eqs. 8-9).
    """
    for name in ["and", "sum", "redundant", "synergy", "unique 1",
                  "diff", "reduced or", "cat"]:
        d = bivariates[name]
        deg = PID_Deg(d, ((0,), (1,)), (2,))
        mc = PID_MC(d, ((0,), (1,)), (2,))
        mmi = PID_MMI(d, ((0,), (1,)), (2,))

        r_deg = deg[((0,), (1,))]
        r_mc = mc[((0,), (1,))]
        r_mmi = mmi[((0,), (1,))]

        assert r_deg <= r_mc + 1e-3, (
            f"I_d^∩ > I_mc^∩ on '{name}': {r_deg:.4f} > {r_mc:.4f}"
        )
        assert r_mc <= r_mmi + 1e-3, (
            f"I_mc^∩ > I_MMI on '{name}': {r_mc:.4f} > {r_mmi:.4f}"
        )


@pytest.mark.flaky(reruns=5)
def test_pid_mc_consistent():
    """PID_MC is consistent on standard distributions."""
    for name in ["and", "sum", "redundant", "synergy", "unique 1"]:
        d = bivariates[name]
        pid = PID_MC(d, ((0,), (1,)), (2,))
        assert pid.consistent, f"PID_MC not consistent on '{name}'"


@pytest.mark.flaky(reruns=5)
def test_pid_mc_nonnegative():
    """PID_MC produces nonnegative decompositions on standard distributions."""
    for name in ["and", "sum", "redundant", "synergy", "unique 1", "cat"]:
        d = bivariates[name]
        pid = PID_MC(d, ((0,), (1,)), (2,))
        assert pid.nonnegative, f"PID_MC not nonneg on '{name}'"
