"""
Tests for dit.pid.ibroja.
"""

import pytest

from dit import Distribution
from dit.algorithms import broja_method
from dit.algorithms.distribution_optimizers import BROJABivariateOptimizer
from dit.pid.distributions import bivariates, trivariates
from dit.pid.measures.ibroja import PID_BROJA
from dit.pid.measures.iproj import PID_Proj

# AND with its sources made independent given the target: p(y) p(x0 | y) p(x1 | y)
AND_COUPLING = Distribution(["000", "010", "100", "110", "111"], [1 / 3, 1 / 6, 1 / 6, 1 / 12, 1 / 4])


@pytest.mark.flaky(reruns=5)
@pytest.mark.parametrize("method", ["scipy", "admui", "auto"])
def test_pid_broja1_method(method):
    """
    Test ibroja on a generic distribution across bivariate solvers.
    """
    d = bivariates["diff"]
    pid = PID_BROJA(d, ((0,), (1,)), (2,), method=method)
    assert pid[((0,), (1,))] == pytest.approx(0.12255624891826589, abs=1e-4)
    assert pid[((0,),)] == pytest.approx(0.18872187554086706, abs=1e-4)
    assert pid[((1,),)] == pytest.approx(0.18872187554086706, abs=1e-4)
    assert pid[((0, 1),)] == pytest.approx(0.0, abs=1e-4)


@pytest.mark.flaky(reruns=5)
def test_pid_broja1_cone():
    pytest.importorskip("ecos")
    test_pid_broja1_method("cone")


@pytest.mark.flaky(reruns=5)
def test_pid_broja1():
    test_pid_broja1_method("scipy")


@pytest.mark.flaky(reruns=5)
def test_pid_broja2():
    """
    Test ibroja on another generic distribution.
    """
    d = trivariates["anddup"]
    pid = PID_BROJA(d, [[0], [1], [2]], [3])
    for atom in pid._lattice:
        if atom == ((0,), (1,), (2,)):
            assert pid[atom] == pytest.approx(0.31127812445913305, abs=1e-4)
        elif atom == ((0, 1), (1, 2)):
            assert pid[atom] == pytest.approx(0.5, abs=1e-4)
        else:
            assert pid[atom] == pytest.approx(0.0, abs=1e-4)


@pytest.mark.flaky(reruns=5)
def test_pid_broja3():
    """
    Test that iproj and ibroja are the same on reduced or.
    """
    d = bivariates["reduced or"]
    pid1 = PID_BROJA(d)
    pid2 = PID_Proj(d)
    assert pid1 == pid2


@pytest.mark.flaky(reruns=5)
def test_pid_broja4():
    """
    Test that xor cat is incomplete, but nonnegative and consistent.
    """
    d = trivariates["xor cat"]
    pid = PID_BROJA(d)
    assert not pid.complete
    assert pid.nonnegative
    assert pid.consistent


def test_broja_optimizer_free_entries():
    """
    The extra constraints indexed the dense pmf by positions among the outcomes of
    positive probability. On the coupling of AND they fixed (1, 0, 0) at its starting
    value and left the determined (1, 1, 1) free, so SLSQP could not move.
    """
    opt = BROJABivariateOptimizer(AND_COUPLING, [[0], [1]], [2])
    outcomes = list(opt.dist.outcomes)
    assert sorted(outcomes[i] for i in opt._free) == [
        (("0",), ("0",), ("0",)),
        (("0",), ("1",), ("0",)),
        (("1",), ("0",), ("0",)),
        (("1",), ("1",), ("0",)),
    ]


@pytest.mark.flaky(reruns=5)
@pytest.mark.parametrize("method", ["scipy", "admui", "auto"])
def test_pid_broja_coupling(method):
    """
    BROJA only depends on the source-target marginals, so the coupling of AND has no
    unique information, as AND does. SLSQP used to report 0.2296 for each source.
    """
    pid = PID_BROJA(AND_COUPLING, method=method)
    assert pid[((0,),)] == pytest.approx(0.0, abs=1e-4)
    assert pid[((1,),)] == pytest.approx(0.0, abs=1e-4)


def test_pid_broja_default_solver():
    """
    SLSQP can stop well short of the minimum. Here it gives unique informations of 0.154
    and 0.190, the first above I(X0 : Y) = 0.018, while the exact cone program and admUI
    give 0 and 0.036. So 'auto' does not use it.
    """
    d = Distribution(["000", "001", "100", "110", "111"], [3 / 13, 2 / 13, 4 / 13, 2 / 13, 2 / 13])
    pid = PID_BROJA(d)
    assert pid[((0,),)] == pytest.approx(0.0, abs=1e-4)
    assert pid[((1,),)] == pytest.approx(0.035934, abs=1e-4)
    assert broja_method.select_broja_method([2, 2, 2]) == ("cone" if broja_method.ecos_available() else "admui")


def test_broja_unconverged_admui(monkeypatch):
    """An admUI iterate that has not converged is kept, not replaced by SLSQP."""

    def unconverged(d, sources, target, maxiter):
        return {sources[0]: 0.5, sources[1]: 0.25}, {"converged": False, "method": "admui"}

    monkeypatch.setattr(broja_method, "_solve_admui", unconverged)
    uniques, meta = broja_method.broja_solve_bivariate(bivariates["and"], [(0,), (1,)], (2,), method="admui")
    assert meta["method"] == "admui"
    assert uniques[(0,)] == 0.5
