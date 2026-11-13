"""
Tests for dit.pid.ibroja.
"""

import numpy as np
import pytest
from hypothesis import assume, given, settings
from hypothesis.strategies import booleans, composite, floats, lists, one_of

from dit import Distribution
from dit.algorithms import broja_method
from dit.algorithms.distribution_optimizers import BROJABivariateOptimizer
from dit.pid.distributions import bivariates, trivariates
from dit.pid.measures.ibroja import PID_BROJA
from dit.pid.measures.iproj import PID_Proj
from dit.shannon import mutual_information
from dit.utils.testing import distributions

# AND with its sources made independent given the target: p(y) p(x0 | y) p(x1 | y)
AND_COUPLING = Distribution(["000", "010", "100", "110", "111"], [1 / 3, 1 / 6, 1 / 6, 1 / 12, 1 / 4])


def _coupling(d):
    """
    p(y) p(x0 | y) p(x1 | y), which has the source-target marginals of ``d``.
    """
    alphabets = [sorted(a) for a in d.alphabet]
    p = np.zeros([len(a) for a in alphabets])
    for outcome, prob in d.zipped():
        p[tuple(a.index(x) for a, x in zip(alphabets, outcome, strict=True))] = prob
    py = p.sum(axis=(0, 1))
    q = p.sum(axis=1)[:, None, :] * p.sum(axis=0)[None, :, :] / np.where(py > 0, py, 1)[None, None, :]
    return Distribution.from_ndarray(q)


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


def test_pid_broja3():
    """
    Test that iproj and ibroja are the same on reduced or.
    """
    d = bivariates["reduced or"]
    pid1 = PID_BROJA(d)
    pid2 = PID_Proj(d)
    assert pid1 == pid2


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


def test_broja_optimizer_free_entries_w():
    """
    On W the target value 1 determines both sources, at the first outcome of positive
    probability, 001. Indexing the dense pmf by that position fixed 000 instead, which
    left no freedom given Y = 0, so SLSQP returned W itself.
    """
    opt = BROJABivariateOptimizer(bivariates["w"], [[0], [1]], [2])
    outcomes = list(opt.dist.outcomes)
    assert sorted(outcomes[i] for i in opt._free) == [
        (("0",), ("0",), ("0",)),
        (("0",), ("1",), ("0",)),
        (("1",), ("0",), ("0",)),
        (("1",), ("1",), ("0",)),
    ]


@pytest.mark.flaky(reruns=5)
@pytest.mark.parametrize("method", ["scipy", "admui", "auto"])
def test_pid_broja_w(method):
    """
    W has no unique information: given Y = 0 the sources can be coupled to agree, which
    makes each source's information about Y redundant. The old SLSQP result was
    U = 0.667 and a redundancy of -0.415.
    """
    pid = PID_BROJA(bivariates["w"], method=method)
    assert pid[((0,),)] == pytest.approx(0.0, abs=1e-4)
    assert pid[((1,),)] == pytest.approx(0.0, abs=1e-4)
    assert pid[((0,), (1,))] == pytest.approx(0.25162916738782287, abs=1e-4)
    assert pid[((0, 1),)] == pytest.approx(2 / 3, abs=1e-4)


@pytest.mark.parametrize("name", sorted(bivariates))
def test_pid_broja_bivariates_nonnegative(name):
    """
    BROJA is nonnegative and consistent on every bivariate example.
    """
    pid = PID_BROJA(bivariates[name])
    assert pid.nonnegative
    assert pid.consistent


@composite
def sparse_distributions(draw):
    """
    Distributions on three bits with a random support, so that some target values
    determine both sources, as Y = 1 does in W.
    """
    mask = np.array(draw(lists(booleans(), min_size=8, max_size=8))).reshape(2, 2, 2)
    assume(mask.sum() >= 3)
    weights = np.array(draw(lists(floats(0.05, 1.0), min_size=8, max_size=8))).reshape(2, 2, 2)
    pmf = mask * weights
    return Distribution.from_ndarray(pmf / pmf.sum())


@pytest.mark.flaky(reruns=2)
@pytest.mark.parametrize("method", ["scipy", "admui", "auto"])
@settings(max_examples=40, deadline=None)
@given(dist=one_of(distributions(alphabets=(2, 2, 2), nondegenerate=True), sparse_distributions()))
def test_pid_broja_depends_on_marginals(method, dist):
    """
    BROJA depends only on the source-target marginals, so a distribution and its
    coupling p(y) p(x0 | y) p(x1 | y) have the same unique informations, and neither
    has negative atoms. SLSQP can stop a few millibits short of the minimum, so its
    tolerance is looser; the indexing bug it guards against was off by tenths of a bit.
    """
    pid = PID_BROJA(dist, method=method)
    coupled = PID_BROJA(_coupling(dist), method=method)
    tol = 1e-2 if method == "scipy" else 1e-4
    for atom in [((0,),), ((1,),)]:
        assert pid[atom] == pytest.approx(coupled[atom], abs=tol)
    assert pid.nonnegative
    assert coupled.nonnegative


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
    """
    An admUI iterate that has not converged is kept, not replaced by SLSQP, as long as it
    is a possible solution: here a redundancy of 0.261 bits, between the certified bounds
    0.082 and 0.311 of AND.
    """

    def unconverged(d, sources, target, maxiter):
        return {sources[0]: 0.05, sources[1]: 0.05}, {"converged": False, "method": "admui"}

    monkeypatch.setattr(broja_method, "_solve_admui", unconverged)
    uniques, meta = broja_method.broja_solve_bivariate(bivariates["and"], [(0,), (1,)], (2,), method="admui")
    assert meta["method"] == "admui"
    assert meta["guard"] == "ok"
    assert uniques[(0,)] == 0.05


def test_pid_broja_three_sources_without_ecos(monkeypatch):
    """
    Without ECOS, as on CI, the unique informations of three sources are computed by admUI,
    and the decomposition is consistent.
    """

    def no_cone(*args, **kwargs):  # pragma: no cover
        raise AssertionError("the cone solver needs ECOS")

    monkeypatch.setattr(broja_method, "ecos_available", lambda: False)
    monkeypatch.setattr(broja_method, "_solve_cone", no_cone)
    pid = PID_BROJA(trivariates["anddup"])
    assert pid.consistent
    assert pid[((0,), (1,), (2,))] == pytest.approx(0.31127812445913305, abs=1e-4)
    assert pid[((0, 1), (1, 2))] == pytest.approx(0.5, abs=1e-4)


def test_pid_broja_ecos_options():
    """
    Options of the ECOS solver are passed on to it.
    """
    pytest.importorskip("ecos")
    pid = PID_BROJA(bivariates["and"], method="cone", feastol=1e-10, max_iters=200)
    assert pid[((0,), (1,))] == pytest.approx(0.31127812445913305, abs=1e-6)


def test_pid_broja_unknown_option():
    """
    A misspelt option is an error: the cone solver would reject it and the fallback to
    admUI hide the mistake.
    """
    with pytest.raises(TypeError, match="PID_BROJA: unknown options"):
        PID_BROJA(bivariates["and"], feastl=1e-10)
    with pytest.raises(TypeError, match="unknown options"):
        broja_method.broja_solve_bivariate(bivariates["and"], [(0,), (1,)], (2,), feastl=1e-10)


@pytest.mark.parametrize("method", ["admui", "scipy"])
def test_broja_keeps_small_probabilities(method):
    """
    The 1.7e-6 outcome 001 splits into two 8e-7 entries of the optimal distribution.
    Zeroing entries below 1e-6 put U_0 3e-5 bits above I(X_0 : Y), so the guard rejected
    both solvers, and without ECOS nothing was left. Found by
    test_pid_broja_depends_on_marginals.
    """
    pmf = np.array([1, 1e-5, 1, 1, 1, 1, 1])
    d = Distribution(["000", "001", "010", "100", "101", "110", "111"], pmf / pmf.sum())
    uniques, meta = broja_method.broja_solve_bivariate(d, [(0,), (1,)], (2,), method=method, rng=0)
    assert meta["method"] == method  # not rescued by another solver, such as the cone program
    # the redundancy lies between 1e-12 and 4e-12, so U_0 = I(X_0 : Y)
    assert uniques[(0,)] == pytest.approx(mutual_information(d, [0], [2]), abs=1e-6)


def test_broja_guard_lifts_suboptimal_result(monkeypatch):
    """
    A result below the certified lower bound is lifted to it. On AND the bound is
    I_q(X0 : X1) = 0.082 bits for q = p(y) p(x0|y) p(x1|y), above the co-information.
    """

    def suboptimal(d, sources, target, maxiter):
        # the unique informations of p itself, whose redundancy is the co-information
        return {sources[0]: 0.5, sources[1]: 0.5}, {"converged": True, "method": "admui"}

    monkeypatch.setattr(broja_method, "_solve_admui", suboptimal)
    uniques, meta = broja_method.broja_solve_bivariate(bivariates["and"], [(0,), (1,)], (2,), method="admui")
    assert meta["method"] == "admui"
    assert meta["guard"] == "lifted"
    assert uniques[(0,)] == pytest.approx(0.311278124459133 - 0.081704165945510, abs=1e-9)
    assert uniques[(1,)] == pytest.approx(uniques[(0,)])


@pytest.mark.parametrize(
    "uniques",
    [
        {(0,): -0.1, (1,): -0.1},  # redundancy above min_i I(X_i : Y)
        {(0,): 0.5, (1,): 0.25},  # different redundancies for the two sources
    ],
)
def test_broja_guard_rejects_impossible_result(monkeypatch, uniques):
    """An impossible result is a solver failure, so the next solver is tried."""
    calls = []

    def impossible(d, sources, target, maxiter):
        calls.append("admui")
        return dict(uniques), {"converged": True, "method": "admui"}

    monkeypatch.setattr(broja_method, "_solve_admui", impossible)
    result, meta = broja_method.broja_solve_bivariate(bivariates["and"], [(0,), (1,)], (2,), method="admui")
    assert calls == ["admui"]
    assert meta["method"] != "admui"
    assert result[(0,)] == pytest.approx(0.0, abs=1e-3)


@pytest.mark.parametrize("seed", range(6))
def test_broja_within_certified_bounds(seed):
    """BROJA's redundancy lies between the co-informations of p and q and min_i I(X_i : Y)."""
    from dit.algorithms.pid_references import source_target_bounds

    rng = np.random.default_rng(seed)
    d = Distribution.from_ndarray(rng.dirichlet(np.ones(12)).reshape((2, 3, 2)))
    lower, upper, _ = source_target_bounds(d, [(0,), (1,)], (2,))
    red = PID_BROJA(d)[((0,), (1,))]
    assert lower - 1e-9 <= red <= upper + 1e-5
