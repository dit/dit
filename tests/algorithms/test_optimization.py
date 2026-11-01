"""
Tests for dit.algorithms.optimization, the scipy optimizer machinery shared by dit's measures.

Two toy optimizers keep every optimization tiny: a squared-distance objective whose unique
minimizer is the joint pmf, and an auxiliary-variable channel maximizing a mutual
information, which has several local optima of differing quality.
"""

import os
import threading
from types import MethodType

import numpy as np
import pytest
from scipy.optimize import OptimizeResult, approx_fprime

from dit import Distribution
from dit.algorithms import optimization
from dit.algorithms.distribution_optimizers import MaxEntOptimizer
from dit.algorithms.optimization import (
    BaseAuxVarOptimizer,
    BaseNonConvexOptimizer,
    BaseOptimizer,
    _resolve_n_jobs,
    parallel_sweep,
)
from dit.exceptions import OptimizationException
from dit.multivariate import entropy as H
from dit.multivariate import residual_entropy, total_correlation


class _SquaredDistance(BaseNonConvexOptimizer):
    """
    Toy optimizer: minimize the squared distance between ``x`` and the joint pmf of the
    (compressed) variables. Its unique minimum, zero, is attained at the pmf itself.
    """

    construct_initial = BaseOptimizer.construct_uniform_initial

    def __init__(self, dist, rvs=None, crvs=None, objective_bound=None):
        super().__init__(dist, rvs=rvs, crvs=crvs)
        self._target = self._pmf.ravel().copy()
        self._optvec_size = self._target.size
        self._default_hops = 3
        if objective_bound is not None:
            self._objective_bound = objective_bound

    def _objective(self):
        target = self._target

        def objective(self, x):
            return float(np.sum((x - target) ** 2))

        return objective


class _MaxInformation(BaseAuxVarOptimizer):
    """
    Toy auxiliary-variable optimizer: the channel ``p(U | X)``, ``|U| <= bound``, that
    maximizes ``I[X : U]``. For ``p(X) = (0.5, 0.3, 0.2)`` and ``bound=2`` the optimum,
    ``I = 1`` bit, separates the first symbol from the other two; the two other binary
    partitions are strictly worse local optima.
    """

    def __init__(self, dist, bound=2, objective_bound=None):
        super().__init__(dist, rvs=[[0]], crvs=[])
        self._construct_auxvars([(self._rvs, bound)])
        if objective_bound is not None:
            self._objective_bound = objective_bound

    def _objective(self):
        mutual_information = self._mutual_information(self._rvs, self._arvs)

        def objective(self, x):
            return -mutual_information(self.construct_joint(x))

        return objective


def _three_symbols():
    """A single variable with p = (0.5, 0.3, 0.2)."""
    return Distribution(["0", "1", "2"], [0.5, 0.3, 0.2])


def _two_bits():
    """Two dependent bits."""
    return Distribution(["00", "01", "10", "11"], [0.4, 0.3, 0.2, 0.1])


def _three_bits():
    """Three dependent bits with full support."""
    outcomes = ["000", "001", "010", "011", "100", "101", "110", "111"]
    return Distribution(outcomes, [0.2, 0.05, 0.1, 0.15, 0.05, 0.1, 0.25, 0.1])


def _bind(opt):
    """Bind the objective and return the minimizer kwargs that ``optimize`` would build."""
    opt.objective = MethodType(opt._objective(), opt)
    return {"bounds": [(0, 1)] * opt._optvec_size, "callback": None, "constraints": opt.constraints, "options": {}}


# ── worker resolution and parallel sweeps ────────────────────────────────


def test_resolve_n_jobs_automatic(monkeypatch):
    """
    Unset or zero ``DIT_OPT_JOBS`` uses one worker per task, capped at the CPU count
    (one worker if the CPU count is unknown).
    """
    monkeypatch.setattr(os, "cpu_count", lambda: 3)
    monkeypatch.setenv("DIT_OPT_JOBS", "0")
    assert _resolve_n_jobs(2) == 2
    assert _resolve_n_jobs(10) == 3
    monkeypatch.delenv("DIT_OPT_JOBS")
    assert _resolve_n_jobs(10) == 3
    monkeypatch.setattr(os, "cpu_count", lambda: None)
    assert _resolve_n_jobs(10) == 1


def test_parallel_sweep_generator_seed():
    """
    A Generator base seed contributes exactly one draw, and task ``i`` then gets the
    stream ``default_rng(seed + i)``, just as for that integer seed.
    """

    def draw(item, rng):
        return item, float(rng.random())

    seed = int(np.random.default_rng(7).integers(0, 2**31 - 1))
    generator = np.random.default_rng(7)
    results = parallel_sweep(draw, range(4), base_seed=generator)

    assert results == parallel_sweep(draw, range(4), base_seed=seed)
    assert results == [(i, float(np.random.default_rng(seed + i).random())) for i in range(4)]
    reference = np.random.default_rng(7)
    reference.integers(0, 2**31 - 1)
    assert generator.random() == reference.random()


# ── pmf-level measures and their gradients ───────────────────────────────


def test_measure_factories_match_dit_measures():
    """
    The total correlation (no conditioning) and the conditional residual entropy built
    on the compressed pmf agree with dit's measures on the original distribution.
    """
    d = _three_bits()
    opt = _SquaredDistance(d, rvs=[[0], [1]], crvs=[2])
    pmf = opt._pmf
    assert opt._total_correlation({0, 1})(pmf) == pytest.approx(total_correlation(d, [[0], [1]]))
    assert opt._residual_entropy({0, 1}, {2})(pmf) == pytest.approx(residual_entropy(d, [[0], [1]], [2]))


@pytest.mark.parametrize(
    ("measure", "args"),
    [
        ("_total_correlation", ({0, 1},)),
        ("_residual_entropy", ({0, 1}, {2})),
        ("_conditional_mutual_information", ({0}, {1}, set())),
    ],
)
def test_gradient_builders_match_finite_differences(measure, args):
    """
    Each analytic pmf-gradient builder agrees with finite differences of its measure,
    including the defaulted conditioning set and the unconditioned CMI shortcut.
    """
    opt = _SquaredDistance(_three_bits(), rvs=[[0], [1]], crvs=[2])
    pmf = opt._pmf
    func = getattr(opt, measure)(*args)
    grad = getattr(opt, f"{measure}_grad")(*args)(pmf)
    assert grad.shape == pmf.shape
    fd = approx_fprime(pmf.ravel(), lambda v: func(v.reshape(pmf.shape)), 1e-7)
    np.testing.assert_allclose(grad.ravel(), fd, atol=1e-5)


def test_unconditioned_cmi_gradient_is_mi_gradient():
    """With an empty conditioning set the CMI gradient is exactly the MI gradient."""
    opt = _SquaredDistance(_three_bits(), rvs=[[0], [1]], crvs=[2])
    cmi_grad = opt._conditional_mutual_information_grad({0}, {1}, set())(opt._pmf)
    np.testing.assert_array_equal(cmi_grad, opt._mutual_information_grad({0}, {1})(opt._pmf))


def test_total_variation_between_marginals():
    """
    ``_total_variation`` is the total variation distance between the marginals of two
    variables over the same alphabet: p(X0) = (3/4, 1/4) and p(X1) = (5/8, 3/8).
    """
    opt = _SquaredDistance(Distribution(["00", "01", "10", "11"], [0.5, 0.25, 0.125, 0.125]))
    assert opt._total_variation({0}, {1})(opt._pmf) == pytest.approx(0.125)
    assert opt._total_variation({1}, {0})(opt._pmf) == pytest.approx(0.125)
    assert opt._total_variation({0}, {0})(opt._pmf) == pytest.approx(0.0)


# ── analytic jacobian wiring ──────────────────────────────────────────────


def test_pmf_gradient_without_parametrization_jacobian_uses_finite_differences():
    """
    A pmf gradient alone is not handed to scipy: without a ``_jacobian`` to compose it
    with, finite differences are kept and the optimization still succeeds.
    """

    class WithPmfGradient(_SquaredDistance):
        def _objective_gradient(self):
            return lambda pmf: 2 * (pmf - self._pmf)

    opt = WithPmfGradient(_two_bits())
    kwargs = {}
    opt._apply_analytic_jacobians(kwargs)
    assert "jac" not in kwargs
    opt.optimize(niter=1, rng=0)
    np.testing.assert_allclose(opt._optima, [0.4, 0.3, 0.2, 0.1], atol=1e-6)

    meo = MaxEntOptimizer(_two_bits(), [[0], [1]])
    kwargs = {}
    meo._apply_analytic_jacobians(kwargs)
    assert kwargs["jac"] == meo._jacobian


# ── the multi-start shotgun ───────────────────────────────────────────────


def test_shotgun_without_initial_condition(monkeypatch):
    """
    Without ``x0`` the shotgun minimizes exactly ``niter`` starts drawn from the
    optimizer's generator and returns the best successful local minimum.
    """
    opt = _MaxInformation(_three_symbols())
    kwargs = _bind(opt)
    opt._rng = np.random.default_rng(0)
    starts, outcomes = [], []
    minimize = optimization.minimize

    def spy(fun, x0, **kw):
        starts.append(x0.copy())
        res = minimize(fun=fun, x0=x0, **kw)
        outcomes.append((res.fun, res.success))
        return res

    monkeypatch.setattr(optimization, "minimize", spy)
    result = opt._optimize_shotgun(None, kwargs, niter=4)

    reference = _MaxInformation(_three_symbols())
    reference._rng = np.random.default_rng(0)
    np.testing.assert_allclose(starts, [reference.construct_random_initial() for _ in range(4)])
    assert result.fun == pytest.approx(min(fun for fun, success in outcomes if success))
    assert result.fun == pytest.approx(-1.0)


def test_shotgun_parallel_matches_serial(monkeypatch):
    """
    Without an objective bound the random starts run on worker threads when
    ``DIT_OPT_JOBS > 1``; the starts are drawn up front, so the result is unchanged.
    """
    minimize = optimization.minimize

    def run(jobs):
        monkeypatch.setenv("DIT_OPT_JOBS", jobs)
        threads, lock = set(), threading.Lock()

        def spy(**kw):
            with lock:
                threads.add(threading.get_ident())
            return minimize(**kw)

        monkeypatch.setattr(optimization, "minimize", spy)
        opt = _MaxInformation(_three_symbols())
        kwargs = _bind(opt)
        opt._rng = np.random.default_rng(3)
        return opt._optimize_shotgun(None, kwargs, niter=6), threads

    serial, serial_threads = run("1")
    parallel, parallel_threads = run("3")

    assert serial_threads == {threading.get_ident()}
    assert parallel_threads
    assert threading.get_ident() not in parallel_threads
    assert serial.fun == pytest.approx(-1.0)
    assert parallel.fun == pytest.approx(serial.fun)
    np.testing.assert_allclose(parallel.x, serial.x)


@pytest.mark.parametrize("jobs", ["1", "3"])
def test_shotgun_falls_back_to_best_unsuccessful_start(monkeypatch, jobs):
    """
    If no start reports success, whether run serially or on worker threads, the shotgun
    returns the lowest-objective feasible candidate instead of giving up.
    """
    monkeypatch.setenv("DIT_OPT_JOBS", jobs)
    outcomes, lock = [], threading.Lock()
    minimize = optimization.minimize

    def spy(**kw):
        res = minimize(**kw)
        with lock:
            outcomes.append(res)
        return res

    monkeypatch.setattr(optimization, "minimize", spy)
    opt = _MaxInformation(_three_symbols())
    kwargs = _bind(opt)
    kwargs["options"] = {"maxiter": 1}
    opt._rng = np.random.default_rng(5)
    result = opt._optimize_shotgun(None, kwargs, niter=4)

    assert len(outcomes) == 4
    assert not any(res.success for res in outcomes)
    assert result is min(outcomes, key=lambda res: res.fun)


def test_best_feasible_accepts_a_single_constraint_dict():
    """
    A lone constraint dict is treated like a one-element list; infeasible, point-less
    and non-finite results are skipped, and None is returned when nothing is feasible.
    """
    con = {"type": "eq", "fun": lambda x: np.sum(x) - 1}
    results = [
        OptimizeResult(x=np.array([0.1, 0.1]), fun=-5.0),
        OptimizeResult(x=np.array([0.5, 0.5]), fun=1.0),
        OptimizeResult(x=np.array([0.3, 0.7]), fun=0.5),
        OptimizeResult(x=None, fun=-10.0),
        OptimizeResult(x=np.array([0.4, 0.6]), fun=np.nan),
    ]
    assert BaseOptimizer._best_feasible(results, {"constraints": con}) is results[2]
    assert BaseOptimizer._best_feasible(results, {"constraints": [con]}) is results[2]
    assert BaseOptimizer._best_feasible(results[:1], {"constraints": con}) is None
    assert BaseOptimizer._best_feasible(results, {"constraints": None}) is results[0]


def test_best_feasible_inequality_tolerance():
    """An inequality constraint may be violated by at most ``ftol``."""
    con = {"type": "ineq", "fun": lambda x: x - 0.5}
    results = [
        OptimizeResult(x=np.array([0.5 - 1e-3, 0.9]), fun=0.0),
        OptimizeResult(x=np.array([0.5 - 1e-8, 0.9]), fun=1.0),
    ]
    assert BaseOptimizer._best_feasible(results, {"constraints": con}) is results[1]
    assert BaseOptimizer._best_feasible(results, {"constraints": con}, ftol=1e-2) is results[0]


# ── fallbacks when scipy reports failure ──────────────────────────────────


def test_convex_backend_retries_with_more_starts(monkeypatch):
    """
    If the first shotgun yields nothing, the convex backend retries once with at least
    25 starts; if that fails too, optimize() reports that no optimum was found.
    """
    xor = Distribution(["000", "011", "101", "110"], [0.25] * 4)
    meo = MaxEntOptimizer(xor, [[0], [1], [2]])
    calls = []
    shotgun = meo._optimize_shotgun

    def fail_once(x0, minimizer_kwargs, niter):
        calls.append(niter)
        return None if len(calls) == 1 else shotgun(x0, minimizer_kwargs, niter)

    monkeypatch.setattr(meo, "_optimize_shotgun", fail_once)
    meo.optimize()
    assert calls == [2, 25]
    assert H(meo.construct_dist()) == pytest.approx(3.0, abs=1e-6)

    failing = MaxEntOptimizer(xor, [[0], [1], [2]])
    attempts = []
    monkeypatch.setattr(failing, "_optimize_shotgun", lambda x0, minimizer_kwargs, niter: attempts.append(niter))
    with pytest.raises(OptimizationException, match="No optima found"):
        failing.optimize(niter=30)
    assert attempts == [30, 30]


@pytest.mark.parametrize(("shotgun", "expected_calls"), [(False, [25]), (4, [4])])
def test_basinhopping_failure_fallbacks(monkeypatch, shotgun, expected_calls):
    """
    When basin hopping fails without recording a feasible minimum, the result of the
    seeding shotgun is used if there was one, and otherwise a 25-start shotgun is run.
    """

    def failed_basinhopping(func, x0, **kwargs):
        lowest = OptimizeResult(x=x0, fun=func(x0), success=False, message="forced failure")
        return OptimizeResult(x=x0, fun=lowest.fun, lowest_optimization_result=lowest)

    monkeypatch.setattr(optimization, "basinhopping", failed_basinhopping)

    class Toy(_MaxInformation):
        _shotgun = shotgun

    opt = Toy(_three_symbols(), objective_bound=-1.0)
    calls = []
    run_shotgun = opt._optimize_shotgun

    def spy(x0, minimizer_kwargs, niter):
        calls.append(niter)
        return run_shotgun(x0, minimizer_kwargs, niter)

    monkeypatch.setattr(opt, "_optimize_shotgun", spy)
    opt.optimize(niter=2, rng=0)
    assert calls == expected_calls
    assert opt.objective(opt._optima) == pytest.approx(-1.0)


# ── alternative global backends ───────────────────────────────────────────


@pytest.mark.parametrize("niter", [None, 2])
@pytest.mark.parametrize("backend", ["shgo", "dual_annealing", "brute"])
def test_alternative_backends_locate_minimum(backend, niter):
    """
    The shgo, dual-annealing and brute-force backends each find the unique minimizer of
    the toy objective, using ``_default_hops`` when ``niter`` is not given.
    """

    class Toy(_SquaredDistance):
        _optimization_backend = getattr(BaseNonConvexOptimizer, f"_optimization_{backend}")

    opt = Toy(_two_bits())
    result = opt.optimize(niter=niter, rng=0)
    assert result.success
    assert result.fun == pytest.approx(0.0, abs=1e-10)
    np.testing.assert_allclose(opt._optima, [0.4, 0.3, 0.2, 0.1], atol=1e-6)


@pytest.mark.parametrize("rng", [0, None])
def test_dual_annealing_wiring(monkeypatch, rng):
    """
    Dual annealing receives the optimizer's generator (none without ``rng``), ``niter`` as
    its iteration cap and a callback that stops once the objective bound is reached; a
    valid bound that is never attained leaves the result unchanged.
    """
    captured = {}
    dual_annealing = optimization.dual_annealing

    def spy(**kwargs):
        captured.update(kwargs)
        return dual_annealing(**kwargs)

    monkeypatch.setattr(optimization, "dual_annealing", spy)

    class Toy(_SquaredDistance):
        _optimization_backend = BaseNonConvexOptimizer._optimization_dual_annealing

    np.random.seed(0)
    opt = Toy(_two_bits(), objective_bound=-1.0)
    result = opt.optimize(niter=2, rng=rng)

    if rng is None:
        assert "rng" not in captured
    else:
        assert captured["rng"] is opt._rng
    assert captured["maxiter"] == 2
    assert captured["callback"](result.x, -1.0)
    assert not captured["callback"](result.x, 0.5)
    assert result.success
    np.testing.assert_allclose(opt._optima, [0.4, 0.3, 0.2, 0.1], atol=1e-6)


@pytest.mark.parametrize("backend", ["shgo", "dual_annealing"])
def test_alternative_backends_discard_failures(monkeypatch, backend):
    """An unsuccessful scipy result is discarded, so optimize() finds no optimum."""
    monkeypatch.setattr(optimization, backend, lambda **kwargs: OptimizeResult(x=None, fun=np.inf, success=False))

    class Toy(_SquaredDistance):
        _optimization_backend = getattr(BaseNonConvexOptimizer, f"_optimization_{backend}")

    with pytest.raises(OptimizationException, match="No optima found"):
        Toy(_two_bits()).optimize(niter=1, rng=0)


# ── auxiliary-variable optimizers ─────────────────────────────────────────


def test_auxvar_uniform_initial_is_an_independent_channel():
    """
    The uniform initial condition gives every row of ``p(U | X)`` mass ``1 / bound``, so
    U is independent of X; the copy initial condition, by contrast, makes U a copy of X.
    """
    d = _three_symbols()
    opt = _MaxInformation(d, bound=3)
    mutual_information = opt._mutual_information(opt._rvs, opt._arvs)

    x = opt.construct_uniform_initial()
    assert x.shape == (opt._optvec_size,) == (9,)
    np.testing.assert_allclose(x, 1 / 3)
    joint = opt.construct_joint(x)
    np.testing.assert_allclose(joint.sum(axis=-1), opt._pmf)
    assert mutual_information(joint) == pytest.approx(0.0, abs=1e-12)

    assert mutual_information(opt.construct_joint(opt.construct_copy_initial())) == pytest.approx(H(d))


def test_integer_rng_is_reseeded_on_every_call():
    """
    An integer ``rng`` is re-seeded on each optimize() call, so repeated calls reproduce
    the same optimum; basin hopping reaches the global one (1 bit).
    """
    opt = _MaxInformation(_three_symbols())
    opt.optimize(niter=3, rng=11)
    first = opt._optima.copy()
    opt.optimize(niter=3, rng=11)
    np.testing.assert_array_equal(opt._optima, first)
    assert opt.objective(first) == pytest.approx(-1.0)
