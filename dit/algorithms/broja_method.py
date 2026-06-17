"""
Method selection and fallback routing for bivariate BROJA solvers.
"""

from __future__ import annotations

import logging
from math import prod

from ..exceptions import OptimizationException

__all__ = (
    "ADMUI_MIN_JOINT",
    "SCIPY_MAX_JOINT",
    "broja_solve_bivariate",
    "ecos_available",
    "select_broja_method",
)

logger = logging.getLogger(__name__)

# Tunable via bench/broja_methods_bench.py (initial conservative defaults).
# Quick check (repeat=1): admui beats scipy at |XYZ|>=125 (e.g. uniform 5^3);
# scipy remains faster on small/binary cases (|XYZ|<=64).
SCIPY_MAX_JOINT = 64
ADMUI_MIN_JOINT = 125

_BROJA_METHODS = frozenset({"scipy", "admui", "cone", "auto"})


def ecos_available() -> bool:
    try:
        import ecos  # noqa: F401
    except ImportError:
        return False
    return True


def select_broja_method(alphabet_sizes, method: str = "auto") -> str:
    """
    Choose a bivariate BROJA solver.

    Parameters
    ----------
    alphabet_sizes : sequence of int
        Alphabet sizes on the coalesced (source0, source1, target) distribution.
    method : str
        ``'scipy'``, ``'admui'``, ``'cone'``, or ``'auto'``.
    """
    if method not in _BROJA_METHODS:
        msg = f"Unknown BROJA method {method!r}; expected one of {sorted(_BROJA_METHODS)}."
        raise ValueError(msg)

    if method != "auto":
        if method == "cone" and not ecos_available():
            msg = "method='cone' requires the optional 'ecos' package. Install with: pip install dit[broja]"
            raise ImportError(msg)
        return method

    n_joint = prod(alphabet_sizes)
    if n_joint < SCIPY_MAX_JOINT:
        return "scipy"
    if n_joint >= ADMUI_MIN_JOINT:
        return "admui"
    return "scipy"


def _uniques_from_optimizer(broja, pmf, sources):
    u0 = float(broja._conditional_mutual_information({0}, {2}, {1})(pmf))
    u1 = float(broja._conditional_mutual_information({1}, {2}, {0})(pmf))
    return {sources[0]: u0, sources[1]: u1}


def _solve_scipy(d, sources, target, maxiter, rng):
    from .broja_util import optimized_pmf
    from .distribution_optimizers import BROJABivariateOptimizer

    sources = list(sources)
    target = list(target)
    broja = BROJABivariateOptimizer(d, sources, target)
    broja.optimize(niter=1, maxiter=maxiter, rng=rng)
    pmf = optimized_pmf(broja)
    uniques = _uniques_from_optimizer(broja, pmf, sources)
    meta = {"converged": True, "method": "scipy"}
    return uniques, meta


def _solve_admui(d, sources, target, maxiter):
    from .broja_util import optimized_pmf
    from .distribution_optimizers import BROJAAdmUIOptimizer

    sources = list(sources)
    target = list(target)
    broja = BROJAAdmUIOptimizer(d, sources, target, maxiter=maxiter)
    broja.optimize()
    pmf = optimized_pmf(broja)
    uniques = _uniques_from_optimizer(broja, pmf, sources)
    meta = {"converged": broja._admui_meta["converged"], "method": "admui", "n_iters": broja._admui_meta["n_iters"]}
    return uniques, meta


def _solve_cone(d, sources, target, **ecos_kwargs):
    from .broja_cone import broja_cone_solve

    sources = list(sources)
    target = list(target)
    _, meta = broja_cone_solve(d, sources, target, **ecos_kwargs)
    uniques = {sources[0]: meta["ui_source0"], sources[1]: meta["ui_source1"]}
    meta = {**meta, "method": "cone"}
    return uniques, meta


def _methods_fallback_chain(primary: str) -> list[str]:
    chain = [primary]
    if primary != "scipy":
        chain.append("scipy")
    if ecos_available() and primary != "cone":
        chain.append("cone")
    return chain


def broja_solve_bivariate(d, sources, target, *, maxiter=1000, method="auto", rng=None, **ecos_kwargs):
    """
    Compute BROJA unique informations with automatic fallback.

    Returns
    -------
    uniques : dict
        Unique information per source.
    meta : dict
        Solver metadata including the method actually used.
    """
    from .pid_broja import prepare_dist as broja_prepare_dist

    prepared = broja_prepare_dist(d, list(sources), list(target))
    alphabet_sizes = [len(a) for a in prepared.alphabet]
    primary = select_broja_method(alphabet_sizes, method)

    errors = []
    results = {}

    for meth in _methods_fallback_chain(primary):
        try:
            if meth == "scipy":
                uniques, meta = _solve_scipy(d, sources, target, maxiter, rng)
            elif meth == "admui":
                uniques, meta = _solve_admui(d, sources, target, maxiter)
                if not meta.get("converged", True):
                    raise OptimizationException("admUI did not converge")
            elif meth == "cone":
                uniques, meta = _solve_cone(d, sources, target, **ecos_kwargs)
                if not meta.get("converged", True):
                    raise OptimizationException("cone solver feasibility check failed")
            else:
                continue
            results[meth] = uniques
        except Exception as exc:
            errors.append((meth, exc))
            continue

        if len(results) >= 2:
            refs = list(results.values())
            delta = max(abs(refs[0][s] - refs[1][s]) for s in refs[0])
            if delta > 1e-3 and ecos_available() and meth != "cone":
                logger.warning(
                    "BROJA methods disagree (Δ=%.4g); retrying with cone solver.",
                    delta,
                )
                try:
                    uniques, meta = _solve_cone(d, sources, target, **ecos_kwargs)
                    return uniques, meta
                except Exception:
                    pass

        return uniques, meta

    msg = "All BROJA solvers failed: " + "; ".join(f"{m}: {e}" for m, e in errors)
    raise OptimizationException(msg)
