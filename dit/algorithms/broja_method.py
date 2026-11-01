"""
Method selection and fallback routing for bivariate BROJA solvers.
"""

from __future__ import annotations

from ..exceptions import OptimizationException

__all__ = (
    "broja_solve_bivariate",
    "ecos_available",
    "select_broja_method",
)

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

    ``'auto'`` picks the exact exponential cone program when ECOS is installed, and
    admUI otherwise. SLSQP (``'scipy'``) is only used when asked for: it often stops
    short of the minimum, overstating the unique information.

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

    return "cone" if ecos_available() else "admui"


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
    """The primary method, then the others from the most reliable to the least."""
    order = ["cone", "admui", "scipy"] if ecos_available() else ["admui", "scipy"]
    return [primary] + [meth for meth in order if meth != primary]


# the options of the ECOS solver (https://github.com/embotech/ecos-python)
ECOS_OPTIONS = frozenset(
    {
        "verbose",
        "feastol",
        "abstol",
        "reltol",
        "feastol_inacc",
        "abstol_inacc",
        "reltol_inacc",
        "max_iters",
        "nitref",
    }
)


def broja_solve_bivariate(d, sources, target, *, maxiter=1000, method="auto", rng=None, **ecos_kwargs):
    """
    Compute BROJA unique informations with automatic fallback.

    If a solver raises, the next most reliable one is tried. admUI returns its last
    iterate, which matches the marginals, even if it has not converged within
    `maxiter` iterations.

    Returns
    -------
    uniques : dict
        Unique information per source.
    meta : dict
        Solver metadata including the method actually used.
    """
    from .pid_broja import prepare_dist as broja_prepare_dist

    unknown = sorted(set(ecos_kwargs) - ECOS_OPTIONS)
    if unknown:
        # the cone solver would fail on them, and the fallback hide the mistake
        msg = f"unknown options {unknown}; the solver options are those of ECOS: {sorted(ECOS_OPTIONS)}"
        raise TypeError(msg)

    prepared = broja_prepare_dist(d, list(sources), list(target))
    alphabet_sizes = [len(a) for a in prepared.alphabet]
    primary = select_broja_method(alphabet_sizes, method)

    errors = []

    for meth in _methods_fallback_chain(primary):
        try:
            if meth == "scipy":
                uniques, meta = _solve_scipy(d, sources, target, maxiter, rng)
            elif meth == "admui":
                uniques, meta = _solve_admui(d, sources, target, maxiter)
            elif meth == "cone":
                uniques, meta = _solve_cone(d, sources, target, **ecos_kwargs)
                if not meta.get("converged", True):
                    raise OptimizationException("cone solver feasibility check failed")
            else:
                continue
        except Exception as exc:
            errors.append((meth, exc))
            continue

        return uniques, meta

    msg = "All BROJA solvers failed: " + "; ".join(f"{m}: {e}" for m, e in errors)
    raise OptimizationException(msg)
