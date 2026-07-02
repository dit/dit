"""
Symbolic Wyner and Exact common-information solves.

Strategy (per ``Agents/Plans/symbolic-common-information-backend.md``):

1. Analytic short-circuits (Phase 2): the common informations are bounded
   between the dual total correlation and the joint entropy, so when those
   coincide the answer is immediate; products give zero.
2. KKT / reduced-gradient solve (Phase 3): substitute the feasibility manifold
   (marginal-match + simplex) into the objective, then solve the stationarity
   equations over the remaining free unknowns with a layered sympy solver.
3. Cardinality sweep (Phase 4): try increasing ``k`` up to the measure's
   Caratheodory bound, stopping when the numeric optimum saturates.
4. XOR structural ansatz (Phase 5): handled in :mod:`symbolic_ansatz`.

Returns exact sympy expressions (possibly ``Min``/``Piecewise``); raises
:class:`SymbolicOptimizationError` when no closed form is reachable.
"""

import numpy as np

from ...symbolic import evaluate, symbolic_min
from ...symbolic.distributions import _require_sympy
from ..dual_total_correlation import dual_total_correlation
from ..entropy import entropy
from .symbolic_markov import (
    SymbolicOptimizationError,
    build_mixture_model,
    min_feasible_cardinality,
)

__all__ = ("symbolic_common_information",)


# Interior (p, q) grid points reused for the numeric saturation stop rule and
# for filtering critical points. Kept rational-free (floats) since they only
# drive numeric comparisons.
_GRID = (
    {"p": 0.3, "q": 0.2},
    {"p": 0.55, "q": 0.4},
    {"p": 0.7, "q": 0.65},
)

# The generic feasibility solve grows combinatorially with the auxiliary
# cardinality (the polynomial system has ``prod(sizes)`` marginal-match
# equations); past k=2 sympy.solve becomes impractically slow. Larger
# cardinalities are handled by the structural ansatz instead.
_GENERIC_K_CAP = 2


def _log2(expr):
    sympy = _require_sympy()
    return sympy.log(expr) / sympy.log(2)


def _wyner_objective(model):
    """I[X:V] = H(X) - sum_v pi_v * sum_i H(r_i(.|v)), as a sympy expression."""
    sympy = _require_sympy()
    hx = -sum(c * _log2(c) for c in model.joint.ravel() if c != 0)
    cond = 0
    for v in range(model.k):
        h_v = 0
        for arr in model.r:
            size = arr.shape[0]
            h_v += -sum(arr[a, v] * _log2(arr[a, v]) for a in range(size))
        cond += model.pi[v] * h_v
    return sympy.expand(hx) - cond


def _exact_objective(model):
    """H[V] = -sum_v pi_v log2 pi_v, as a sympy expression."""
    return -sum(model.pi[v] * _log2(model.pi[v]) for v in range(model.k))


_OBJECTIVES = {
    "wyner": _wyner_objective,
    "exact": _exact_objective,
}


def _numeric_subs(expr, point, symbol_map):
    """Evaluate ``expr`` at a grid ``point`` (dict of name->float)."""
    subs = {sym: point[name] for name, sym in symbol_map.items() if name in point}
    try:
        return evaluate(expr, subs)
    except (TypeError, ValueError, KeyError):
        return None


def _collect_symbols(dist):
    """Return a ``{name: Symbol}`` map of all free symbols in ``dist``."""
    symbol_map = {}
    if dist.is_symbolic():
        for cell in dist.data.values.ravel():
            for s in getattr(cell, "free_symbols", ()):
                symbol_map[s.name] = s
    return symbol_map


def _grid_points(symbol_map):
    """Yield interior evaluation points restricted to the distribution's symbols."""
    names = set(symbol_map)
    for point in _GRID:
        yield {n: v for n, v in point.items() if n in names}


def _is_zero(expr, symbol_map):
    """
    Whether ``expr`` is identically zero.

    Tries a symbolic ``simplify`` first; when that is inconclusive (common for
    log-sum expressions sympy cannot canonicalise), falls back to numeric
    verification at the interior grid.
    """
    sympy = _require_sympy()
    expr = sympy.sympify(expr)
    if not expr.free_symbols:
        return bool(sympy.simplify(expr) == 0)
    if sympy.simplify(expr) == 0:
        return True
    checked = False
    for point in _grid_points(symbol_map):
        val = _numeric_subs(expr, point, symbol_map)
        if val is None or not np.isfinite(val):
            return False
        if abs(val) > 1e-9:
            return False
        checked = True
    return checked


def _score(expr, symbol_map):
    """Mean objective value across the interior grid, for saturation checks."""
    vals = []
    for point in _grid_points(symbol_map):
        v = _numeric_subs(expr, point, symbol_map)
        if v is not None and np.isfinite(v):
            vals.append(v)
    if not vals:
        return None
    return float(np.mean(vals))


def _rational_in(exprs, symbols):
    """Whether every expr is free of transcendentals in ``symbols`` (log/exp)."""
    sympy = _require_sympy()
    syms = set(symbols)
    for expr in exprs:
        for node in sympy.preorder_traversal(expr):
            if isinstance(node, (sympy.log, sympy.exp)) and node.free_symbols & syms:
                return False
    return True


def _solve_layered(equations, unknowns):
    """Try ``solve``, then ``nonlinsolve``, returning a list of solution dicts."""
    sympy = _require_sympy()
    try:
        sols = sympy.solve(equations, list(unknowns), dict=True)
        if sols:
            return sols
    except Exception:
        pass
    try:
        res = sympy.nonlinsolve(equations, list(unknowns))
        return [dict(zip(unknowns, tup, strict=True)) for tup in res if tup]
    except Exception:
        return []


def _solve_at_cardinality(dist, rvs, measure, k, param_symbols):
    """
    Attempt the symbolic solve at a fixed auxiliary cardinality ``k``.

    Returns
    -------
    expr : sympy expression or None
        The objective value at the optimum, or None if the solve did not close.
    """
    sympy = _require_sympy()
    model = build_mixture_model(dist, rvs=rvs, k=k)
    objective = _OBJECTIVES[measure](model)

    # Substitute the feasibility manifold into the objective.
    feasible = _solve_layered(model.equations, model.symbols)
    if not feasible:
        return None

    candidates = []
    for sol in feasible:
        reduced = objective.subs(sol)
        free = sorted(reduced.free_symbols - set(param_symbols.values()), key=str)
        if not free:
            candidates.append(sympy.simplify(reduced))
            continue
        # Reduced-gradient stationarity over the remaining gauge freedom.
        grads = [sympy.diff(reduced, f) for f in free]
        if not _rational_in(grads, free):
            # Transcendental stationarity (log terms): sympy.solve can hang or
            # fail on these. Defer to the structural ansatz rather than risk it.
            continue
        crit = _solve_layered(grads, free)
        for c in crit:
            # A critical point may still leave some free vars (degenerate
            # manifold); only keep fully-determined, parameter-only values.
            val = reduced.subs(c)
            if val.free_symbols - set(param_symbols.values()):
                continue
            candidates.append(sympy.simplify(val))
        # Also consider the boundary (deterministic channels handled by the
        # short-circuits / ansatz), so a missing interior optimum is not fatal.

    candidates = [c for c in candidates if c is not None]
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    return symbolic_min(candidates)


def symbolic_common_information(dist, measure, rvs=None, crvs=None, bound=None):
    """
    Compute the ``measure`` common information symbolically.

    Parameters
    ----------
    dist : Distribution
        A symbolic distribution.
    measure : str
        ``'wyner'`` or ``'exact'``.
    rvs, crvs : list, None
        Variable specs. ``crvs`` is unsupported in v1.
    bound : int, None
        Optional cap on the auxiliary cardinality sweep.

    Returns
    -------
    ci : sympy expression
        The symbolic common information.
    """
    sympy = _require_sympy()
    if crvs:
        raise NotImplementedError("Symbolic common information does not support crvs (v1).")

    # Phase 2: analytic short-circuits.
    dtc = sympy.sympify(dual_total_correlation(dist, rvs, crvs))
    ent = sympy.sympify(entropy(dist, rvs, crvs))
    symbol_map = _collect_symbols(dist)
    if _is_zero(ent - dtc, symbol_map):
        # CI is squeezed between DTC and H; equal bounds pin it.
        return sympy.simplify(dtc)
    if _is_zero(dtc, symbol_map):
        # The rvs are already independent: no common information is needed.
        return sympy.Integer(0)

    from .symbolic_ansatz import ansatz_common_information
    from .symbolic_markov import _grouped_joint

    _, sizes = _grouped_joint(dist, normalize_sizes_rvs(dist, rvs))
    k_min = min_feasible_cardinality(sizes)
    k_max = int(np.prod(sizes))
    if bound is not None:
        k_max = min(k_max, bound)

    # Phase 5: structural (symmetry-injected) ansatz. This is the reliable path
    # for symmetric sources (e.g. the doubly-symmetric binary source), where the
    # generic KKT stationarity is transcendental and does not close in radicals.
    ansatz = ansatz_common_information(dist, measure, rvs=rvs, symbol_map=symbol_map)
    if ansatz is not None:
        return ansatz

    # Phases 3-4: generic KKT solve, swept over k. The full feasibility solve
    # blows up combinatorially in k (the polynomial system grows as prod(sizes)),
    # so the generic path is capped at ``_GENERIC_K_CAP``; larger cardinalities
    # rely on the ansatz above.
    best = None
    best_score = None
    for k in range(k_min, min(k_max, _GENERIC_K_CAP) + 1):
        expr = _solve_at_cardinality(dist, rvs, measure, k, symbol_map)
        if expr is None:
            continue
        score = _score(expr, symbol_map)
        if best is None or (score is not None and best_score is not None and score < best_score - 1e-9):
            best, best_score = expr, score
        elif score is not None and best_score is not None and score >= best_score - 1e-9:
            # Saturated: increasing k did not improve the numeric optimum.
            break

    if best is None:
        raise SymbolicOptimizationError(
            f"Could not close the symbolic {measure} common information for this distribution."
        )
    return best


def normalize_sizes_rvs(dist, rvs):
    """Return an rvs spec suitable for :func:`_grouped_joint`."""
    from ...helpers import normalize_rvs

    rvs, _ = normalize_rvs(dist, rvs, None)
    return rvs
