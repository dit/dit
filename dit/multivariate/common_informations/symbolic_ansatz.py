"""
Structural (symmetry-injected) ansatz solves for common information.

The generic KKT stationarity for the common informations is transcendental (the
objective carries ``log`` terms), so ``sympy.solve`` does not close it in
radicals for the interesting symmetric sources. For these, injecting the known
symmetry collapses the problem to a small algebraic system that *does* close.

v1 covers the doubly-symmetric binary source (DSBS): two binary variables with
joint

    p(00) = p(11) = (1 - a) / 2,   p(01) = p(10) = a / 2

whose Wyner common information admits the closed form

    C_wyner = 1 + h(a) - 2 h(a0),   a0 (1 - a0) = a / 2,   a0 in (0, 1/2)

where ``h`` is the binary entropy (bits). Here the optimal auxiliary is a fair
bit ``V`` feeding two identical binary symmetric channels with crossover ``a0``.
The *Exact* common information of the DSBS does not reduce to this form (its
optimal auxiliary need not be uniform) and is left to the generic solver, which
in turn may fail to close it — an honest ``None``/error is preferred to a wrong
closed form.

Sources that do not match a known template return ``None`` so the caller can
fall back to the generic solver or raise.
"""

from ...symbolic.distributions import _require_sympy
from .symbolic_markov import _grouped_joint

__all__ = ("ansatz_common_information",)


def _binary_entropy(x):
    """Binary entropy ``h(x)`` in bits."""
    sympy = _require_sympy()
    return -x * sympy.log(x) / sympy.log(2) - (1 - x) * sympy.log(1 - x) / sympy.log(2)


def _match_dsbs(joint):
    """
    If ``joint`` is a 2x2 doubly-symmetric binary source, return its crossover
    parameter ``a`` (a sympy expression); otherwise ``None``.

    A DSBS has ``joint == [[(1-a)/2, a/2], [a/2, (1-a)/2]]``: equal diagonal
    entries, equal off-diagonal entries, and the two differing.
    """
    sympy = _require_sympy()
    if tuple(joint.shape) != (2, 2):
        return None
    p00, p01 = joint[0, 0], joint[0, 1]
    p10, p11 = joint[1, 0], joint[1, 1]
    if sympy.simplify(p00 - p11) != 0:
        return None
    if sympy.simplify(p01 - p10) != 0:
        return None
    if sympy.simplify(p00 - p01) == 0:
        return None  # uniform: independent, handled by the short-circuit
    # a = p01 + p10 (total off-diagonal mass).
    a = sympy.simplify(p01 + p10)
    return a


def ansatz_common_information(dist, measure, rvs=None, symbol_map=None):
    """
    Attempt a structural-ansatz closed form for the ``measure`` common
    information of ``dist``.

    Parameters
    ----------
    dist : Distribution
        A symbolic distribution.
    measure : str
        ``'wyner'`` or ``'exact'``.
    rvs : list, None
        Variable specs (single-variable groups only in v1).
    symbol_map : dict, None
        Unused placeholder for symmetry with the generic solver signature.

    Returns
    -------
    ci : sympy expression or None
        The closed form if a template matched, else ``None``.
    """
    sympy = _require_sympy()
    from .symbolic_solve import normalize_sizes_rvs

    joint, _ = _grouped_joint(dist, normalize_sizes_rvs(dist, rvs))

    a = _match_dsbs(joint)
    if a is not None and measure == "wyner":
        # Wyner CI of the DSBS (Wyner 1975): the optimal auxiliary is a fair bit
        # feeding two identical binary symmetric channels with crossover a0,
        # where a0 (1 - a0) = a / 2 and a0 in (0, 1/2).
        a0 = sympy.Rational(1, 2) - sympy.sqrt(1 - 2 * a) / 2
        return sympy.simplify(1 + _binary_entropy(a) - 2 * _binary_entropy(a0))

    # NOTE: the Exact common information of the DSBS does *not* reduce to the
    # Wyner form (its optimal auxiliary need not be a fair bit); it has no simple
    # closed form here. Return None so the caller fails honestly rather than
    # emitting a wrong expression.
    return None
