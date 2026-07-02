"""
Constructors and helpers for symbolic (sympy-backed) distributions.
"""

from ..distribution import Distribution

__all__ = (
    "simplify",
    "symbolic_distribution",
    "symbolic_min",
    "symbols",
)


def _require_sympy():
    """Import sympy or raise a helpful error."""
    try:
        import sympy
    except ImportError as err:  # pragma: no cover
        raise ImportError("Symbolic distributions require sympy. Install with: pip install sympy") from err
    return sympy


def symbols(names, positive=True, **assumptions):
    """
    Create sympy symbols suitable for use as probabilities.

    A thin wrapper around :func:`sympy.symbols` that defaults to
    ``positive=True`` (appropriate for probabilities, and helpful for
    simplification).

    Parameters
    ----------
    names : str
        Symbol names, e.g. ``'p'`` or ``'a b c'`` (see :func:`sympy.symbols`).
    positive : bool
        Whether the symbols are assumed positive. Defaults to ``True``.
    **assumptions
        Additional sympy assumptions forwarded to :func:`sympy.symbols`.

    Returns
    -------
    syms : Symbol or tuple of Symbol
    """
    sympy = _require_sympy()
    return sympy.symbols(names, positive=positive, **assumptions)


def symbolic_distribution(outcomes, pmf, rv_names=None, validate=True, **kwargs):
    """
    Construct a :class:`~dit.distribution.Distribution` with symbolic probabilities.

    This is a convenience wrapper around :class:`Distribution` that sympifies
    the ``pmf`` entries so that the resulting distribution stores exact,
    symbolic probabilities.

    Parameters
    ----------
    outcomes : sequence
        The outcomes, as accepted by :class:`Distribution` (e.g. a list of
        strings such as ``['00', '11']``).
    pmf : sequence
        The probabilities, as numbers and/or sympy expressions. Each entry is
        passed through :func:`sympy.sympify`.
    rv_names : list of str, optional
        Names for each random variable.
    validate : bool
        If True, validate normalisation after construction. For pmfs
        containing free symbols normalisation is not decidable and is skipped.
    **kwargs
        Additional keyword arguments forwarded to :class:`Distribution`.

    Returns
    -------
    d : Distribution
        A distribution with ``d.is_symbolic()`` True.
    """
    sympy = _require_sympy()
    sym_pmf = [sympy.sympify(p) for p in pmf]
    return Distribution(
        list(outcomes),
        sym_pmf,
        rv_names=rv_names,
        validate=validate,
        **kwargs,
    )


def simplify(expr, **kwargs):
    """
    Simplify a symbolic measure result.

    A thin wrapper around :func:`sympy.simplify` for convenience, so callers
    need not import sympy directly.

    Parameters
    ----------
    expr : sympy expression
        The expression to simplify (e.g. the return value of a measure).
    **kwargs
        Forwarded to :func:`sympy.simplify`.

    Returns
    -------
    simplified : sympy expression
    """
    sympy = _require_sympy()
    return sympy.simplify(expr, **kwargs)


def symbolic_min(args):
    """Return ``sympy.Min`` over ``args``, robust to unsimplified constants.

    ``sympy.Min`` raises ``ValueError`` when an argument is a constant it cannot
    immediately decide is comparable (e.g. ``2 - log(4)/log(2)``, which equals
    ``0``). Simplifying each argument first resolves such constants while
    leaving genuinely symbolic arguments intact.

    Parameters
    ----------
    args : iterable of sympy expressions
        The values to minimise over.

    Returns
    -------
    m : sympy expression
        ``Min`` of the (simplified) arguments.
    """
    sympy = _require_sympy()
    return sympy.Min(*[sympy.simplify(a) for a in args])


def evaluate(expr, subs):
    """Numerically evaluate a symbolic measure result at a point.

    This is a robust alternative to ``expr.subs(subs)`` for expressions that
    contain ``Min``/``Max`` (as produced by e.g. ``I_min``/``I_mmi`` or CAEKL).
    ``sympy``'s ``Min``/``Max`` can raise ``ValueError`` ("not comparable")
    when a plain ``.subs`` leaves unsimplified constant arguments; evaluating
    through :func:`sympy.lambdify` sidesteps that by comparing the arguments
    numerically.

    Parameters
    ----------
    expr : sympy expression or number
        The expression to evaluate (e.g. a measure result).
    subs : dict
        Mapping of symbols to numeric values.

    Returns
    -------
    value : float
        The numeric value of ``expr`` at ``subs``.
    """
    sympy = _require_sympy()
    if not hasattr(expr, "free_symbols"):
        return float(expr)
    symbols_ = tuple(expr.free_symbols)
    if not symbols_:
        # No free symbols: still route through lambdify to resolve Min/Max.
        func = sympy.lambdify((), expr, "math")
        return float(func())
    values = [subs[s] for s in symbols_]
    func = sympy.lambdify(symbols_, expr, "math")
    return float(func(*values))
