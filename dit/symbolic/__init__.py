"""
Symbolic (sympy-backed) distributions and helpers.

This subpackage lets you build :class:`~dit.distribution.Distribution` objects
whose probabilities are sympy expressions rather than floats. Information
measures (entropy, mutual information, and the multivariate/closed-form family)
then return sympy expressions that can be manipulated and simplified exactly.

Example
-------
>>> from dit.symbolic import symbolic_distribution, symbols
>>> from dit.multivariate import total_correlation
>>> p = symbols('p')
>>> d = symbolic_distribution(['00', '11'], [p, 1 - p])
>>> total_correlation(d)  # doctest: +SKIP
-p*log(p)/log(2) + (p - 1)*log(1 - p)/log(2)
"""

from .distributions import (
    evaluate,
    simplify,
    symbolic_distribution,
    symbolic_min,
    symbols,
)

__all__ = (
    "evaluate",
    "simplify",
    "symbolic_distribution",
    "symbolic_min",
    "symbols",
)
