.. symbolic.rst

********
Symbolic
********

In addition to numeric (floating-point) probabilities, :mod:`dit` supports
*symbolic* distributions whose probabilities are `sympy
<https://www.sympy.org>`_ expressions. Information measures then return sympy
expressions that can be manipulated, evaluated, and simplified exactly.

Symbolic support requires ``sympy`` (an optional dependency)::

   pip install dit[symbolic]

Constructing a symbolic distribution
====================================

The :mod:`dit.symbolic` subpackage provides convenience constructors. Use
:func:`~dit.symbolic.symbols` to create probability symbols (positive by
default) and :func:`~dit.symbolic.symbolic_distribution` to build a
distribution:

.. ipython::

   In [1]: from dit.symbolic import symbolic_distribution, symbols, simplify

   In [2]: p = symbols('p')

   In [3]: d = symbolic_distribution(['0', '1'], [p, 1 - p])

   In [4]: d.is_symbolic()
   Out[4]: True

The pmf preserves the symbolic values rather than coercing them to floats:

.. ipython::

   In [5]: list(d.pmf)
   Out[5]: [p, 1 - p]

Computing measures
==================

Shannon and the multivariate/closed-form measures return sympy expressions.
For example, the entropy of a symbolic coin is the binary entropy function:

.. ipython::

   In [6]: from dit.multivariate import entropy, total_correlation

   In [7]: entropy(d)

Results are returned *unsimplified*; call :func:`~dit.symbolic.simplify` (a thin
wrapper around :func:`sympy.simplify`) when a canonical form is desired, or
substitute a concrete value with :meth:`sympy.Expr.subs`:

.. ipython::

   In [8]: entropy(d).subs(p, sympy.Rational(1, 2))
   Out[8]: 1

For measures that involve a minimum (e.g. ``I_min``, ``I_mmi``, CAEKL), a plain
``.subs`` can occasionally fail with a sympy "not comparable" error when it
leaves unsimplified constant arguments inside a ``Min``. Use
:func:`~dit.symbolic.evaluate` to numerically evaluate such a result at a point
robustly:

.. ipython::

   In [9]: from dit.symbolic import evaluate

   In [10]: evaluate(entropy(d), {p: 0.25})

For a "giant bit" (two perfectly correlated bits), the joint entropy, mutual
information, and total correlation all equal :math:`H(p)`:

.. ipython::

   In [9]: gb = symbolic_distribution(['00', '11'], [p, 1 - p])

   In [10]: simplify(total_correlation(gb))

Supported measures
==================

Symbolic computation is supported for the closed-form measures:

- **Shannon**: entropy, conditional entropy, mutual information.
- **Multivariate**: co-information, total correlation, dual total correlation,
  interaction information, O-information, TSE complexity, cohesion.
- **Divergences**: cross entropy, Kullback-Leibler divergence.
- **Common informations**: Gács-Körner and the other combinatorial forms.
- **PID**: the closed-form redundancy measures (e.g. ``I_min``, ``I_mmi``).

Optimization-based measures (e.g. Wyner/exact common information, BROJA, and
the secret-key-agreement rates) are **not** available symbolically: their
optima have no closed form and continue to require numeric solvers.

Notes and limitations
=====================

- Only **linear** probability space is supported for symbolic distributions.
- Because the sign of a free symbol is not decidable, a probability is treated
  as a structural zero only when it is *literally* zero. Normalisation is not
  checked when free symbols are present.
- Simplification of logarithms (e.g. recognising ``log(1/p) == -log(p)``) may
  require sympy assumptions; probability symbols created via
  :func:`~dit.symbolic.symbols` are ``positive=True`` to help.

.. py:currentmodule:: dit.symbolic

API
===

.. autofunction:: symbolic_distribution

.. autofunction:: symbols

.. autofunction:: simplify

.. autofunction:: evaluate
