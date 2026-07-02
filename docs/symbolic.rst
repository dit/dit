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

Optimization-based common informations
======================================

The Wyner and Exact common informations are *variational* (a minimisation over
an auxiliary variable) and have no general closed form, so the numeric backends
use iterative solvers. A best-effort ``backend="symbolic"`` is nonetheless
available for **small** symbolic distributions:

.. ipython::

   In [1]: from dit.multivariate import wyner_common_information

   In [2]: a = symbols('a')

   In [3]: dsbs = symbolic_distribution(['00', '01', '10', '11'],
      ...:                              [(1 - a) / 2, a / 2, a / 2, (1 - a) / 2])

   In [4]: wyner_common_information(dsbs, backend="symbolic")  # doctest: +SKIP

The symbolic backend proceeds by (a) analytic short-circuits — the common
informations are squeezed between the dual total correlation and the joint
entropy, so equal bounds (e.g. a giant bit, the XOR source) or independence
give the answer immediately; (b) a generic KKT / reduced-gradient solve at
small auxiliary cardinality; and (c) structural (symmetry-injected) ansätze for
recognised symmetric sources such as the doubly-symmetric binary source (whose
Wyner common information is :math:`1 + h(a) - 2h(a_0)` with
:math:`a_0(1 - a_0) = a/2`). A symbolic distribution is routed to this backend
automatically; pass ``backend="symbolic"`` explicitly for a numeric
distribution.

When no closed form is reachable (e.g. the Exact common information of the
doubly-symmetric binary source, which has no simple closed form), the backend
raises ``SymbolicOptimizationError`` rather than returning an approximation.
Other optimization-based measures (BROJA, ``I_proj``/``I_IG``/``I_GH``, and the
secret-key-agreement rates) remain numeric-only.

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
