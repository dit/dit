.. o_information.rst
.. py:module:: dit.multivariate.o_information

.. _o_information:

*************
O-Information
*************

The O-information :cite:`rosas2019quantifying` is the difference between the
:ref:`total_correlation` and the :ref:`dual_total_correlation`:

.. math::

   \O{X_{0:n}} = \T{X_{0:n}} - \B{X_{0:n}}

Positive values indicate that redundant (shared) structure dominates;
negative values indicate that synergistic structure dominates. On a
three-variable giant bit it is :math:`+1` bit; on three-bit parity (xor)
it is :math:`-1` bit:

.. ipython::

   In [1]: from dit.multivariate import o_information

   In [2]: from dit.example_dists import giant_bit, n_mod_m, Xor

   @doctest float
   In [3]: o_information(giant_bit(3, 2))
   Out[3]: 1.0

   @doctest float
   In [4]: o_information(Xor())
   Out[4]: -1.0

The :doc:`cohesion` interpolates between :math:`\T{}` and :math:`\B{}` at
finite order :math:`k`; the O-information is the single-number summary
:math:`T - B`.

API
===

.. autofunction:: o_information
