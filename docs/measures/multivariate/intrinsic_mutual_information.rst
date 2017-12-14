.. intrinsic_mutual_information.rst
.. py:module:: dit.multivariate.intrinsic_mutual_information

****************************
Intrinsic Mutual Information
****************************

The intrinsic mutual information was defined in :cite:`maurer1997intrinsic` as:

.. math::

   \I{X : Y \downarrow Z} = \min_{p(\overline{z} | z)} \I{X : Y | \overline{Z}}

The intrinsic mutual information was defined as an upper bound on the rate of secret key agreement.

Generalizations
===============

The intrinsic mutual information generalizes in the obvious ways, replacing the :ref:`mutual_information` with most of its generalizations.

Intrinsic Total Correlation
---------------------------

The intrinsic form of the :doc:`total_correlation`:

.. math::

   \T{X_{0:n} \downarrow Z} = \min_{p(\overline{z} | z)} \T{X_{0:n} | \overline{Z}}

The following example is given in :cite:`maurer1997intrinsic`:

.. ipython::

   In [1]: from dit.multivariate import intrinsic_total_correlation, total_correlation

   In [2]: d = dit.example_dists.intrinsic import intrinsic_1

   @doctest float
   In [3]: total_correlation(d, [[0], [1]], [2])
   Out[3]: 0.5

   @doctest float
   In [4]: intrinsic_total_correlation(d, [[0], [1]], [2])
   Out[4]: 0.0

We see that although the `xor` part of the distribution contributes 0.5 bits of conditional dependence, it can be "fuzzed out" in the intrinsic mutual information.

Intrinsic Dual Total Correlation
--------------------------------

The intrinsic form of the :doc:`dual_total_correlation`:

.. math::

   \B{X_{0:n} \downarrow Z} = \min_{p(\overline{z} | z)} \B{X_{0:n} | \overline{Z}}

Intrinsic CAEKL Mutual Information
----------------------------------

The intrinsic form of the :doc:`caekl_mutual_information`:

.. math::

   \J{X_{0:n} \downarrow Z} = \min_{p(\overline{z} | z)} \J{X_{0:n} | \overline{Z}}

Intrinsic Co-Information
------------------------

Since the :doc:`coinformation` is not non-negative nor does a zero value imply a lack of :math:`n`-way interactions, it is unclear how to construct an "intrinsic" form of the co-information.

API
===

.. autofunction:: intrinsic_total_correlation

.. autofunction:: intrinsic_dual_total_correlation

.. autofunction:: intrinsic_caekl_mutual_information
