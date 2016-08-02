.. intrinsic_mutual_information.rst
.. py:module:: dit.multivariate.intrinsic_mutual_information

****************************
Intrinsic Mutual Information
****************************

The intrinsic mutual information was defined in :cite:`maurer1997intrinsic` as:

.. math::

   \I\left[X:Y\downarrow Z\right] = \min_{p(\overline{z}|z)} \I\left[X:Y|\overline{Z}\right]

The intrinsic mutual information was defined as an upper bound on the rate of secret key agreement.

Information Flow
================

Compared to the transfer entropy, it is also a plausibly superior measure of information flow:

.. math::

   T^\prime_{X \rightarrow Y} = \I\left[X_0^t : Y_t \downarrow Y_0^t\right]

Generalizations
===============

The intrinsic mutual information generalizes in the obvious ways, replacing the :ref:`mutual_information` with most of its generalizations.

Intrinsic Total Correlation
---------------------------

The intrinsic form of the :doc:`total_correlation`:

.. math::

   \T\left[X_{0:n} \downarrow Z\right] = \min_{p(\overline{z}|z)} \T\left[X_{0:n} | \overline{Z}\right]

The following example is given in :cite:`maurer1997intrinsic`:

.. ipython::

   In [1]: from dit.multivariate import intrinsic_total_correlation, total_correlation

   In [2]: d = dit.Distribution(['000', '011', '101', '110', '222', '333'], [1/8]*4 + [1/4]*2)

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

   \B\left[X_{0:n} \downarrow Z\right] = \min_{p(\overline{z}|z)} \B\left[X_{0:n} | \overline{Z}\right]

Intrinsic CAEKL Mutual Information
----------------------------------

The intrinsic form of the :doc:`caekl_mutual_information`:

.. math::

   \J\left[X_{0:n} \downarrow Z\right] = \min_{p(\overline{z}|z)} \J\left[X_{0:n} | \overline{Z}\right]

Intrinsic Co-Information
------------------------

Since the :doc:`coinformation` is not non-negative nor does a zero value imply a lack of :math:`n`-way interactions, it is unclear how to construct an "intrinsic" form of the co-information.

API
===

.. autofunction:: intrinsic_total_correlation

.. autofunction:: intrinsic_dual_total_correlation

.. autofunction:: intrinsic_caekl_mutual_information
