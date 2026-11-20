.. renyi_total_correlation.rst
.. py:currentmodule:: dit.multivariate

.. _renyi_total_correlation:

***********************
Rényi Total Correlation
***********************

The Rényi total correlation generalizes the :ref:`total_correlation` :cite:`watanabe1960information` by replacing each Shannon entropy with a :doc:`/measures/other/renyi_entropy` of order :math:`\alpha` :cite:`berta2015renyi`:

.. math::

   \op{T_\alpha}{X_{0:n}} = \sum_{i} \RE{\alpha}{X_i} - \RE{\alpha}{X_{0:n}}

At :math:`\alpha = 1` this is exactly the total correlation.

.. ipython::

   In [1]: from dit.multivariate import renyi_total_correlation as T

   In [2]: d = dit.example_dists.And()

   @doctest float
   In [3]: T(d, 1)
   Out[3]: 0.8112781244591329

   @doctest float
   In [4]: T(d, 2)
   Out[4]: 0.6780719051126378

Only joint Rényi entropies appear, so no conditional Rényi entropy is needed. This is what lets the :ref:`renyi_dual_total_correlation` and the :ref:`renyi_caekl_mutual_information` be built on top of it.

Independent variables share nothing at any order:

.. ipython::

   In [5]: d = dit.Distribution(['00', '01', '10', '11'], [1/4]*4)

   @doctest float
   In [6]: T(d, 5)
   Out[6]: 0.0

API
===

.. autofunction:: renyi_total_correlation
   :no-index:
