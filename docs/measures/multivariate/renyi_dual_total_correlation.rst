.. renyi_dual_total_correlation.rst
.. py:currentmodule:: dit.multivariate

.. _renyi_dual_total_correlation:

*****************************
Rényi Dual Total Correlation
*****************************

The Rényi dual total correlation generalizes the :ref:`dual_total_correlation` :cite:`Han1975linear` by replacing each Shannon entropy with a :doc:`/measures/other/renyi_entropy` of order :math:`\alpha` :cite:`berta2015renyi`:

.. math::

   \op{B_\alpha}{X_{0:n}} = \sum_{i} \RE{\alpha}{X_{\{0:n\} \setminus i}} - (n-1) \RE{\alpha}{X_{0:n}}

At :math:`\alpha = 1` this is exactly the dual total correlation.

.. ipython::

   In [1]: from dit.multivariate import renyi_dual_total_correlation as B

   In [2]: d = dit.example_dists.And()

   @doctest float
   In [3]: B(d, 1)
   Out[3]: 1.0

   @doctest float
   In [4]: B(d, 2)
   Out[4]: 0.8300749985576878

Avoiding a Conditional Rényi Entropy
====================================

The dual total correlation is usually written using conditional entropies:

.. math::

   \B{X_{0:n}} = \H{X_{0:n}} - \sum_{i} \H{X_i \mid X_{\{0:n\} \setminus i}}

That form cannot be carried over directly, because there is no single agreed-upon definition of a conditional Rényi entropy. Instead, ``dit`` accumulates the value from :ref:`renyi_total_correlation` using the identity

.. math::

   \op{B_\alpha}{X_{0:n}} = (n-1) \op{T_\alpha}{X_{0:n}} - \sum_{i} \op{T_\alpha}{X_{\{0:n\} \setminus i}}

Expanding each total correlation makes the single-variable terms cancel, leaving the definition above. Every term is therefore a *joint* Rényi entropy, and no conditional Rényi entropy is needed.

.. ipython::

   In [5]: from dit.multivariate import renyi_total_correlation as T

   In [6]: rvs = [[0], [1], [2]]

   @doctest float
   In [7]: 2 * T(d, 2) - sum(T(d, 2, rvs=rvs[:i] + rvs[i+1:]) for i in range(3))
   Out[7]: 0.8300749985576878

API
===

.. autofunction:: renyi_dual_total_correlation
   :no-index:
