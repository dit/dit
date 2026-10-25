.. dual_total_correlation.rst
.. py:currentmodule:: dit.multivariate

.. _dual_total_correlation:
.. _dual total correlation:

**********************
Dual Total Correlation
**********************

The dual total correlation :cite:`Han1975linear`, or binding information :cite:`Abdallah2012`, is yet another generalization of the :ref:`mutual_information`. It is the amount of information that is shared among the variables. It is defined as:

.. math::

   \B{X_{0:n}} &= \H{X_{0:n}} - \sum \H{X_i | X_{\{0..n\}/i}} \\
               &= - \sum_{x_{0:n} \in X_{0:n}} p(x_{0:n}) \log_2 \frac{p(x_{0:n})}{\prod p(x_i|x_{\{0:n\}/i})}

In a sense the binding information captures the same information that the :doc:`/measures/multivariate/total_correlation` does, in that both measures are zero or non-zero together. However, the two measures take on very different quantitative values for different distributions. By way of example, the type of distribution that maximizes the total correlation is a "giant bit":

.. ipython::

   In [1]: from dit.multivariate import binding_information, total_correlation

   In [2]: d = dit.Distribution(['000', '111'], [1/2, 1/2])

   @doctest float
   In [3]: total_correlation(d)
   Out[3]: 2.0

   @doctest float
   In [4]: binding_information(d)
   Out[4]: 1.0

For the same distribution, the dual total correlation takes on a relatively low value. On the other hand, the type of distribution that maximizes the dual total correlation is a "parity" distribution:

.. ipython::

   In [5]: from dit.example_dists import n_mod_m

   In [6]: d = n_mod_m(3, 2)

   @doctest float
   In [7]: total_correlation(d)
   Out[7]: 1.0

   @doctest float
   In [8]: binding_information(d)
   Out[8]: 2.0


Relationship to Other Measures
==============================

The dual total correlation obeys particular bounds related to both the :doc:`/measures/multivariate/entropy` and the :doc:`/measures/multivariate/total_correlation`:

.. math::

   0 \leq & \B{X_{0:n}} \leq \H{X_{0:n}} \\
   \frac{\T{X_{0:n}}}{n-1} \leq & \B{X_{0:n}} \leq (n-1)\T{X_{0:n}}


Visualization
=============

The binding information, as seen below, consists equally of the information shared among the variables.

.. image:: /images/idiagrams/b_xyz.png
   :alt: The dual total correlation :math:`\B{X : Y : Z}`
   :width: 357px
   :align: center


Generalized Dual Total Correlation
==================================

The dual total correlation is the sum of every atom of the information diagram
shared by at least two variables. Allowing a different threshold gives the
generalized dual total correlation: the sum of all (conditional)
co-informations over at least ``order`` variables. Order 1 is the joint
entropy, order 2 is the dual total correlation, and order :math:`n` is the
:ref:`coinformation`:

.. ipython::

   In [9]: from dit.multivariate import generalized_dual_total_correlation

   In [10]: xor = dit.example_dists.Xor()

   @doctest float
   In [11]: generalized_dual_total_correlation(xor, 1)
   Out[11]: 2.0

   @doctest float
   In [12]: generalized_dual_total_correlation(xor, 2)
   Out[12]: 2.0

   @doctest float
   In [13]: generalized_dual_total_correlation(xor, 3)
   Out[13]: -1.0


API
===

:py:func:`binding_information` is an alias for :py:func:`dual_total_correlation`.

.. autofunction:: dual_total_correlation
   :no-index:

.. autofunction:: generalized_dual_total_correlation
   :no-index:
