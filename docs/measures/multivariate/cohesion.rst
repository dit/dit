.. cohesion.rst
.. py:module:: dit.multivariate.cohesion

********
Cohesion
********

The cohesion :cite:`rosas2016understanding` is a parameterized multivariate
mutual information which interpolates between the :ref:`total_correlation`
(:math:`k = 1`) and the :ref:`dual_total_correlation` (:math:`k = n-1`):

.. math::

   C_k[X_0 : X_1 : \dotsc : X_n] = \sum_{\substack{A \subset [n]}{|A| = k}} \H{X_A} - \binom{n - 1}{k - 1} \H{X_0, X_1, \dotsc, X_n}

The later O-information of :cite:`rosas2019quantifying` is the single
difference :math:`\T{} - \B{}`; see :doc:`o_information`.

On a three-variable giant bit, :math:`C_1 = 2` (total correlation) and
:math:`C_2 = 1` (dual total correlation):

.. ipython::

   In [1]: from dit.multivariate import cohesion

   In [2]: from dit.example_dists import giant_bit

   In [3]: d = giant_bit(3, 2)

   @doctest float
   In [4]: cohesion(d, 1)
   Out[4]: 2.0

   @doctest float
   In [5]: cohesion(d, 2)
   Out[5]: 1.0

API
===

.. autofunction:: cohesion
