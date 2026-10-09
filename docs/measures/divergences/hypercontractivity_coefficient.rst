.. hypercontractivity_coefficient.rst
.. py:module:: dit.divergences.hypercontractivity_coefficient

******************************
Hypercontractivity Coefficient
******************************

The hypercontractivity coefficient :cite:`beigi2016phi` of a pair of variables
is

.. math::

   s^*(X \Vert Y) = \max_{U - X - Y} \frac{I(U:Y)}{I(U:X)}

It is zero when :math:`X` and :math:`Y` are independent.

The supremum is often approached only as :math:`U` becomes independent of :math:`X`, where the ratio tends to the squared maximal correlation, a lower bound on :math:`s^*` :cite:`anantharam2013maximal`.
Equivalently, :math:`s^*` is the least :math:`\lambda` for which :math:`H(Y) - \lambda H(X)`, as a function of the distribution of :math:`X`, touches its lower convex envelope at :math:`p(x)` :cite:`anantharam2013maximal`.
For :math:`|X| \leq 4`, ``dit`` also evaluates this envelope on a grid and takes the best of the resulting bound, the squared maximal correlation, and the direct optimization.

.. ipython::

   In [1]: from dit.divergences import hypercontractivity_coefficient

   In [2]: from dit.example_dists import Xor

   @doctest float
   In [3]: hypercontractivity_coefficient(Xor(), [[0], [1]])
   Out[3]: 0.0

API
===

.. autofunction:: hypercontractivity_coefficient
