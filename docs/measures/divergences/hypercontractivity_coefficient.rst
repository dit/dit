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

.. ipython::

   In [1]: from dit.divergences import hypercontractivity_coefficient

   In [2]: from dit.example_dists import Xor

   @doctest float
   In [3]: hypercontractivity_coefficient(Xor(), [[0], [1]])
   Out[3]: 0.0

API
===

.. autofunction:: hypercontractivity_coefficient
