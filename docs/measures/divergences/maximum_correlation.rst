.. maximum_correlation.rst
.. py:module:: dit.divergences.maximum_correlation

*******************
Maximum Correlation
*******************

The Hirschfeld–Gebelein–Rényi maximal correlation of a pair of random
variables is

.. math::

   \rho_m(X:Y) = \max_{f,g} \mathbb{E}[f(X)g(Y)]

subject to zero-mean, unit-variance :math:`f` and :math:`g`. It is 1 if the
variables are a deterministic function of each other (a giant bit) and 0 if
they are independent.

.. ipython::

   In [1]: from dit.divergences import maximum_correlation

   In [2]: from dit.example_dists import giant_bit, Xor

   @doctest
   In [3]: abs(maximum_correlation(giant_bit(2, 2), [[0], [1]]) - 1.0) < 1e-8
   Out[3]: True

   @doctest
   In [4]: abs(maximum_correlation(Xor(), [[0], [1]])) < 1e-10
   Out[4]: True

API
===

.. autofunction:: maximum_correlation
