.. exact_common_information.rst
.. py:module:: dit.multivariate.exact_common_information

************************
Exact Common Information
************************

The exact common information :cite:`kumar2014exact` is the entropy of the smallest variable :math:`V` which renders all variables of interest independent:

.. math::

   \G[X_{0:n}|Y_{0:m}] = \min_{\ind X_{0:n} \mid Y_{0:m}, V} \H[V | Y_{0:m}]

Subadditivity of Independent Variables
======================================

Kumar **et. al.** :cite:`kumar2014exact` have shown that the exact common information of a pair of independent pairs of variables can be less than the sum of their individual exact common informations. Here we verify this claim:

.. ipython::

   In [1]: from dit.multivariate import exact_common_information as G

   In [2]: d = dit.Distribution([(0,0), (0,1), (1,0)], [1/3]*3)

   In [3]: d2 = d.__matmul__(d) # RTD doesn't use python 3

   In [4]: print(d2)
   Class:          Distribution
   Alphabet:       (0, 1) for all rvs
   Base:           linear
   Outcome Class:  tuple
   Outcome Length: 4
   RV Names:       None

   x              p(x)
   (0, 0, 0, 0)   1/9
   (0, 0, 0, 1)   1/9
   (0, 0, 1, 0)   1/9
   (0, 1, 0, 0)   1/9
   (0, 1, 0, 1)   1/9
   (0, 1, 1, 0)   1/9
   (1, 0, 0, 0)   1/9
   (1, 0, 0, 1)   1/9
   (1, 0, 1, 0)   1/9

   @doctest float
   In [5]: 2*G(d)
   Out[5]: 1.8365916681089791

   @doctest float
   In [6]: G(d2, [[0, 2], [1, 3]])
   Out[6]: 1.7527152736283176

API
===

.. autofunction:: exact_common_information
