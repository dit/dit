.. tse_complexity.rst
.. py:module:: dit.multivariate.tse_complexity

**************
TSE Complexity
**************

The Tononi-Sporns-Edelmans (TSE) complexity :cite:`Tononi1994` is a complexity
measure for distributions. It is designed so that it maximized by distributions
where small subsets of random variables are loosely coupled but the overall
distribution is tightly coupled.

.. math::

   \TSE[X|Z] = \sum_{k=1}^{|X|} \left( {N \choose k}^{-1} \sum_{\substack{y \subseteq X \\ |y|=k}} \left( \H[y|Z] \right) - \frac{k}{|X|}\H[X|Z] \right)

Two distributions which might be considered tightly coupled are the "giant bit"
and the "parity" distributions:

.. ipython::

   In [54]: from dit.multivariate import tse_complexity

   In [55]: from dit.example_dists import Xor

   In [56]: d1 = Xor()

   @doctest float
   In [57]: tse_complexity(d1)
   Out[57]: 1.0

   In [58]: d2 = dit.Distribution(['000', '111'], [1/2, 1/2])

   @doctest float
   In [59]: tse_complexity(d2)
   Out[59]: 1.0

The TSE Complexity assigns them both a value of :math:`1.0` bits, which is the
maximal value the TSE takes over trivariate, binary alphabet distributions.

API
===

.. autofunction:: tse_complexity
