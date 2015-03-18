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

.. todo::

   Come up with some examples to verify the claim about coupling.


API
===

.. autofunction:: tse_complexity
