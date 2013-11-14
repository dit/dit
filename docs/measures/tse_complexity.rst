.. tse_complexity.rst
.. py:module::dit.algorithms.tse_complexity

**************
TSE Complexity
**************

The Tononi-Sporns-Edelmans (TSE) complexity :cite:`Tononi1994` is a complexity
measure for distributions. It is designed so that it maximized by distributions
where small subsets of random variables are loosely coupled but the overall
distribution is tightly coupled.

.. math::

   \TSE[X] = \sum_{k=1}^{|X|} \left( {N \choose k}^{-1} \sum_\stackrel{y \subseteq X}{|y|=k} \left( \H[y] \right) - \frac{k}{|X|}\H[X] \right)

.. todo::

   Come up with some examples to verify the claim about coupling.

.. autofunction:: dit.algorithms.tse_complexity.tse_complexity
