.. cumulative_residual_entropy.rst
.. py:module:: dit.other.cumulative_residual_entropy

***************************
Cumulative Residual Entropy
***************************

The cumulative residual entropy :cite:`Rao2004` is an alternative to the
differential Shannon entropy. The differential entropy has many issues,
including that it can be negative even for simple distributions such as the
uniform distribution; and that if one takes discrete estimates that limit to the
continuous distribution, the discrete entropy does not limit to the differential
(continuous) entropy. It also attempts to provide meaningful differences between
numerically different random variables, such as a die labeled [1, 2, 3, 4, 5, 6]
and one lebeled [1, 2, 3, 4, 5, 100].

.. note::

   The Cumulative Residual Entropy is unrelated to
   :doc:`../multivariate/residual_entropy`.

.. math::

   \CRE[X] = -\int_{0}^{\infty} p(|X| > x) \log_{2} p(|X| > x) dx

.. ipython::

   In [1]: from dit.other import cumulative_residual_entropy

   In [2]: d1 = dit.ScalarDistribution([1, 2, 3, 4, 5, 6], [1/6]*6)

   In [3]: d2 = dit.ScalarDistribution([1, 2, 3, 4, 5, 100], [1/6]*6)

   @doctest float
   In [4]: cumulative_residual_entropy(d1)
   Out[4]: 2.0683182557028439

   @doctest float
   In [5]: cumulative_residual_entropy(d2)
   Out[5]: 22.672680046016705

Generalized Cumulative Residual Entropy
=======================================

The genearlized form of the cumulative residual entropy integrates over the
intire set of reals rather than just the positive ones:

.. math::

   \GCRE[X] = -\int_{-\infty}^{\infty} p(X > x) \log_{2} p(X > x) dx

.. ipython::

   In [6]: from dit.other import generalized_cumulative_residual_entropy

   @doctest float
   In [7]: generalized_cumulative_residual_entropy(d1)
   Out[7]: 2.0683182557028439

   In [8]: d3 = dit.ScalarDistribution([-2, -1, 0, 1, 2], [1/5]*5)

   @doctest float
   In [9]: cumulative_residual_entropy(d3)
   Out[9]: 0.90656497547719606

   @doctest float
   In [10]: generalized_cumulative_residual_entropy(d3)
   Out[10]: 1.6928786893420307


Conditional Cumulative Residual Entropy
=======================================

The conditional cumulative residual entropy :math:`\CRE[X|Y]` is a distribution
with the same probability mass function as :math:`Y`, and the outcome associated
with :math:`p(y)` is equal to the cumulative residual entropy over probabilities
conditioned on :math:`Y = y`. In this sense the conditional cumulative residual
entropy is more akin to a distribution over :math:`\H[X|Y=y]` than the single
scalar quantity :math:`\H[X|Y]`.

.. math::

   \CRE[X|Y] = - \int_{0}^{\infty} p(|X| > x | Y) \log_{2} p(|X| > x | Y) dx


Conditional Generalized Cumulative Residual Entropy
---------------------------------------------------

Conceptually the conditional generalized cumulative residual entropy is the same
as the non-generalized form, but integrated over the entire real line rather
than just the positive:

.. math::

   \GCRE[X|Y] = - \int_{-\infty}^{\infty} p(X > x | Y) \log_{2} p(X > x | Y) dx


API
===

.. autofunction:: cumulative_residual_entropy

.. autofunction:: generalized_cumulative_residual_entropy


Conditional Forms
-----------------

.. autofunction:: conditional_cumulative_residual_entropy

.. autofunction:: conditional_generalized_cumulative_residual_entropy
