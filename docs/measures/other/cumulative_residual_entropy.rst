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

The genearlized form integrates over the intire set of reals rather than just
the positive ones:

.. math::

   \GCRE[X] = -\int_{-\infty}^{\infty} p(X > x) \log_{2} p(X > x) dx

.. todo::

   Compute some examples.

.. todo::

   discuss the conditional forms.


.. autofunction:: cumulative_residual_entropy

.. autofunction:: generalized_cumulative_residual_entropy


Conditional Cumulative Residual Entropy
=======================================

.. autofunction:: conditional_cumulative_residual_entropy

.. autofunction:: conditional_generalized_cumulative_residual_entropy
