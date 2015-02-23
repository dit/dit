.. cumulative_residual_entropy.rst
.. py:module:: dit.other.cumulative_residual_entropy

*******
Cumulative Residual Entropy
*******

The cumulative residual entropy :cite:`Rao2004` is an alternative to the
differential Shannon entropy. The differential entropy has many issues,
including that it can be negative even for simple distributions such as the
uniform distribution; and that if one takes discrete estimates that limit to the
continuous distribution, the discrete entropy does not limit to the differential
(continuous) entropy. It also attempts to provide meaningful differences between
numerically different random variables, such as a die labeled [1, 2, 3, 4, 5, 6]
and one lebeled [1, 2, 3, 4, 5, 100].

.. math::

   \CRE[X] = -\sum_{x \in X} p(|X| > x) \log_2 p(|X| > x)

.. todo::

   Generalized form.

.. todo::

   Compute some examples.

.. autofunction:: dit.other.cumulative_residual_entropy.cumulative_residual_entropy
