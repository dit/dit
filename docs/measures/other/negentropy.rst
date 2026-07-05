.. negentropy.rst
.. py:module:: dit.other.negentropy

**********
Negentropy
**********

The negentropy :cite:`brillouin1953negentropy` measures how far a distribution is from uniformity. It is defined as the difference between the entropy of a uniform distribution over the same alphabet and the entropy of the distribution itself:

.. math::

   \N{X} = \sum_{i} \log_2 |\mathcal{X}_i| - \H{X}

where :math:`|\mathcal{X}_i|` is the cardinality of the alphabet of the :math:`i`\ th random variable. Since the uniform distribution maximizes the entropy, the negentropy is non-negative, and is zero if and only if the distribution is uniform.

Unlike most measures in :mod:`dit`, the negentropy depends on the cardinality of the alphabet and not just the probabilities. For example, the ``xor`` distribution is one bit away from uniform:

.. ipython::

   In [1]: from dit.other import negentropy

   In [2]: from dit.example_dists import Xor

   @doctest float
   In [3]: negentropy(Xor())
   Out[3]: 1.0


API
===

.. autofunction:: negentropy
