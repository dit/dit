.. renyi_entropy.rst
.. py:module:: dit.other.renyi_entropy

*************
Rényi Entropy
*************

The Rényi entropy is a spectrum of generalizations to the Shannon entropy:

.. math::

   \RE[X] = \frac{1}{1-\alpha} \log_2 \left( \sum{x \in \mathcal{X}} p(x)^\alpha \right)

.. todo::

   discuss special values of alpha: 0, (1/2)?, 1, 2, inf

.. autofunction:: dit.other.renyi_entropy.renyi_entropy
