.. kullback_leiber_divergence.rst
.. py:module:: dit.divergences.kullback_leiber_divergence

***************************
Kullback-Leibler Divergence
***************************

The Kullback-Leibler divergence, sometimes also called the *relative entropy*, is defined as:

.. math::

   \DKL(p || q) = \sum_{x \in \mathcal{X}} p(x) \log_2 \frac{p(x)}{q(x)}

.. todo::

   More discussion, examples.

.. autofunction:: dit.divergences.kullback_leibler_divergence.kullback_leibler_divergence
