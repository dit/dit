.. kullback_leiber_divergence.rst
.. py:module:: dit.divergences.kullback_leibler_divergence

***************************
Kullback-Leibler Divergence
***************************

The Kullback-Leibler divergence, sometimes also called the *relative entropy*,
of a distribution :math:`p` from a distribution :math:`q` is defined as:

.. math::

   \DKL[p || q] = \sum_{x \in \mathcal{X}} p(x) \log_2 \frac{p(x)}{q(x)}

The Kullback-Leibler divergence quantifies the average number of *extra* bits
required to represent a distribution :math:`p` when using an arbitrary
distribution :math:`q`. This can be seen through the following identity:

.. math::

   \DKL[p || q] = \xH[p || q] - \H[p]

Where the :doc:`cross_entropy` quantifies the total cost of encoding :math:`p`
using :math:`q`, and the :doc:`../multivariate/entropy` quantifies the true,
minimum cost of encoding :math:`p`. For example, let's consider the cost of
representing a biased coin by a fair one:

.. ipython::

   In [1]: from dit.divergences import kullback_leibler_divergence

   In [2]: p = dit.Distribution(['0', '1'], [3/4, 1/4])

   In [3]: q = dit.Distribution(['0', '1'], [1/2, 1/2])

   @doctest float
   In [4]: kullback_leibler_divergence(p, q)
   Out[4]: 0.18872187554086717

That is, it costs us :math:`0.1887` bits of wasted overhead by using a
mismatched distribution.


Not a Metric
============

Although the Kullback-Leibler divergence is often used to see how "different"
two distributions are, it is not a metric. Importantly, it is neither symmetric
nor does it obey the triangle inequality. It does, however, have the following
property:

.. math::

   \DKL[p || q] \ge 0

with equality if and only if :math:`p = q`. This makes it a `premetric
<http://en.wikipedia.org/wiki/Metric_(mathematics)#Premetrics>`_.


API
===

.. autofunction:: kullback_leibler_divergence
