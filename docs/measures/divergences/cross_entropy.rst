.. cross_entropy.rst
.. py:module:: dit.divergences.cross_entropy

*************
Cross Entropy
*************

The cross entropy between two distributions :math:`p(x)` and :math:`q(x)` is
given by:

.. math::

   \xH[p || q] = -\sum_{x \in \mathcal{X}} p(x) \log_2 q(x)

This quantifies the average cost of representing a distribution defined by the
probabilities :math:`p(x)` using the probabilities :math:`q(x)`. For example,
the cross entropy of a distribution with itself is the entropy of that
distribion because the entropy quantifies the average cost of representing a
distribution:

.. ipython::

   In [1]: from dit.divergences import cross_entropy

   In [2]: p = dit.Distribution(['0', '1'], [1/2, 1/2])

   @doctest float
   In [3]: cross_entropy(p, p)
   Out[3]: 1.0

If, however, we attempted to model a fair coin with a biased on, we could
compute this mis-match with the cross entropy:

.. ipython::

   In [4]: q = dit.Distribution(['0', '1'], [3/4, 1/4])

   @doctest float
   In [5]: cross_entropy(p, q)
   Out[5]: 1.207518749639422

Meaning, we will on average use about :math:`1.2` bits to represent the flips of
a fair coin. Turning things around, what if we had a biased coin that we
attempted to represent with a fair coin:

.. ipython::

   @doctest float
   In [6]: cross_entropy(q, p)
   Out[6]: 1.0

So although the entropy of :math:`q` is less than :math:`1`, we will use a full
bit to represent its outcomes. Both of these results can easily be seen by
considering the following identity:

.. math::

   \xH[p || q] = \H[p] + \DKL[p || q]

So in representing :math:`p` using :math:`q`, we of course must at least use
:math:`\H[p]` bits -- the minimum required to represent :math:`p` -- plus the
Kullback-Leibler divergence of :math:`q` from :math:`p`.

.. autofunction:: cross_entropy
