.. perplexity.rst
.. py:module:: dit.other.perplexity

**********
Perplexity
**********

The perplexity is a trivial measure to make the :doc:`../multivariate/entropy` more intuitive:

.. math::

   \P{X} = 2^{\H{X}}

The perplexity of a random variable is the size of a uniform distribution that would have the same entropy. For example, a distribution with 2 bits of entropy has a perplexity of 4, and so could be said to be "as random" as a four-sided die.

The conditional perplexity is defined in the natural way:

.. math::

   \P{X | Y} = 2^{\H{X | Y}}

We can see that the `xor` distribution is "4-way" perplexed:

.. ipython::

   In [1]: from dit.other import perplexity

   In [2]: from dit.example_dists import Xor

   @doctest float
   In [3]: perplexity(Xor())
   Out[3]: 4.0


API
===

.. autofunction:: perplexity
