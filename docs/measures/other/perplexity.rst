.. perplexity.rst
.. py:module:: dit.other.perplexity

**********
Perplexity
**********

The perplexity is a trivial measure to make the entropy more intuitive.

.. math::

   \P[X] = 2^{\H[X]}

The perplexity of a random variable is the size of a uniform distribution that
would have the same entropy. For example, a distribution with 2 bits of entropy
has a perplexity of 4, and so could be said to be "as random" as a four-sided
die.

The conditional perplexity is defined in the natural way:

.. math::

   \P[X|Y] = 2^{\H[X|Y]}

.. todo::

   Add some good examples.


API
===

.. autofunction:: perplexity
