.. extropy.rst
.. py:module:: dit.other.extropy

*******
Extropy
*******

The extropy :cite:`Lad2011` is a dual to the :doc:`../multivariate/entropy`. It
is defined by:

.. math::

   \J[X] = -\sum_{x \in X} (1-p(x)) \log_2 (1-p(x))

The entropy and the extropy satisify the following relationship:

.. math::

   \H[X] + \J[X] = \sum_{x \in \mathcal{X}} \H[p(x), 1-p(x)] = \sum_{x \in \mathcal{X}} \J[p(x), 1-p(x)]

 Unfortunately, the extropy does not yet have any intuitive interpretation.

.. ipython::

   In [1]: from dit.other import extropy

   In [2]: from dit.example_dists import Xor

   @doctest float
   In [3]: extropy(Xor())
   Out[3]: 1.2451124978365313

   @doctest float
   In [4]: extropy(Xor(), [0])
   Out[4]: 1.0


API
===

.. autofunction:: extropy
