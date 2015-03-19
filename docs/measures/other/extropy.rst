.. extropy.rst
.. py:module:: dit.other.extropy

*******
Extropy
*******

The extropy :cite:`Lad2011` is a dual to the entropy. It is defined by:

.. math::

   \J[X] = -\sum_{x \in X} (1-p(x)) \log_2 (1-p(x))

.. math::

   \H[X] + \J[X] = \sum_{x \in \mathcal{X}} \H[p(x), 1-p(x)] = \sum_{x \in \mathcal{X}} \J[p(x), 1-p(x)]

.. todo::

   discuss extropy.

.. todo::

   Compute some examples from the paper.


API
===

.. autofunction:: extropy
