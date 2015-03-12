.. extropy.rst
.. py:module:: dit.other.extropy

*******
Extropy
*******

The extropy :cite:`Lad2011` is a dual to the entropy. It is defined by:

.. math::

   \J[X] = -\sum_{x \in X} (1-p(x)) \log_2 (1-p(x))

.. todo::

   discuss extropy.

.. todo::

   Compute some examples from the paper.

.. autofunction:: dit.other.extropy.extropy
