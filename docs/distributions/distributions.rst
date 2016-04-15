.. distributions.rst

*************
Distributions
*************

Distributions in :mod:`dit` come in two different flavors: :class:`~dit.ScalarDistribution` and :class:`~dit.Distribution`. :class:`~dit.ScalarDistribution` is used for representing distributions over real numbers, and have many features related to that. :class:`~dit.Distribution` is used for representing joint distributions, and therefore has many features related to marginalizing, conditioning, and otherwise exploring the relationships between random variables.

.. toctree::

   npscalardist.rst
   npdist.rst
