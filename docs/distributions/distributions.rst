.. distributions.rst

*************
Distributions
*************

Distributions in :mod:`dit` come in several flavors:

* :class:`~dit.ScalarDistribution` is used for representing distributions over real numbers, and has many features related to that.
* :class:`~dit.Distribution` is used for representing joint distributions, and therefore has many features related to marginalizing, conditioning, and otherwise exploring the relationships between random variables.
* :class:`~dit.xrdist.XRDistribution` (requires the ``xarray`` extra) is an xarray-backed distribution class that tracks named dimensions, supports natural algebraic operations on conditional distributions (e.g. ``p(X,Y) * p(Z|X,Y) = p(X,Y,Z)``), and makes conditioning and marginalization straightforward via named variables.

.. toctree::

   npscalardist.rst
   npdist.rst
