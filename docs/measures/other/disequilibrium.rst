.. other.rst
.. py:module:: dit.other.disequilibrium

**************************************
Disequilibrium and the LMPR Complexity
**************************************

Lamberti, Martin, Plastino, and Rosso have proposed a complexity measure :cite:`Lamberti2004` disigned around the idea of being a measure of "distance from equilibrium", or disequilibrium, multiplied by a measure of "randomness". Here, they measure "randomness" by the (normalized) :doc:`../multivariate/entropy`:

.. math::

   \H[X]/\log_2{|X|}

and the disequilibrium as a (normalized) :doc:`../divergences/jensen_shannon_divergence`:

.. math::

   \JSD[X || P_e] / Q_0

where :math:`P_e` is a uniform distribution over the same outcome space as :math:`X`, and :math:`Q_0` is the maximum possible value of the Jensen-Shannon divergence of a distribution with :math:`P_e`.

The LMPR complexity does not necessarily behave as one might intuitively hope. For example, the LMPR complexity of the ``xor`` and "double bit" with independent bit are identical:

.. ipython::

   In [1]: from dit.other.disequilibrium import *

   In [2]: d1 = dit.Distribution(['000', '001', '110', '111'], [1/4]*4)

   In [3]: d2 = dit.Distribution(['000', '011', '101', '110'], [1/4]*4)

   In [4]: LMPR_complexity(d1)
   Out[4]: 0.28945986160258008

   In [5]: LMPR_complexity(d2)
   Out[5]: 0.28945986160258008

This is because they are both equally "far from equilibrium" with four equiprobable events over the space of three binary variables, and both have the same entropy of two bits.

This implies that the LMPR complexity is perhaps best applied to a :class:`ScalarDistribution`, and is not suitable for measuring the complexity of dependencies between variables.

API
===

.. autofunction:: disequilibrium

.. autofunction:: LMPR_complexity
