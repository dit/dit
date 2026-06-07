.. transmission.rst
.. py:module:: dit.multivariate.transmission

.. _transmission:

************
Transmission
************

The transmission :cite:`zwick2004overview`, denoted :math:`T`, is the central error measure of *reconstructability analysis*. Given a *structure* (a set of marginals, or "projections", to hold fixed), it quantifies how much of a distribution's constraint is lost when the distribution is reconstructed from those marginals alone. It is the Kullback-Leibler divergence from the data to the maximum entropy distribution consistent with the chosen marginals:

.. math::

   T(\text{structure}) = \DKL{p}{q_{\text{structure}}}
                       = U(q_{\text{structure}}) - U(p)

where :math:`q_{\text{structure}}` is the maximum entropy reconstruction (see :doc:`/optimization`) matching the specified marginals of :math:`p`, and :math:`U` is the Shannon entropy.

A structure is specified as a list of marginals, each a list of variable indices. For example, ``[[0, 1], [1, 2]]`` is the structure ``AB:BC``.

.. ipython::

   In [1]: from dit.multivariate import transmission

   In [2]: xor = dit.example_dists.Xor()

The default structure is the *independence model* (each variable on its own), for which the transmission equals the :doc:`total_correlation`:

.. ipython::

   @doctest float
   In [3]: transmission(xor)
   Out[3]: 1.0

   @doctest float
   In [4]: from dit.multivariate import total_correlation

   @doctest float
   In [5]: total_correlation(xor)
   Out[5]: 1.0

For the *saturated model* (all variables held jointly), nothing is lost and the transmission is zero:

.. ipython::

   @doctest float
   In [6]: transmission(xor, [[0, 1, 2]])
   Out[6]: 0.0

Because the parity information in ``xor`` lives entirely in the three-way interaction, any proper decomposition loses all of it:

.. ipython::

   @doctest float
   In [7]: transmission(xor, [[0, 1], [1, 2]])
   Out[7]: 1.0

Model Complexity
================

Reconstructability analysis trades transmission (model error) against model complexity, measured by the degrees of freedom of the structure: the number of free parameters needed to specify the maximum entropy distribution. Together, transmission and degrees of freedom for every structure in the lattice form the "decomposition spectrum"; see the dependency decomposition in :doc:`/profiles`.

.. ipython::

   In [8]: from dit.algorithms import degrees_of_freedom

   In [9]: degrees_of_freedom(xor, [[0, 1], [1, 2]])
   Out[9]: 5

API
===

.. autofunction:: transmission

.. autofunction:: dit.algorithms.degrees_of_freedom
