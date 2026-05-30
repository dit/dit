.. delta_gamma.rst
.. py:module:: dit.multivariate.delta_gamma

.. _delta_gamma:

***********************
Delta^k and Gamma^k
***********************

The :math:`\Delta^k` and :math:`\Gamma^k` measures :cite:`varley2026faces` form a parameterized family of higher-order information measures. They unify several existing multivariate measures by recognizing them as special cases tuned by an integer order parameter :math:`k`.

:math:`\Delta^k` is defined in terms of the S-information :math:`\mathcal{S}` and the :doc:`total_correlation` :math:`\mathcal{T}`:

.. math::

   \Delta^k(X_{0:n}) = \mathcal{S}(X_{0:n}) - k\mathcal{T}(X_{0:n})

Since the S-information is the sum of the total correlation and the :doc:`dual_total_correlation` :math:`\mathcal{D}`, this is equivalent to :math:`\Delta^k = \mathcal{D} + (1 - k)\mathcal{T}`. It is arranged into a hierarchy of increasingly high-order synergies: if :math:`\Delta^k(X) < 0` the system is dominated by interactions of order greater than :math:`k`, while if :math:`\Delta^k(X) > 0` it is dominated by interactions of order lower than :math:`k`.

:math:`\Gamma^k` is the entropic conjugate of :math:`\Delta^k`, obtained by exchanging the roles of the total correlation and dual total correlation:

.. math::

   \Gamma^k(X_{0:n}) = \mathcal{S}(X_{0:n}) - k\mathcal{D}(X_{0:n})

equivalently :math:`\Gamma^k = \mathcal{T} + (1 - k)\mathcal{D}`. It is arranged into a hierarchy of increasingly high-order redundancies.

For particular values of :math:`k`, both measures recover known quantities:

+----------------+-------------------------------------+-------------------------------------+
| :math:`k`      | :math:`\Delta^k`                    | :math:`\Gamma^k`                    |
+================+=====================================+=====================================+
| 0              | S-information                       | S-information                       |
+----------------+-------------------------------------+-------------------------------------+
| 1              | dual total correlation              | total correlation                   |
+----------------+-------------------------------------+-------------------------------------+
| 2              | negative O-information              | O-information                       |
+----------------+-------------------------------------+-------------------------------------+

.. ipython::

   In [1]: from dit.multivariate import delta_k, gamma_k

   In [2]: from dit.example_dists import n_mod_m

   In [3]: d = n_mod_m(5, 2)

   @doctest float
   In [4]: delta_k(d, 2)
   Out[4]: 3.0

   @doctest float
   In [5]: gamma_k(d, 2)
   Out[5]: -3.0


API
===

.. autofunction:: delta_k
.. autofunction:: gamma_k
