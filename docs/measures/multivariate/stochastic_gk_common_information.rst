.. stochastic_gk_common_information.rst
.. py:module:: dit.multivariate.common_informations.stochastic_gk_common_information

.. _stochastic_gk_common_information:

*****************************************
Stochastic Gács–Körner Common Information
*****************************************

The stochastic Gács–Körner common information :cite:`kleinman2022gacs` is the
maximum :math:`I(X_i ; Z)` over stochastic variables :math:`Z` satisfying
:math:`p(Z \mid X_i) = p(Z \mid X_j)` for all jointly occurring
:math:`(X_i, X_j)`. It relaxes the deterministic common variable of the
:ref:`gács-körner common information`.

.. ipython::

   In [1]: from dit.multivariate import stochastic_gk_common_information

   In [2]: from dit.example_dists import Xor

   In [3]: stochastic_gk_common_information(Xor(), niter=4)

API
===

.. autofunction:: stochastic_gk_common_information
