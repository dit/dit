.. beta_common_information.rst
.. py:module:: dit.multivariate.common_informations.beta_common_information

.. _beta_common_information:

************************
Beta Common Information
************************

Yu, Li & Chen :cite:`yu2017generalized` interpolate between the
:doc:`wyner_common_information` and the :ref:`gács-körner common information`
using a bound on the conditional maximal correlation:

.. math::

   C_{\beta}(X_1 : \ldots : X_n \mid Z)
     = \inf_{\substack{P_{U \mid X_{1:n} Z} \\ \max_{i \neq j} \rho_m(X_i; X_j \mid U, Z) \le \beta}}
       I(X_{1:n} ; U \mid Z)

Special cases: :math:`\beta = 0` recovers Wyner common information;
:math:`\beta \to 1` recovers Gács–Körner common information.

.. ipython::

   In [1]: from dit.multivariate import beta_common_information

   In [2]: from dit.example_dists import Xor

   In [3]: beta_common_information(Xor(), beta=0.0, niter=4)

API
===

.. autofunction:: beta_common_information
