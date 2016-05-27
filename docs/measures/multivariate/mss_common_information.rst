.. mss_common_information.rst
.. py:module:: dit.multivariate.mss_common_information

**********************
MSS Common Information
**********************

The Minimal Sufficient Statistic Common Information is the entropy of the join of the minimal sufficient statistic of each variable about the others:

.. math::

   \M[X_{0:n}] = \H\left[ \joinop_i X_i \mss X_\overline{\{i\}} \right]

API
===

.. autofunction:: mss_common_information
