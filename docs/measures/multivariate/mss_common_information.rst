.. mss_common_information.rst
.. py:module:: dit.multivariate.mss_common_information

**********************
MSS Common Information
**********************

The Minimal Sufficient Statistic Common Information is the entropy of the join of the minimal sufficient statistic of each variable about the others:

.. math::

   \M{X_{0:n}} = \H{ \join_i \left(X_i \mss X_\overline{\{i\}}\right) }

The distribution that the MSS common information is the entroy of is also known "information trim" of the original distribution, and is accessable via :py:func:`dit.algorithms.minimal_sufficient_statistic.info_trim`.

API
===

.. autofunction:: mss_common_information
