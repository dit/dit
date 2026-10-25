.. mss_common_information.rst
.. py:currentmodule:: dit.multivariate

**********************
MSS Common Information
**********************

The Minimal Sufficient Statistic Common Information is the entropy of the join of the minimal sufficient statistic of each variable about the others:

.. math::

   \M{X_{0:n}} = \H{ \join_i \left(X_i \mss X_{\overline{\{i\}}}\right) }

The distribution whose entropy is the MSS common information is also known as
the *information trim* of the original distribution, and is accessible via
:py:func:`dit.algorithms.minimal_sufficient_statistic.info_trim`.

For the distribution uniform over :math:`\{00, 01, 11\}`, each variable's
minimal sufficient statistic about the other is the variable itself, so the
MSS common information is the full joint entropy :math:`\log_2 3`:

.. ipython::

   In [1]: from dit.multivariate import mss_common_information

   In [2]: d = dit.Distribution(['00', '01', '11'], [1/3] * 3)

   @doctest float
   In [3]: mss_common_information(d)
   Out[3]: 1.584962500721156

API
===

.. autofunction:: mss_common_information
   :no-index:
