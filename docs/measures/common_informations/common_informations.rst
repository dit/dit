.. common_informations.rst
.. py:currentmodule:: dit.multivariate

.. _common informations:

*******************
Common Informations
*******************

These measures all somehow measure shared information, but do not equal the
mutual information in the bivariate case. All of them are importable from
:mod:`dit.multivariate`.

.. toctree::
   :maxdepth: 1

   gk_common_information
   wyner_common_information
   channel_synthesis
   exact_common_information
   functional_common_information
   kamath_common_information
   maxent_function
   mss_common_information
   beta_common_information
   stochastic_gk_common_information
   tension_common_information

The directed Kamath–Anantharam quantity
:func:`~dit.multivariate.directed_kamath_common_information` is documented on
the :doc:`/measures/common_informations/kamath_common_information` page.

Several of these quantities are corner points of the Gray–Wyner region; see
:doc:`/channels/gray_wyner` for the operational picture.

Ordering
========

The common information measures (together with the
:doc:`/measures/multivariate/dual_total_correlation` and
:doc:`/measures/multivariate/caekl_mutual_information`) form an ordering:

.. math::

   \K{X_{0:n}} \leq \J{X_{0:n}}
               \leq \B{X_{0:n}}
               \leq \C{X_{0:n}}
               \leq \G{X_{0:n}}
               \leq \F{X_{0:n}}
               \leq U(X_{0:n})
               \leq \M{X_{0:n}}
