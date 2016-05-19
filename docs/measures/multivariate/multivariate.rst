.. multivariate.rst
.. py:module:: dit.multivariate

************
Multivariate
************

Multivariate measures of information generally attempt to capture some global property of a joint distribution. For example, they might attempt to quantify how much information is shared among the random variables, or quantify how "non-indpendent" in the joint distribution is.

Total Information
=================
These quantities, currently just the Shannon entropy, measure the total amount of information contained in a set of joint variables.

.. toctree::
   :maxdepth: 1

   entropy

Mutual Informations
===================
These measuares all reduce to the standard Shannon mutual information for bivariate distributions.

.. toctree::
   :maxdepth: 1

   coinformation
   interaction_information
   caekl_mutual_information
   total_correlation
   dual_total_correlation

Common Informations
===================
These measures all somehow measure shared information, but do not equal the mutual information in the bivaraite case.

.. toctree::
   :maxdepth: 1

   gk_common_information
   tse_complexity

Others
======
These measures quantify other aspects of a joint distribution.

.. toctree::
   :maxdepth: 1

   residual_entropy
