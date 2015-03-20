.. multivariate.rst
.. py:module:: dit.multivariate

************
Multivariate
************

Multivariate measures of information generally attempt to capture some global
property of a joint distribution. For example, they might attempt to quantify
how much information is shared among the random variables, or quantify how
"non-indpendent" in the joint distribution is.

Shannon Measures
================
The first batch are all linear combinations of atoms from an I-diagram, and are
therefore computable using only the entropy of subsets of the distribution.

.. toctree::
   :maxdepth: 1

   entropy
   coinformation
   interaction_information
   total_correlation
   binding_information
   residual_entropy
   tse_complexity

Non-Shannon Measures
====================
The next set of measures are based on the definition of an auxiliary random
variable, generally one which in some way captures the information common to the
random variables.

.. toctree::
   :maxdepth: 1

   gk_common_information
