.. multivariate.rst
.. py:currentmodule:: dit.multivariate

************
Multivariate
************

Multivariate measures of information generally attempt to capture some global property of a joint distribution. For example, they might attempt to quantify how much information is shared among the random variables, or quantify how "non-independent" the joint distribution is.

Total Information
=================
These quantities, currently just the Shannon entropy, measure the total amount of information contained in a set of joint variables.

.. toctree::
   :maxdepth: 1

   entropy

Mutual Informations
===================
These measures all reduce to the standard Shannon :ref:`mutual_information` for bivariate distributions.

.. toctree::
   :maxdepth: 1

   coinformation
   total_correlation
   dual_total_correlation
   cohesion
   o_information
   caekl_mutual_information
   kirkwood
   frustration
   interaction_information
   s_information
   deweese
   higher_order_common_information
   union_information
   quax_synergy
   synergistic_disclosure
   logarithmic_decomposition

It is perhaps illustrative to consider how each of these measures behaves on two canonical distributions: the giant bit and parity.

+-----------+----------------------------------------+-------------------------------------------------------------+
|           | giant bit                              | parity                                                      |
+-----------+---+----------------+-----------+---+---+----------------+----+---+-----------+-----------------------+
| size      | I | II             | T         | B | J | I              | II | T | B         | J                     |
+-----------+---+----------------+-----------+---+---+----------------+----+---+-----------+-----------------------+
| 2         | 1 | 1              | 1         | 1 | 1 | 1              | 1  | 1 | 1         | 1                     |
+-----------+---+----------------+-----------+---+---+----------------+----+---+-----------+-----------------------+
| 3         | 1 | -1             | 2         | 1 | 1 | -1             | 1  | 1 | 2         | :math:`\frac{1}{2}`   |
+-----------+---+----------------+-----------+---+---+----------------+----+---+-----------+-----------------------+
| 4         | 1 | 1              | 3         | 1 | 1 | 1              | 1  | 1 | 3         | :math:`\frac{1}{3}`   |
+-----------+---+----------------+-----------+---+---+----------------+----+---+-----------+-----------------------+
| 5         | 1 | -1             | 4         | 1 | 1 | -1             | 1  | 1 | 4         | :math:`\frac{1}{4}`   |
+-----------+---+----------------+-----------+---+---+----------------+----+---+-----------+-----------------------+
| :math:`n` | 1 | :math:`(-1)^n` | :math:`n` | 1 | 1 | :math:`(-1)^n` | 1  | 1 | :math:`n` | :math:`\frac{1}{n-1}` |
+-----------+---+----------------+-----------+---+---+----------------+----+---+-----------+-----------------------+

Measures of shared information that do *not* reduce to the mutual information
in the bivariate case (Gács–Körner, Wyner, exact, …) are collected on the
:ref:`common informations` page.

Others
======
These measures quantify other aspects of a joint distribution.

.. toctree::
   :maxdepth: 1

   residual_entropy
   tse_complexity
   necessary_conditional_entropy
   delta_gamma
   alpha_synergy
   transmission
   cross_mutual_information
