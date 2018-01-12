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
These measures all reduce to the standard Shannon :ref:`mutual_information` for bivariate distributions.

.. toctree::
   :maxdepth: 1

   coinformation
   interaction_information
   caekl_mutual_information
   total_correlation
   dual_total_correlation

It is perhaps illustrative to consider how each of these measures behaves on two canonical distributions: the giant bit and parity.

+-----------+----------------------------------------+-------------------------------------------------------------+
|           | giant bit                              | parity                                                      |
+-----------+---+----------------+-----------+---+---+----------------+----+---+-----------+-----------------------+
|           | I | II             | T         | B | J | I              | II | T | B         | J                     |
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


Common Informations
===================
These measures all somehow measure shared information, but do not equal the mutual information in the bivaraite case.

.. toctree::
   :maxdepth: 1

   gk_common_information
   wyner_common_information
   exact_common_information
   functional_common_information
   mss_common_information

Ordering
--------

The common information measures (together with the :doc:`dual_total_correlation` and :doc:`caekl_mutual_information`) form an ordering:

.. math::

   \K{X_{0:n}} \leq \J{X_{0:n}}
               \leq \B{X_{0:n}}
               \leq \C{X_{0:n}}
               \leq \G{X_{0:n}}
               \leq \F{X_{0:n}}
               \leq \M{X_{0:n}}

Others
======
These measures quantify other aspects of a joint distribution.

.. toctree::
   :maxdepth: 1

   residual_entropy
   tse_complexity
   necessary_conditional_entropy
