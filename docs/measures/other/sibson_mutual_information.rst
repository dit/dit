.. sibson_mutual_information.rst
.. py:module:: dit.other.sibson_mutual_information

************************
Sibson Mutual Information
************************

Sibson (or :math:`\alpha`-) mutual information generalizes Shannon mutual
information. At :math:`\alpha = 1` it equals Shannon MI; at
:math:`\alpha = \infty` it equals **maximal leakage** from :math:`X` to
:math:`Y`.

.. math::

   I_\alpha(X;Y) = \min_{Q_Y} D_\alpha(P_{XY} \| P_X \otimes Q_Y)
                 = \frac{\alpha}{\alpha - 1} \log_2 \sum_y
                   \left(\sum_x P(x)\, P(y|x)^\alpha\right)^{1/\alpha}

The measure is **asymmetric** in :math:`(X, Y)`: the first argument is the
source whose marginal :math:`P(x)` appears in the sum.

.. ipython::

   In [1]: from dit.other import sibson_mutual_information, maximal_leakage

   In [2]: from dit.example_dists import Xor

   In [3]: from dit import Distribution

   In [4]: d = Distribution(["00", "11"], [0.5, 0.5])

   @doctest float
   In [5]: sibson_mutual_information(d, [0], [1], 2)
   Out[5]: 1.0

   @doctest float
   In [6]: maximal_leakage(d, [0], [1])
   Out[6]: 1.0


Conditional variants
====================

Two conditional Sibson measures from Esposito et al. (2021) are provided:

* ``sibson_conditional_mutual_information_y_given_z`` — minimizes over
  :math:`Q_{Y|Z}`; reduces to unconditional Sibson MI when :math:`Z` is
  constant.
* ``sibson_conditional_mutual_information_z`` — minimizes over :math:`Q_Z`;
  symmetric in :math:`X` and :math:`Y`.

.. autofunction:: sibson_mutual_information

.. autofunction:: sibson_mutual_information_pmf

.. autofunction:: maximal_leakage

.. autofunction:: sibson_conditional_mutual_information_y_given_z

.. autofunction:: sibson_conditional_mutual_information_z
