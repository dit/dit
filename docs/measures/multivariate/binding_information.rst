.. binding_information.rst
.. py:module:: dit.multivariate.binding_information

*******************
Binding Information
*******************

The binding information :cite:`Abdallah2012`, or dual total correlation, is yet
another generalization of the mutual information. It is the amount of
information that is shared among the variables. It is defined as:

.. math::

   \B[X_{0:n}] &= \H[X_{0:n}] - \sum \H[X_i | X_{\{0..n\}/i}] \\
               &= - \sum_{x_{0:n} \in X_{0:n}} p(x_{0:n}) \log_2 \frac{p(x_{0:n})}{\prod p(x_i|x_{\{0:n\}/i})}


Relationship to Other Measures
===============================

The binding information obeys particular bounds related to both the
:doc:`entropy` and the :doc:`total_correlation`:

.. math::

   0 \leq & \B[X_{0:n}] \leq \H[X_{0:n}] \\
   \frac{\T[X_{0:n}]}{n-1} \leq & \B[X_{0:n}] \leq (n-1)\T[X_{0:n}]

Visualization
=============

The binding information, as seen below, consists equally of the information
shared among the variables.

.. image:: ../../images/idiagrams/b_xyz.png
   :alt: The binding information :math:`\B[X:Y:Z]`
   :width: 357px
   :align: center

.. todo::

   Show some examples, perhaps from the Abdallah paper.

.. autofunction:: binding_information
