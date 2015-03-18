.. residual_entropy.rst
.. py:currentmodule:: dit.multivariate.binding_information

****************
Residual Entropy
****************

The residual entropy, or erasure entropy, is a dual to the
:doc:`binding_information`. It is dual in the sense that together they form the
entropy of the distribution.

.. math::

   \R[X_{0:n}] &= \sum \H[X_i | X_{\{0..n\}/i}] \\
               &= -\sum_{x_{0:n} \in X_{0:n}} p(x_{0:n}) \log_2 \prod p(x_i|x_{\{0:n\}/i})

The residual entropy was originally proposed in :cite:`Verdu2008` to quantify
the information lost by sporatic erasures in a channel. The idea here is that
only the information uncorrelated with other random variables is lost if that
variable is erased.

.. todo::

   Add some good examples.

Visualization
=============

The residual entropy consists of all the unshared information in the
distribution. That is, it is the information in each variable not overlapping
with any other.

.. image:: ../../images/idiagrams/r_xy.png
   :alt: The residual entropy :math:`\R[X:Y]`
   :width: 342px
   :align: center

.. image:: ../../images/idiagrams/r_xyz.png
   :alt: The residual entropy :math:`\R[X:Y:Z]`
   :width: 357px
   :align: center

.. autofunction:: residual_entropy
