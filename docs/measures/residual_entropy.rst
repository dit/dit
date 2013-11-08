.. residual_entropy.rst
.. py:module :: dit.algorithms.binding

****************
Residual Entropy
****************

The residual entropy, or erasure entropy, is a dual to the binding information.

.. math::

   \R[X_{0:n}] &= \sum \H[X_i | X_{\{0..n\}/i}] \\
               &= - \sum_{x_{0:n} \in X_{0:n}} p(x_{0:n}) \log_2 \prod p(x_i|x_{\{0:n\}/i})

.. image:: ../images/idiagrams/r_xy.png
   :alt: The residual entropy :math:`\R[X:Y]`
   :align: center

.. image:: ../images/idiagrams/r_xyz.png
   :alt: The residual entropy :math:`\R[X:Y:Z]`
   :align: center

.. todo::

   Add discussion, history (erasure entropy)

.. todo::

   Add some good examples.

.. autofunction:: dit.algorithms.binding.residual_entropy
