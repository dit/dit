.. coinformation.rst
.. py:module:: dit.algorithms.coinformation

**************
Co-Information
**************

The co-information :cite:`Bell2003` is one generalization of the mutual
information to multiple variables. The co-information quantifies the amount of
infomration that *all* variables participate in. It is defined via an
inclusion/exclusion sum:

.. math::

   \I[X_{0:n}] &= -\sum_{y \in \mathcal{P}(\{0..n\})} (-1)^{|y|} \H[X_y] \\
               &= \sum_{x_{0:n} \in X_{0:n}} p(x_{0:n}) \log_2 \prod_{y \in \mathcal{P}(\{0..n\})} p(y)^{(-1)^{|y|}}

One notable property of the co-information is that for :math:`n \geq 3` it can
be negative.

.. image:: ../images/idiagrams/i_xy.png
   :alt: The co-information :math:`\I[X:Y]`
   :width: 342px
   :align: center

.. image:: ../images/idiagrams/i_xyz.png
   :alt: The co-information :math:`\I[X:Y:Z]`
   :width: 357px
   :align: center

.. todo::

   add examples, preferably from Bell's paper.

.. autofunction:: dit.algorithms.coinformation.coinformation
