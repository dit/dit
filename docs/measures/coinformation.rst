.. coinformation.rst

**************
Co-Information
**************

The co-information is one generalization of the mutual information to multiple
variables. It is defined via an inclusion/exclusion sum:

.. math::

   \I[X_{0:n}] = -\sum_{y \subseteq \{0..n\}} (-1)^{|y|} \H[X_y]

.. autofunction:: dit.algorithms.coinformation
