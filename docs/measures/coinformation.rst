.. coinformation.rst

**************
Co-Information
**************

The co-information :cite:`Bell2003` is one generalization of the mutual information to multiple
variables. It is defined via an inclusion/exclusion sum:

.. math::

   \I[X_{0:n}] &= -\sum_{y \in \mathcal{P}(\{0..n\})} (-1)^{|y|} \H[X_y] \\
               &= \sum_{x_{0:n} \in X_{0:n}} p(x_{0:n}) \log_2 \prod_{y \in \mathcal{P}(\{0..n\})} p(y)^{(-1)^{|y|}}

.. autofunction:: dit.algorithms.coinformation.coinformation
