.. shannon.rst

**********************
Basic Shannon measures
**********************

Entropy
=======

.. math::

   \H[X] = - \sum_{x \in X} p(x) \log_2 p(x)

.. autofunction:: dit.algorithms.entropy

Conditional Entropy
===================

.. math::

   \H[X|Y] = \sum_{x \in X, y \in Y} p(x, y) \log_2 p(x|y)

.. autofunction:: dit.algorithms.conditional_entropy

Mutual Information
==================

.. math::

   \I[X:Y] &= \H[X,Y] - \H[X|Y] - \H[Y|X] \\
           &= \H[X] + \H[Y] - \H[X,Y] \\
           &= \sum_{x \in X, y \in Y} p(x, y) \log_2 \frac{p(x, y)}{p(x)p(y)}

.. autofunction:: dit.algorithms.mutual_information
