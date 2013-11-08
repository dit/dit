.. shannon.rst
.. py:module :: dit.algorithms.shannon

**********************
Basic Shannon measures
**********************

The information on this page is drawn from :cite:`Cover2006`.

Entropy
=======

The entropy measures how much information is in a random variable :math:`X`.

.. math::

   \H[X] = - \sum_{x \in X} p(x) \log_2 p(x)

.. autofunction:: dit.algorithms.shannon.entropy

Conditional Entropy
===================

The conditional entropy is the amount of information in variable :math:`X`
beyond that which is in variable :math:`Y`.

.. math::

   \H[X|Y] = \sum_{x \in X, y \in Y} p(x, y) \log_2 p(x|y)

.. autofunction:: dit.algorithms.shannon.conditional_entropy

Mutual Information
==================

The mutual information is the amount of information shared by :math:`X` and
:math:`Y`.

.. math::

   \I[X:Y] &= \H[X,Y] - \H[X|Y] - \H[Y|X] \\
           &= \H[X] + \H[Y] - \H[X,Y] \\
           &= \sum_{x \in X, y \in Y} p(x, y) \log_2 \frac{p(x, y)}{p(x)p(y)}

.. todo::

   Add i-diagrams.

.. todo::

   Add discussion.

.. todo::

   Add examples.

.. autofunction:: dit.algorithms.shannon.mutual_information
