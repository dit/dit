.. cross_entropy.rst
.. py:module:: dit.divergences.cross_entropy

*************
Cross Entropy
*************

The cross entropy between two distributions :math:`p(x)` and :math:`q(x)` is given by:

.. math::

   \xH(p, q) = -\sum_{x \in \mathcal{X}} p(x) \log_2 q(x)

.. todo::

   Add some examples and motivation. 

.. autofunction:: dit.divergences.cross_entropy.cross_entropy
