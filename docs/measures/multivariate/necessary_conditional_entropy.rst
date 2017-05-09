.. necessary_conditional_entropy.rst
.. py:module:: dit.multivariate.necessary_conditional_entropy

*****************************
Necessary Conditional Entropy
*****************************

The necessary conditional entropy :cite:`cuff2010coordination` quantifies the amount of information that a random variable :math:`X` necessarily must carry above and beyond the mutual information :math:`\I[X:Y]` to actually contain that mutual information:

.. math::

   \H[X \dagger Y] = \H[ X \mss Y | Y]

API
===

.. autofunction:: necessary_conditional_entropy
