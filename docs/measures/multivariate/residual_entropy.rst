.. residual_entropy.rst
.. py:currentmodule:: dit.multivariate.dual_total_correlation

****************
Residual Entropy
****************

The residual entropy, or erasure entropy, is a dual to the :doc:`dual_total_correlation`. It is dual in the sense that together they form the entropy of the distribution.

.. math::

   \R{X_{0:n}} &= \sum \H{X_i | X_{\{0..n\}/i}} \\
               &= -\sum_{x_{0:n} \in X_{0:n}} p(x_{0:n}) \log_2 \prod p(x_i|x_{\{0:n\}/i})

The residual entropy was originally proposed in :cite:`Verdu2008` to quantify the information lost by sporatic erasures in a channel. The idea here is that only the information uncorrelated with other random variables is lost if that variable is erased.

If a joint distribution consists of independent random variables, the residual entropy is equal to the :doc:`entropy`:

.. ipython::

   In [1]: from dit.multivariate import entropy, residual_entropy

   In [2]: d = dit.uniform_distribution(3, 2)

   @doctest
   In [3]: entropy(d) == residual_entropy(d)
   Out[3]: True

Another simple example is a distribution where one random variable is independent of the others:

.. ipython::

   In [1]: d = dit.uniform(['000', '001', '110', '111'])

   @doctest float
   In [2]: residual_entropy(d)
   Out[2]: 1.0

If we ask for the residual entropy of only the latter two random variables, the middle one is now independent of the others and so the residual entropy grows:

.. ipython::

   @doctest float
   In [4]: residual_entropy(d, [[1], [2]])
   Out[4]: 2.0


Visualization
=============

The residual entropy consists of all the unshared information in the distribution. That is, it is the information in each variable not overlapping with any other.

.. image:: ../../images/idiagrams/r_xy.png
   :alt: The residual entropy :math:`\R{X : Y}`
   :width: 342px
   :align: center

.. image:: ../../images/idiagrams/r_xyz.png
   :alt: The residual entropy :math:`\R{X : Y : Z}`
   :width: 357px
   :align: center


API
===

.. autofunction:: residual_entropy
