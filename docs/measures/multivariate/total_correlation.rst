.. total_correlation.rst
.. py:module:: dit.multivariate.total_correlation

*****************
Total Correlation
*****************

The total correlation :cite:`watanabe1960information`, denoted :math:`\T{}`, also known as the multi-information or integration, is one generalization of the :ref:`mutual_information`. It is defined as the amount of information each individual variable carries above and beyond the joint entropy, e.g. the difference between the whole and the sum of its parts:

.. math::

   \T{X_{0:n}} &= \sum \H{X_i} - \H{X_{0:n}} \\
               &= \sum_{x_{0:n} \in X_{0:n}} p(x_{0:n}) \log_2 \frac{p(x_{0:n})}{\prod p(x_i)}

Two nice features of the total correlation are that it is non-negative and that it is zero if and only if the random variables :math:`X_{0:n}` are all independent. Some baseline behavior is good to note also. First its behavior when applied to "giant bit" distributions:

.. ipython::

   In [1]: from dit import Distribution as D

   In [2]: from dit.multivariate import total_correlation as T

   @doctest float
   In [3]: [ T(D(['0'*n, '1'*n], [0.5, 0.5])) for n in range(2, 6) ]
   Out[3]: [1.0, 2.0, 3.0, 4.0]

So we see that for giant bit distributions, the total correlation is equal to one less than the number of variables. The second type of distribution to consider is general parity distributions:

.. ipython::

   In [4]: from dit.example_dists import n_mod_m

   @doctest float
   In [5]: [ T(n_mod_m(n, 2)) for n in range(3, 6) ]
   Out[5]: [1.0, 1.0, 1.0]

   @doctest float
   In [6]: [ T(n_mod_m(3, m)) for m in range(2, 5) ]
   Out[6]: [1.0, 1.5849625007211565, 2.0]

Here we see that the total correlation is equal to :math:`\log_2{m}` regardless of :math:`n`.

The total correlation follows a nice decomposition rule. Given two sets of (not necessarily independent) random variables, :math:`A` and :math:`B`, the total correaltion of :math:`A \cup B` is:

.. math::

   \T{A \cup B} = \T{A} + \T{B} + \I{A : B}

.. ipython::

   In [18]: from dit.multivariate import coinformation as I

   In [19]: d = n_mod_m(4, 3)

   @doctest
   In [20]: T(d) == T(d, [[0], [1]]) + T(d, [[2], [3]]) + I(d, [[0, 1], [2, 3]])
   Out[20]: True


Visualization
=============

The total correlation consists of all information that is shared among the variables, and weights each piece according to how many variables it is shared among.

.. image:: ../../images/idiagrams/t_xyz.png
   :alt: The total correlation :math:`\T{X : Y : Z}`
   :width: 357px
   :align: center


API
===

.. autofunction:: total_correlation
