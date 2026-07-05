.. s_information.rst
.. py:module:: dit.multivariate.s_information

.. _s_information:

*************
S-Information
*************

The S-information (also known as the exogenous information) :cite:`rosas2019quantifying` quantifies the total amount of dependency between each individual variable and the rest of a system. It is defined as the sum of the :doc:`total_correlation` :math:`\T{}` and the :doc:`dual_total_correlation` :math:`\B{}`:

.. math::

   \S{X_{0:n}} = \T{X_{0:n}} + \B{X_{0:n}}

Equivalently, it is the sum, over each variable, of the mutual information between that variable and all the others:

.. math::

   \S{X_{0:n}} = \sum_{i=0}^{n-1} \I{X_i : X_{\{0:n\} \setminus i}}

The S-information is a special case of both the :math:`\Delta^k` and :math:`\Gamma^k` measures at :math:`k = 0`; see :doc:`delta_gamma`.

.. ipython::

   In [1]: from dit.multivariate import s_information

   In [2]: from dit.example_dists import n_mod_m

   In [3]: d = n_mod_m(5, 2)

   @doctest float
   In [4]: s_information(d)
   Out[4]: 5.0


API
===

.. autofunction:: s_information
