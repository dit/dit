.. deweese.rst
.. py:module:: dit.multivariate.deweese

*********************
DeWeese-like Measures
*********************

Mike DeWeese has introduced a family of multivariate information measures based on a multivariate extension of the data processing inequality. The general idea is the following: local modification of a single variable can not increase the amount of correlation or dependence it has with the other variables. Consider, however, the triadic distribution:

.. ipython::

   In [1]: from dit.example_dists import dyadic, triadic

   In [2]: print(triadic)
   Class:          Distribution
   Alphabet:       ('0', '1', '2', '3') for all rvs
   Base:           linear
   Outcome Class:  str
   Outcome Length: 3
   RV Names:       None

   x     p(x)
   000   1/8
   022   1/8
   111   1/8
   133   1/8
   202   1/8
   220   1/8
   313   1/8
   331   1/8


This particular distribution has zero :ref:`coinformation`:

.. ipython::

   In [3]: from dit.multivariate import coinformation

   @doctest float
   In [4]: coinformation(triadic)
   Out[4]: 0.0

Yet the distribution is a product of a giant bit (coinformation :math:`1.0`) and the xor (coinformation :math:`-1.0`), and so there exists within it the capability of having a coinformation of :math:`1.0` if the xor component were dropped. This is exactly what the DeWeese construction captures:

.. math::

   \ID{X_0 : \ldots : X_n} = \max_{p(x'_i | x_i)} \I{X'_0 : \ldots : X'_n}

.. ipython::

   In [5]: from dit.multivariate import deweese_coinformation

   @doctest float
   In [6]: deweese_coinformation(triadic)
   Out[6]: 1.0

DeWeese version of the :ref:`total_correlation`, :ref:`dual_total_correlation`, and :ref:`caekl_mutual_information` are also available, and operate on an arbitrary number of variables with optional conditional variables.

API
===

.. autofunction:: deweese_coinformation
.. autofunction:: deweese_total_correlation
.. autofunction:: deweese_dual_total_correlation
.. autofunction:: deweese_caekl_mutual_information
