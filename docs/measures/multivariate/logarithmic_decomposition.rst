.. logarithmic_decomposition.rst
.. py:module:: dit.multivariate.logarithmic_decomposition

.. _logarithmic_decomposition:

**************************
Logarithmic Decomposition
**************************

Down & Mediano :cite:`down2024logarithmic` refine Yeung's I-measure into
*logarithmic atoms* — one for every subset of the joint outcome space with
two or more elements. Each atom has an intrinsic sign given by its degree
(even positive, odd negative) and an interior-loss measure :math:`\mu` that
sums to entropy, mutual information, co-information, and so on.

The number of atoms is :math:`2^{|\Omega|} - |\Omega| - 1`, so the
decomposition is practical only for small supports.

.. ipython::

   In [1]: from dit.multivariate import LogarithmicDecomposition

   In [2]: from dit.example_dists import Xor

   In [3]: ld = LogarithmicDecomposition(Xor())

   @doctest
   In [4]: ld
   Out[4]: LogarithmicDecomposition(|Omega|=4, atoms=11)

   @doctest
   In [5]: abs(ld.coinformation() - (-1.0)) < 1e-10
   Out[5]: True

API
===

:class:`~dit.multivariate.LogarithmicDecomposition` and
:func:`~dit.multivariate.logarithmic_decomposition` live in
:mod:`dit.multivariate.logarithmic_decomposition`.

.. autofunction:: logarithmic_decomposition
