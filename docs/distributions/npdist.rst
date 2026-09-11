.. npdist.rst

.. py:currentmodule:: dit

Construction
============

The primary method of constructing a distribution is by supplying both the
outcomes and the probability mass function. Each outcome is an indexable
sequence whose length is the number of random variables:

.. ipython::

   In [1]: from dit import Distribution

   In [2]: outcomes = ['000', '011', '101', '110']

   In [3]: pmf = [1/4]*4

   In [4]: xor = Distribution(outcomes, pmf)

   @doctest
   In [5]: print(xor)
   Class:    Distribution
   Alphabet: (('0', '1'), ('0', '1'), ('0', '1'))
   Base:     linear
   <BLANKLINE>
   x                 p(X0,X1,X2)
   ('0', '0', '0')   0.25
   ('0', '1', '1')   0.25
   ('1', '0', '1')   0.25
   ('1', '1', '0')   0.25

A dictionary mapping outcomes to probabilities is equivalent:

.. ipython::

   In [6]: xor2 = Distribution({'000': 1/4, '011': 1/4, '101': 1/4, '110': 1/4})

   @doctest
   In [7]: xor.is_approx_equal(xor2)
   Out[7]: True

An ndarray is interpreted as a dense pmf, with each axis a random variable
and the index along that axis the variable's value:

.. ipython::

   In [8]: pmf = [[0.5, 0.25], [0.25, 0]]

   In [9]: d = Distribution.from_ndarray(pmf)

   @doctest
   In [10]: print(d)
   Class:    Distribution
   Alphabet: ((0, 1), (0, 1))
   Base:     linear
   <BLANKLINE>
   x        p(X0,X1)
   (0, 0)   0.5
   (0, 1)   0.25
   (1, 0)   0.25

An :class:`~xarray.DataArray` can be passed directly, which is the native
storage format. Dimension names become random-variable names:

.. ipython::

   In [11]: import numpy as np

   In [12]: import xarray as xr

   In [13]: arr = np.zeros((2, 2, 2))

   In [14]: arr[0, 0, 0] = arr[0, 1, 1] = arr[1, 0, 1] = arr[1, 1, 0] = 0.25

   In [15]: data = xr.DataArray(arr, dims=['X', 'Y', 'Z'], coords={'X': ['0', '1'], 'Y': ['0', '1'], 'Z': ['0', '1']})

   In [16]: dx = Distribution(data)

   In [17]: xor.set_rv_names('XYZ')

   @doctest
   In [18]: dx.is_approx_equal(xor)
   Out[18]: True

:meth:`Distribution.from_array` is the same idea with an explicit alphabet
list. :meth:`Distribution.from_factors` rebuilds a joint from a marginal and
a compatible conditional (the inverse of chain-rule multiplication; see
:doc:`algebra`). :meth:`Distribution.from_rv_discrete` wraps a frozen
``scipy.stats.rv_discrete``.

Sparse vs dense
---------------

Zero-probability outcomes can be dropped from the printed table
(:meth:`~dit.Distribution.make_sparse`) or filled back in
(:meth:`~dit.Distribution.make_dense`). :meth:`~dit.Distribution.validate`
checks that free-variable slices are normalized.

Symbolic probabilities are constructed with :mod:`dit.symbolic`; see
:doc:`../symbolic`.

API
===

.. automethod:: Distribution.__init__

.. automethod:: Distribution.from_ndarray

.. automethod:: Distribution.from_array

.. automethod:: Distribution.from_factors

.. automethod:: Distribution.from_rv_discrete

.. automethod:: Distribution.is_approx_equal
