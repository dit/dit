.. accessors.rst

.. py:currentmodule:: dit

Accessors
=========

.. ipython::

   In [1]: from dit import Distribution

   In [2]: d = Distribution(['000', '011', '101', '110'], [1/4]*4)

   In [3]: d.set_rv_names('XYZ')

Indexing
--------

Outcomes can be looked up as concatenated strings or as tuples of coordinate
values. :meth:`~Distribution.sel` selects by named coordinate:

.. ipython::

   @doctest float
   In [4]: d['000']
   Out[4]: 0.25

   @doctest float
   In [5]: d[('0', '0', '0')]
   Out[5]: 0.25

   @doctest float
   In [6]: d.sel(X='0', Y='0', Z='0')
   Out[6]: 0.25

An event (a set of outcomes) is summed by
:meth:`~Distribution.event_probability`:

.. ipython::

   @doctest float
   In [7]: d.event_probability([('0', '0', '0'), ('0', '1', '1')])
   Out[7]: 0.5

Arrays and tables
-----------------

.. ipython::

   @doctest
   In [8]: d.outcomes
   Out[8]: (('0', '0', '0'), ('0', '1', '1'), ('1', '0', '1'), ('1', '1', '0'))

   In [9]: d.pmf

   In [10]: d.alphabet

   In [11]: d.to_dict()

The underlying :class:`~xarray.DataArray` is ``d.data``.
:meth:`~Distribution.to_numpy` returns the dense ndarray.

Base, copy, sampling
--------------------

:meth:`~Distribution.set_base` converts between linear probabilities and log
probabilities (base ``2``, ``'e'``, or any positive float).
:meth:`~Distribution.copy` duplicates a distribution, optionally changing
base. :meth:`~Distribution.rand` draws outcomes.
:meth:`~Distribution.normalize` renormalizes the free-variable slices.

Queries
-------

- :meth:`~Distribution.is_conditional` — whether ``given_vars`` is nonempty
- :meth:`~Distribution.is_symbolic` — sympy probabilities
- :meth:`~Distribution.is_numerical` — numeric probabilities
- :meth:`~Distribution.is_approx_equal` — compare two distributions

API
===

.. automethod:: Distribution.__getitem__

.. automethod:: Distribution.sel

.. automethod:: Distribution.event_probability

.. automethod:: Distribution.set_rv_names

.. automethod:: Distribution.set_base

.. automethod:: Distribution.copy

.. automethod:: Distribution.rand

.. automethod:: Distribution.normalize
