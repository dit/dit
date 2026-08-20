.. distributions.rst

*************
Distributions
*************

:class:`~dit.Distribution` is the core class in :mod:`dit`. It represents a
probability distribution over discrete random variables as an `xarray
<https://docs.xarray.dev/>`_ :class:`~xarray.DataArray`: each dimension is a
random variable, the coordinates along that dimension are the variable's
alphabet, and the array values are probabilities.

The class tracks which dimensions are **free** (being described) versus
**given** (conditioned on). A joint :math:`p(X,Y,Z)` has three free variables
and no given variables; a conditional :math:`p(Z \mid X,Y)` has free variable
:math:`Z` and given variables :math:`X,Y`. That metadata is what makes the
chain-rule algebra in :doc:`algebra` work.

By default, unnamed variables are called ``X0``, ``X1``, … . Call
:meth:`~dit.Distribution.set_rv_names` to give them readable names.

Walkthrough
===========

Construct the exclusive-or distribution, name its variables, take a marginal,
and condition:

.. ipython::

   In [1]: from dit import Distribution

   In [2]: d = Distribution(['000', '011', '101', '110'], [1/4]*4)

   In [3]: d.set_rv_names('XYZ')

   @doctest
   In [4]: print(d)
   Class:    Distribution
   Alphabet: (('0', '1'), ('0', '1'), ('0', '1'))
   Base:     linear
   <BLANKLINE>
   x                 p(X,Y,Z)
   ('0', '0', '0')   0.25
   ('0', '1', '1')   0.25
   ('1', '0', '1')   0.25
   ('1', '1', '0')   0.25

   @doctest
   In [5]: print(d.marginal('X', 'Y'))
   Class:    Distribution
   Alphabet: (('0', '1'), ('0', '1'))
   Base:     linear
   <BLANKLINE>
   x            p(X,Y)
   ('0', '0')   0.25
   ('0', '1')   0.25
   ('1', '0')   0.25
   ('1', '1')   0.25

   @doctest
   In [6]: print(d.condition_on('X', 'Y'))
   Class:    Distribution
   Alphabet: (('0', '1'), ('0', '1'), ('0', '1'))
   Base:     linear
   <BLANKLINE>
   x                 p(Z|X,Y)
   ('0', '0', '0')   1.0
   ('0', '1', '1')   1.0
   ('1', '0', '1')   1.0
   ('1', '1', '0')   1.0

The header ``p(Z|X,Y)`` is the distribution's notation: free variables before
the bar, given variables after. Native :meth:`~dit.Distribution.condition_on`
returns that single conditional, not a list of slices.

See also the :doc:`../symbolic` page for sympy-valued probabilities.

.. toctree::

   npdist.rst
   algebra.rst
   accessors.rst
   npscalardist.rst
   constructors.rst
