.. operations.rst

**********
Operations
**********

There are several operations possible on joint random variables. Let's consider the standard ``xor`` distribution:

.. ipython::

   In [1]: d = dit.Distribution(['000', '011', '101', '110'], [1/4]*4)

   In [2]: d.set_rv_names('XYZ')


.. py:currentmodule:: dit

Marginal
========

:meth:`~Distribution.marginal` returns a distribution containing only the
random variables specified; :meth:`~Distribution.marginalize` returns one
containing all free variables *except* the ones specified. Both accept
positional names or a single list. See :doc:`distributions/algebra` for the
chain-rule product and quotient that reconstruct joints from these pieces.

.. ipython::
   :doctest:

   In [3]: print(d.marginal(['X', 'Y']))
   Class:    Distribution
   Alphabet: (('0', '1'), ('0', '1'))
   Base:     linear
   <BLANKLINE>
   x            p(X,Y)
   ('0', '0')   0.25
   ('0', '1')   0.25
   ('1', '0')   0.25
   ('1', '1')   0.25

   In [4]: print(d.marginalize(['X', 'Y']))
   Class:    Distribution
   Alphabet: (('0', '1'),)
   Base:     linear
   <BLANKLINE>
   x        p(Z)
   ('0',)   0.5
   ('1',)   0.5

.. automethod:: Distribution.marginal
   :no-index:
.. automethod:: Distribution.marginalize
   :no-index:

Conditional
===========

Native :meth:`~Distribution.condition_on` takes positional variable names and
returns a **single** conditional :class:`~dit.Distribution`:

.. ipython::

   In [5]: p_z_xy = d.condition_on('X', 'Y')

   @doctest
   In [6]: print(p_z_xy)
   Class:    Distribution
   Alphabet: (('0', '1'), ('0', '1'), ('0', '1'))
   Base:     linear
   <BLANKLINE>
   x                 p(Z|X,Y)
   ('0', '0', '0')   1.0
   ('0', '1', '1')   1.0
   ('1', '0', '1')   1.0
   ('1', '1', '0')   1.0

Passing a list or the ``crvs`` / ``rvs`` keywords uses the older
compatibility form, which returns a pair
``(marginal, list_of_conditionals)`` — one slice per outcome of the
conditioning variables:

.. ipython::

   In [7]: marginal, cdists = d.condition_on(['X', 'Y'], rvs=['Z'])

   @doctest
   In [8]: print(marginal)
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
   In [9]: print(cdists[0])
   Class:    Distribution
   Alphabet: (('0', '1'),)
   Base:     linear
   <BLANKLINE>
   x        p(Z)
   ('0',)   1.0

.. automethod:: Distribution.condition_on
   :no-index:

.. py:module:: dit.algorithms.lattice

Join
====

We can construct the join of two random variables:

.. math::

   X \join Y = \min \{ V | V \imore X \land V \imore Y \}

Where :math:`\min` is understood to be minimizing with respect to the entropy.

.. ipython::

   In [10]: from dit.algorithms.lattice import join

   @doctest
   In [11]: print(join(d, ['XY']))
   Class:    Distribution
   Alphabet: (0, 1, 2, 3)
   Base:     linear
   <BLANKLINE>
   x   p(x)
   0   0.25
   1   0.25
   2   0.25
   3   0.25

.. autofunction:: join
.. autofunction:: insert_join

Meet
====

We can construct the meet of two random variables:

.. math::

   X \meet Y = \max \{ V | V \iless X \land V \iless Y \}

Where :math:`\max` is understood to be maximizing with respect to the entropy.

.. ipython::

   In [12]: from dit.algorithms.lattice import meet

   In [13]: outcomes = ['00', '01', '10', '11', '22', '33']

   In [14]: d2 = dit.Distribution(outcomes, [1/8]*4 + [1/4]*2)

   In [15]: d2.set_rv_names('XY')

   @doctest
   In [16]: print(meet(d2, ['X', 'Y']))
   Class:    Distribution
   Alphabet: (0,)
   Base:     linear
   <BLANKLINE>
   x   p(x)
   0   1.0

.. autofunction:: meet
.. autofunction:: insert_meet

.. py:module:: dit.algorithms.minimal_sufficient_statistic

Minimal Sufficient Statistic
============================

This method constructs the minimal sufficient statistic of :math:`X` about
:math:`Y`: :math:`X \mss Y`:

.. math::

   X \mss Y = \min \{ V | V \iless X \land \I{X:Y} = \I{V:Y} \}

.. ipython::

   In [17]: from dit.algorithms import insert_mss

   In [18]: d2 = dit.Distribution(['00', '01', '10', '11', '22', '33'], [1/8]*4 + [1/4]*2)

   @doctest
   In [19]: print(insert_mss(d2, -1, [0], [1]))
   Class:    Distribution
   Alphabet: (('0', '1', '2', '3'), ('0', '1', '2', '3'), (0, 1, 2))
   Base:     linear
   <BLANKLINE>
   x               p(X0,X1,X2)
   ('0', '0', 2)   0.125
   ('0', '1', 2)   0.125
   ('1', '0', 2)   0.125
   ('1', '1', 2)   0.125
   ('2', '2', 0)   0.25
   ('3', '3', 1)   0.25

Again, :math:`\min` is understood to be over entropies.

.. autofunction:: mss
.. autofunction:: insert_mss
