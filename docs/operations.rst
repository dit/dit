.. operations.rst

**********
Operations
**********

There are several operations possible on joint random variables. Let's consider the standard ``xor`` distribution:

.. ipython::

   In [1]: d = dit.Distribution(['000', '011', '101', '110'], [1/4]*4)

   In [2]: d.set_rv_names('XYZ')


.. py:module:: dit.npdist

Marginal
========

:mod:`dit` supports two ways of selecting only a subset of random variables. :meth:`~Distribution.marginal` returns a distribution containing only the random variables specified, whereas :meth:`~Distribution.marginalize` return a distribution containing all random variables *except* the ones specified:

.. ipython:: python
   :doctest:

   In [3]: print(d.marginal('XY'))
   Class:          Distribution
   Alphabet:       ('0', '1') for all rvs
   Base:           linear
   Outcome Class:  str
   Outcome Length: 2
   RV Names:       ('X', 'Y')

   x    p(x)
   00   1/4
   01   1/4
   10   1/4
   11   1/4

   In [4]: print(d.marginalize('XY'))
   Class:          Distribution
   Alphabet:       ('0', '1') for all rvs
   Base:           linear
   Outcome Class:  str
   Outcome Length: 1
   RV Names:       ('Z',)

   x   p(x)
   0   1/2
   1   1/2

.. automethod:: Distribution.marginal
.. automethod:: Distribution.marginalize

Conditional
===========

We can also condition on a subset of random variables:

.. ipython:: python

   In [5]: marginal, cdists = d.condition_on('XY')

   @doctest
   In [6]: print(marginal)
   Class:          Distribution
   Alphabet:       ('0', '1') for all rvs
   Base:           linear
   Outcome Class:  str
   Outcome Length: 2
   RV Names:       ('X', 'Y')

   x    p(x)
   00   1/4
   01   1/4
   10   1/4
   11   1/4

   @doctest
   In [7]: print(cdists[0]) # XY = 00
   Class:          Distribution
   Alphabet:       ('0', '1') for all rvs
   Base:           linear
   Outcome Class:  str
   Outcome Length: 1
   RV Names:       ('Z',)

   x   p(x)
   0   1

   @doctest
   In [8]: print(cdists[1]) # XY = 01
   Class:          Distribution
   Alphabet:       ('0', '1') for all rvs
   Base:           linear
   Outcome Class:  str
   Outcome Length: 1
   RV Names:       ('Z',)

   x   p(x)
   1   1

   @doctest
   In [9]: print(cdists[2]) # XY = 10
   Class:          Distribution
   Alphabet:       ('0', '1') for all rvs
   Base:           linear
   Outcome Class:  str
   Outcome Length: 1
   RV Names:       ('Z',)

   x   p(x)
   1   1

   @doctest
   In [10]: print(cdists[3]) # XY = 11
   Class:          Distribution
   Alphabet:       ('0', '1') for all rvs
   Base:           linear
   Outcome Class:  str
   Outcome Length: 1
   RV Names:       ('Z',)

   x   p(x)
   0   1

.. automethod:: Distribution.condition_on

.. py:module:: dit.algorithms.lattice

Join
====

We can construct the join of two random variables:

.. math::

   X \join Y = \min \{ V | V \imore X \land V \imore Y \}

Where :math:`\min` is understood to be minimizing with respect to the entropy.

.. ipython:: python

   In [11]: from dit.algorithms.lattice import join

   @doctest
   In [12]: print(join(d, ['XY']))
   Class:    ScalarDistribution
   Alphabet: (0, 1, 2, 3)
   Base:     linear

   x   p(x)
   0   1/4
   1   1/4
   2   1/4
   3   1/4

.. autofunction:: join
.. autofunction:: insert_join

Meet
====

We can construct the meet of two random variabls:

.. math::

   X \meet Y = \max \{ V | V \iless X \land V \iless Y \}

Where :math:`\max` is understood to be maximizing with respect to the entropy.

.. ipython:: python

   In [13]: from dit.algorithms.lattice import meet

   In [14]: outcomes = ['00', '01', '10', '11', '22', '33']

   In [15]: d2 = dit.Distribution(outcomes, [1/8]*4 + [1/4]*2, sample_space=outcomes)

   In [16]: d2.set_rv_names('XY')

   @doctest
   In [17]: print(meet(d2, ['X', 'Y']))
   Class:    ScalarDistribution
   Alphabet: (0, 1, 2)
   Base:     linear

   x   p(x)
   0   1/4
   1   1/4
   2   1/2

.. autofunction:: meet
.. autofunction:: insert_meet

.. py:module:: dit.algorithms.minimal_sufficient_statistic

Minimal Sufficient Statistic
============================

This method constructs the minimal sufficient statistic of :math:`X` about
:math:`Y`: :math:`X \mss Y`:

.. math::

   X \mss Y = \min \{ V | V \iless X \land \I[X:Y] = \I[V:Y] \}

.. ipython:: python

   In [18]: from dit.algorithms import insert_mss

   In [19]: d2 = dit.Distribution(['00', '01', '10', '11', '22', '33'], [1/8]*4 + [1/4]*2)

   @doctest
   In [20]: print(insert_mss(d2, -1, [0], [1]))
   Class:          Distribution
   Alphabet:       (('0', '1', '2', '3'), ('0', '1', '2', '3'), ('0', '1', '2'))
   Base:           linear
   Outcome Class:  str
   Outcome Length: 3
   RV Names:       None

   x     p(x)
   002   1/8
   012   1/8
   102   1/8
   112   1/8
   220   1/4
   331   1/4

Again, :math:`\min` is understood to be over entropies.

.. autofunction:: mss
.. autofunction:: insert_mss
