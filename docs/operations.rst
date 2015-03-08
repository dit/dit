.. operations.rst

**********
Operations
**********

There are several operations possible on joint random variables.

Marginal
========

.. automethod:: dit.npdist.Distribution.marginal

Conditional
===========

.. automethod:: dit.npdist.Distribution.condition_on

Join
====

We can construct the join of two random variables:

.. math::

   X \join Y = \max \{ V | V \imore X \land V \imore Y \}

Where :math:`\max` is understood to be maximizing with respect to the entropy.

.. autofunction:: dit.algorithms.lattice.join
.. autofunction:: dit.algorithms.lattice.insert_join

Meet
====

We can construct the meet of two random variabls:

.. math::

   X \meet Y = \max \{ V | V \iless X \land V \iless Y \}

Where :math:`\min` is understood to be minimizing with respect to the entropy.

.. autofunction:: dit.algorithms.lattice.meet
.. autofunction:: dit.algorithms.lattice.insert_meet

Minimal Sufficient Statistic
============================

This method constructs the minimal sufficient statistic of :math:`X` about
:math:`Y`: :math:`X \mss Y`:

.. math::

   X \mss Y = \min \{ V | V \iless X \land \I[X:Y] = \I[V:Y] \}

Again, :math:`\min` is understood to be over entropies.

.. autofunction:: dit.algorithms.minimal_sufficient_statistic.mss
.. autofunction:: dit.algorithms.minimal_sufficient_statistic.insert_mss
