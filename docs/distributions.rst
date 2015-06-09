.. distributions.rst
.. py:module:: dit.distribution

*************
Distributions
*************

Here we describe how to create, modify, and manipulate distribution objects.

.. py:module:: dit.npscalardist

Numpy-based ScalarDistribution
==============================

ScalarDistribution specific stuff.

.. automethod:: ScalarDistribution.__init__


.. py:module:: dit.npdist

Numpy-based Distribution
========================

The primary method of constructing a distribution is by supplying both the
outcomes and the probability mass function:

.. ipython::

   In [1]: from dit import Distribution

   In [2]: outcomes = ['000', '011', '101', '110']

   In [3]: pmf = [1/4]*4

   In [4]: xor = Distribution(outcomes, pmf)

   In [5]: print(xor)
   Class:          Distribution
   Alphabet:       ('0', '1') for all rvs
   Base:           linear
   Outcome Class:  str
   Outcome Length: 3
   RV Names:       None

   x     p(x)
   000   0.25
   011   0.25
   101   0.25
   110   0.25

Another way to construct a distribution is by supplying a dictionary mapping
outcomes to probabilities:

.. ipython::

   In [6]: outcomes_probs = {'000': 1/4, '011': 1/4, '101': 1/4, '110': 1/4}

   In [7]: xor2 = Distribution(outcomes_probs)

   In [8]: print(xor2)
   Class:          Distribution
   Alphabet:       ('0', '1') for all rvs
   Base:           linear
   Outcome Class:  str
   Outcome Length: 3
   RV Names:       None

   x     p(x)
   000   0.25
   011   0.25
   101   0.25
   110   0.25

.. automethod:: Distribution.__init__

To verify that these two distributions are the same, we can use the
`is_approx_equal` method:

.. ipython::

   @doctest
   In [9]: xor.is_approx_equal(xor2)
   Out[9]: True

.. automethod:: Distribution.is_approx_equal
