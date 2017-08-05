.. hypothesis.rst
.. py:function:: dit.util.testing.distributions

****************
Finding Examples
****************

What if you'd like to find a distribution that has a particular property? For example, what if I'd like to find a distribution with a :ref:`coinformation` less that :math:`-0.5`? This is where Hypothesis comes in:

.. ipython::

   In [1]: from hypothesis import find

   In [2]: from dit.utils.testing import distributions

   In [3]: find(distributions(3, 2), lambda d: dit.multivariate.coinformation(d) < -0.5)
   Out[3]:
   Class:          Distribution
   Alphabet:       (0, 1) for all rvs
   Base:           linear
   Outcome Class:  tuple
   Outcome Length: 3
   RV Names:       None

   x           p(x)
   (0, 0, 0)   0.25
   (0, 1, 1)   0.25
   (1, 0, 1)   0.25
   (1, 1, 0)   0.25

What hypothesis has done is use the :py:func:`distributions` *strategy* to randomly test distributions. Once it finds a distribution satisfying the criteria we specified (coinformation less than :math:`-0.5`) it then simplifies the example as much as possible. Here, we see that even though it could have found any distribution, it found the exclusive or distribution, and simplified the probabilities to be uniform.
