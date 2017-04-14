.. optimization.rst
.. py:module:: dit.algorithms.scipy_optimizers

************
Optimization
************

It is often useful to construct a distribution :math:`d^\prime` which is consistent with some marginal aspects of :math:`d`, but otherwise optimizes some information measure. For example, perhaps we are interested in constructing a distribution which matches pairwise marginals with another, but otherwise has maximum entropy:

.. ipython::

    In [1]: from dit.algorithms.scipy_optimizers import MaxEntOptimizer

    In [2]: d = dit.example_dists.Xor()

    In [3]: meo = MaxEntOptimizer(d, [[0,1], [0,2], [1,2]])

    In [4]: meo.optimize()

    In [5]: dp = meo.construct_dist()

    In [6]: print(dp)
    Class:          Distribution
    Alphabet:       ('0', '1') for all rvs
    Base:           linear
    Outcome Class:  str
    Outcome Length: 3
    RV Names:       None

    x     p(x)
    000   0.125
    001   0.125
    010   0.125
    011   0.125
    100   0.125
    101   0.125
    110   0.125
    111   0.125

