.. optimization.rst
.. py:module:: dit.algorithms.scipy_optimizers

************
Optimization
************

It is often useful to construct a distribution :math:`d^\prime` which is consistent with some marginal aspects of :math:`d`, but otherwise optimizes some information measure. For example, perhaps we are interested in constructing a distribution which matches pairwise marginals with another, but otherwise has maximum entropy:

.. ipython::

    In [1]: from dit.algorithms.scipy_optimizers import MaxEntOptimizer

    In [2]: xor = dit.example_dists.Xor()

    In [3]: meo = MaxEntOptimizer(xor, [[0,1], [0,2], [1,2]])

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

================
Helper Functions
================

There are three special functions to handle common optimization problems:

.. ipython::

    In [7]: from dit.algorithms import maxent_dist, marginal_maxent_dists, pid_broja

The first is maximum entropy distributions with specific fixed marginals. It encapsulates the steps run above:

.. ipython::

    In [8]: print(maxent_dist(xor, [[0,1], [0,2], [1,2]]))
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

The second constructs several maximum entropy distributions, each with all subsets of variables of a particular size fixed:

.. ipython::

    In [9]: k0, k1, k2, k3 = marginal_maxent_dists(xor)

where ``k0`` is the maxent dist corresponding the same alphabets as ``xor``; ``k1`` fixes :math:`p(x_0)`, :math:`p(x_1)`, and :math:`p(x_2)`; ``k2`` fixes :math:`p(x_0, x_1)`, :math:`p(x_0, x_2)`, and :math:`p(x_1, x_2)` (as in the ``maxent_dist`` example above), and finally ``k3`` fixes :math:`p(x_0, x_1, x_2)` (e.g. is the distribution we started with).

Partial Information Decomposition
---------------------------------

Finally, we have :py:func:`pid_broja`. This computes the 2 input, 1 output partial information decomposition as defined :cite:`bertschinger2014quantifying`. We can compute the partial information decomposition where :math:`X_0` and :math:`X_1` are interpreted as inputs, and :math:`X_2` as the output, with the following code:

.. ipython::

    In [10]: sources = [[0], [1]]

    In [11]: target = [2]

    In [12]: pid_broja(xor, sources, target)
    Out[12]: PID(R=0.0, U0=0.0, U1=0.0, S=1.0)

indicating that the redundancy (R) is zero, neither input provides unique informaiton (U0, U1), and there is 1 bit of synergy (S).

===========================
Creating Your Own Optimizer
===========================

``dit.algorithms.scipy_optimizers`` provides two optimization classes for optimizing some quantity while matching arbitrary margins from a reference distribution. The first, :py:class:`dit.algorithms.scipy_optimizers.BaseConvexOptimizer`, is for use when the objective is convex, while the second, :py:class:`dit.algorithms.scipy_optimizers.BaseNonConvexOptimizer` is for use when the objective is non-convex. Simply subclass one of these two and impliment the ``objective`` method and it is good to go.
