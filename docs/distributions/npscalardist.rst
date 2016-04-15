.. npscalardist.rst
.. py:module:: dit.npscalardist

Numpy-based ScalarDistribution
==============================

ScalarDistributions are used to represent distributions over real numbers, for example a six-sided die or the number of heads when flipping 100 coins.

Playing with ScalarDistributions
--------------------------------

First we will enable two optional features: printing fractions by default, and using :func:`__str__` as :func:`__repr__`. Be careful using either of these options, they can incur significant performance hits on some distributions.

.. ipython::

   In [1]: dit.ditParams['print.exact'] = dit.ditParams['repr.print'] = True

We next construct a six-sided die:

.. ipython::

   In [2]: from dit.example_dists import uniform

   In [3]: d6 = uniform(1, 7)

   In [4]: d6
   Out[4]:
   Class:    ScalarDistribution
   Alphabet: (1, 2, 3, 4, 5, 6)
   Base:     linear

   x   p(x)
   1   1/6
   2   1/6
   3   1/6
   4   1/6
   5   1/6
   6   1/6

We can perform standard mathematical operations with scalars, such as adding, subtracting from or by, multiplying, taking the modulo, or testing inequalities.

.. ipython::

   In [5]: d6 + 3
   Out[5]:
   Class:    ScalarDistribution
   Alphabet: (4, 5, 6, 7, 8, 9)
   Base:     linear

   x   p(x)
   4   1/6
   5   1/6
   6   1/6
   7   1/6
   8   1/6
   9   1/6

   In [6]: d6 - 1
   Out[6]:
   Class:    ScalarDistribution
   Alphabet: (0, 1, 2, 3, 4, 5)
   Base:     linear

   x   p(x)
   0   1/6
   1   1/6
   2   1/6
   3   1/6
   4   1/6
   5   1/6

   In [7]: 10 - d6
   Out[7]:
   Class:    ScalarDistribution
   Alphabet: (4, 5, 6, 7, 8, 9)
   Base:     linear

   x   p(x)
   4   1/6
   5   1/6
   6   1/6
   7   1/6
   8   1/6
   9   1/6

   In [8]: 2 * d6
   Out[8]:
   Class:    ScalarDistribution
   Alphabet: (2, 4, 6, 8, 10, 12)
   Base:     linear

   x    p(x)
   2    1/6
   4    1/6
   6    1/6
   8    1/6
   10   1/6
   12   1/6

   In [9]: d6 % 2
   Out[9]:
   Class:    ScalarDistribution
   Alphabet: (0, 1)
   Base:     linear

   x   p(x)
   0   1/2
   1   1/2

   In [10]: (d6 % 2).is_approx_equal(d6 <= 3)
   Out[10]: True

Furthermore, we can perform such operations with two distributions:

.. ipython::

   In [11]: d6 + d6
   Out[11]:
   Class:    ScalarDistribution
   Alphabet: (2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)
   Base:     linear

   x    p(x)
   2    1/36
   3    1/18
   4    1/12
   5    1/9
   6    5/36
   7    1/6
   8    5/36
   9    1/9
   10   1/12
   11   1/18
   12   1/36

   In [12]: (d6 + d6) % 4
   Out[12]:
   Class:    ScalarDistribution
   Alphabet: (0, 1, 2, 3)
   Base:     linear

   x   p(x)
   0   1/4
   1   2/9
   2   1/4
   3   5/18

   In [13]: d6 // d6
   Out[13]:
   Class:    ScalarDistribution
   Alphabet: (0, 1, 2, 3, 4, 5, 6)
   Base:     linear

   x   p(x)
   0   5/12
   1   1/3
   2   1/9
   3   1/18
   4   1/36
   5   1/36
   6   1/36

   In [14]:  d6 % (d6 % 2 + 1)
   Out[14]:
   Class:    ScalarDistribution
   Alphabet: (0, 1)
   Base:     linear

   x   p(x)
   0   3/4
   1   1/4

There are also statistical functions which can be applied to :class:`~dit.ScalarDistributions`:

.. ipython::

   In [15]: from dit.algorithms.stats import *

   @doctest float
   In [16]: median(d6+d6)
   Out[16]: 7.0

   In [17]: from dit.example_dists import binomial

   In [18]: d = binomial(10, 1/3)

   In [19]: d
   Out[19]:
   Class:    ScalarDistribution
   Alphabet: (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
   Base:     linear

   x    p(x)
   0    409/23585
   1    4302/49615
   2    1280/6561
   3    5120/19683
   4    4480/19683
   5    896/6561
   6    1120/19683
   7    320/19683
   8    20/6561
   9    9/26572
   10   1/59046

   @doctest float
   In [20]: mean(d)
   Out[20]: 3.3333333333333335

   @doctest float
   In [21]: median(d)
   Out[21]: 3.0

   @doctest float
   In [22]: standard_deviation(d)
   Out[22]: 1.4907119849998596


API
---

.. automethod:: ScalarDistribution.__init__
