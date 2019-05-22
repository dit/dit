.. entropy.rst
.. py:module:: dit.multivariate.entropy

*******
Entropy
*******

The entropy measures the total amount of information contained in a set of random variables, :math:`X_{0:n}`, potentially excluding the information contain in others, :math:`Y_{0:m}`.

.. math::

   \H{X_{0:n} | Y_{0:m}} =  -\sum_{\substack{x_{0:n} \in \mathcal{X}_{0:n} \\ y_{0:m} \in \mathcal{Y}_{0:m}}} p(x_{0:n}, y_{0:m}) \log_2 p(x_{0:n}|y_{0:m})

Let's consider two coins that are interdependent: the first coin fips fairly, and if the first comes up heads, the other is fair, but if the first comes up tails the other is certainly tails:

.. ipython::

   In [1]: d = dit.Distribution(['HH', 'HT', 'TT'], [1/4, 1/4, 1/2])

We would expect that entropy of the second coin conditioned on the first coin would be :math:`0.5` bits, and sure enough that is what we find:

.. ipython::

   In [2]: from dit.multivariate import entropy

   @doctest float
   In [2]: entropy(d, [1], [0])
   Out[2]: 0.5

And since the first coin is fair, we would expect it to have an entropy of :math:`1` bit:

.. ipython::

   @doctest float
   In [3]: entropy(d, [0])
   Out[3]: 1.0

Taken together, we would then expect the joint entropy to be :math:`1.5` bits:

.. ipython::

   @doctest float
   In [4]: entropy(d)
   Out[4]: 1.5


Visualization
=============

Below we have a pictoral representation of the joint entropy for both 2 and 3 variable joint distributions.

.. image:: ../../images/idiagrams/h_xy.png
   :alt: The entropy :math:`\H{X, Y}`
   :width: 342px
   :align: center

.. image:: ../../images/idiagrams/h_xyz.png
   :alt: The entropy :math:`\H{X, Y, Z}`
   :width: 357px
   :align: center


API
===

.. autofunction:: entropy
