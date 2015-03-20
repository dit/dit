.. renyi_entropy.rst
.. py:module:: dit.other.renyi_entropy

*************
Rényi Entropy
*************

The Rényi entropy is a spectrum of generalizations to the Shannon
:doc:`../multivariate/entropy`:

.. math::

   \RE[X] = \frac{1}{1-\alpha} \log_2 \left( \sum_{x \in \mathcal{X}} p(x)^\alpha \right)

.. ipython::

   In [1]: from dit.other import renyi_entropy

   In [2]: from dit.example_dists import binomial

   In [3]: d = binomial(15, 0.4)

   @doctest float
   In [4]: renyi_entropy(d, 3)
   Out[4]: 2.6611840717104625


Special Cases
=============

For several values of :math:`\alpha`, the Rényi entropy takes on particular
values.


:math:`\alpha = 0`
------------------

When :math:`\alpha = 0` the Rényi entropy becomes what is known as the Hartley
entropy:

.. math::

    \H_{0}[X] = \log_2 |X|

.. ipython::

   @doctest float
   In [5]: renyi_entropy(d, 0)
   Out[5]: 4.0


:math:`\alpha = 1`
------------------

When :math:`\alpha = 1` the Rényi entropy becomes the standard Shannon entropy:

.. math::

    \H_{1}[X] = \H[X]

.. ipython::

   @doctest float
   In [6]: renyi_entropy(d, 1)
   Out[6]: 2.9688513169509623


:math:`\alpha = 2`
------------------

When :math:`\alpha = 2`, the Rényi entropy becomes what is known as the
collision entropy:

.. math::

    \H_{2}[X] = - \log_2 p(X = Y)

where :math:`Y` is an IID copy of X. This is basically the surprisal of "rolling
doubles"

.. ipython::

   @doctest float
   In [7]: renyi_entropy(d, 2)
   Out[7]: 2.7607270851693615


:math:`\alpha = \infty`
-----------------------

Finally, when :math:`\alpha = \infty` the Rényi entropy picks out the
probability of the most-probable event:

.. math::

    \H_{\infty}[X] = - \log_2 \max_{x \in \mathcal{X}} p(x)

.. ipython::

   @doctest float
   In [8]: renyi_entropy(d, np.inf)
   Out[8]: 2.275104563096674


General Properies
=================

In general, the Rényi entropy is a monotonically decreasing function in
:math:`\alpha`:

.. math::

    \H_{\alpha}[X] \ge \H_{\beta}[X], \quad \beta > \alpha

Further, the following inequality holds in the other direction:

.. math::

    \H_{2}[X] \le 2 \H_{\infty}[X]


API
===

.. autofunction:: renyi_entropy
