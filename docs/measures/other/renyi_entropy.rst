.. renyi_entropy.rst
.. py:module:: dit.other.renyi_entropy

*************
Rényi Entropy
*************

The Rényi entropy is a spectrum of generalizations to the Shannon entropy:

.. math::

   \RE[X] = \frac{1}{1-\alpha} \log_2 \left( \sum{x \in \mathcal{X}} p(x)^\alpha \right)

Special Cases
=============

For several values of :math:`\alpha`, the Rényi entropy takes on particular values.

:math:`\alpha = 0`
------------------

When :math:`\alpha = 0` the Rényi entropy becomes what is known as the Hartley
entropy:

.. math::

    \H_{0}[X] = \log_2 |X|

:math:`\alpha = 1`
------------------

When :math:`\alpha = 1` the Rényi entropy becomes the standard Shannon entropy:

.. math::

    \H_{1}[X] = \H[X]

:math:`\alpha = 2`
------------------

When :math:`\alpha = 2`, the Rényi entropy becomes what is known as the
collision entropy:

.. math::

    \H_{2}[X] = - \log_2 p(X = Y)

where :math:`Y` is an IID copy of X. This is basically the surprisal of "rolling
doubles"

:math:`\alpha = \infty`
-----------------------

Finally, when :math:`\alpha = \infty` the Rényi entropy picks out the
probability of the most-probable event:

.. math::

    \H_{\infty}[X] = - \log_2 \max_{x \in \mathcal{X}} p(x)

General Properies
=================

In general, the Rényi entropy is a monotonically decreasing function in
:math:`\alpha`:

.. math::

    \H_{\alpha} \ge \H_{\beta}, \beta > \alpha

Further, the following inequalities holds in the other direction:

.. math::

    \H_{2} \le 2 \H_{\infty}

.. autofunction:: renyi_entropy
