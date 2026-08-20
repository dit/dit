.. union_information.rst
.. py:module:: dit.multivariate.union_information

.. _union_information:

*****************
Union Information
*****************

Finn & Lizier :cite:`finn2020generalised` decompose joint entropy using
pointwise maxima and minima of the marginal surprisals
:math:`h(x_i) = -\log_2 p(x_i)`:

.. math::

   H^{\cup}(X_{0:n}) &= \mathbb{E}\bigl[\max_i h(x_i)\bigr] \\
   H^{\cap}(X_{0:n}) &= \mathbb{E}\bigl[\min_i h(x_i)\bigr] \\
   H^{+}(X_{0:n})    &= \H{X_{0:n}} - H^{\cup}(X_{0:n})

These are the *union entropy*, *intersection entropy*, and *synergistic
entropy*. The *unique entropy* of one group relative to another is
:math:`H^{\cup}(X,Y) - H(Y)`.

.. ipython::

   In [1]: from dit.multivariate import union_entropy, intersection_entropy, synergistic_entropy, unique_entropy

   In [2]: from dit.example_dists import Xor

   In [3]: d = Xor()

   @doctest float
   In [4]: union_entropy(d)
   Out[4]: 1.0

   @doctest float
   In [5]: intersection_entropy(d)
   Out[5]: 1.0

   @doctest float
   In [6]: synergistic_entropy(d)
   Out[6]: 1.0

   @doctest float
   In [7]: unique_entropy(d, [[0], [1]])
   Out[7]: 0.0

The related partial entropy decomposition :math:`H_{\mathrm{mos}}` is
documented with the :doc:`../pid`.

API
===

.. autofunction:: union_entropy

.. autofunction:: intersection_entropy

.. autofunction:: synergistic_entropy

.. autofunction:: unique_entropy
