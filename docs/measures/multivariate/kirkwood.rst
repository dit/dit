.. kirkwood.rst
.. py:module:: dit.multivariate.kirkwood

.. _kirkwood_mutual_information:
.. _kirkwood:

**********************************************
Kirkwood and Ouroboros Mutual Informations
**********************************************

Both measures in this module are the Kullback-Leibler divergence :cite:`Cover2006` from a joint distribution to a *normalized* log-linear product of its marginals,

.. math::

   D_{KL}\left(p \,\middle\|\, \frac{\tilde{p}}{Z}\right), \qquad \tilde{p}(x) = \prod_S p(x_S)^{c_S}, \qquad Z = \sum_x \tilde{p}(x).

They differ only in the exponents :math:`c_S`. Wherever a marginal used by :math:`\tilde{p}` vanishes, :math:`\tilde{p}` is set to zero, so the support of :math:`p` is always contained in the support of the approximation and the divergence is finite.

Kirkwood Mutual Information
===========================

The generalized Kirkwood superposition approximation :cite:`watanabe1960information` is the Möbius product over proper subsets of the variables:

.. math::

   \tilde{p}(x_{0:n}) = \prod_{\emptyset \neq S \subsetneq \{0, \ldots, n-1\}} p(x_S)^{(-1)^{n-1-|S|}}

For three variables this is the classical :math:`p(x, y) p(x, z) p(y, z) / \left(p(x) p(y) p(z)\right)`, and the divergence from its normalization was introduced by :cite:`wan2010boost` as a fast screening statistic for gene-gene interactions. The Kirkwood mutual information :math:`K` relates to the :ref:`coinformation` through the log normalizer :cite:`kubkowski2020asymptotic`:

.. math::

   K[X_0 : \cdots : X_{n-1}] = (-1)^n \I{X_0 : \cdots : X_{n-1}} + \log_2 Z

For two variables the approximation is the product of marginals, so :math:`K` is the :ref:`mutual_information`. The giant bit is reproduced exactly by its Kirkwood approximation, and parity carries exactly one bit:

.. ipython::

   In [1]: from dit.example_dists import giant_bit, n_mod_m

   In [2]: from dit.multivariate import kirkwood_mutual_information as K

   @doctest float
   In [3]: [K(giant_bit(n, 2)) for n in range(3, 6)]
   Out[3]: [0.0, 0.0, 0.0]

   @doctest float
   In [4]: [K(n_mod_m(n, 2)) for n in range(3, 6)]
   Out[4]: [1.0, 1.0, 1.0]

Ouroboros Mutual Information
============================

For three variables the Kirkwood approximation is a loop of channels, :math:`p(x \mid y) p(y \mid z) p(z \mid x)`, each variable feeding the next. For :math:`n \geq 4` no such reading exists. A product of conditional distributions has exponents summing to zero, but the Kirkwood exponents sum to :math:`1 + (-1)^n`. Even for odd :math:`n \geq 5`, the :math:`\binom{n}{2}` subsets of size :math:`n-2` would each need a distinct channel output among only :math:`n` subsets of size :math:`n-1`.

The order-:math:`k` ouroboros approximation instead takes the symmetric geometric mean over every wiring in which each variable is the output of a channel fed by :math:`k` of the other variables:

.. math::

   \tilde{p}_k(x_{0:n}) = \frac{\prod_{|S| = k+1} p(x_S)^{n / \binom{n}{k+1}}}{\prod_{|S| = k} p(x_S)^{n / \binom{n}{k}}}, \qquad 1 \leq k \leq n - 2

For :math:`n = 3` and :math:`k = 1` this is the Kirkwood approximation. For :math:`k = 1` it is the :math:`(n-1)`-th root of the pairwise conditional composite likelihood :cite:`varin2011overview`. Because the approximation is log-linear in the :math:`(k+1)`-marginals, :math:`O_k` upper bounds the divergence from the maximum entropy distribution consistent with those marginals. No canonical literature source is known for this construction.

Low orders detect structure that the Kirkwood approximation ignores. A triadic distribution alongside an independent bit has no four-way structure, but its pairwise ouroboros approximation misses the triad:

.. ipython::

   In [5]: from dit import Distribution as D

   In [6]: from dit.example_dists import triadic

   In [7]: from dit.multivariate import ouroboros_mutual_information as O

   In [8]: d = triadic @ D(['0', '1'], [1/2, 1/2])

   @doctest float
   In [9]: [K(d), O(d, order=1), O(d, order=2)]
   Out[9]: [0.0, 1.0, 0.0]

API
===

.. autofunction:: kirkwood_mutual_information

.. autofunction:: ouroboros_mutual_information
