.. pointwise.rst
.. py:currentmodule:: dit.multivariate

.. _pointwise:

******************
Pointwise Measures
******************

The entropy is the average surprisal :math:`h(x) = -\log_2 p(x)` of the outcomes of a random variable :cite:`Cover2006`, and the mutual information is the average of the pointwise mutual information of its outcomes :cite:`fano1961transmission`:

.. math::

   i(x : y) = \log_2 \frac{p(x, y)}{p(x) p(y)} = h(x) + h(y) - h(x, y)

Every measure that is a signed sum of conditional entropies has such a pointwise, or local, form :cite:`lizier2008local`: replace each :math:`\H{A | C}` with the local entropy :math:`h(a | c) = -\log_2 p(a | c)`. A pointwise measure assigns a value to each outcome, and averaging it over the distribution gives the measure itself. ``dit`` provides the pointwise forms of the following measures, where :math:`x_{0:n}` is an outcome of :math:`n` groups of variables and :math:`z` an outcome of the variables conditioned on:

.. math::

   h(x_{0:n} | z) &= -\log_2 p(x_{0:n} | z) \\
   i(x_0 : \ldots : x_{n-1} | z) &= \sum_{\emptyset \neq S \subseteq \{0:n\}} (-1)^{|S| + 1} h(x_S | z) \\
   t(x_{0:n} | z) &= \sum_{i} h(x_i | z) - h(x_{0:n} | z) \\
   r(x_{0:n} | z) &= \sum_{i} h(x_i | x_{\{0:n\} \setminus i}, z) \\
   b(x_{0:n} | z) &= h(x_{0:n} | z) - r(x_{0:n} | z) \\
   \mathit{ii}(x_{0:n} | z) &= (-1)^n i(x_0 : \ldots : x_{n-1} | z) \\
   o(x_{0:n} | z) &= t(x_{0:n} | z) - b(x_{0:n} | z) \\
   s(x_{0:n} | z) &= t(x_{0:n} | z) + b(x_{0:n} | z)

These are the pointwise :doc:`entropy </measures/multivariate/entropy>`, :doc:`coinformation </measures/multivariate/coinformation>`, :doc:`total correlation </measures/multivariate/total_correlation>`, :doc:`residual entropy </measures/multivariate/residual_entropy>`, :doc:`dual total correlation </measures/multivariate/dual_total_correlation>`, :doc:`interaction information </measures/multivariate/interaction_information>`, :doc:`O-information </measures/multivariate/o_information>`, and :doc:`S-information </measures/multivariate/s_information>`. Unlike the mutual information, the pointwise mutual information can be negative: an outcome has negative pointwise mutual information when it is less likely than it would be were the variables independent.

.. ipython::

   In [1]: from dit.multivariate import pointwise_mutual_information

   In [2]: d = dit.Distribution(['00', '01', '10', '11'], [0.4, 0.1, 0.1, 0.4])

   In [3]: pmi = pointwise_mutual_information(d)

   In [4]: print(pmi)

   @doctest float
   In [5]: pmi.expectation()
   Out[5]: 0.27807190511263785

Each function returns a :class:`PointwiseMeasure`. Its values can be looked up by an outcome of the variables the measure depends on, by an outcome of the whole distribution it was computed from, or by a dict giving the value of each variable. :meth:`~PointwiseMeasure.to_dict` lists the value of each outcome in the support, and ``pmi.data`` holds the values as an ``xarray.DataArray``, nan off the support.

.. ipython::

   @doctest float
   In [6]: pmi['01']
   Out[6]: -1.3219280948873622

   @doctest float
   In [7]: pmi[{'X0': '1', 'X1': '1'}]
   Out[7]: 0.6780719051126378

Local O-information
===================

The O-information is positive when a distribution is dominated by redundancy and negative when it is dominated by synergy, and its pointwise form classifies individual outcomes the same way :cite:`scagliarini2022quantifying`. Consider three spins with :math:`p(s) \propto \exp(J (s_0 s_1 + s_0 s_2 + s_1 s_2))`. The O-information has the sign of the coupling :math:`J`, but the two aligned configurations are redundancy-dominated and the other six synergy-dominated whatever its sign. With :math:`J = -1`:

.. ipython::

   In [8]: import itertools

   In [9]: from dit.multivariate import pointwise_o_information

   In [10]: outcomes = [''.join(o) for o in itertools.product('01', repeat=3)]

   In [11]: weights = [np.exp(-sum((2 * int(a) - 1) * (2 * int(b) - 1) for a, b in itertools.combinations(o, 2))) for o in outcomes]

   In [12]: ising = dit.Distribution(outcomes, np.array(weights) / sum(weights))

   In [13]: omega = pointwise_o_information(ising)

   In [14]: print(omega)

   @doctest float
   In [15]: omega.expectation()
   Out[15]: -0.1381030490641982

Partial averages
================

By default :meth:`~PointwiseMeasure.expectation` averages over every variable. Passing ``over`` averages over those variables only, conditional on the others, and returns the result as a pointwise measure of the variables that remain. Averaging the pointwise mutual information over :math:`X` gives the *specific surprise* about :math:`X` from an outcome :math:`y` :cite:`deweese1999measure`:

.. math::

   \sum_x p(x | y) \, i(x : y) = \sum_x p(x | y) \log_2 \frac{p(x | y)}{p(x)} = \DKL{p(X | y) || p(X)}

It is never negative, but it is not the change in uncertainty about :math:`X`, :math:`\H{X} - \H{X | Y = y}`, which DeWeese and Meister call the *specific information*. Their example is a test for a disease with prevalence :math:`\epsilon`, which errs with probability :math:`\epsilon`. A positive result makes the disease as likely as not: a large surprise, but it raises the uncertainty about the disease by almost a bit.

.. ipython::

   In [16]: from dit.multivariate import pointwise_entropy

   In [17]: eps = 0.01

   In [18]: test = dit.Distribution(['00', '01', '10', '11'], [(1 - eps)**2, (1 - eps) * eps, eps**2, eps * (1 - eps)])

   In [19]: surprise = pointwise_mutual_information(test).expectation(over=[0])

   @doctest float
   In [20]: surprise['1']
   Out[20]: 2.3291778797349196

   In [21]: uncertainty = pointwise_entropy(test, [0], [1]).expectation(over=[0])

   @doctest float
   In [22]: pointwise_entropy(test, [0]).expectation() - uncertainty['1']
   Out[22]: -0.9192068641040888

Local information dynamics
==========================

Because a :class:`PointwiseMeasure` accepts the outcomes of the distribution it was computed from, it can assign a value to each step of a time series. For example, the local transfer entropy :math:`i(y_t : x_{t-1} | y_{t-1})` from :math:`X` to :math:`Y` is negative at the steps where :math:`Y` fails to copy :math:`X` :cite:`lizier2008local`:

.. ipython::

   In [23]: from dit.multivariate import pointwise_coinformation

   In [24]: rng = np.random.default_rng(0)

   In [25]: x = rng.integers(0, 2, 1000)

   In [26]: y = np.roll(x, 1) ^ (rng.random(1000) < 0.1)

   In [27]: ts = dit.inference.dist_from_timeseries(np.column_stack([x, y]))

   In [28]: te = pointwise_coinformation(ts, [[3], [0]], [1])

   In [29]: [round(te[(x[t - 1], y[t - 1], x[t], y[t])], 3) for t in range(1, 7)]

   In [30]: y[1:7] == x[:6]

API
===

.. autoclass:: PointwiseMeasure
   :members: expectation, to_dict, to_string
   :no-index:

.. autofunction:: pointwise_entropy
   :no-index:

.. autofunction:: pointwise_coinformation
   :no-index:

.. autofunction:: pointwise_total_correlation
   :no-index:

.. autofunction:: pointwise_dual_total_correlation
   :no-index:

.. autofunction:: pointwise_residual_entropy
   :no-index:

.. autofunction:: pointwise_interaction_information
   :no-index:

.. autofunction:: pointwise_o_information
   :no-index:

.. autofunction:: pointwise_s_information
   :no-index:
