.. rate_distortion.rst
.. py:currentmodule:: dit.rate_distortion

Rate Distortion Theory
======================

.. note::

   We use :math:`p` to denote fixed probability distributions, and :math:`q` to denote probability distributions that are optimized.

Rate-distortion theory :cite:`Cover2006` is a framework for studying optimal lossy compression. Given a distribution :math:`p(x)`, we wish to find :math:`q(\hat{x}|x)` which compresses :math:`X` as much as possible while limiting the amount of user-defined distortion, :math:`d(x, \hat{x})`. The minimum rate (effectively, code book size) at which :math:`X` can be compressed while maintaining a fixed distortion is known as the rate-distortion curve:

.. math::

   R(D) = \min_{q(\hat{x}|x), \langle d(x, \hat{x}) \rangle = D} \I{X : \hat{X}}

By introducing a Lagrange multiplier, we can transform this constrained optimization into an unconstrained one:

.. math::

   \mathcal{L} = \I{X : \hat{X}} + \beta \langle d(x, \hat{x}) \rangle

where minimizing at each :math:`\beta` produces a point on the curve.

``RDCurve`` minimizes :math:`\mathcal{L}` with the Blahut-Arimoto algorithm
:cite:`Blahut1972,Arimoto1972`, which alternates
:math:`q(\hat{x}|x) \propto q(\hat{x}) 2^{-\beta d(x, \hat{x})}` with
:math:`q(\hat{x}) = \sum_x p(x) q(\hat{x}|x)` (see also :cite:`Cover2006`,
Section 10.8). When the distortion does not depend on :math:`q` (as for the
Hamming distortion) the problem is convex, so two deterministic starts suffice,
and the iteration stops once Blahut's upper and lower bounds on
:math:`\min \mathcal{L}` agree. With :math:`c(x) = \sum_{\hat{x}} q(\hat{x}) 2^{-\beta d(x, \hat{x})}`,
those bounds differ by
:math:`\log_2 \max_{\hat{x}} \sum_x p(x) 2^{-\beta d(x, \hat{x})} / c(x)`.
Distortions that depend on :math:`q` (residual entropy, the information
bottleneck) are not convex and keep multiple random restarts.

Example
-------

It is known that under the Hamming distortion (:math:`d(x, \hat{x}) = \left[ x \neq \hat{x} \right]`) the rate-distortion function for a biased coin has the following solution: :math:`R(D) = \H{p} - \H{D}`:

.. ipython::
   :okwarning:

   In [1]: from dit.rate_distortion import RDCurve

   In [2]: d = dit.Distribution(['0', '1'], [1/2, 1/2])

   @savefig rate_distortion.png width=1000 align=center
   In [3]: RDCurve(d, beta_num=26).plot();

Distortion Measures
-------------------

``RDCurve`` takes its distortion through the ``distortion`` argument. Three are
provided in :mod:`dit.rate_distortion`:

* ``hamming`` (the default): :math:`d(x, \hat{x}) = \left[ x \neq \hat{x} \right]`.
* ``residual_entropy``: :math:`d(x, \hat{x}) = -\log_2 q(x | \hat{x}) - \log_2 q(\hat{x} | x)`,
  whose average is the :doc:`residual entropy </measures/multivariate/residual_entropy>` of :math:`X\hat{X}`.
* ``maximum_correlation``: penalizes the :doc:`maximum correlation
  </measures/divergences/maximum_correlation>` between :math:`X` and :math:`\hat{X}`.

Each is a named tuple of a display name, a distortion matrix (when one
exists), and the optimizer class that handles it. Note that the
``hamming`` *distortion* here is unrelated to the Hamming *code*
:func:`dit.coding.hamming`.

.. code-block:: python

   from dit.rate_distortion import RDCurve, residual_entropy

   RDCurve(d, distortion=residual_entropy)

Information Bottleneck
======================

The information bottleneck :cite:`tishby2000information` is a form of rate-distortion where the distortion measure is given by:

.. math::

   d(x, \hat{x}) = D\left[~p(Y | x)~\mid\mid~q(Y | \hat{x})~\right]

where :math:`D` is an arbitrary divergence measure, and :math:`\hat{X} - X - Y` form a Markov chain. Traditionally, :math:`D` is the :doc:`/measures/divergences/kullback_leibler_divergence`, in which case the average distortion takes a particular form:

.. math::

   \langle d(x, \hat{x}) \rangle &= \sum_{x, \hat{x}} q(x, \hat{x}) \DKL{ p(Y | x) || q(Y | \hat{x}) } \\
                                 &= \sum_{x, \hat{x}} q(x, \hat{x}) \sum_{y} p(y | x) \log_2 \frac{p(y | x)}{q(y | \hat{x})} \\
                                 &= \sum_{x, \hat{x}, y} q(x, \hat{x}, y) \log_2 \frac{p(y | x) p(x) p(y) q(\hat{x})}{q(y | \hat{x}) p(x) p(y) q(\hat{x})} \\
                                 &= \sum_{x, \hat{x}, y} q(x, \hat{x}, y) \log_2 \frac{p(y | x) p(x)}{p(x) p(y)} \frac{p(y)q(\hat{x})}{q(y | \hat{x}) q(\hat{x})} \\
                                 &= \I{X : Y} - \I{\hat{X} : Y}

Since :math:`\I{X : Y}` is constant over :math:`q(\hat{x} | x)`, it can be removed from the optimization. Furthermore,

.. math::

   \I{X : Y} - \I{\hat{X} : Y} &= (\I{X : Y | \hat{X}} + \I{X : Y : \hat{X}}) - (\I{Y : \hat{X} | X} + \I{X : Y : \hat{X}}) \\
                               &= \I{X : Y | \hat{X}} - \I{Y : \hat{X} | X} \\
                               &= \I{X : Y | \hat{X}}

where the final equality is due to the Markov chain. Due to all this, Information Bottleneck utilizes a "relevance" term, :math:`\I{\hat{X} : Y}`, which replaces the average distortion in the Lagrangian:

.. math::

   \mathcal{L} = \I{X : \hat{X}} - \beta \I{\hat{X} : Y}
   ~.

Though :math:`\I{X : Y | \hat{X}}` is the most simplified form of the average distortion, it is faster to compute :math:`\I{\hat{X} : Y}` during optimization.

Variants and Algorithms
-----------------------

The standard information bottleneck uses :math:`\I{X : \hat{X}}` as the
compression term. The generalized information bottleneck replaces this with

.. math::

   \H{\hat{X}} - \alpha \H{\hat{X} | X}

so that :math:`\alpha = 1` recovers the standard bottleneck, while
:math:`\alpha = 0` gives the deterministic information bottleneck
:cite:`strouse2016deterministic`. The deterministic endpoint minimizes

.. math::

   \H{\hat{X}} - \beta \I{\hat{X} : Y}

up to the constant :math:`\beta\I{X : Y}`, and its optima are hard clusterings
of :math:`X`.

``IBCurve`` supports these variants through ``variant='ib'``,
``variant='gib'``, and ``variant='dib'``. The legacy ``alpha`` argument is still
accepted: ``alpha=1`` is standard IB, ``alpha=0`` is DIB, and intermediate
values are generalized IB. The default ``method='sp'`` uses the generic
optimizer. ``method='ba'`` uses the Blahut-Arimoto-style finite-alphabet
iteration for the standard, unconditional IB. ``method='sequential'`` and
``method='agglomerative'`` use finite-alphabet hard-clustering algorithms for
the unconditional DIB; the latter follows the information-based clustering
viewpoint of building a hierarchy by merging clusters
:cite:`slonim2005information`.

For deterministic prediction tasks, where :math:`Y` is a function of
:math:`X`, sweeps over the IB Lagrangian can miss portions of the IB curve and
can include trivial solutions. Interpret kinks and beta sweeps in those cases
with care :cite:`kolchinsky2018caveats`.


Example
-------

Consider this distribution:

.. ipython::

   In [4]: d = dit.Distribution(['00', '02', '12', '21', '22'], [1/5]*5)

There are effectively three features that the first index, :math:`X`, has regarding the second index, :math:`Y`. We can find them using the standard information bottleneck:

.. ipython::
   :verbatim:

   In [5]: from dit.rate_distortion import IBCurve

   In [6]: IBCurve(d, beta_num=26).plot();

.. image:: /images/information_bottleneck_1.png
   :alt: The information bottleneck curve of the distribution.
   :width: 1000px
   :align: center

We can also find them utilizing the total variation:

.. ipython::
   :verbatim:

   In [7]: from dit.divergences.pmf import variational_distance

   In [8]: IBCurve(d, divergence=variational_distance).plot();

.. image:: /images/information_bottleneck_2.png
   :alt: The information bottleneck curve of the distribution, using the non-standard total variation divergence measure for distortion.
   :width: 1000px
   :align: center

.. note::

   The spiky behavior at low :math:`\beta` values is due to numerical imprecision.

Passing ``divergence`` makes ``IBCurve`` use
:class:`InformationBottleneckDivergence`, the bottleneck whose distortion is
:math:`D[p(Y | x) \mid\mid q(Y | \hat{x})]` for an arbitrary divergence
:math:`D` operating on probability vectors (see :mod:`dit.divergences.pmf`).

The deterministic information bottleneck can be computed with a hard-clustering
solver:

.. ipython::
   :verbatim:

   In [9]: IBCurve(d, variant='dib', method='sequential', beta_num=26).plot();

Privacy Funnel
==============

The privacy funnel :cite:`makhdoumi2014information` is the information
bottleneck turned around. Given private data :math:`S` and public data
:math:`X`, we release :math:`T`, a randomized function of :math:`X` (so that
:math:`S - X - T` form a Markov chain), which is informative about :math:`X`
but reveals as little as possible about :math:`S`:

.. math::

   \mathrm{PF}(R) = \min_{q(t|x), \I{X : T} \geq R} \I{S : T}

The leakage :math:`\I{S : T}` and utility :math:`\I{X : T}` are both convex in
:math:`q(t|x)`, so their difference is not, and the privacy funnel is a
non-convex problem. Its Lagrangian is

.. math::

   \mathcal{L} = \I{S : T} - \beta \I{X : T}

Since :math:`\I{S : T} \leq \I{X : T}`, every :math:`\beta \geq 1` is
minimized by releasing :math:`T = X`, so the Lagrangian is swept over
:math:`\beta \in [0, 1]`.

For a fixed :math:`p(s, x)`, the pairs :math:`(\I{X : T}, \I{S : T})` over
all releases :math:`T` form a convex set. Its upper boundary is the information
bottleneck curve of :math:`X` with respect to :math:`S`, and its lower boundary
is the privacy funnel curve :cite:`asoodeh2020bottleneck`. So the privacy funnel
curve is convex and increasing. Since it often has a narrow range of slopes, a
sweep over :math:`\beta` resolves few of its points, and ``PFCurve`` sweeps the
rate :math:`R` instead. The size of :math:`T` is bounded by :math:`|X| + 1`;
unlike the information bottleneck, :math:`|X|` does not suffice in general
:cite:`asoodeh2020bottleneck`.

``greedy_privacy_funnel`` implements the greedy algorithm of
:cite:`makhdoumi2014information`: starting from :math:`T = X`, it repeatedly
merges the two symbols of :math:`T` whose merger most reduces the leakage while
the utility is at least :math:`R`. Each step is locally optimal, but the result
need not be optimal overall.

Perfect Privacy
---------------

A release with non-zero utility and zero leakage exists if and only if the
vectors :math:`p(S | x)` are linearly dependent :cite:`calmon2015fundamental`;
in particular, whenever :math:`|X| > |S|`. The largest utility of such a
release,

.. math::

   g_0 = \max_{q(t|x), \I{S : T} = 0} \I{X : T}

is computed exactly by ``perfect_privacy_utility``. Every such :math:`T`
writes :math:`p(X)` as a mixture of distributions :math:`p(X | t)` under which
:math:`S` is distributed as :math:`p(S)`. These form a polytope, and since
entropy is concave, the optimal mixture uses only its vertices; the mixing
weights then solve a linear program :cite:`rassouli2021perfect`.

Example
-------

Let :math:`S` be binary and :math:`X` ternary:

.. ipython::

   In [10]: from dit.rate_distortion import admits_perfect_privacy, perfect_privacy_utility, privacy_funnel_rate

   In [11]: d = dit.Distribution(['00', '01', '12', '10'], [0.3, 0.2, 0.3, 0.2])

   In [12]: admits_perfect_privacy(d)

   In [13]: perfect_privacy_utility(d)

   In [14]: privacy_funnel_rate(d, 1.2)

``BottleneckRegionPlotter`` computes both curves and plots the region between
them:

.. ipython::
   :okwarning:

   In [15]: from dit.rate_distortion.plotting import BottleneckRegionPlotter

   @savefig privacy_funnel_region.png width=800 align=center
   In [16]: BottleneckRegionPlotter.from_dist(d).plot();

See Also
========

The :doc:`/channels/gray_wyner` extends rate-distortion ideas to a one-encoder, many-decoder network, and recovers the :doc:`common informations </measures/multivariate/multivariate>` as operating points.

APIs
====

.. autoclass:: RDCurve
   :no-index:

.. autoclass:: IBCurve
   :no-index:

.. autoclass:: InformationBottleneck
   :no-index:

.. autoclass:: GeneralizedInformationBottleneck
   :no-index:

.. autoclass:: DeterministicInformationBottleneck
   :no-index:

.. autoclass:: InformationBottleneckDivergence
   :no-index:

.. autoclass:: PFCurve
   :no-index:

.. autoclass:: PrivacyFunnel
   :no-index:

.. autoclass:: PrivacyFunnelRate
   :no-index:

.. autofunction:: privacy_funnel
   :no-index:

.. autofunction:: privacy_funnel_rate
   :no-index:

.. autofunction:: greedy_privacy_funnel
   :no-index:

.. autoclass:: PrivacyFunnelResult
   :no-index:

.. autofunction:: admits_perfect_privacy
   :no-index:

.. autofunction:: perfect_privacy_utility
   :no-index:

.. autoclass:: dit.rate_distortion.plotting.BottleneckRegionPlotter
   :no-index:
   :members: from_dist, plot
