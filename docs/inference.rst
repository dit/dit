.. inference.rst
.. py:module:: dit.inference

*********
Inference
*********

:mod:`dit.inference` estimates distributions and information quantities from
samples. Plug-in counts live alongside bias-corrected entropy estimators,
Markov-order tests and selection, surrogate-data significance tests,
stationary-bootstrap confidence intervals, and k-nearest-neighbor /
Kraskov–Stögbauer–Grassberger estimators for differential entropy and total
correlation.

The kNN / KSG implementations optionally use ``scikit-learn`` (installed
with the ``dit[optional]`` extra) when it is available.

Every randomized function takes a ``prng`` argument: ``None`` (draw from
``dit.math.prng``), an integer seed, a :class:`numpy.random.Generator`, or a
:class:`numpy.random.RandomState`.

From samples
============

:func:`distribution_from_data` builds a joint over words of length ``L``
from a sequence of symbols. :func:`dist_from_timeseries` treats each
column of a multivariate series as a variable and appends a
``history_length`` past together with the present. Passing
``history_length='auto'`` chooses the history length with
:func:`select_markov_order` (see below).

.. ipython::

   In [1]: from dit.inference import distribution_from_data, entropy_0, entropy_1

   In [2]: data = [0, 0, 0, 1, 1, 1]

   In [3]: d = distribution_from_data(data, L=1, base='linear')

   @doctest
   In [4]: d.outcomes
   Out[4]: (0, 1)

   @doctest float
   In [5]: entropy_0(data)
   Out[5]: 1.0

The distributions returned here are plug-in (maximum-likelihood) estimates, so
any measure computed from them inherits the plug-in bias. When the number of
distinct words is not small compared to the number of samples, prefer the
count-level estimators below. A :class:`~dit.Distribution` indexes outcomes
densely by each variable's alphabet, so for large alphabets and word lengths the
count-level estimators are also the only practical option.

Independent trials
------------------

Wrap separate recordings in :class:`Trials` to pool them. Counts are summed
over trials, but no word ever spans the boundary between two trials, which
concatenating them would do. Every sample-based function in this module accepts
:class:`Trials`: block entropies, Markov-order tests (each trial gets its own
Whittle surrogate set), transfer entropy, surrogates, and the stationary
bootstrap, which resamples whole trials.

.. code-block:: python

   from dit.inference import Trials, select_markov_order

   select_markov_order(Trials([run_1, run_2, run_3]), max_order=4)

Undersampling
-------------

Estimators raise an :class:`UndersamplingWarning` when there are more than a
fifth as many distinct words as windows, or when the word space
``len(alphabet) ** L`` exceeds the number of windows. Test results report their
effective sample size (``MarkovOrderTest.n_windows``,
``SurrogateTest.n_samples``).

Entropy estimators
==================

:func:`entropy_from_counts` estimates an entropy from outcome counts, and
:func:`block_entropy` and :func:`conditional_entropy_rate` apply it to the
overlapping words of a sequence. The ``estimator`` argument selects:

* ``'plugin'`` — the maximum-likelihood estimate (as :func:`entropy_0`).
* ``'miller_madow'`` — plug-in plus :math:`(K - 1) / 2N` nats for :math:`K`
  observed outcomes :cite:`Miller1955`.
* ``'digamma'`` — :math:`\psi(N) - \sum_i \frac{n_i}{N} \psi(n_i)`
  :cite:`Grassberger1988` (as :func:`entropy_1`).
* ``'grassberger'`` — Grassberger's (2003) estimator :cite:`Grassberger2003`.
  :func:`entropy_2` is the same estimator with :math:`\psi(N)` in place of
  :math:`\ln N`.
* ``'chao_shen'`` — the coverage-adjusted estimator, which accounts for unseen
  outcomes :cite:`Chao2003`.
* ``'nsb'`` — the Nemenman–Shafee–Bialek estimator, a mixture of Dirichlet
  priors chosen to make the prior on the entropy nearly uniform
  :cite:`Nemenman2002`. It needs the number of possible outcomes
  (``alphabet_size``); :func:`block_entropy` supplies ``len(alphabet) ** L``.

:func:`conditional_entropy_rate` estimates
:math:`h_L = \operatorname{H}[X_L \mid X_{0:L}]`, which decreases to the entropy
rate and equals it once :math:`L` reaches the Markov order :cite:`Cover2006`.

Two further entropy-rate estimates check it:

* :func:`lz_entropy_rate` uses match lengths :cite:`Kontoyiannis1998`. It needs
  no history length, but converges slowly and usually from below.
* :func:`entropy_rate_posterior` draws transition matrices of an order-:math:`k`
  chain from their Dirichlet posterior and reports the entropy rate of each draw
  :cite:`Strelioff2007`, giving credible intervals.

Markov order
============

Choosing a history length by hand conflates estimator bias with dynamics.
:func:`markov_order_test` tests the null hypothesis that a sequence is
:math:`n`-th order Markov against order :math:`n + 1`.

With ``method='exact'`` (the default), the null distribution comes from
:func:`whittle_surrogates`: sequences drawn uniformly from all those sharing
the observed :math:`(n + 1)`-gram counts and initial word. Those counts are
sufficient for an :math:`n`-th order chain, so the test is exact at any sample
size :cite:`Pethel2014`. Whittle's formula :cite:`Whittle1955,Billingsley1961`
(:func:`whittle_count`) gives the size of that set. Pethel & Hahs show the
asymptotic chi-squared test :cite:`Anderson1957` (``method='asymptotic'``) is
badly anti-conservative at short lengths and higher orders. Surrogates are
sampled in linear time as random Eulerian trails of the de Bruijn multigraph
:cite:`Kandel1996`.

:func:`select_markov_order` tests :math:`n = 0, 1, \ldots` in turn and returns
the first order that is not rejected (``'exact'`` or ``'chi2'``), or minimizes
AIC or BIC over orders fitted on common targets :cite:`Tong1975,Katz1981`.
Processes with infinite Markov order, such as strictly sofic processes, have no
true order; all methods return longer histories as the sample grows.

.. ipython::

   In [6]: import numpy as np

   In [7]: from dit.inference import markov_order_test, select_markov_order

   In [8]: rng = np.random.default_rng(0)

   In [9]: x = np.cumsum(rng.random(500) < 0.2) % 2  # a sticky first-order chain

   In [10]: markov_order_test(x, order=0, prng=0).pvalue < 0.05
   Out[10]: True

   In [11]: select_markov_order(x, max_order=4, prng=0)
   Out[11]: 1

Significance and uncertainty
============================

:func:`transfer_entropy` :cite:`Schreiber2000` and
:func:`conditional_mutual_information` estimate from samples with any of the
entropy estimators above. Transfer entropy takes separate target and source
history lengths (``history_length``, ``source_history``), a ``lag`` between the
source past and the target present, and ``conditions``: further series whose
pasts are conditioned on, which removes a common driver. ``history_length='auto'``
sets the target history to its estimated Markov order.
:func:`transfer_entropy_test` compares the estimate against surrogates:

* ``null='conditional'`` permutes the source past within strata of the target
  past, which targets exactly :math:`T_{X \to Y} = 0`.
* ``null='whittle'`` replaces the source with :func:`whittle_surrogates`,
  preserving its own Markov structure.
* ``null='shift'`` (:func:`shift_surrogates`) circularly shifts the source.
* ``null='block'`` (:func:`block_surrogates`) shuffles blocks of the source.

The last three test the stronger null that the two series are independent,
while keeping the source's memory. An i.i.d. shuffle would inflate false
positives whenever the source is autocorrelated.
:func:`conditional_mutual_information_test` is the analogous permutation test
for paired samples. For continuous data,
:func:`conditional_mutual_information_test_knn` pairs the KSG estimator with
local-permutation surrogates, which shuffle :math:`X` among nearest neighbours
in :math:`Z` :cite:`Runge2018`.

When many tests are run at once, for example every pair of series in a
network, :func:`benjamini_hochberg` controls the false discovery rate
:cite:`Benjamini1995`. Pass ``dependent=True`` for the Benjamini–Yekutieli
correction, which is valid under arbitrary dependence :cite:`Benjamini2001`.

:func:`stationary_bootstrap` :cite:`Politis1994` resamples a series in blocks
of geometrically distributed length, and :func:`bootstrap_ci` turns any
statistic into a confidence interval. Each block junction creates words that
were never observed, so statistics of lagged windows are biased toward
independence unless the mean block length is much longer than the window.
:func:`transfer_entropy_ci` avoids this by resampling the aligned windows
themselves.

API
===

.. autoclass:: Trials

.. autoclass:: UndersamplingWarning

.. autofunction:: distribution_from_data

.. autofunction:: dist_from_timeseries

.. autofunction:: entropy_0

.. autofunction:: entropy_1

.. autofunction:: entropy_2

.. autofunction:: entropy_from_counts

.. autofunction:: block_entropy

.. autofunction:: conditional_entropy_rate

.. autofunction:: lz_entropy_rate

.. autofunction:: entropy_rate_posterior

.. autoclass:: EntropyRatePosterior

.. autofunction:: markov_order_test

.. autoclass:: MarkovOrderTest

.. autofunction:: select_markov_order

.. autofunction:: whittle_count

.. autofunction:: whittle_surrogates

.. autofunction:: shift_surrogates

.. autofunction:: block_surrogates

.. autofunction:: transfer_entropy

.. autofunction:: transfer_entropy_test

.. autofunction:: transfer_entropy_ci

.. autofunction:: conditional_mutual_information

.. autofunction:: conditional_mutual_information_test

.. autoclass:: SurrogateTest

.. autofunction:: benjamini_hochberg

.. autofunction:: stationary_bootstrap

.. autofunction:: bootstrap_ci

.. autofunction:: differential_entropy_knn

.. autofunction:: total_correlation_ksg

.. autofunction:: conditional_mutual_information_test_knn
