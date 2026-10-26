.. inference.rst
.. py:currentmodule:: dit.inference

*********
Inference
*********

:mod:`dit.inference` estimates distributions and information quantities from
samples. Plug-in counts live alongside bias-corrected entropy estimators,
Markov-order tests and selection, surrogate and bootstrap generators, and
k-nearest-neighbor / Kraskov–Stögbauer–Grassberger estimators for differential
entropy and total correlation.

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
Whittle surrogate set), surrogates, and the stationary bootstrap, which
resamples whole trials.

.. code-block:: python

   from dit.inference import Trials, select_markov_order

   select_markov_order(Trials([run_1, run_2, run_3]), max_order=4)

Undersampling
-------------

Estimators raise an :class:`UndersamplingWarning` when there are more than a
fifth as many distinct words as windows, or when the word space
``len(alphabet) ** L`` exceeds the number of windows. Markov-order test results
report their effective sample size (``MarkovOrderTest.n_windows``).

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
* ``'zhang'`` — Zhang's estimator, whose bias decays exponentially in the
  sample size :cite:`Zhang2012`, computed as in :cite:`Lozano2017`.
* ``'chao_wang_jost'`` — an estimator built from the rate at which new outcomes
  are discovered :cite:`Chao2013`.
* ``'shrink'`` — the James–Stein shrinkage of the empirical distribution toward
  uniform :cite:`Hausser2009`.
* ``'bayes_jeffreys'``, ``'bayes_laplace'``, ``'bayes_perks'``,
  ``'bayes_minimax'`` — the plug-in entropy of the Dirichlet posterior mean with
  concentration :math:`1/2` :cite:`Krichevsky1981`, :math:`1`, :math:`1/K` and
  :math:`\sqrt{N}/K` :cite:`Hausser2009`.
* ``'ansb'`` — the asymptotic NSB estimate for :math:`N \ll K`, from the number
  of coincidences :cite:`Nemenman2004`.
* ``'bonachela'`` — a short-series estimator that balances bias and variance
  :cite:`Bonachela2008`.
* ``'good_turing'`` — the plug-in entropy of the simple Good–Turing
  probabilities :cite:`Gale1995`.
* ``'jackknife'`` — the delete-one jackknife of the plug-in estimate
  :cite:`Efron1981`. :func:`jackknife_entropy` jackknifes any estimator and
  also returns the jackknife variance.

For heavily undersampled data a further group trades speed for accuracy:

* ``'panzeri_treves'`` — a Miller–Madow-style correction that counts the
  relevant outcomes with a Bayesian estimate :cite:`Panzeri1996`.
* ``'quadratic_extrapolation'`` — extrapolates the plug-in entropy of
  subsamples of size :math:`N`, :math:`N/2` and :math:`N/4` to infinite data
  :cite:`Strong1998`; dit uses the exact expected subsample entropies.
* ``'wu_yang'`` — the minimax-rate polynomial estimator :cite:`Wu2016`.
* ``'bub'`` — Paninski's best-upper-bound estimator :cite:`Paninski2003`.
* ``'unseen'`` — Valiant & Valiant's linear-programming estimate of the
  unseen part of the distribution :cite:`Valiant2017`.
* ``'pym'`` — the Pitman–Yor mixture posterior mean, for heavy-tailed
  distributions on unbounded alphabets :cite:`Archer2014`.

The accepted names are the keys of the ``dit.inference.ENTROPY_ESTIMATORS``
registry, so ``list(ENTROPY_ESTIMATORS)`` enumerates them; the registry's values
are internal implementations, so always select an estimator by name.
:func:`get_counts` produces the count vector these estimators consume from a
sequence, counting every overlapping word of a given length:

.. ipython::

   In [6]: from dit.inference import get_counts, entropy_from_counts

   In [7]: counts = get_counts([0, 1, 0, 0, 1, 1, 0, 1] * 4, 2)

   In [8]: counts

   In [9]: entropy_from_counts(counts, estimator='miller_madow')

``'nsb'``, ``'shrink'``, the ``'bayes_*'`` estimators, ``'good_turing'``,
``'panzeri_treves'``, ``'wu_yang'`` and ``'bub'`` depend on the number of
possible outcomes :math:`K`. Pass ``alphabet_size`` (or
a count vector including zeros); :func:`conditional_mutual_information` takes
``alphabet_sizes`` and uses products of them for its joint terms.

:func:`conditional_entropy_rate` estimates
:math:`h_L = \operatorname{H}[X_L \mid X_{0:L}]`, which decreases to the entropy
rate and equals it once :math:`L` reaches the Markov order :cite:`Cover2006`.

Two further entropy-rate estimates check it:

* :func:`lz_entropy_rate` uses match lengths :cite:`Kontoyiannis1998`. It needs
  no history length, but converges slowly and usually from below.
* :func:`entropy_rate_posterior` draws transition matrices of an order-:math:`k`
  chain from their Dirichlet posterior and reports the entropy rate of each draw
  :cite:`Strelioff2007`, giving credible intervals.

Ordinal patterns
================

:func:`ordinal_patterns` maps each delay window
:math:`(x_{t-(m-1)\tau}, \ldots, x_t)` of a real-valued series to the permutation
that sorts it :cite:`Bandt2002`, giving :math:`m!` symbols that are invariant to
monotone transforms of the series. :func:`relative_rank` encodes the *present*
:math:`x_t` as its rank among the :math:`m` preceding values, so that a present
symbol shares no values with the preceding pattern; this avoids the leakage that
biases symbolic transfer entropy :cite:`Kugiumtzis2012`. Ties are ranked by order
of occurrence, broken by negligible noise, or kept as their own patterns
:cite:`Bian2012`.

:func:`permutation_entropy` is the entropy of the pattern distribution, and
:func:`weighted_permutation_entropy` weights each window by its variance, which
restores some amplitude information :cite:`Fadlallah2013` and serves as a
model-free predictability measure :cite:`Garland2014`.
:func:`multiscale_permutation_entropy` repeats the permutation entropy on
block-averaged versions of the series at successive scales :cite:`Aziz2005,Costa2002`.
:func:`dispersion_entropy` instead classes the values themselves into :math:`c`
levels and takes the entropy of the length-:math:`m` class patterns
:cite:`Rostaghi2016`, keeping amplitude information and avoiding ties.

For symbolic sequences, :func:`lz76_complexity` counts the phrases of the
Lempel–Ziv (1976) exhaustive history :cite:`Lempel1976,Kaspar1987`; normalized, it
is a compression-based estimate of the entropy rate, like :func:`lz_entropy_rate`.

Consecutive windows with :math:`\tau = 1` share :math:`m - 1` values, so each
pattern can be followed by only :math:`m` of the :math:`m!` patterns. Those
forbidden transitions are a property of the encoding, not of the process, so
history-based estimates on ordinal symbols should use non-overlapping patterns.

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

   In [10]: import numpy as np

   In [11]: from dit.inference import markov_order_test, select_markov_order

   In [12]: rng = np.random.default_rng(0)

   In [13]: x = np.cumsum(rng.random(500) < 0.2) % 2  # a sticky first-order chain

   In [14]: markov_order_test(x, order=0, prng=0).pvalue < 0.05
   Out[14]: True

   In [15]: select_markov_order(x, max_order=4, prng=0)
   Out[15]: 1

Conditional mutual information and resampling
=============================================

:func:`conditional_mutual_information` estimates :math:`I[X : Y \mid Z]` from
paired samples with any of the entropy estimators above; lagged copies of a
series give time-delayed and transfer-entropy-style quantities.

:func:`stationary_bootstrap` :cite:`Politis1994` resamples a series in blocks of
geometrically distributed length (whole trials for :class:`Trials`). Each block
junction creates words that were never observed, so statistics of lagged windows
are biased toward independence unless the mean block length is much longer than
the window.

Transfer entropy, its surrogate tests and confidence intervals, false discovery
rate control, and network inference live in the companion package
`infoflow <https://github.com/dit/infoflow>`_, which builds on these estimators
and null generators.

Continuous and mixed data
=========================

:func:`differential_entropy_knn` and :func:`total_correlation_ksg` estimate
differential entropies and (conditional) total correlations from nearest-neighbor
distances. The same distances give:

* :func:`cross_entropy_knn` and :func:`kullback_leibler_divergence_knn` — the
  cross entropy and divergence of two sampled densities :cite:`Leonenko2008,Wang2009`.
* :func:`jensen_shannon_divergence_knn` — the Jensen–Shannon divergence
  :cite:`Lin1991`, from the entropy of the pooled samples.
* :func:`renyi_entropy_knn` and :func:`tsallis_entropy_knn` — differential
  Rényi and Tsallis entropies of order :math:`q < k + 1` :cite:`Leonenko2008`.
  Orders below 1 converge slowly.
* :func:`mutual_information_mixed_ksg` — the mixed KSG estimator, which handles
  variables that are discrete, continuous, or a mixture (with point masses)
  :cite:`Gao2017`.

:func:`gaussian_copula_mutual_information` instead rank-transforms each variable
to a standard normal and computes the information of the Gaussian with the
resulting covariance :cite:`Ince2017`. It lower-bounds the information, is
unaffected by monotone transformations of the marginals, and stays stable in
many dimensions.

API
===

.. autoclass:: Trials
   :no-index:

.. autoclass:: UndersamplingWarning
   :no-index:

.. autofunction:: distribution_from_data
   :no-index:

.. autofunction:: binned
   :no-index:

.. autofunction:: ordinal_patterns
   :no-index:

.. autofunction:: relative_rank
   :no-index:

.. autofunction:: permutation_entropy
   :no-index:

.. autofunction:: weighted_permutation_entropy
   :no-index:

.. autofunction:: multiscale_permutation_entropy
   :no-index:

.. autofunction:: dispersion_entropy
   :no-index:

.. autofunction:: lz76_complexity
   :no-index:

.. autofunction:: dist_from_timeseries
   :no-index:

.. autofunction:: entropy_0
   :no-index:

.. autofunction:: entropy_1
   :no-index:

.. autofunction:: entropy_2
   :no-index:

.. autofunction:: entropy_from_counts
   :no-index:

.. autofunction:: get_counts
   :no-index:

.. autofunction:: jackknife_entropy
   :no-index:

.. autofunction:: block_entropy
   :no-index:

.. autofunction:: conditional_entropy_rate
   :no-index:

.. autofunction:: lz_entropy_rate
   :no-index:

.. autofunction:: entropy_rate_posterior
   :no-index:

.. autoclass:: EntropyRatePosterior
   :no-index:

.. autofunction:: markov_order_test
   :no-index:

.. autoclass:: MarkovOrderTest
   :no-index:

.. autofunction:: select_markov_order
   :no-index:

.. autofunction:: whittle_count
   :no-index:

.. autofunction:: whittle_surrogates
   :no-index:

.. autofunction:: shift_surrogates
   :no-index:

.. autofunction:: block_surrogates
   :no-index:

.. autofunction:: conditional_mutual_information
   :no-index:

.. autofunction:: stationary_bootstrap
   :no-index:

.. autofunction:: differential_entropy_knn
   :no-index:

.. autofunction:: total_correlation_ksg
   :no-index:

.. autofunction:: cross_entropy_knn
   :no-index:

.. autofunction:: kullback_leibler_divergence_knn
   :no-index:

.. autofunction:: jensen_shannon_divergence_knn
   :no-index:

.. autofunction:: renyi_entropy_knn
   :no-index:

.. autofunction:: tsallis_entropy_knn
   :no-index:

.. autofunction:: mutual_information_mixed_ksg
   :no-index:

.. autofunction:: gaussian_copula_mutual_information
   :no-index:

.. autofunction:: copula_normalize
   :no-index:
