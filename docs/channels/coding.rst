.. coding.rst
.. py:currentmodule:: dit.coding

Coding
======

The :mod:`dit.coding` module builds both *source codes* (which compress a source
:class:`~dit.Distribution`) and *channel codes* (which add redundancy to protect
against channel noise), and exposes their code-theoretic properties.

Source coding
-------------

A *source code* maps the outcomes of a source to codewords over a ``radix``-ary
alphabet (binary by default), with the goal of minimizing the expected number of
code symbols per source symbol -- the *rate*. By the source coding theorem
:cite:`Cover2006`, no uniquely decodable code can have a rate below the source
entropy :math:`\H{X}`.

Base classes
------------

All source codes derive from :class:`SourceCoding`, which defines the
:meth:`~SourceCoding.encode`, :meth:`~SourceCoding.decode`, and
:meth:`~SourceCoding.rate` interface and provides the comparisons to the source
entropy that are common to every source code:

- :meth:`~SourceCoding.source_entropy` -- :math:`\H{X}` in ``radix``-ary digits,
- :meth:`~SourceCoding.redundancy` -- ``rate - source_entropy``,
- :meth:`~SourceCoding.efficiency` -- ``source_entropy / rate``.

Channel codes derive from the companion :class:`ChannelCoding` base class,
described under :ref:`channel-coding` below.

Symbol codes
------------

A :class:`SymbolCode` assigns one codeword to each source outcome. The classic
constructions are available:

- :func:`shannon` -- lengths :math:`\lceil \log_D 1/p \rceil`, codewords from the
  cumulative distribution,
- :func:`fano` -- top-down balanced partitioning,
- :func:`shannon_fano_elias` -- lengths :math:`\lceil \log_D 1/p \rceil + 1`,
  codewords from the midpoint cumulative distribution,
- :func:`huffman` -- the optimal symbol code,
- :func:`length_limited_huffman` -- the optimal code subject to a maximum codeword
  length (via package-merge),
- :func:`golomb` / :func:`rice` -- optimal codes for geometric integer sources.

Beyond the rate-based properties, a :class:`SymbolCode` reports the Kraft sum
(:meth:`~SymbolCode.kraft_sum`), whether it is complete
(:meth:`~SymbolCode.is_complete`), prefix-free (:meth:`~SymbolCode.is_prefix_free`),
uniquely decodable (:meth:`~SymbolCode.is_uniquely_decodable`, via
Sardinas-Patterson), and optimal (:meth:`~SymbolCode.is_optimal`).

Example
-------

.. ipython::

   In [1]: from dit.coding import huffman, shannon

   In [2]: d = dit.Distribution(['a', 'b', 'c', 'd', 'e'], [0.4, 0.2, 0.2, 0.1, 0.1])

   In [3]: code = huffman(d)

   In [4]: code.average_length(), code.source_entropy()

   In [5]: code.is_optimal(), code.is_prefix_free(), code.is_complete()

   In [6]: seq = list(d.outcomes)

   In [7]: code.decode(code.encode(seq)) == seq

Huffman is optimal, so its average length is no larger than the Shannon code's:

.. ipython::

   In [8]: shannon(d).average_length()

Universal integer codes
-----------------------

The universal codes encode the positive integers without reference to a source
distribution, yet stay within a constant factor of optimal across a wide range of
sources: :func:`unary`, :func:`elias_gamma`, :func:`elias_delta`,
:func:`elias_omega`, and :func:`fibonacci`. The :func:`universal_code` helper wraps
a chosen family into a :class:`SymbolCode` over an integer-valued source.

Tunstall codes
--------------

Where a symbol code maps each symbol to a variable-length codeword, a
:func:`tunstall` code parses the source into variable-length *words* and maps each
word to a fixed-length codeword. It is the variable-to-fixed dual of Huffman and
is realized by :class:`TunstallCode`.

Polar source coding
-------------------

Source polarization :cite:`arikan2010source` applies the Arikan transform on the
*source* side. For :math:`N = 2^m` i.i.d. copies of a binary source :math:`X`
(optionally with side information :math:`Y`), the transform
:math:`U^N = X^N G_N` produces synthesized coordinates whose conditional
entropies :math:`\H{U_i \mid U^{i-1}, Y^N}` polarize toward :math:`0`
(deterministic given the past) or :math:`1` (uniform given the past) as :math:`N`
grows. The coordinates that stay near one -- the *high-entropy set* -- are exactly
what a lossless source code must store; the rest are recovered by sequential
decisions and the inverse transform. With side information this is the polar route
to Slepian-Wolf coding.

The finite-block utilities are exact (no density evolution), so they are limited
to small :math:`N`:

- :func:`source_bhattacharyya` -- the source Bhattacharyya parameter
  :math:`Z(X \mid Y) = 2 \sum_y \sqrt{p(0, y) p(1, y)}`, small when :math:`X` is
  nearly determined by :math:`Y` and near one when :math:`X` is nearly uniform,
- :func:`source_polarization_profile` -- the per-coordinate conditional entropy
  and source Bhattacharyya (and an optional Goela-style
  ``max_correlation_with_past`` diagnostic :cite:`goela2014polarized`),
- :func:`source_high_entropy_set` -- the coordinates a code keeps; by default the
  lossless set (every coordinate whose conditional entropy exceeds a tolerance).

The :func:`polar_source` constructor builds a :class:`PolarSourceCode` -- an exact
finite-block source code (binary source, power-of-two block length, optional
decoder side information). Its :meth:`~PolarSourceCode.encode` returns the
high-entropy coordinates and :meth:`~PolarSourceCode.decode` fills the low-entropy
coordinates by exact maximum a posteriori decisions before inverting the
transform. Because the joint table is enumerated exactly, a ``max_states`` guard
prevents accidental exponential blowups.

.. ipython::

   In [1]: from dit.coding import polar_source, source_polarization_profile

   In [2]: dsbs = dit.Distribution(['00', '01', '10', '11'], [0.45, 0.05, 0.05, 0.45])

   In [3]: [round(row['entropy'], 3) for row in source_polarization_profile(dsbs, 4, rv=0, crvs=[1])]

   In [4]: code = polar_source(dsbs, 8, rv=0, crvs=[1])

   In [5]: code.rate(), code.high_entropy_set

.. _channel-coding:

Channel coding
--------------

A *channel code* adds redundancy to a message so that it can be recovered after
transmission over a noisy channel. All channel codes in :mod:`dit.coding` are
binary (over :math:`\mathrm{GF}(2)`). The usual channel is a conditional
:class:`~dit.Distribution` :math:`p(Y \mid X)` -- a discrete memoryless channel
applied independently to each transmitted symbol. The :doc:`/channels/channels` page
catalogs ready-made channels, including :func:`~dit.example_channels.binary_symmetric_channel`
and :func:`~dit.example_channels.binary_erasure_channel` (re-exported from
:mod:`dit.coding` for convenience). Channels with memory can be attached too; see
:ref:`channel-protocol`.

Every channel code derives from :class:`ChannelCoding`, which provides the
information-theoretic evaluation common to all of them:

- :meth:`~ChannelCoding.capacity_gap` -- ``channel_capacity(channel) - rate``, the
  gap to the channel coding theorem's limit :cite:`Cover2006` (capacity is
  computed with the Blahut–Arimoto solver described on :doc:`/channels/channels`;
  a channel that only brackets its capacity yields a ``(lower, upper)`` bracket
  on the gap),
- :meth:`~ChannelCoding.probability_of_error` -- the block-error probability under
  the code's own decoder, computed exactly for small codes (enumerating every
  message and received word) or by Monte Carlo otherwise.

Linear block codes
~~~~~~~~~~~~~~~~~~~

A :class:`LinearCode` is specified by a generator matrix ``G``; the parity-check
matrix ``H`` is derived as a basis for its null space. It supports
hard-decision syndrome decoding (and maximum-likelihood decoding when a channel is
given), and reports the :meth:`~LinearCode.minimum_distance`,
:meth:`~LinearCode.weight_enumerator`, and
:meth:`~LinearCode.error_correcting_capability`. The classical families are
available as constructors: :func:`repetition`, :func:`parity_check`,
:func:`hamming`, :func:`reed_muller`, and :func:`golay`.

Modern codes
~~~~~~~~~~~~

Three modern codes specialize the decoder:

- :func:`ldpc` / :func:`gallager` build an :class:`LDPCCode` decoded by
  sum-product belief propagation over its Tanner graph :cite:`Cover2006`,
- :func:`polar` builds a :class:`PolarCode`, freezing the least reliable
  synthesized bit-channels (by Bhattacharyya parameter) and decoding by successive
  cancellation,
- :func:`convolutional` builds a :class:`ConvolutionalCode` from generator
  polynomials, decoded with the Viterbi algorithm (soft-decision when a channel is
  given).

Channel coding example
~~~~~~~~~~~~~~~~~~~~~~~

The Hamming ``[7, 4, 3]`` code corrects any single error; over a binary symmetric
channel its exact block-error probability matches the closed form
:math:`1 - (1-p)^7 - 7p(1-p)^6`:

.. ipython::

   In [1]: from dit.coding import hamming, binary_symmetric_channel

   In [2]: code = hamming(3)

   In [3]: code.length, code.dimension, code.minimum_distance()

   In [4]: bsc = binary_symmetric_channel(0.05)

   In [5]: code.capacity_gap(bsc)

   In [6]: code.probability_of_error(bsc, method='exact')

   In [7]: 1 - 0.95**7 - 7 * 0.05 * 0.95**6

.. _channel-protocol:

Channels with memory
~~~~~~~~~~~~~~~~~~~~

Codes do not require a :class:`~dit.Distribution`: every channel-aware method
first calls :func:`as_channel`, which wraps a conditional distribution as a
:class:`MemorylessChannel` and accepts any other object implementing a small
duck-typed protocol:

- ``inputs`` and ``outputs`` -- the alphabets, as tuples,
- ``log_likelihood(codeword, received)`` -- :math:`\log p(y^n \mid x^n)`, used
  by maximum-likelihood decoding and by the exact block-error probability,
- ``sample(codeword, prng)`` -- one received word, used by Monte Carlo,
- ``symbol_log_likelihoods(received)`` -- an ``(n, len(inputs))`` array of
  :math:`\log p(y^n \mid x_i = a)` up to a per-row constant, from which belief
  propagation, successive cancellation, and soft Viterbi form their
  per-position metrics,
- optionally ``capacity(**kwargs)`` (a float or a ``(lower, upper)`` bracket) and
  ``bhattacharyya()`` (needed only to *design* a polar code),
- optionally ``feedback_initial()``, ``feedback_step(belief, x)`` and
  ``feedback_capacity(**kwargs)``, used only by the
  :ref:`feedback codes <feedback-codes>`.

A finite-state channel :math:`T(y, s' \mid s, x)` computes ``log_likelihood``
with the forward algorithm and ``symbol_log_likelihoods`` with the
forward-backward (BCJR) recursions :cite:`Bahl1974` under uniform i.i.d. inputs
-- the standard separate detect-then-decode approach. Maximum-likelihood decoding
and the exact error probability remain exact over such a channel; the per-symbol
decoders are approximate because they ignore the correlations the channel
induces between positions.

An object may also define ``__dit_channel__()`` returning a protocol object, so
another library can make its own channel types attachable without :mod:`dit`
knowing about them. For example, :mod:`sofic` finite-state channels (Mealy
machines such as the trapdoor channel) work directly:

.. code-block:: python

   from dit.coding import hamming
   from sofic.examples import trapdoor

   code = hamming(3)
   code.probability_of_error(trapdoor(), method='exact')
   code.capacity_gap(trapdoor())   # a (lower, upper) bracket

.. _feedback-codes:

Codes with feedback
~~~~~~~~~~~~~~~~~~~

With a noiseless feedback link the transmitter sees each output before
choosing the next input, so a code is a strategy
:math:`x_t = f(m, y_{0:t})` and the number of channel uses :math:`N` may be
random. Feedback does not increase the capacity of a memoryless channel
:cite:`Cover2006`, but it makes capacity-achieving schemes much simpler and
improves the error exponent :cite:`Burnashev1976`. On channels with memory
it can raise the capacity itself; the trapdoor channel goes from about
:math:`0.58` bits without feedback to :math:`\log_2 \varphi \approx 0.694`
with it :cite:`Permuter2008`.

A :class:`FeedbackCode` keeps a state that both ends know, updated from each
output alone. The channel is stepped one use at a time through
``feedback_initial()`` and ``feedback_step(belief, x)``, which returns every
``(y, p(y | x, belief), next_belief)``; :class:`MemorylessChannel` provides
both, and a finite-state channel uses the normalized forward vector as its
belief. The exact block-error probability and :math:`\mathbb{E}[N]` come from
walking the tree of outputs, merging branches that reach the same state, and
the rate is :math:`k / \mathbb{E}[N]`.

- :func:`erasure_arq` resends each bit until it arrives unerased. On the binary
  erasure channel it is zero-error with :math:`\mathbb{E}[N] = k / (1 - \epsilon)`,
  so it achieves capacity exactly.
- :func:`stop_and_wait_arq` resends a block code's codeword until the received
  word is a codeword; it fails only when the error pattern is itself a codeword.
- :func:`horstein` is Horstein's bisection scheme :cite:`Horstein1963`: send
  whether the message lies above the receiver's posterior median, stopping
  after ``n`` uses or once the posterior reaches ``1 - delta``. Posterior
  matching :cite:`Shayevitz2011` generalizes it, and the Schalkwijk--Kailath
  scheme :cite:`Schalkwijk1966` is its Gaussian-channel analogue (not
  implemented, since :mod:`dit` channels are discrete).

.. ipython::

   In [1]: from dit.coding import erasure_arq, horstein, binary_erasure_channel, binary_symmetric_channel

   In [2]: erasure_arq(3).performance(binary_erasure_channel(0.3), method='exact')

   In [3]: 3 / 0.7

   In [4]: bsc = binary_symmetric_channel(0.1)

   In [5]: [horstein(3, n=n).probability_of_error(bsc, method='exact') for n in (4, 8, 12)]

APIs
====

.. autoclass:: SourceCoding
   :no-index:
   :members:

.. autoclass:: ChannelCoding
   :no-index:
   :members:

.. autoclass:: SymbolCode
   :no-index:
   :members:

.. autoclass:: TunstallCode
   :no-index:
   :members:

.. autoclass:: PolarSourceCode
   :no-index:
   :members:

.. autofunction:: polar_source
   :no-index:

.. autofunction:: source_bhattacharyya
   :no-index:

.. autofunction:: source_polarization_profile
   :no-index:

.. autofunction:: source_high_entropy_set
   :no-index:

.. autofunction:: shannon
   :no-index:

.. autofunction:: fano
   :no-index:

.. autofunction:: shannon_fano_elias
   :no-index:

.. autofunction:: huffman
   :no-index:

.. autofunction:: length_limited_huffman
   :no-index:

.. autofunction:: golomb
   :no-index:

.. autofunction:: rice
   :no-index:

.. autofunction:: tunstall
   :no-index:

.. autofunction:: universal_code
   :no-index:

.. autofunction:: unary
   :no-index:

.. autofunction:: elias_gamma
   :no-index:

.. autofunction:: elias_delta
   :no-index:

.. autofunction:: elias_omega
   :no-index:

.. autofunction:: fibonacci
   :no-index:

.. autofunction:: as_channel
   :no-index:

.. autoclass:: MemorylessChannel
   :no-index:
   :members:

.. autoclass:: LinearCode
   :no-index:
   :members:

.. autoclass:: LDPCCode
   :no-index:
   :members:

.. autoclass:: PolarCode
   :no-index:
   :members:

.. autoclass:: ConvolutionalCode
   :no-index:
   :members:

.. autofunction:: repetition
   :no-index:

.. autofunction:: parity_check
   :no-index:

.. autofunction:: hamming
   :no-index:

.. autofunction:: reed_muller
   :no-index:

.. autofunction:: golay
   :no-index:

.. autofunction:: ldpc
   :no-index:

.. autofunction:: gallager
   :no-index:

.. autofunction:: polar
   :no-index:

.. autofunction:: convolutional
   :no-index:

.. autoclass:: FeedbackCode
   :no-index:
   :members: start, next_input, update, decode, transmit, performance, probability_of_error, expected_uses, rate, capacity_gap

.. autoclass:: ErasureARQ
   :no-index:

.. autoclass:: StopAndWaitARQ
   :no-index:

.. autoclass:: HorsteinCode
   :no-index:

.. autofunction:: erasure_arq
   :no-index:

.. autofunction:: stop_and_wait_arq
   :no-index:

.. autofunction:: horstein
   :no-index:
