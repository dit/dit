.. coding.rst
.. py:module:: dit.coding

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
binary (over :math:`\mathrm{GF}(2)`), and a channel is supplied as a conditional
:class:`~dit.Distribution` :math:`p(Y \mid X)` -- a discrete memoryless channel
applied independently to each transmitted symbol. The :doc:`channels` page
catalogs ready-made channels, including :func:`~dit.example_channels.binary_symmetric_channel`
and :func:`~dit.example_channels.binary_erasure_channel` (re-exported from
:mod:`dit.coding` for convenience).

Every channel code derives from :class:`ChannelCoding`, which provides the
information-theoretic evaluation common to all of them:

- :meth:`~ChannelCoding.capacity_gap` -- ``channel_capacity(channel) - rate``, the
  gap to the channel coding theorem's limit :cite:`Cover2006`,
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

APIs
====

.. autoclass:: SourceCoding
   :members:

.. autoclass:: ChannelCoding
   :members:

.. autoclass:: SymbolCode
   :members:

.. autoclass:: TunstallCode
   :members:

.. autoclass:: PolarSourceCode
   :members:

.. autofunction:: polar_source

.. autofunction:: source_bhattacharyya

.. autofunction:: source_polarization_profile

.. autofunction:: source_high_entropy_set

.. autofunction:: shannon

.. autofunction:: fano

.. autofunction:: shannon_fano_elias

.. autofunction:: huffman

.. autofunction:: length_limited_huffman

.. autofunction:: golomb

.. autofunction:: rice

.. autofunction:: tunstall

.. autofunction:: universal_code

.. autofunction:: unary

.. autofunction:: elias_gamma

.. autofunction:: elias_delta

.. autofunction:: elias_omega

.. autofunction:: fibonacci

.. autoclass:: LinearCode
   :members:

.. autoclass:: LDPCCode
   :members:

.. autoclass:: PolarCode
   :members:

.. autoclass:: ConvolutionalCode
   :members:

.. autofunction:: repetition

.. autofunction:: parity_check

.. autofunction:: hamming

.. autofunction:: reed_muller

.. autofunction:: golay

.. autofunction:: ldpc

.. autofunction:: gallager

.. autofunction:: polar

.. autofunction:: convolutional
