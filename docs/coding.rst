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

.. _channel-coding:

Channel coding
--------------

A *channel code* adds redundancy to a message so that it can be recovered after
transmission over a noisy channel. All channel codes in :mod:`dit.coding` are
binary (over :math:`\mathrm{GF}(2)`), and a channel is supplied as a conditional
:class:`~dit.Distribution` :math:`p(Y \mid X)` -- a discrete memoryless channel
applied independently to each transmitted symbol. The convenience constructors
:func:`binary_symmetric_channel` and :func:`binary_erasure_channel` build the two
standard binary channels (the erasure channel uses the integer ``2`` for an
erasure).

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

.. autofunction:: binary_symmetric_channel

.. autofunction:: binary_erasure_channel
