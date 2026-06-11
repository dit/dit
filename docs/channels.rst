.. channels.rst
.. py:module:: dit.example_channels

Channels
========

The :mod:`dit.example_channels` module is a catalog of canonical discrete
memoryless channels (DMCs). Each constructor returns a *conditional*
:class:`~dit.Distribution` :math:`p(Y \mid X)` -- the representation consumed by
:func:`~dit.algorithms.channel_capacity` and by the channel-coding evaluation
layer in :mod:`dit.coding`. Alphabets are integer-valued; an erasure is the
integer just past the input alphabet (binary erasure ``2``, q-ary erasure ``q``).

The catalog
-----------

Binary-input channels:

- :func:`binary_symmetric_channel` -- each bit is flipped with probability ``p``;
  capacity :math:`1 - H_b(p)`,
- :func:`binary_erasure_channel` -- each bit is erased with probability
  :math:`\epsilon`; capacity :math:`1 - \epsilon`,
- :func:`z_channel` -- a ``0`` is received perfectly while a ``1`` is flipped to
  ``0`` with probability ``p`` (the canonical asymmetric channel),
- :func:`binary_asymmetric_channel` -- independent crossover probabilities for
  ``0`` and ``1`` (the BSC and Z-channel are special cases),
- :func:`binary_symmetric_erasure_channel` -- errors and erasures together.

q-ary channels:

- :func:`q_ary_symmetric_channel` -- correct with probability ``1 - p``, else a
  uniform wrong symbol; capacity :math:`\log_2 q - H_b(p) - p \log_2(q - 1)`,
- :func:`q_ary_erasure_channel` -- erased with probability :math:`\epsilon`;
  capacity :math:`(1 - \epsilon) \log_2 q`,
- :func:`noisy_typewriter` -- each letter maps to itself or the next, each with
  probability one half; capacity :math:`\log_2(n / 2)`.

Trivial endpoints:

- :func:`identity_channel` -- the noiseless channel, capacity :math:`\log_2 n`,
- :func:`useless_channel` -- output independent of input, capacity ``0``.

Example
-------

A channel can be fed directly to :func:`~dit.algorithms.channel_capacity` or to a
code's :meth:`~dit.coding.ChannelCoding.probability_of_error`:

.. ipython::

   In [1]: from dit.example_channels import binary_symmetric_channel

   In [2]: from dit.algorithms import channel_capacity

   In [3]: bsc = binary_symmetric_channel(0.1)

   In [4]: channel_capacity(bsc)[0]

   In [5]: from dit.coding import hamming

   In [6]: hamming(3).probability_of_error(bsc, method='exact')

APIs
====

.. autofunction:: binary_symmetric_channel

.. autofunction:: binary_erasure_channel

.. autofunction:: z_channel

.. autofunction:: binary_asymmetric_channel

.. autofunction:: binary_symmetric_erasure_channel

.. autofunction:: q_ary_symmetric_channel

.. autofunction:: q_ary_erasure_channel

.. autofunction:: noisy_typewriter

.. autofunction:: identity_channel

.. autofunction:: useless_channel
