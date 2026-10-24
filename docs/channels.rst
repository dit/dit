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

Capacity
========

:func:`~dit.algorithms.channel_capacity` takes a list of conditionals
:math:`p(Y \mid X=x)` (or a channel from this catalog) and returns the
capacity together with a capacity-achieving input. For a joint
distribution, use :func:`~dit.algorithms.channel_capacity_joint`.

The capacity is computed with the Blahut-Arimoto algorithm
:cite:`Blahut1972,Arimoto1972` (see also :cite:`Cover2006`, Section 10.8).
Write :math:`q = r P` for the output distribution induced by an input
:math:`r`, and :math:`D_x = D(P_x \| q)`. Then

.. math::

   \sum_x r(x) D_x = \I{X : Y} \leq C \leq \max_x D_x,

and the iteration stops once these two bounds agree to within
``max(atol, rtol * C)`` bits. The returned capacity is therefore certified
rather than merely stationary. Three options trade speed for generality:

- ``method='accelerated'`` (the default) takes the step
  :math:`r \leftarrow r\, e^{\mu D}` of :cite:`matz2004information` with the
  adaptive :math:`\mu_k = D(r_k \| r_{k-1}) / D(q_k \| q_{k-1}) \geq 1`. When
  that step would decrease the mutual information, the classic step
  :math:`\mu = 1` is taken instead. (The paper prints the reciprocal ratio,
  which is at most 1 by data processing and contradicts its own monotonicity
  bound.) ``method='ba'`` is the classic iteration.
- ``eliminate=True`` iterates on a growing active set of inputs
  :cite:`sayir2000iterating`. It adds inputs that violate the Kuhn-Tucker
  condition :math:`D_x \leq C` and stops when none do. That condition is the
  upper bound above, taken over the full alphabet. This pays off when only a
  few of many inputs are used at capacity.
- Weakly symmetric channels (:cite:`Cover2006`, Theorem 7.2.1) are solved in
  closed form. So are square nonsingular channels whose optimal input uses
  every symbol: there :math:`D_x = C` for all :math:`x` is a linear system in
  :math:`\log q`. That closed form is used only when its own bound gap
  certifies it. No canonical reference for it is cited here.

Duplicate inputs and outputs that are never produced are removed before
iterating.

:func:`~dit.algorithms.channel_capacity_cost` adds an average cost constraint
:math:`\mathbb{E}[c(X)] \leq B` through Blahut's Lagrangian form
:cite:`Blahut1972`, and :func:`~dit.algorithms.sibson_capacity` computes
Sibson's capacity of order :math:`\alpha`. The latter is the maximum over inputs of
:func:`~dit.other.sibson_mutual_information`, and it equals the Rényi radius
:math:`\min_q \max_x D_\alpha(P_x \| q)` :cite:`csiszar1995generalized`.

.. ipython::

   In [1]: from dit.algorithms import channel_capacity_cost, sibson_capacity

   In [2]: channel_capacity_cost(bsc, cost=[0, 1], budget=0.2)[0]

   In [3]: sibson_capacity(bsc, order=2)[0]

.. autofunction:: dit.algorithms.channel_capacity

.. autofunction:: dit.algorithms.channel_capacity_joint

.. autofunction:: dit.algorithms.channel_capacity_cost

.. autofunction:: dit.algorithms.sibson_capacity
