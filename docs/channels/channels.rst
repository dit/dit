.. channels.rst
.. py:currentmodule:: dit.example_channels

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
   :no-index:

.. autofunction:: binary_erasure_channel
   :no-index:

.. autofunction:: z_channel
   :no-index:

.. autofunction:: binary_asymmetric_channel
   :no-index:

.. autofunction:: binary_symmetric_erasure_channel
   :no-index:

.. autofunction:: q_ary_symmetric_channel
   :no-index:

.. autofunction:: q_ary_erasure_channel
   :no-index:

.. autofunction:: noisy_typewriter
   :no-index:

.. autofunction:: identity_channel
   :no-index:

.. autofunction:: useless_channel
   :no-index:

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
   :no-index:

.. autofunction:: dit.algorithms.channel_capacity_joint
   :no-index:

.. autofunction:: dit.algorithms.channel_capacity_cost
   :no-index:

.. autofunction:: dit.algorithms.sibson_capacity
   :no-index:

Commitment capacity
===================

In bit commitment, Alice commits to a message by sending Bob information about it, and
reveals the message later. The commitment must be *concealing*, so that Bob learns almost
nothing about the message before it is revealed, and *binding*, so that Alice cannot reveal
a different message. Without computational assumptions this is impossible over noiseless
channels, but noisy channels allow it. The commitment capacity of a discrete memoryless
channel is the largest rate at which messages can be committed to with vanishing errors,
even with unlimited noiseless two-way communication on the side
:cite:`winter2003commitment`.

An input is *redundant* when its output distribution :math:`P_x` is a convex combination of
the other inputs' output distributions. Alice can then simulate it by randomizing over the
others without Bob noticing, so it cannot bind her. Removing every redundant input leaves
one input for each vertex of the convex hull of the :math:`P_x`, the *non-redundant*
channel :math:`\tilde{P}`. Winter, Nascimento and Imai show that the commitment capacity
is the maximal equivocation of the non-redundant channel (:cite:`winter2003commitment`,
Theorem 2):

.. math::

   C_{\mathrm{com}} = \max_{p(x)} \H{X | Y}, \qquad p(y \mid x) = \tilde{P}(y \mid x).

It is zero exactly when the output of the non-redundant channel determines its input, so
every other channel allows bit commitment. It is also a lower bound on the rate at which the
channel allows coin tossing. :func:`~dit.algorithms.nonredundant_inputs` finds the
remaining inputs with one linear program per input, and
:func:`~dit.algorithms.maximal_equivocation` maximizes the equivocation of any channel.
Without the reduction, the maximal equivocation is the secrecy capacity of a wiretap
channel :cite:`wyner1975wire` whose legitimate receiver sees the input noiselessly and
whose eavesdropper sees it through the channel.

The equivocation is concave in the input. With :math:`q = rP` and
:math:`D_x = D(P_x \| q)` as above, its gradient is :math:`g_x = -\log_2 r(x) - D_x`, and
:math:`\sum_x r(x) g_x = \H{X | Y}`. So

.. math::

   \H{X | Y} \leq \max_{p(x)} \H{X | Y} \leq \max_x g_x,

the counterpart of Blahut's bounds, and the iteration again stops once they meet. Its
update :math:`r(x) \propto 2^{-D_x}` maximizes :math:`\H{X} - \sum_x r(x) D(P_x \| q)` with
:math:`q` held fixed. That function equals :math:`\H{X | Y} - D(rP \| q)`, so the
equivocation never decreases. Inputs whose output supports overlap are solved together,
and separately from the others: the output reveals which of these blocks was used, so
all the probability goes to the best block.

The paper's examples follow. The binary symmetric channel's commitment capacity is
:math:`H_b(p)`, so the noise that hurts transmission helps commitment:

.. ipython::

   In [1]: from dit.algorithms import commitment_capacity, maximal_equivocation, nonredundant_inputs

   In [2]: commitment_capacity(bsc)

In the next channel, the first input is the average of the other two, which form a noiseless
channel. Mixing it in gives one bit of equivocation, but the commitment capacity is zero:

.. ipython::

   In [3]: import numpy as np

   In [4]: P = np.array([[1 / 2, 1 / 2], [1, 0], [0, 1]])

   In [5]: maximal_equivocation(P)

   In [6]: nonredundant_inputs(P)

   In [7]: commitment_capacity(P)

Transmission and commitment can favour very different inputs. On the Z-channel with
``p = 1/2`` together they exceed the one bit of input entropy:

.. ipython::

   In [8]: from dit.example_channels import z_channel

   In [9]: z = z_channel(0.5)

   In [10]: channel_capacity(z)

   In [11]: commitment_capacity(z)

.. note::

   The commitment capacity is not continuous in the channel. As the crossover probability
   of the binary symmetric channel tends to one half, its commitment capacity tends to one
   bit, but at one half both inputs have the same output distribution, one of them is
   redundant, and the capacity is zero. Inputs whose output distributions lie within
   ``redundancy_tol`` (in :math:`\ell_1`) of the convex hull of the others' count as
   redundant.

.. autofunction:: dit.algorithms.commitment_capacity
   :no-index:

.. autofunction:: dit.algorithms.maximal_equivocation
   :no-index:

.. autofunction:: dit.algorithms.nonredundant_inputs
   :no-index:
