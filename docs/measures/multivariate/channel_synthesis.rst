.. channel_synthesis.rst
.. py:module:: dit.multivariate.common_informations.channel_synthesis

*****************
Channel Synthesis
*****************

In distributed channel synthesis :cite:`cuff2013distributed`, an encoder observes an i.i.d. source :math:`X` and sends a message at rate :math:`R` to a decoder, which must produce :math:`Y` so that :math:`(X^n, Y^n)` is indistinguishable (in total variation) from i.i.d. draws of :math:`p(x, y)`. The two nodes also share common randomness at rate :math:`R_0`, independent of :math:`X`. The achievable rate pairs are exactly those for which some :math:`U` with :math:`X - U - Y` satisfies

.. math::

   R \geq \I{X : U}, \qquad R + R_0 \geq \I{X, Y : U}.

:func:`channel_synthesis_rate` computes the minimal communication rate for a given :math:`R_0`,

.. math::

   R(R_0) = \min_{X - U - Y} \max\left\{ \I{X : U},\ \I{X, Y : U} - R_0 \right\}.

Its two extremes are familiar quantities :cite:`cuff2010coordination`. Without common randomness, :math:`R(0)` is the :doc:`wyner_common_information` :math:`\C{X : Y}`. Common randomness lowers the rate until it reaches the mutual information :math:`\I{X : Y}`, which happens once :math:`R_0` is at least the :doc:`necessary_conditional_entropy` :math:`\H{Y \dagger X}`.

Variants
========

Broadcast channels
   With several decoders that receive the same message and produce :math:`Y_1, \ldots, Y_m`, :math:`Y` is replaced by :math:`Y_1, \ldots, Y_m` and :math:`U` must render :math:`X, Y_1, \ldots, Y_m` mutually independent. Pass ``outputs`` as a list of lists, one per decoder.

Public channels
   If the message is public and must look independent of the synthesized variables, the sum-rate bound becomes :math:`R_0 \geq \I{X, Y : U}` (pass ``public=True``). The common randomness must then be at least :math:`\C{X : Y}`; below that, :func:`channel_synthesis_rate` raises an exception.

Local randomness
   The decoder's private randomness rate is :math:`R_L \geq \H{Y \mid U}`. At a corner point the common and local randomness together supply exactly :math:`\H{Y \mid X}`. The optimizer class exposes :math:`\H{Y \mid U}` at its optimum via ``ChannelSynthesis.local_randomness()``.

Erasure Channel
===============

For a fair bit through an erasure channel with erasure probability :math:`p` (:func:`~dit.example_dists.erasure_channel`), the rate falls from :math:`\C{X : Y}` (one bit for :math:`p < 1/2`) to :math:`\I{X : Y} = 1 - p` as :math:`R_0` grows to :math:`\H{Y \dagger X} = h(p)`:

.. ipython::
   :verbatim:

   In [1]: from dit.example_dists import erasure_channel

   In [2]: from dit.multivariate import channel_synthesis_rate

   In [3]: d = erasure_channel(0.3)

   In [4]: [round(channel_synthesis_rate(d, [0], [1], common_randomness=t), 3) for t in (0.0, 0.2, 0.5, 0.9)]
   Out[4]: [1.0, 0.897, 0.783, 0.7]

The :doc:`/gray_wyner` extension profile also draws this curve (``ExtensionProfile.channel_synthesis``), approximated from points on the profile's boundary for plotting; :func:`channel_synthesis_rate` optimizes over :math:`U` directly.

API
===

.. autofunction:: channel_synthesis_rate
