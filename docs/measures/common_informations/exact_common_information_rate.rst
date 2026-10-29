.. exact_common_information_rate.rst
.. py:currentmodule:: dit.multivariate

*****************************
Exact Common Information Rate
*****************************

The :doc:`exact_common_information` :math:`\G{X : Y}` is the least entropy of a variable rendering :math:`X` and :math:`Y` independent: the cost, in bits of common randomness, of generating one copy of :math:`(X, Y)` exactly. Generating many copies at once can be cheaper per copy, because :math:`\G{X : Y}` is subadditive. The exact common information rate :cite:`kumar2014exact` is the per-copy cost in the limit,

.. math::

   \bar{G}[X : Y] = \lim_{n \to \infty} \frac{1}{n} \G{X^n : Y^n},

the least rate of common randomness, compressed with variable-length codes, from which two parties can separately generate i.i.d. copies of :math:`(X, Y)` with exactly the right joint distribution. It lies between the :doc:`wyner_common_information`, which suffices for approximate generation, and the one-shot quantity:

.. math::

   \C{X : Y} \leq \bar{G}[X : Y] \leq \G{X : Y}.

Bounds
======

No single-letter formula for :math:`\bar{G}` is known. Yu and Tan :cite:`yu2020on,yu2020exact` bound it using the :doc:`/measures/divergences/maximal_cross_entropy` :math:`\mathbb{H}`:

.. math::

   \Gamma^{(o)}[X : Y] \leq \bar{G}[X : Y] \leq \Gamma[X : Y]
     = \min_{X - W - Y} -\H{X, Y | W} + \sum_w p(w) \mathbb{H}(p_{X|w}, p_{Y|w} \| p_{XY}),

where the lower bound replaces the pairing of :math:`p_{X|w}` with :math:`p_{Y|w}` by the cheapest coupling :math:`q` of :math:`p_W` with itself,

.. math::

   \Gamma^{(o)}[X : Y] = \min_{X - W - Y} -\H{X, Y | W}
     + \min_{q \in \mathcal{C}(p_W, p_W)} \sum_{w, w'} q(w, w') \mathbb{H}(p_{X|w}, p_{Y|w'} \| p_{XY}).

Replacing :math:`\mathbb{H}` by the cross entropy of the product coupling turns :math:`\Gamma` into Wyner's :math:`\I{X, Y : W}`, so the upper bound is at least the Wyner common information, and it is at most :math:`\H{W}` for every :math:`W`, so it never exceeds :math:`\G{X : Y}`. It equals the Wyner common information when some :math:`W` achieving the Wyner common information makes one of :math:`p_{X|w}, p_{Y|w}` deterministic for every :math:`w`, as for the erasure channel. For the doubly symmetric binary source with crossover :math:`p` the bounds meet:

.. math::

   \bar{G}[X : Y] = \log_2 \frac{2}{1 - p} + 2a \log_2 \frac{1 - p}{p} - 2 h(a), \qquad a = \frac{1 - \sqrt{1 - 2p}}{2},

where :math:`h` is the binary entropy, strictly between :math:`\C{X : Y}` and :math:`\G{X : Y}`:

.. ipython::
   :verbatim:

   In [1]: from dit.multivariate import exact_common_information, exact_common_information_rate_bounds, wyner_common_information

   In [2]: d = dit.Distribution(['00', '01', '10', '11'], [0.4, 0.1, 0.1, 0.4])

   In [3]: [round(v, 4) for v in (wyner_common_information(d), *exact_common_information_rate_bounds(d), exact_common_information(d))]
   Out[3]: [0.7059, 0.7567, 0.7567, 0.9544]

The distribution Kumar **et. al.** used to show that :math:`\G{X : Y}` is subadditive (see :doc:`exact_common_information`) has a one-shot exact common information of :math:`0.9183` bits, and about :math:`0.876` bits per copy for two copies, yet its rate is its Wyner common information, :math:`2/3`:

.. ipython::
   :verbatim:

   In [4]: from dit.multivariate import exact_common_information_rate

   In [5]: d = dit.Distribution([(0, 0), (0, 1), (1, 0)], [1/3] * 3)

   In [6]: round(exact_common_information_rate(d), 4)
   Out[6]: 0.6667

Computation
===========

:func:`exact_common_information_rate` minimizes :math:`\Gamma`, starting from the optima of the Wyner and one-shot exact common informations. With the conditionals :math:`p_{X|w}, p_{Y|w}` fixed, the objective and the constraint that the auxiliary variable reproduces :math:`p(x, y)` are both linear in :math:`p(w)`, so the result is the best mixture of every state the optimizer visits, found by a linear program. Every auxiliary variable reproducing the distribution achieves its value of :math:`\Gamma`, so the result is a valid upper bound even when the optimizer misses the minimum. With ``block_length=n`` it also bounds :math:`n` i.i.d. copies of the pair and returns the smallest :math:`\Gamma[X^k : Y^k] / k` for :math:`k \leq n`, which decreases to :math:`\bar{G}`; the cost grows exponentially in :math:`n`, so pass ``bound`` beyond small alphabets.

:func:`exact_common_information_rate_bounds` also returns the larger of the Wyner common information and :math:`\Gamma^{(o)}` as a lower bound. A lower bound holds only at a global minimum over auxiliary variables (here with at most ``bound`` states), which local optimization cannot certify, so treat it as an estimate.

The rate of exact channel synthesis generalizes the upper bound to common randomness shared by the encoder and decoder; see :doc:`channel_synthesis`.

API
===

.. autofunction:: exact_common_information_rate
   :no-index:

.. autofunction:: exact_common_information_rate_bounds
   :no-index:
