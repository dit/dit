.. maximal_cross_entropy.rst
.. py:currentmodule:: dit.divergences

*********************
Maximal Cross Entropy
*********************

The maximal cross entropy :cite:`yu2020exact` of a pair of marginals :math:`p_X, p_Y` relative to a joint distribution :math:`q(x, y)` is the largest :doc:`cross entropy <cross_entropy>`, relative to :math:`q`, of any joint distribution with those marginals:

.. math::

   \mathbb{H}(p_X, p_Y \| q) = \max_{r \in \mathcal{C}(p_X, p_Y)} -\sum_{x, y} r(x, y) \log_2 q(x, y),

where :math:`\mathcal{C}(p_X, p_Y)` is the set of couplings of :math:`p_X` and :math:`p_Y`. It is an optimal transport problem with cost :math:`\log_2 q(x, y)` (compare the :doc:`earth_movers_distance`), and it is infinite when some coupling places mass where :math:`q` vanishes.

It measures how unlikely, under :math:`q`, the least likely pairs with the given marginal statistics are. This is what makes exact simulation expensive: when :math:`X^n` and :math:`Y^n` are generated separately from shared information, every pair of sequences compatible with it gets generated, including the least likely ones, and each must be generated with exactly the right probability. Yu and Tan use it to bound the :doc:`/measures/common_informations/exact_common_information_rate` and the rate of exact :doc:`/measures/common_informations/channel_synthesis`.

For the doubly symmetric binary source, uniform marginals can be coupled so that all of the mass sits on the two unlikely pairs, each of probability :math:`0.1`, so the maximal cross entropy is :math:`\log_2 10`, well above the joint entropy:

.. ipython::

   In [1]: from dit.divergences import maximal_cross_entropy

   In [2]: d = dit.Distribution(['00', '01', '10', '11'], [0.4, 0.1, 0.1, 0.4])

   @doctest float
   In [3]: maximal_cross_entropy(d)
   Out[3]: 3.321928094887362

   @doctest float
   In [4]: dit.shannon.entropy(d)
   Out[4]: 1.721928094887362

With :math:`q`'s own marginals the maximal cross entropy is always at least the joint entropy of :math:`q`, exceeding it by at least :math:`\DKL{q || q_X q_Y} + \DKL{q_X q_Y || q}`, so the two are equal exactly when :math:`X` and :math:`Y` are independent. Other marginals are taken from the first argument:

.. ipython::

   In [5]: p = dit.Distribution(['00', '01', '10', '11'], [0.36, 0.24, 0.24, 0.16])

   @doctest float
   In [6]: maximal_cross_entropy(p, d)
   Out[6]: 2.9219280948873623

API
===

.. autofunction:: maximal_cross_entropy
   :no-index:
