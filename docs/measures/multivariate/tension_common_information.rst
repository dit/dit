.. tension_common_information.rst
.. py:module:: dit.multivariate.common_informations.tension_common_information

**************************
Tension Common Information
**************************

The :doc:`gk_common_information` is brittle: it is determined entirely by the connected components of the support graph, so an arbitrarily small perturbation of the distribution can send it to zero. The tension common information :cite:`matveev2026beyond` is a continuous relaxation of it, built from the *entanglement*.

Entanglement
============

The entanglement of a pair measures how far its mutual information is from being extractable:

.. math::

   \Ent{X : Y} = \inf_{W} \left( \I{X : Y \mid W} + \I{X : W \mid Y} + \I{W : Y \mid X} \right)

This is the quantity on the right-hand side of the Makarychev-Makarychev-Romashchenko-Vereshchagin class of non-Shannon inequalities :cite:`makarychev2002class`, and was studied by Zhang :cite:`zhang2003new` (writing it :math:`W(X, Y)`) in connection with the approximate representability of mutual information. The name is due to :cite:`matveev2026beyond`; it is unrelated to quantum entanglement.

The three summands are exactly the coordinates of the :ref:`region of tension <tension region>`, so for a pair the entanglement is the least coordinate-sum over that region — a single weighted-rate query on the :doc:`/gray_wyner`.

For :math:`n` sources the source tensions are :math:`\I{X_{-i} : W \mid X_i}` and the leftover dependence is measured by the :doc:`dual_total_correlation` :math:`\B{X_{0:n} \mid W}`, giving

.. math::

   \Ent{X_{0:n}} = \inf_{W} \left( \B{X_{0:n} \mid W} + \sum_i \I{X_{-i} : W \mid X_i} \right)

which reduces to the pairwise definition at :math:`n = 2`. The dual total correlation keeps the measure on the scale of the common informations: conditioning on the Gács-Körner meet lowers :math:`\B{}` by exactly :math:`\K{}`, while it lowers the total correlation by :math:`(n-1)\K{}`. With a total-correlation residual the three-variable giant bit would have a tension common information of 2 bits, more than its entropy. Because :math:`\B{X_{0:n} \mid W}` is not affine in the Gray-Wyner rates, for :math:`n \geq 3` the probe :math:`W` is optimized directly rather than through the Gray-Wyner region.

The entanglement vanishes exactly when the mutual information is Gács-Körner extractable, and attains :math:`\min\{\H{X \mid Y}, \H{Y \mid X}, \I{X : Y}\}` when the pair is *maximally non-extractable*.

The tension common information
==============================

Because the entanglement is a deficiency, subtracting it from the dual total correlation gives a quantity that runs the same way as a common information:

.. math::

   \Tci{X_{0:n}} = \B{X_{0:n}} - \Ent{X_{0:n}}

For a pair :math:`\B{X : Y} = \I{X : Y}`. Using the Gács-Körner meet as the probe makes every source tension vanish and leaves a residual of :math:`\B{X_{0:n}} - \K{X_{0:n}}`, which sandwiches it:

.. math::

   \K{X_{0:n}} \leq \Tci{X_{0:n}} \leq \B{X_{0:n}}

For two variables this places it in the common-information ordering between the Gács-Körner common information and the mutual information. For three or more it is bounded above by the dual total correlation but is not known to be comparable with the :doc:`caekl_mutual_information`.

Continuity
==========

The distinguishing behaviour is what happens under perturbation. Consider a distribution whose support has two components, so that :math:`\K{X : Y} > 0`, and then add a single low-probability outcome bridging them:

.. ipython::
   :verbatim:

   In [1]: from dit.multivariate import gk_common_information as K, tension_common_information as Theta

   In [2]: blocks = dit.Distribution(['00', '01', '10', '11', '22'], [0.2]*5)

   In [3]: bridged = dit.Distribution(['00', '01', '10', '11', '22', '02'], [0.2, 0.2, 0.2, 0.2, 0.19, 0.01])

   In [4]: K(blocks), K(bridged)
   Out[4]: (0.7219280948873623, 0.0)

   In [5]: Theta(blocks), Theta(bridged)

The Gács-Körner common information collapses to zero; the tension common information does not.

Fano plane
==========

A pair uniform on the point-line incidences of the Fano plane is maximally non-extractable: its entanglement equals its mutual information, so its tension common information is zero.

.. ipython::
   :verbatim:

   In [6]: from dit.multivariate import entanglement, total_correlation

   In [7]: lines = ['012', '034', '056', '136', '145', '235', '246']

   In [8]: fano = dit.Distribution([p + str(i) for i, l in enumerate(lines) for p in l], [1/21]*21)

   In [9]: total_correlation(fano), entanglement(fano)

This is certified spectrally: the Fano incidence graph is an expander, and :func:`~dit.algorithms.spectral_entanglement_bound` reports its nominal bound. See :doc:`/gray_wyner` for the caveats around that bound.

API
===

.. autofunction:: entanglement

.. autofunction:: tension_common_information
