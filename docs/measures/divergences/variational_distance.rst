.. variational_distance.rst
.. py:currentmodule:: dit.divergences

**********************
Variational Distance
**********************

The variational (total variation) distance :cite:`Cover2006` between two
distributions on the same alphabet is

.. math::

   \delta(p, q) = \tfrac{1}{2} \sum_x \lvert p(x) - q(x) \rvert

.. ipython::

   In [1]: from dit.divergences import variational_distance

   In [2]: p = dit.Distribution(['0', '1'], [3/4, 1/4])

   In [3]: q = dit.Distribution(['0', '1'], [1/2, 1/2])

   @doctest float
   In [4]: variational_distance(p, q)
   Out[4]: 0.25

The Bhattacharyya Coefficient
=============================

The rest of this module is built on the Bhattacharyya coefficient
:cite:`bhattacharyya1943`, the overlap between the square roots of the two
pmfs:

.. math::

   BC(p, q) = \sum_x \sqrt{p(x) q(x)}

It ranges from 1, when :math:`p = q`, down to 0, when :math:`p` and :math:`q`
have disjoint support. Geometrically it is the inner product of
:math:`\sqrt{p}` and :math:`\sqrt{q}`, both of which are unit vectors on the
positive orthant of the sphere — so :math:`BC` is the cosine of the angle
between them.

.. ipython::

   In [5]: from dit.divergences import bhattacharyya_coefficient

   @doctest float
   In [6]: bhattacharyya_coefficient(p, q)
   Out[6]: 0.9659258262890682

Derived Quantities
==================

Each of the following is a function of :math:`BC` alone:

.. math::

   D_B(p, q)      &= -\log_2 BC(p, q)          &&\quad \text{Bhattacharyya distance} \\
   \theta(p, q)   &= \arccos BC(p, q)          &&\quad \text{Bhattacharyya angle} \\
   d_{FR}(p, q)   &= 2 \arccos BC(p, q)        &&\quad \text{Fisher-Rao distance} \\
   H(p, q)        &= \sqrt{1 - BC(p, q)}       &&\quad \text{Hellinger distance} \\
   d_M(p, q)      &= \sqrt{2 (1 - BC(p, q))}   &&\quad \text{Matusita distance}

The Bhattacharyya distance is the one that is *not* a metric: it is symmetric
and vanishes only when :math:`p = q`, but it fails the triangle inequality, and
it diverges when the supports are disjoint. The other four are true metrics.

The Bhattacharyya angle is the angle between :math:`\sqrt{p}` and
:math:`\sqrt{q}` on the sphere, and the Fisher-Rao distance
:cite:`rao1945information` is the corresponding geodesic distance — the
distance induced by the Fisher information metric, which on the simplex is the
round metric. The Matusita distance :cite:`matusita1955decision` is instead the
straight-line Euclidean distance between :math:`\sqrt{p}` and :math:`\sqrt{q}`,
making it the chord to the Fisher-Rao arc, and it is :math:`\sqrt{2}` times the
Hellinger distance as ``dit`` defines it.

.. ipython::

   In [7]: from dit.divergences import bhattacharyya_distance, bhattacharyya_angle

   In [8]: from dit.divergences import fisher_rao_distance, hellinger_distance, matusita_distance

   @doctest float
   In [9]: bhattacharyya_distance(p, q)
   Out[9]: 0.05001568652350428

   @doctest float
   In [10]: bhattacharyya_angle(p, q)
   Out[10]: 0.2617993877991498

   @doctest float
   In [11]: fisher_rao_distance(p, q)
   Out[11]: 0.5235987755982996

   @doctest float
   In [12]: hellinger_distance(p, q)
   Out[12]: 0.18459191128251476

   @doctest float
   In [13]: matusita_distance(p, q)
   Out[13]: 0.26105238444010354

The angle-based quantities are reported in radians, so they max out at
:math:`\pi/2` and :math:`\pi` respectively on disjoint distributions, where the
Bhattacharyya distance instead diverges.

.. note::

   The Bhattacharyya coefficient and distance are also reachable through
   :doc:`/measures/divergences/generalized_divergences`: :math:`BC` is the
   Hellinger sum at :math:`\alpha = \tfrac{1}{2}`, and :math:`D_B` is half the
   Rényi divergence of order :math:`\tfrac{1}{2}`.

API
===

.. autofunction:: variational_distance
   :no-index:

.. autofunction:: bhattacharyya_coefficient
   :no-index:

.. autofunction:: bhattacharyya_distance
   :no-index:

.. autofunction:: bhattacharyya_angle
   :no-index:

.. autofunction:: fisher_rao_distance
   :no-index:

.. autofunction:: hellinger_distance
   :no-index:

.. autofunction:: matusita_distance
   :no-index:

.. autofunction:: chernoff_information
   :no-index:
