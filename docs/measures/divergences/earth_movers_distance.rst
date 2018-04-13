.. earth_movers_distance.rst
.. py:module:: dit.divergences.earth_movers_distance

Earth Mover's Distance
======================

The Earth mover's distance is a distance measure between probability distributions. If we consider each probability mass function as a histogram of dirt, it is equal to the amount of work needed to optimally move the dirt of one histogram into the shape of the other.

For categorical data, the "distance" between unequal symbols is unitary. In this case, :math:`1/6` of the probability in symbol '0' needs to be moved to '1', and :math:`1/6` needs to be moved to '2', for a total of :math:`1/3`:

.. ipython::

   In [1]: from dit.divergences import earth_movers_distance

   In [2]: d1 = dit.Distribution(['0', '1', '2'], [2/3, 1/6, 1/6])

   In [3]: d2 = dit.Distribution(['0', '1', '2'], [1/3, 1/3, 1/3])

   @doctest float
   In [4]: earth_movers_distance(d1, d2)
   Out[4]: 0.3333333333333334

 For numerical data, "distance" defaults to the difference between the symbols. In this case, :math:`1/6` of the probability in symbol '0' needs to be moved to '1' (a distance of 1), and :math:`1/6` needs to be moved to '2' (a distance of 2), for a total of :math:`1/2`:

.. ipython::

   In [1]: from dit.divergences import earth_movers_distance

   In [2]: d1 = dit.ScalarDistribution([0, 1, 2], [2/3, 1/6, 1/6])

   In [3]: d2 = dit.ScalarDistribution([0, 1, 2], [1/3, 1/3, 1/3])

   @doctest float
   In [4]: earth_movers_distance(d1, d2)
   Out[4]: 0.5

API
---

.. autofunction:: earth_movers_distance