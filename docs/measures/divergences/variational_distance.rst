.. variational_distance.rst
.. py:module:: dit.divergences.variational_distance

**********************
Variational Distance
**********************

The variational (total variation) distance :cite:`Cover2006` between two
distributions on the same alphabet is

.. math::

   \delta(p, q) = \tfrac{1}{2} \sum_x \lvert p(x) - q(x) \rvert

Related quantities in the same module are the Hellinger distance, the
Bhattacharyya coefficient, and the Chernoff information.

.. ipython::

   In [1]: from dit.divergences import variational_distance, hellinger_distance, bhattacharyya_coefficient

   In [2]: p = dit.Distribution(['0', '1'], [3/4, 1/4])

   In [3]: q = dit.Distribution(['0', '1'], [1/2, 1/2])

   @doctest float
   In [4]: variational_distance(p, q)
   Out[4]: 0.25

   @doctest float
   In [5]: hellinger_distance(p, q)
   Out[5]: 0.18459191128251476

API
===

.. autofunction:: variational_distance

.. autofunction:: hellinger_distance

.. autofunction:: bhattacharyya_coefficient

.. autofunction:: chernoff_information
