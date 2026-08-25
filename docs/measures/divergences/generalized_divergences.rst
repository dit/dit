.. generalized_divergences.rst
.. py:module:: dit.divergences.generalized_divergences

***********************
Generalized Divergences
***********************

Several one-parameter families extend the :doc:`kullback_leibler_divergence`.
``dit`` implements the :math:`\alpha`-divergence, Hellinger divergence,
Rényi divergence, Tsallis divergence, and a generic Csiszár
:math:`f`-divergence :cite:`Cover2006`.

.. ipython::

   In [1]: from dit.divergences import alpha_divergence, hellinger_divergence, renyi_divergence, tsallis_divergence

   In [2]: p = dit.Distribution(['0', '1'], [3/4, 1/4])

   In [3]: q = dit.Distribution(['0', '1'], [1/2, 1/2])

   @doctest float
   In [4]: renyi_divergence(p, q, alpha=2)
   Out[4]: 0.32192809488736235

   @doctest float
   In [5]: tsallis_divergence(p, q, alpha=2)
   Out[5]: 0.25

API
===

.. autofunction:: alpha_divergence

.. autofunction:: hellinger_divergence

.. autofunction:: renyi_divergence

.. autofunction:: tsallis_divergence

.. autofunction:: f_divergence
