.. generalized_divergences.rst
.. py:currentmodule:: dit.divergences

***********************
Generalized Divergences
***********************

Several one-parameter families extend the :doc:`/measures/divergences/kullback_leibler_divergence`.
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

   @doctest float
   In [6]: renyi_divergence(p, q, alpha=float('inf'))
   Out[6]: 0.5849625007211562

At :math:`\alpha = \infty` the Rényi divergence is the max-divergence
:math:`D_\infty(p \| q) = \log_2 \max_x p(x) / q(x)`, the maximum taken over the
support of :math:`p`; above, :math:`\log_2 \frac{3/4}{1/2} = \log_2 1.5`. It is
infinite unless :math:`q` covers the support of :math:`p`.

Several of these divergences are built from the Hellinger sum
:math:`\sum_x p(x)^\alpha q(x)^{1-\alpha}`, which is available directly as
:py:func:`hellinger_sum`.

API
===

.. autofunction:: alpha_divergence
   :no-index:

.. autofunction:: hellinger_divergence
   :no-index:

.. autofunction:: renyi_divergence
   :no-index:

.. autofunction:: tsallis_divergence
   :no-index:

.. autofunction:: f_divergence
   :no-index:

.. autofunction:: hellinger_sum
   :no-index:
