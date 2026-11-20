.. renyi_caekl_mutual_information.rst
.. py:currentmodule:: dit.multivariate

.. _renyi_caekl_mutual_information:

******************************
Rényi CAEKL Mutual Information
******************************

The Rényi CAEKL mutual information generalizes the :ref:`caekl_mutual_information` :cite:`chan2015multivariate` by replacing each Shannon entropy with a :doc:`/measures/other/renyi_entropy` of order :math:`\alpha` :cite:`berta2015renyi`.

The CAEKL mutual information is the smallest per-partition :ref:`total_correlation`, minimized over the non-trivial partitions :math:`\Pi` of the variables. Keeping that definition and swapping in the :ref:`renyi_total_correlation` gives:

.. math::

   \op{J_\alpha}{X_{0:n}} = \min_{\mathcal{P} \in \Pi} ~ \frac{\op{T_\alpha}{X_C : C \in \mathcal{P}}}{|\mathcal{P}| - 1}

At :math:`\alpha = 1` this is exactly the CAEKL mutual information.

.. ipython::

   In [1]: from dit.multivariate import renyi_caekl_mutual_information as J

   In [2]: d = dit.example_dists.And()

   @doctest float
   In [3]: J(d, 1)
   Out[3]: 0.40563906222956647

   @doctest float
   In [4]: J(d, 2)
   Out[4]: 0.3390359525563189

Because each partition's value is a Rényi total correlation, only joint Rényi entropies are involved; no conditional Rényi entropy is required.

For a giant bit every variable is a copy of the others, so every non-trivial partition shares the same single bit no matter the order:

.. ipython::

   @doctest float
   In [5]: J(dit.example_dists.giant_bit(4, 2), 5)
   Out[5]: 1.0

.. note::

   Unlike :py:func:`~dit.multivariate.caekl_mutual_information`, which uses the agglomerative principal-sequence-of-partitions algorithm of :cite:`ChanLiu2017agglomerative`, this minimizes by enumerating partitions: the Rényi entropy is not submodular in general, so that faster algorithm does not apply. The number of partitions grows as the Bell numbers, making this expensive beyond roughly eight variables.

API
===

.. autofunction:: renyi_caekl_mutual_information
   :no-index:
