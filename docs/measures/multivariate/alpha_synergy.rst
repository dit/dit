.. alpha_synergy.rst
.. py:module:: dit.multivariate.alpha_synergy

.. _alpha_synergy:

*********************
Alpha-Synergy Spectra
*********************

Varley's synergy-first backbone decomposition :cite:`varley2024scalable` measures, for each outcome, the least information lost when any :math:`\alpha` of the :math:`n` variables are removed:

.. math::

   h^{syn}_{\alpha}(x) = \min_{|a| = \alpha} h(x_a \mid x_{-a})

This grows from :math:`0` to :math:`h(x)` as :math:`\alpha` goes from :math:`0` to :math:`n`. Its increments :math:`h^{syn}_{\alpha} - h^{syn}_{\alpha - 1}`, averaged over outcomes, form the *partial entropy spectrum*, whose entries sum to the joint entropy. Information that shows up only at large :math:`\alpha` is carried synergistically: it survives the loss of any small group of variables. Unlike lattice decompositions the spectrum has only :math:`n` entries, so it scales to many variables.

Because the local divergence :math:`\log q(x)/p(x)` is a difference of local entropies, the same construction decomposes Kullback-Leibler divergences, and through them the total correlation (against the product of marginals) and the mutual information.

.. ipython::

   In [1]: from dit.multivariate import partial_entropy_spectrum

   In [2]: from dit.example_dists import Xor

   @doctest float
   In [3]: partial_entropy_spectrum(Xor())
   Out[3]: array([0., 1., 1.])

Removing one variable of XOR loses nothing, since it is a function of the other two, but removing any two loses a bit each time.

.. autofunction:: alpha_synergistic_entropy

.. autofunction:: partial_entropy_spectrum

.. autofunction:: partial_kullback_leibler_spectrum

.. autofunction:: partial_total_correlation_spectrum

.. autofunction:: partial_information_spectrum
