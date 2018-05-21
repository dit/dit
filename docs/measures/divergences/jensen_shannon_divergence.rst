.. jensen_shannon_divergence.rst
.. py:module:: dit.divergences.jensen_shannon_divergence

*************************
Jensen-Shannon Divergence
*************************

The Jensen-Shannon divergence is a principled divergence measure which is always finite for finite random variables. It quantifies how "distinguishable" two or more distributions are from each other. In its basic form it is:

.. math::

   \JSD{X || Y} = \H{\frac{X + Y}{2}} - \frac{\H{X} + \H{Y}}{2}

That is, it is the entropy of the mixture minus the mixture of the entropy. This can be generalized to an arbitrary number of random variables with arbitrary weights:

.. math::

   \JSD{X_{0:n}} = \H{\sum w_i X_i} - \sum \left( w_i \H{X_i} \right)

.. ipython::

   In [1]: from dit.divergences import jensen_shannon_divergence

   In [2]: X = dit.ScalarDistribution(['red', 'blue'], [1/2, 1/2])

   In [3]: Y = dit.ScalarDistribution(['blue', 'green'], [1/2, 1/2])

   @doctest float
   In [4]: jensen_shannon_divergence([X, Y])
   Out[4]: 0.5

   @doctest float
   In [5]: jensen_shannon_divergence([X, Y], [3/4, 1/4])
   Out[5]: 0.40563906222956647

   In [13]: Z = dit.ScalarDistribution(['blue', 'yellow'], [1/2, 1/2])

   @doctest float
   In [14]: jensen_shannon_divergence([X, Y, Z])
   Out[14]: 0.79248125036057782

   @doctest float
   In [15]: jensen_shannon_divergence([X, Y, Z], [1/2, 1/4, 1/4])
   Out[15]: 0.75


Derivation
==========

Where does this equation come from? Consider Jensen's inequality:

.. math::

   \Psi \left( \mathbb{E}(x) \right) \geq \mathbb{E} \left( \Psi(x) \right)

where :math:`\Psi` is a concave function. If we consider the *divergence* of the left and right side we find:

.. math::

   \Psi \left( \mathbb{E}(x) \right) - \mathbb{E} \left( \Psi(x) \right) \geq 0

If we make that concave function :math:`\Psi` the Shannon entropy :math:`\H`, we get the Jensen-Shannon divergence. Jensen from Jensen's inequality, and Shannon from the use of the Shannon entropy.

.. note::

   Some people look at the Jensen-Rényi divergence (where :math:`\Psi` is the :doc:`../other/renyi_entropy`) and the Jensen-Tsallis divergence (where :math:`\Psi` is the :doc:`../other/tsallis_entropy`).


Metric
======

The square root of the Jensen-Shannon divergence, :math:`\sqrt{\JSD{}}`, is a true metric between distributions.


Relationship to the Other Measures
==================================

The Jensen-Shannon divergence can be derived from other, more well known information measures; notably the :doc:`kullback_leibler_divergence` and the :ref:`mutual_information`.


Kullback-Leibler divergence
---------------------------

The Jensen-Shannon divergence is the average Kullback-Leibler divergence of :math:`X` and :math:`Y` from their mixture distribution, :math:`M`:

.. math::

   \JSD[X || Y] &= \frac{1}{2} \left( \DKL[X || M] + \DKL[Y || M] \right) \\
   M &= \frac{X + Y}{2}


Mutual Information
------------------

.. math::

   \JSD[X || Y] = \I[Z:M]

where :math:`M` is the mixture distribution as before, and :math:`Z` is an indicator variable over :math:`X` and :math:`Y`. In essence, if :math:`X` and :math:`Y` are each an urn containing colored balls, and I randomly selected one of the urns and draw a ball from it, then the Jensen-Shannon divergence is the mutual information between which urn I drew the ball from, and the color of the ball drawn.


API
===

.. autofunction:: jensen_shannon_divergence
