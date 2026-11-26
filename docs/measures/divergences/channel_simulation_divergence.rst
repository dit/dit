.. channel_simulation_divergence.rst
.. py:currentmodule:: dit.divergences

*****************************
Channel Simulation Divergence
*****************************

For :math:`P` absolutely continuous with respect to :math:`Q`, with likelihood ratio :math:`r = dP/dQ`, the *width function*

.. math::

   w(\ell) = Q\left(r \geq \ell\right), \qquad \ell \geq 0,

is a probability density on :math:`[0, \infty)`, because it integrates to :math:`\mathbb{E}_Q[r] = 1`. The channel simulation divergence :cite:`goc2024channel` is its differential entropy:

.. math::

   D_{CS}(P \| Q) = -\int_0^\infty w(\ell) \log_2 w(\ell) \, d\ell.

It arises in channel simulation with rejection samplers :cite:`goc2024channel`. Averaged over the posteriors :math:`p(X | y)` against the prior :math:`p(X)`, it is the :doc:`functional information </measures/other/functional_information>`. Its properties :cite:`flamich2026functional`:

* It sandwiches the Kullback-Leibler divergence:

  .. math::

     \DKL{P || Q} \leq D_{CS}(P \| Q) \leq \DKL{P || Q} + \log_2\left(\DKL{P || Q} + 1\right) + 1.

* It is convex in :math:`P` for a fixed :math:`Q`.
* It is infinite unless :math:`P \ll Q`.

When :math:`P` is :math:`Q` conditioned on an event :math:`A`, the likelihood ratio is constant on its support, and both divergences equal :math:`-\log_2 Q(A)`. Otherwise :math:`D_{CS}` is usually larger:

.. ipython::

   In [1]: from dit.divergences import channel_simulation_divergence, kullback_leibler_divergence

   In [2]: p = dit.Distribution(['0', '1'], [0.2, 0.8])

   In [3]: q = dit.Distribution(['0', '1'], [0.5, 0.5])

   @doctest float
   In [4]: channel_simulation_divergence(p, q)
   Out[4]: 0.6000000000000001

   @doctest float
   In [5]: kullback_leibler_divergence(p, q)
   Out[5]: 0.2780719051126377

Rényi Channel Simulation Divergence
===================================

For :math:`\alpha \in (0, 1)`, the Rényi channel simulation divergence :cite:`flamich2026functional` is the differential Rényi entropy of the width function:

.. math::

   D^{CS}_\alpha(P \| Q) = \frac{1}{1 - \alpha} \log_2 \int_0^\infty w(\ell)^\alpha \, d\ell.

It decreases with :math:`\alpha` towards :math:`D_{CS}`:

.. ipython::

   @doctest float
   In [6]: channel_simulation_divergence(p, q, alpha=0.5)
   Out[6]: 0.6404566697880599

API
===

The pmf form, ``dit.divergences.pmf.channel_simulation_divergence(p, q, alpha=1)``, takes two aligned probability vectors.

.. autofunction:: channel_simulation_divergence
   :no-index:
