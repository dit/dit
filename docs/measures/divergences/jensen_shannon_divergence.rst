.. jensen_shannon_divergence.rst
.. py:module:: dit.divergences.jensen_shannon_divergence

*************************
Jensen-Shannon Divergence
*************************

The Jensen-Shannon divergence is a principled divergence measure which is always
finite. In its basic form it is:

.. math::

   \JSD[X || Y] = \H\left[\frac{X + Y}{2}\right] - \frac{\H[X] + \H[Y]}{2}

That is, it is the entropy of the mixture minus the mixture of the entropy. This
can be generalized to an arbitrary number of random variables with arbitrary
weights:

.. math::

   \JSD[X_{0:n}] = \H\left[ \sum w_i X_i \right] - \sum \left( w_i \H[X_i] \right)

Where does this equation come from? Consider Jensen's inequality:

.. math::

   \Psi \left( \mathbb{E}(x) \right) \geq \mathbb{E} \left( \Psi(x) \right)

where :math:`\Psi` is a concave function. If we consider the *divergence* of the
left and right side we find:

.. math::

   \Psi \left( \mathbb{E}(x) \right) - \mathbb{E} \left( \Psi(x) \right) \geq 0

If we make that concave function :math:`\Psi` the Shannon entropy :math:`\H`, we
get the Jensen-Shannon divergence. Jensen from Jensen's inequality, and Shannon
from the use of the Shannon entropy.

.. note::

   Some people look at the Jensen-Rényi divergence (where :math:`\Psi` is the
   Rényi entropy) and the Jensen-Tsallis divergence (where :math:`\Psi` is the
   Tsallis entropy).

The square root of the Jensen-Shannon divergence, :math:`\sqrt{\JSD}`, is a true
metric between distributions.

.. todo::

   discuss how JSD = I(X:M) where X is indicator and M is mixture.

.. todo::

   Add examples.

.. autofunction:: dit.divergences.jensen_shannon_divergence.jensen_shannon_divergence
