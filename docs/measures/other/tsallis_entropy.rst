.. tsallis_entropy.rst
.. py:module:: dit.other.tsallis_entropy

***************
Tsallis Entropy
***************

The Tsallis entropy is a generalization of the Shannon (or Boltzmann-Gibbs)
entropy to the case where entropy is nonextensive. It is given by:

.. math::

    \TE[X] = \frac{1}{q - 1} \left( 1 - \sum_{x \in \mathcal{X}} p(x)^q \right)

.. ipython::

   In [1]: from dit.other import tsallis_entropy

   In [2]: from dit.example_dists import n_mod_m

   In [3]: d = n_mod_m(4, 3)

   @doctest float
   In [4]: tsallis_entropy(d, 4)
   Out[4]: 0.33331639824552489


Non-additivity
==============

One interesting property of the Tsallis entropy is the relationship between the
joint Tsallis entropy of two indpendent systems, and the Tsallis entropy of
those subsystems:

.. math::

    \TE[X, Y] = \TE[X] + \TE[Y] + (1-q)\TE[X]\TE[Y]


API
===

.. autofunction:: tsallis_entropy
