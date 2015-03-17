.. tsallis_entropy.rst
.. py:module:: dit.other.tsallis_entropy

***************
Tsallis Entropy
***************

The Tsallis entropy is a generalization of the Shannon (or Boltzmann-Gibbs)
entropy to the case where entropy is nonextensive. It is given by:

.. math::

    \TE[X] = \frac{1}{q - 1} \left( 1 - \sum_{x \in \mathcal{X}} p(x)^q \right)

Non-additivity
==============

One interesting property of the Tsallis entropy is the relationship between the
joint Tsallis entropy of two indpendent systems, and the Tsallis entropy of
those subsystems:

.. math::

    \TE[X, Y] = \TE[X] + \TE[Y] + (1-q)\TE[X]\TE[Y]

.. autofunction:: tsallis_entropy
