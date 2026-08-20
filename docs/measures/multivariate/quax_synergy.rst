.. quax_synergy.rst
.. py:module:: dit.multivariate.quax_synergy

.. _quax_synergy:

************
Quax Synergy
************

Quax, Har-Shemesh & Sloot :cite:`quax2017quantifying` quantify synergistic
information via a *synergistic random variable* (SRV) :math:`S` of the
sources :math:`X`: :math:`I(S:X) > 0` while :math:`I(S:X_i) = 0` for every
source. The synergistic information that a target :math:`Y` stores about
:math:`X` is then :math:`I(Y:S)` for an SRV that maximises :math:`I(S:X)`.

This is **not** a PID synergy atom: synergistic and unique information can
coexist in :math:`Y`.

.. ipython::

   In [1]: from dit.multivariate import quax_synergy

   In [2]: from dit.example_dists import Xor

   In [3]: quax_synergy(Xor(), [[0], [1]], [2], niter=8)

API
===

.. autofunction:: quax_synergy

.. autofunction:: max_synergistic_entropy
