.. synergistic_disclosure.rst
.. py:currentmodule:: dit.multivariate

.. _synergistic_disclosure:

**********************
Synergistic Disclosure
**********************

Rosas, Mediano, Rassouli & Barrett :cite:`rosas2020operational` define
:math:`\alpha`-synergy as the maximum mutual information :math:`I(V;Y)` over
channels :math:`p(V \mid X)` that are independent of every block in a
constraint set :math:`\alpha`:

.. math::

   S_{\alpha}(X \to Y) = \max_{V :\; I(V; X_{\alpha_i})=0\ \forall i} I(V; Y)

``dit`` exposes both the scalar functions in this module and the lattice
decomposition :class:`~dit.pid.syndisc.SynDisc` / :class:`~dit.pid.syndisc.ModifiedSynDisc`
(see :doc:`/decompositions/pid`).

.. ipython::

   In [1]: from dit.multivariate import synergistic_disclosure

   In [2]: from dit.example_dists import Xor

   In [3]: d = Xor()

   In [4]: synergistic_disclosure(d, sources=[[0], [1]], target=[2], alpha=[[0], [1]], niter=8)

API
===

.. autofunction:: synergistic_disclosure
   :no-index:

.. autofunction:: backbone_disclosure
   :no-index:

.. autofunction:: self_synergy
   :no-index:

:func:`~dit.multivariate.modified_synergistic_disclosure` is the
singleton-constraint shortcut used by :class:`~dit.pid.ModifiedSynDisc`.
