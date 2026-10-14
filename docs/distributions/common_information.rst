.. common_information.rst
.. py:module:: dit.example_dists.common_information

Common-Information Witnesses
============================

The common informations of a set of random variables form a total order
:cite:`james2017multivariate`,

.. math::

   0 \le \operatorname{K} \le \operatorname{J} \le \operatorname{B}
     \le \operatorname{C} \le \operatorname{G} \le \operatorname{F}
     \le \operatorname{M} \le \operatorname{H},

from the Gács-Körner :math:`\operatorname{K}`, through the CAEKL
:math:`\operatorname{J}`, dual total correlation :math:`\operatorname{B}`, Wyner
:cite:`wyner1975common` :math:`\operatorname{C}`, exact :cite:`kumar2014exact`
:math:`\operatorname{G}`, functional :math:`\operatorname{F}`, and minimal
sufficient statistic :math:`\operatorname{M}` common informations, up to the joint
entropy :math:`\operatorname{H}`. A *witness* is a single distribution on which
every inequality is strict, certifying that no two of the measures coincide as
functionals.

:func:`symmetric_common_information_witness` builds an exchangeable witness for any
number of variables :math:`n \ge 2` by taking the independent product of three
layers, so that each variable is :math:`X_i = (G, Z \oplus N_i, P_i)`: a shared
"giant bit" :math:`G` (:func:`~dit.example_dists.giant_bit`, which lifts
:math:`0 < \operatorname{K}`), a latent fair bit :math:`Z` seen through an
independent binary symmetric channel with crossover :math:`q` (the workhorse
:func:`noisy_copies`, which separates the middle of the chain), and a private fair
bit :math:`P_i` (which opens :math:`\operatorname{M} < \operatorname{H}`).

.. ipython::

   In [1]: from dit.example_dists import noisy_copies, symmetric_common_information_witness

   In [2]: d = symmetric_common_information_witness(3)

   In [3]: d.outcome_length(), len(d.outcomes), len(d.alphabet[0])

The workhorse already strictly separates the middle of the ordering, while having
no private randomness (so :math:`\operatorname{M} = \operatorname{H}`):

.. ipython::

   In [4]: from dit.multivariate import gk_common_information, caekl_mutual_information

   In [5]: from dit.multivariate import dual_total_correlation, functional_common_information

   In [6]: w = noisy_copies(0.1, 3)

   In [7]: gk_common_information(w), caekl_mutual_information(w), dual_total_correlation(w)

   In [8]: functional_common_information(w)

Between the dual total correlation and the functional common information shown
above sit the Wyner (:func:`~dit.multivariate.wyner_common_information`) and exact
(:func:`~dit.multivariate.exact_common_information`) common informations, with
:math:`\operatorname{B} < \operatorname{C} < \operatorname{G} < \operatorname{F}`
strict (these are optimization-based, so they are omitted from this example). The
giant bit then lifts :math:`\operatorname{K}` away from zero and the private bits
pull :math:`\operatorname{M}` below :math:`\operatorname{H}`, so the assembled
witness strictly separates the whole chain. For ``n = 2`` the CAEKL and dual total
correlation coincide with the mutual information, giving one fewer distinct rung.

APIs
====

.. autofunction:: noisy_copies
.. autofunction:: symmetric_common_information_witness
