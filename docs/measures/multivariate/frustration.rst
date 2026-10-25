.. frustration.rst
.. py:currentmodule:: dit.multivariate

.. _frustration:
.. _pairwise_frustration:

***********
Frustration
***********

Negative :ref:`coinformation` has two sources. Let :math:`\tilde{p}_2` be the maximum entropy distribution with the same pairwise marginals as :math:`p`. It agrees with :math:`p` on every entropy except the joint, which it exceeds by the third-order connected information :math:`\Delta_3 \geq 0` :cite:`Schneidman2003`, so

.. math::

   \I{X : Y : Z}_p = \I{X : Y : Z}_{\tilde{p}_2} - \Delta_3 .

The second term is a genuine three-way constraint, as in xor. The first can also be negative: the W distribution, uniform over :math:`\{001, 010, 100\}`, is its own pairwise maximum entropy distribution, so :math:`\Delta_3 = 0`, yet its co-information is :math:`\log_2 3 - 2`. Its pairwise constraints are *frustrated*: every pair forbids :math:`11`, an antiferromagnetic loop that cannot be satisfied consistently. Frustration measures this through the interactions of the maximum entropy distribution rather than through the sign of a co-information, which redundancy elsewhere in the distribution can mask. The same construction applies to interactions of any order :math:`k`.

Definition
==========

Let :math:`\tilde{p}_k` be the maximum entropy distribution matching the :math:`k`-variable marginals of :math:`p`, so that :math:`D_{KL}(p \| \tilde{p}_k) = \sum_{j > k} \Delta_j`. A *binary sub-model* picks two letters for each variable. Its top-order interaction on a :math:`k`-set :math:`S` is the :math:`k`-th mixed difference of :math:`\log q` over the chosen letters; for :math:`k = 2` it is the conditional log-odds ratio

.. math::

   \lambda_{ij}(A, B) = \log \frac{q(a_1, b_1 \mid \cdot)\, q(a_0, b_0 \mid \cdot)}{q(a_1, b_0 \mid \cdot)\, q(a_0, b_1 \mid \cdot)} .

Relabeling a variable negates the interactions of the :math:`k`-sets containing it, so a product of interactions over a family of :math:`k`-sets is relabeling-invariant exactly when every variable is covered an even number of times: the family is a cycle over GF(2). A model of order at most :math:`k` is *unfrustrated* if every such product, in every binary sub-model, is non-negative; its lower-order interactions are unconstrained. For bits, read each interaction's sign as a parity constraint :math:`\bigoplus_{i \in S} x_i = 0` or :math:`1`; the model is unfrustrated exactly when these constraints are simultaneously satisfiable. For :math:`k = 2` the cycles are the cycles of the graph, and unfrustrated means the signed graph of couplings is balanced :cite:`harary1953notion`: relabeling makes every coupling ferromagnetic, i.e. the model is multivariate totally positive :cite:`fallat2017total`. Writing :math:`\mathcal{U}_k` for the unfrustrated models,

.. math::

   F_k = \min_{q \in \mathcal{U}_k} D_{KL}(\tilde{p}_k \| q) = \min_{q \in \mathcal{U}_k} D_{KL}(p \| q) - \sum_{j > k} \Delta_j ,

where the second form is the Pythagorean theorem for hierarchical log-linear families :cite:`Amari2001`.

Which orders can be frustrated on how many variables is a question of parity. The :math:`k`-subsets of :math:`k + 1` variables form a cycle only when :math:`k` is even, so pairwise frustration first appears on three variables, three-way frustration needs at least five, and four-way frustration first appears on five.

Pairwise Frustration
====================

For three bits and :math:`k = 2`, :math:`F_2` is the conditional mutual information :cite:`Cover2006` of the weakest bond when the loop of couplings is negative, and zero otherwise:

.. math::

   F_2 = \begin{cases} \min_k \I{X_i : X_j | X_k}_{\tilde{p}_2} & \lambda_{XY} \lambda_{XZ} \lambda_{YZ} < 0, \\ 0 & \text{otherwise,} \end{cases}

and :math:`F_2 \geq -\I{X : Y : Z}_{\tilde{p}_2}`. The W distribution has :math:`F_2 = 2/3`, while xor, whose synergy is entirely third-order, has :math:`F_2 = 0`:

.. ipython::

   In [1]: from dit import Distribution as D

   In [2]: from dit.example_dists import Xor

   In [3]: from dit.multivariate import frustration, pairwise_frustration

   In [4]: w = D(['001', '010', '100'], [1/3] * 3)

   @doctest float
   In [5]: round(pairwise_frustration(w), 6)
   Out[5]: 0.666667

   @doctest float
   In [6]: pairwise_frustration(Xor())
   Out[6]: 0.0

Frustration is not masked by redundancy. Combining W with an independent giant bit whose bias makes the total co-information exactly zero leaves the frustration at :math:`2/3`. For three or more letters, information frustration differs from the physical notion of unsatisfiable bonds: the uniform distribution over proper three-colourings of a triangle satisfies every bond, but its co-information at its own fields is :math:`\log_2 \tfrac{3}{4}`, and :math:`F_2 = 1`.

Higher Orders
=============

Four parity constraints on five bits, :math:`x_0 \oplus x_1 \oplus x_2 = 0`, :math:`x_0 \oplus x_1 \oplus x_3 = 0`, :math:`x_0 \oplus x_2 \oplus x_4 = 0` and :math:`x_0 \oplus x_3 \oplus x_4 = 1`, cover every variable an even number of times, so their left-hand sides sum to zero while their right-hand sides sum to one: they cannot all hold. Rewarding each satisfied constraint gives a distribution whose three-way interactions are frustrated, while changing the last right-hand side to zero makes them consistent:

.. ipython::

   In [7]: from itertools import product

   In [8]: clauses = [(0, 1, 2), (0, 1, 3), (0, 2, 4), (0, 3, 4)]

   In [9]: def xorsat(rhs):
      ...:     outcomes = [''.join(map(str, x)) for x in product((0, 1), repeat=5)]
      ...:     weights = [2.718281828 ** sum(sum(x[i] for i in C) % 2 == b for C, b in zip(clauses, rhs)) for x in product((0, 1), repeat=5)]
      ...:     return D(outcomes, [w / sum(weights) for w in weights])

   @doctest float
   In [10]: round(frustration(xorsat([0, 0, 0, 1]), order=3), 4)
   Out[10]: 0.1073

   @doctest float
   In [11]: round(frustration(xorsat([0, 0, 0, 0]), order=3), 4)
   Out[11]: 0.0

Computation
===========

For three bits and :math:`k = 2` the closed form above is used. Otherwise :math:`F_k` is computed exactly by branch-and-bound. Every top-order interaction is linear in the natural parameters, so :math:`\mathcal{U}_k` is a finite union of polyhedral cones indexed by sign patterns of the interactions, and the projection onto each cone is a convex program :cite:`lauritzen2021total`. The constraints are indexed by the circuits (minimal cycles) of the :math:`k`-subsets, of which there are few for small :math:`n`. If the node budget ``max_nodes`` is exhausted before optimality is certified, the best value found is returned with a warning stating the certified bracket.

No canonical literature source is known for this measure.

API
===

.. autofunction:: frustration
   :no-index:

.. autofunction:: pairwise_frustration
   :no-index:
