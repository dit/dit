.. pairwise_frustration.rst
.. py:module:: dit.multivariate.pairwise_frustration

.. _pairwise_frustration:

********************
Pairwise Frustration
********************

Negative :ref:`coinformation` has two sources. Let :math:`\tilde{p}_2` be the maximum entropy distribution with the same pairwise marginals as :math:`p`. It agrees with :math:`p` on every entropy except the joint, which it exceeds by the third-order connected information :math:`\Delta_3 \geq 0` :cite:`Schneidman2003`, so

.. math::

   \I{X : Y : Z}_p = \I{X : Y : Z}_{\tilde{p}_2} - \Delta_3 .

The second term is a genuine three-way constraint, as in xor. The first can also be negative: the W distribution, uniform over :math:`\{001, 010, 100\}`, is its own pairwise maximum entropy distribution, so :math:`\Delta_3 = 0`, yet its co-information is :math:`\log_2 3 - 2`. Its pairwise constraints are *frustrated*: every pair forbids :math:`11`, an antiferromagnetic loop that cannot be satisfied consistently. The pairwise frustration measures this through the couplings of :math:`\tilde{p}_2` rather than through the sign of its co-information, which redundancy elsewhere in the distribution can mask.

Definition
==========

A *binary sub-model* of a pairwise model picks two letters for each variable. Its couplings are the conditional log-odds ratios

.. math::

   \lambda_{ij}(A, B) = \log \frac{q(a_1, b_1 \mid \cdot)\, q(a_0, b_0 \mid \cdot)}{q(a_1, b_0 \mid \cdot)\, q(a_0, b_1 \mid \cdot)}

for letter pairs :math:`A = \{a_0, a_1\}` and :math:`B = \{b_0, b_1\}`, which in a pairwise model do not depend on the other variables. A pairwise model is *unfrustrated* if every binary sub-model is balanced: the product of log-odds ratios around every cycle of variables is non-negative. For bits, this is exactly the condition that relabeling some variables makes every coupling ferromagnetic :cite:`harary1953notion`, i.e. that the model is multivariate totally positive :cite:`fallat2017total`. Every tree-structured model is unfrustrated. Writing :math:`\mathcal{U}` for the unfrustrated pairwise models,

.. math::

   F = \min_{q \in \mathcal{U}} D_{KL}(\tilde{p}_2 \| q) = \min_{q \in \mathcal{U}} D_{KL}(p \| q) - \Delta_3 ,

where the second form is the Pythagorean theorem for the pairwise exponential family :cite:`Amari2001`. Thus the distance from :math:`p` to the nearest unfrustrated model splits into a third-order part and a frustration part.

Three Bits
==========

For three bits, :math:`F` is the conditional mutual information :cite:`Cover2006` of the weakest bond when the loop of couplings is negative, and zero otherwise:

.. math::

   F = \begin{cases} \min_k \I{X_i : X_j | X_k}_{\tilde{p}_2} & \lambda_{XY} \lambda_{XZ} \lambda_{YZ} < 0, \\ 0 & \text{otherwise,} \end{cases}

and :math:`F \geq -\I{X : Y : Z}_{\tilde{p}_2}`. The W distribution has :math:`F = 2/3`, while xor, whose synergy is entirely third-order, has :math:`F = 0`:

.. ipython::

   In [1]: from dit import Distribution as D

   In [2]: from dit.example_dists import Xor

   In [3]: from dit.multivariate import pairwise_frustration as F

   In [4]: w = D(['001', '010', '100'], [1/3] * 3)

   @doctest float
   In [5]: round(F(w), 6)
   Out[5]: 0.666667

   @doctest float
   In [6]: F(Xor())
   Out[6]: 0.0

Frustration is not masked by redundancy. Combining W with an independent giant bit whose bias makes the total co-information exactly zero leaves the frustration at :math:`2/3`.

General Alphabets
=================

For larger alphabets or more variables, :math:`F` is computed exactly by branch-and-bound. Every log-odds ratio is linear in the natural parameters, so :math:`\mathcal{U}` is a finite union of polyhedral cones indexed by sign patterns of the log-odds ratios, and the projection onto each cone is a convex program :cite:`lauritzen2021total`. If the node budget ``max_nodes`` is exhausted before optimality is certified, the best value found is returned with a warning stating the certified bracket.

For three or more letters, information frustration differs from the physical notion of unsatisfiable bonds. The uniform distribution over proper three-colourings of a triangle satisfies every bond, but its co-information at its own fields is :math:`\log_2 \tfrac{3}{4}`, and :math:`F = 1`.

No canonical literature source is known for this measure.

API
===

.. autofunction:: pairwise_frustration
