.. coupling_metrics.rst
.. py:currentmodule:: dit.divergences

Information-Theoretic Couplings
===============================

A **coupling** of marginal distributions :math:`P_1, \ldots, P_k` is a joint
distribution whose marginals match the :math:`P_i`. ``dit`` constructs couplings
that optimize multivariate information measures subject to those marginal
constraints.

These routines are distinct from optimal-transport couplings such as the
:doc:`/measures/divergences/earth_movers_distance`, which minimize expected ground-metric cost
:cite:`chan2015multivariate`.

Example
-------

Couple a fair bit with a three-valued variable of distribution
:math:`(1/2, 1/4, 1/4)`. The minimum residual-entropy coupling makes each
variable as predictable from the other as the marginals allow, so the
optimizer returns (approximately) the coupling in which the bit is a function of
the three-valued variable:

.. ipython::

   In [1]: from dit.divergences import min_residual_entropy_coupling, max_total_correlation_coupling

   In [2]: p = dit.Distribution(['0', '1'], [1/2, 1/2])

   In [3]: q = dit.Distribution(['0', '1', '2'], [1/2, 1/4, 1/4])

   In [4]: c = min_residual_entropy_coupling([p, q])

   In [5]: print(c)

   In [6]: dit.multivariate.residual_entropy(c)

   In [7]: dit.multivariate.total_correlation(max_total_correlation_coupling([p, q]))

Coupling constructors
---------------------

.. autofunction:: min_residual_entropy_coupling
   :no-index:

.. autofunction:: max_total_correlation_coupling
   :no-index:

.. autofunction:: max_dual_total_correlation_coupling
   :no-index:

.. autofunction:: max_caekl_coupling
   :no-index:

Scalar summaries
----------------

.. autofunction:: coupling_min_residual_entropy
   :no-index:

.. autofunction:: coupling_metric
   :no-index:

The legacy :func:`coupling_metric` returns the residual entropy of the
*minimum joint-entropy* coupling (not a direct minimization of residual entropy).

Entropy is concave, so the minimum joint-entropy coupling lies at a vertex of the
polytope of couplings. The search is seeded with vertices from linear programs with
random costs and, for disjoint marginals, with the greedy coupling of
:cite:`kocaoglu2017entropic`, whose entropy is within one bit of the minimum
:cite:`cicalese2019minimum`.
