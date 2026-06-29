.. coupling_metrics.rst
.. py:module:: dit.divergences.coupling_metrics

Information-Theoretic Couplings
===============================

A **coupling** of marginal distributions :math:`P_1, \ldots, P_k` is a joint
distribution whose marginals match the :math:`P_i`. ``dit`` constructs couplings
that optimize multivariate information measures subject to those marginal
constraints.

These routines are distinct from optimal-transport couplings such as the
:doc:`earth_movers_distance`, which minimize expected ground-metric cost
:cite:`chan2015multivariate`.

Coupling constructors
---------------------

.. autofunction:: min_residual_entropy_coupling

.. autofunction:: max_total_correlation_coupling

.. autofunction:: max_dual_total_correlation_coupling

.. autofunction:: max_caekl_coupling

Scalar summaries
----------------

.. autofunction:: coupling_min_residual_entropy

.. autofunction:: coupling_metric

The legacy :func:`coupling_metric` returns the residual entropy of the
*minimum joint-entropy* coupling (not a direct minimization of residual entropy).
