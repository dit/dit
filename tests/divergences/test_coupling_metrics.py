"""
Tests for dit.divergences.coupling_metrics
"""

import pytest

from dit import Distribution
from dit.divergences.coupling_metrics import (
    _coupling_problem,
    coupling_metric,
    coupling_min_residual_entropy,
    max_caekl_coupling,
    max_dual_total_correlation_coupling,
    max_total_correlation_coupling,
    min_residual_entropy_coupling,
)
from dit.multivariate import caekl_mutual_information
from dit.multivariate import dual_total_correlation as B
from dit.multivariate import residual_entropy as R
from dit.multivariate import total_correlation as T


def _marginals_match(joint, dists):
    lengths = [0]
    for d in dists:
        lengths.append(lengths[-1] + len(d.rvs))
    for i, d in enumerate(dists):
        rv = list(range(lengths[i], lengths[i + 1]))
        assert joint.marginal(rv).is_approx_equal(d, rtol=1e-3, atol=1e-3)


@pytest.mark.flaky(reruns=10)
def test_coupling_metric_reversed_binary():
    """
    Legacy coupling_metric: R at the min-H coupling for reversed marginals.
    """
    d1 = Distribution(["0", "1"], [1 / 3, 2 / 3])
    d2 = Distribution(["a", "b"], [2 / 3, 1 / 3])
    cm = coupling_metric([d1, d2], p=1.0)
    assert cm == pytest.approx(0.0, abs=1e-3)


@pytest.mark.flaky(reruns=10)
def test_min_residual_entropy_coupling_reversed_binary():
    d1 = Distribution(["0", "1"], [1 / 3, 2 / 3])
    d2 = Distribution(["a", "b"], [2 / 3, 1 / 3])
    joint = min_residual_entropy_coupling([d1, d2], niter=75)
    _marginals_match(joint, [d1, d2])
    assert R(joint, [[0], [1]]) == pytest.approx(0.0, abs=1e-2)


@pytest.mark.flaky(reruns=10)
def test_max_total_correlation_coupling_dependent():
    """Max TC coupling is highly dependent (min joint entropy)."""
    d1 = Distribution(["0", "1"], [0.5, 0.5])
    d2 = Distribution(["a", "b"], [0.5, 0.5])
    joint = max_total_correlation_coupling([d1, d2], niter=75)
    _marginals_match(joint, [d1, d2])
    assert T(joint, [[0], [1]]) == pytest.approx(1.0, abs=1e-2)


@pytest.mark.flaky(reruns=10)
def test_min_r_lower_than_independent_product():
    """Min-R coupling is more dependent than the independent product."""
    d1 = Distribution(["0", "1"], [0.5, 0.5])
    d2 = Distribution(["a", "b"], [0.5, 0.5])
    product, dist_ids = _coupling_problem([d1, d2])
    min_r = min_residual_entropy_coupling([d1, d2], niter=75)
    assert R(min_r, dist_ids) < R(product, dist_ids) - 0.5


@pytest.mark.flaky(reruns=10)
def test_max_dual_total_correlation_coupling():
    from dit.distconst import uniform

    d = uniform(["000", "111"])
    joint = max_dual_total_correlation_coupling([d.marginal([0]), d.marginal([1]), d.marginal([2])], niter=75)
    assert B(joint, [[0], [1], [2]]) == pytest.approx(2.0, abs=1e-2)


@pytest.mark.flaky(reruns=10)
def test_max_caekl_coupling():
    from dit.distconst import uniform

    d = uniform(["000", "111"])
    joint = max_caekl_coupling([d.marginal([0]), d.marginal([1]), d.marginal([2])], niter=100)
    assert caekl_mutual_information(joint) == pytest.approx(1.0, abs=2e-2)


@pytest.mark.flaky(reruns=10)
def test_coupling_min_residual_entropy_scalar():
    d1 = Distribution(["0", "1"], [1 / 3, 2 / 3])
    d2 = Distribution(["a", "b"], [2 / 3, 1 / 3])
    assert coupling_min_residual_entropy([d1, d2], niter=75) == pytest.approx(0.0, abs=1e-2)
