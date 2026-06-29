"""
Optimizer CAEKL backends should agree with dit.multivariate.caekl_mutual_information.
"""

import numpy as np
import pytest

from dit import Distribution
from dit.algorithms.distribution_optimizers import MaxCAEKLMutualInformationOptimizer
from dit.distconst import uniform
from dit.multivariate import caekl_mutual_information


def _optimizer_caekl_value(dist, rvs):
    opt = MaxCAEKLMutualInformationOptimizer(dist, rvs)
    caekl_fn = opt._caekl_mutual_information(opt._rvs)
    pmf = opt._full_pmf.reshape(opt._full_shape)
    return float(caekl_fn(pmf))


@pytest.mark.parametrize("rvs", [[[0], [1], [2]], [[0, 1], [2]], [[0], [1, 2]]])
def test_numpy_optimizer_caekl_matches_measure(rvs):
    dist = uniform([f"{i:03b}" for i in range(8)])
    expected = caekl_mutual_information(dist, rvs)
    assert _optimizer_caekl_value(dist, rvs) == pytest.approx(expected, abs=1e-10)


def test_auxvar_optimizer_has_caekl_grad():
    """Sanity: numpy analytic CAEKL grad exists on a small problem."""
    dist = Distribution([(0, 0), (0, 1), (1, 0), (1, 1)], [0.25] * 4)
    opt = MaxCAEKLMutualInformationOptimizer(dist, [[0], [1]])
    grad_fn = opt._caekl_mutual_information_grad(opt._rvs)
    pmf = opt._full_pmf.reshape(opt._full_shape)
    g = grad_fn(pmf)
    assert g.shape == pmf.shape
    assert np.all(np.isfinite(g))
