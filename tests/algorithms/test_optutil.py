"""
Tests for dit.algorithms.optutil.
"""

import numpy as np
import pytest

from dit.algorithms import degrees_of_freedom, ipf_dist, marginal_maxent_dists
from dit.algorithms.distribution_optimizers import MaxEntOptimizer, MinEntOptimizer
from dit.algorithms.maxentropy import MarginalMaximumEntropy
from dit.algorithms.optutil import prepare_dist
from dit.example_dists import Xor


def _sparse_log_xor():
    """A sparse, log-base distribution, which prepare_dist must convert."""
    d = Xor()
    d.set_rv_names("XYZ")
    d.set_base(2)
    return d


def _state(d):
    return d.get_base(), d.is_dense(), d.outcomes, d.pmf.tolist(), d.get_rv_names()


def test_prepare_dist_returns_dense_linear_copy():
    """prepare_dist returns a dense, linear copy and leaves its input alone."""
    d = _sparse_log_xor()
    before = _state(d)
    pd = prepare_dist(d)
    assert pd is not d
    assert pd.get_base() == "linear"
    assert pd.is_dense()
    np.testing.assert_allclose(pd.pmf, [1 / 4, 0, 0, 1 / 4, 0, 1 / 4, 1 / 4, 0])
    assert _state(d) == before


@pytest.mark.parametrize(
    "func",
    [
        pytest.param(marginal_maxent_dists, id="marginal_maxent_dists"),
        pytest.param(lambda d: ipf_dist(d, [[0, 1], [1, 2]]), id="ipf_dist"),
        pytest.param(lambda d: MaxEntOptimizer(d, [[0], [1], [2]]), id="MaxEntOptimizer"),
        pytest.param(lambda d: MinEntOptimizer(d, [[0], [1], [2]]), id="MinEntOptimizer"),
        pytest.param(lambda d: MarginalMaximumEntropy(d, 2), id="MarginalMaximumEntropy"),
        pytest.param(degrees_of_freedom, id="degrees_of_freedom"),
    ],
)
def test_input_is_not_modified(func):
    """Functions built on prepare_dist do not modify the caller's distribution (#86)."""
    d = _sparse_log_xor()
    before = _state(d)
    func(d)
    assert _state(d) == before
