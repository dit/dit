"""
Regression tests for optimizer bugs found in review.
"""

from types import MethodType

import numpy as np
import pytest

import dit
from dit.algorithms.convex_maximization import maximize_convex_function
from dit.algorithms.distribution_optimizers import MinResidualEntropyOptimizer
from dit.multivariate import residual_entropy
from dit.utils.optimization import BasinHoppingCallBack


def test_min_residual_entropy_objective_is_residual_entropy():
    d = dit.random_distribution(3, 2, prng=np.random.RandomState(1))
    opt = MinResidualEntropyOptimizer(d, [[0], [1], [2]])
    opt.objective = MethodType(opt._objective(), opt)
    pmf = opt.construct_joint(opt._vpmf[opt._free])
    assert opt._residual_entropy(opt._rvs)(pmf) == pytest.approx(residual_entropy(d))


@pytest.mark.parametrize(
    ("A", "b", "expected"),
    [
        (np.array([[1.0, 0], [0, 1], [-1, 0], [0, -1]]), np.array([1.0, 1, 0, 0]), 2.0),
        (np.array([[-1.0, 0], [0, -1], [1, 1]]), np.array([0.0, 0, 1]), 1.0),
    ],
)
def test_maximize_convex_function_keeps_inequalities(A, b, expected):
    pytest.importorskip("pypoman")
    _, val = maximize_convex_function(lambda x: (x**2).sum(), A, b)
    assert val == pytest.approx(expected)


def test_basinhopping_callback_rejects_negative_equality_residual():
    constraints = [{"type": "eq", "fun": lambda x: x.sum() - 1}]
    cb = BasinHoppingCallBack(constraints, None)
    cb(np.array([0.1, 0.1]), -10.0, True)
    cb(np.array([0.5, 0.5]), 0.0, True)
    assert np.allclose(cb.minimum().x, [0.5, 0.5])
