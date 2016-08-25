from __future__ import division

import numpy as np

import dit
from dit.algorithms.maxentropy import (
    marginal_constraints_generic, marginal_constraints
)


def test_marginal_constraints():
    d = dit.uniform_distribution(3, 2)
    d.make_dense()

    A, b = marginal_constraints(d, 2)

    A_ = np.array([
        [ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.],
        [ 1.,  1.,  0.,  0.,  0.,  0.,  0.,  0.],
        [ 0.,  0.,  1.,  1.,  0.,  0.,  0.,  0.],
        [ 0.,  0.,  0.,  0.,  1.,  1.,  0.,  0.],
        [ 0.,  0.,  0.,  0.,  0.,  0.,  1.,  1.],
        [ 1.,  0.,  1.,  0.,  0.,  0.,  0.,  0.],
        [ 0.,  1.,  0.,  1.,  0.,  0.,  0.,  0.],
        [ 0.,  0.,  0.,  0.,  1.,  0.,  1.,  0.],
        [ 0.,  0.,  0.,  0.,  0.,  1.,  0.,  1.],
        [ 1.,  0.,  0.,  0.,  1.,  0.,  0.,  0.],
        [ 0.,  1.,  0.,  0.,  0.,  1.,  0.,  0.],
        [ 0.,  0.,  1.,  0.,  0.,  0.,  1.,  0.],
        [ 0.,  0.,  0.,  1.,  0.,  0.,  0.,  1.]
    ])

    b_ = np.array([1] + [0.25] * 12)

    assert np.allclose(A, A_)
    assert np.allclose(b, b_)
