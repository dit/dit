"""
Tests for dit.abstractdist.
"""

import numpy as np

from dit.abstractdist import get_abstract_dist, distribution_constraint
from dit.example_dists import Xor


def test_distribution_constraint1():
    """
    Test the xor distribution.
    """
    d = Xor()
    ad = get_abstract_dist(d)
    A, b = distribution_constraint([0], [1], ad)
    true_A = np.array([[0, 0, 1, 1, -1, -1, 0, 0],
                       [0, 0, -1, -1, 1, 1, 0, 0]])
    true_b = np.array([0, 0, 0, 0, 0, 0, 0, 0])
    assert (A == true_A).all()
    assert (b == true_b).all()


def test_distribution_constraint2():
    """
    Test the xor distribution.
    """
    d = Xor()
    ad = get_abstract_dist(d)
    A, b = distribution_constraint([0, 1], [1, 2], ad)
    true_A = np.array([[0, 1, 0, 0, -1, 0, 0, 0],
                       [0, -1, 1, 1, 0, -1, 0, 0],
                       [0, 0, -1, 0, 1, 1, -1, 0],
                       [0, 0, 0, -1, 0, 0, 1, 0]])
    true_b = np.array([0, 0, 0, 0, 0, 0, 0, 0])
    assert (A == true_A).all()
    assert (b == true_b).all()
