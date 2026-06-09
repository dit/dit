"""
Tests for dit.abstractdist.
"""

import numpy as np
import pytest

from dit.abstractdist import distribution_constraint, get_abstract_dist
from dit.example_dists import Xor


def test_distribution_constraint1():
    """
    Test the xor distribution.
    """
    d = Xor()
    ad = get_abstract_dist(d)
    A, b = distribution_constraint([0], [1], ad)
    true_A = np.array([[0, 0, 1, 1, -1, -1, 0, 0], [0, 0, -1, -1, 1, 1, 0, 0]])
    true_b = np.array([0, 0, 0, 0, 0, 0, 0, 0])
    assert (true_A == A).all()
    assert (b == true_b).all()


def test_parameter_array_internal_cache():
    """parameter_array builds a private cache when none is supplied."""
    ad = get_abstract_dist(Xor())
    p = ad.parameter_array([0])
    assert p.shape[0] == ad.n_symbols


def test_parameter_array_invalid_indexes():
    ad = get_abstract_dist(Xor())
    with pytest.raises(Exception, match="Invalid indexes"):
        ad.parameter_array([ad.n_variables])


def test_marginal():
    ad = get_abstract_dist(Xor())
    m = ad.marginal([0, 1])
    assert m.n_variables == 2
    assert m.n_symbols == ad.n_symbols


def test_marginal_invalid_indexes():
    ad = get_abstract_dist(Xor())
    with pytest.raises(Exception, match="Invalid indexes"):
        ad.marginal([-1])


def test_distribution_constraint_incompatible():
    ad = get_abstract_dist(Xor())
    with pytest.raises(Exception, match="Incompatible distributions"):
        distribution_constraint([0], [1, 2], ad)


def test_distribution_constraint2():
    """
    Test the xor distribution.
    """
    d = Xor()
    ad = get_abstract_dist(d)
    A, b = distribution_constraint([0, 1], [1, 2], ad)
    true_A = np.array(
        [[0, 1, 0, 0, -1, 0, 0, 0], [0, -1, 1, 1, 0, -1, 0, 0], [0, 0, -1, 0, 1, 1, -1, 0], [0, 0, 0, -1, 0, 0, 1, 0]]
    )
    true_b = np.array([0, 0, 0, 0, 0, 0, 0, 0])
    assert (true_A == A).all()
    assert (b == true_b).all()
