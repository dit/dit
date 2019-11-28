"""
Tests for dit.example_dists.nonsignalling_boxes.
"""

from __future__ import division

from itertools import product

from dit import Distribution
from dit.example_dists import pr_box


def test_pr_1():
    """
    Test
    """
    d1 = Distribution(list(product([0, 1], repeat=4)), [1/16]*16)
    d2 = pr_box(0.0)
    assert d1.is_approx_equal(d2)


def test_pr_2():
    """
    Test
    """
    d1 = Distribution([(0, 0, 0, 0),
                       (0, 0, 1, 1),
                       (0, 1, 0, 0),
                       (0, 1, 1, 1),
                       (1, 0, 0, 0),
                       (1, 0, 1, 1),
                       (1, 1, 0, 1),
                       (1, 1, 1, 0)], [1/8]*8)
    d2 = pr_box(1.0, name=True)
    assert d1.is_approx_equal(d2)
