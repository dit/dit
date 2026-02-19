"""
Tests for dit.math.combinatorics
"""

import numpy as np
import pytest

from dit.math.combinatorics import slots, unitsum_tuples


def test_unitsum_tuples1():
    s = np.asarray(list(unitsum_tuples(3, 2, .2, .8)))
    s_ = np.asarray([[.2, .8], [.4, .6], [.6, .4], [.8, .2]])
    assert np.allclose(s, s_)


def test_unitsum_tuples2():
    s = np.asarray(list(unitsum_tuples(4, 3, -1, 3)))
    s_ = np.asarray([
        (-1.0, -1.0, 3.0),
        (-1.0, 0.0, 2.0),
        (-1.0, 1.0, 1.0),
        (-1.0, 2.0, 0.0),
        (-1.0, 3.0, -1.0),
        (0.0, -1.0, 2.0),
        (0.0, 0.0, 1.0),
        (0.0, 1.0, 0.0),
        (0.0, 2.0, -1.0),
        (1.0, -1.0, 1.0),
        (1.0, 0.0, 0.0),
        (1.0, 1.0, -1.0),
        (2.0, -1.0, 0.0),
        (2.0, 0.0, -1.0),
        (3.0, -1.0, -1.0),
    ])
    assert np.allclose(s, s_)


def test_unitsum_tuples3():
    # Violate: 1 = mx + (k-1) * mn
    g = unitsum_tuples(3, 3, 1, 0)
    with pytest.raises(Exception, match="Specified min and max will not create unitsum tuples."):
        next(g)


def test_slots1():
    x = list(slots(3, 2))
    x_ = [(0, 3), (1, 2), (2, 1), (3, 0)]
    assert x == x_


def test_slots2():
    x = np.asarray(list(slots(3, 2, normalized=True)))
    x_ = np.asarray([(0, 1), (1 / 3, 2 / 3), (2 / 3, 1 / 3), (1, 0)])
    assert np.allclose(x, x_)
