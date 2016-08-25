# -*- coding: utf-8 -*-
from __future__ import division

import pytest

import numpy as np

from dit.exceptions import ditException

from dit.math.combinatorics import unitsum_tuples, slots

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
     with pytest.raises(Exception):
         next(g)

def test_slots1():
    x = list(slots(3, 2))
    x_ = [(0, 3), (1, 2), (2, 1), (3, 0)]
    assert x == x_

def test_slots2():
    x = np.asarray(list(slots(3, 2, normalized=True)))
    x_ = np.asarray([(0, 1), (1/3, 2/3), (2/3, 1/3), (1, 0)])
    assert np.allclose(x, x_)
