# -*- coding: utf-8 -*-

"""
Tests for dit.multivariate.o_information.
"""

import pytest

from dit.example_dists import giant_bit, n_mod_m
from dit.multivariate.o_information import o_information


d1 = giant_bit(5, 2)
d2 = n_mod_m(5, 2)


@pytest.mark.parametrize(['dist', 'rvs', 'crvs', 'value'], [
    (d1, [[0], [1], [2], [3], [4]], [], 3),
    (d1, [[0], [1], [2], [3]], [4], 0),
    (d1, [[0], [1], [2]], [3, 4], 0),
    (d2, [[0], [1], [2], [3], [4]], [], -3),
    (d2, [[0], [1], [2], [3]], [4], -2),
    (d2, [[0], [1], [2]], [3, 4], -1),
])
def test_o_information_1(dist, rvs, crvs, value):
    """
    Test the o-information against known values.
    """
    assert o_information(dist=dist, rvs=rvs, crvs=crvs) == pytest.approx(value)
