"""
Tests for dit.profiles.MUIProfile. Known examples taken from http://arxiv.org/abs/1409.4708 .
"""

from __future__ import division

import pytest

import numpy as np

from dit import Distribution
from dit.profiles import MUIProfile

ex1 = Distribution(['000', '001', '010', '011', '100', '101', '110', '111'], [1/8]*8)
ex2 = Distribution(['000', '111'], [1/2]*2)
ex3 = Distribution(['000', '001', '110', '111'], [1/4]*4)
ex4 = Distribution(['000', '011', '101', '110'], [1/4]*4)
examples = [ex1, ex2, ex3, ex4]


pytest.importorskip('scipy')

@pytest.mark.parametrize(('d', 'prof', 'width'), [
    (ex1, {0.0: 1.0}, [3.0]),
    (ex2, {0.0: 3.0}, [1.0]),
    (ex3, {0.0: 2.0, 1.0: 1.0}, [1.0, 1.0]),
    (ex4, {0.0: 3/2}, [2.0]),
])
def test_mui_profile(d, prof, width):
    """
    Test against known examples.
    """
    mui = MUIProfile(d)
    assert mui.profile == prof
    assert np.allclose(mui.widths, width)
