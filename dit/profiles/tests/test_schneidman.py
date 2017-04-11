"""
Tests for dit.profiles.SchneidmanProfile. Known examples taken from http://arxiv.org/abs/1409.4708 .
"""

from __future__ import division

import pytest

import numpy as np

from dit import Distribution
from dit.profiles import SchneidmanProfile

ex1 = Distribution(['000', '001', '010', '011', '100', '101', '110', '111'], [1/8]*8)
ex2 = Distribution(['000', '111'], [1/2]*2)
ex3 = Distribution(['000', '001', '110', '111'], [1/4]*4)
ex4 = Distribution(['000', '011', '101', '110'], [1/4]*4)
ex4.set_rv_names('XYZ')


@pytest.mark.parametrize(('ex', 'prof'), [
    (ex1, (0.0, 0.0, 0.0)),
    (ex2, (0.0, 2.0, 0.0)),
    (ex3, (0.0, 1.0, 0.0)),
    (ex4, (0.0, 0.0, 1.0)),
])
def test_schneidman_profile(ex, prof):
    """
    Test against known examples.
    """
    sp = SchneidmanProfile(ex)
    assert np.allclose([sp.profile[i] for i in (1,2,3)], prof, atol=1e-5)
