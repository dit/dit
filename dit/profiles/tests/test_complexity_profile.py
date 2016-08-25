"""
Tests for dit.profiles.ComplexityProfile. Known examples taken from http://arxiv.org/abs/1409.4708 .
"""

from __future__ import division

import pytest

from dit import Distribution
from dit.profiles import ComplexityProfile

ex1 = Distribution(['000', '001', '010', '011', '100', '101', '110', '111'], [1/8]*8)
ex2 = Distribution(['000', '111'], [1/2]*2)
ex3 = Distribution(['000', '001', '110', '111'], [1/4]*4)
ex4 = Distribution(['000', '011', '101', '110'], [1/4]*4)

@pytest.mark.parametrize(('ex', 'prof'), [
    (ex1, {1: 3.0, 2: 0.0, 3: 0.0}),
    (ex2, {1: 1.0, 2: 1.0, 3: 1.0}),
    (ex3, {1: 2.0, 2: 1.0, 3: 0.0}),
    (ex4, {1: 2.0, 2: 2.0, 3: -1.0}),
])
def test_complexity_profile(ex, prof):
    """
    Test against known examples.
    """
    cp = ComplexityProfile(ex)
    assert cp.profile == prof
