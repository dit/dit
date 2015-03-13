"""
Tests for dit.other.tsallis_entropy.
"""

from __future__ import division

from nose.tools import assert_almost_equal

import numpy as np

from dit import Distribution
from dit.other import tsallis_entropy

def test_tsallis_entropy_1():
    """
    Test the pseudo-additivity property.
    """
    d = Distribution(['00', '01', '02', '10', '11', '12'], [1/6]*6)
    for q in np.arange(-2, 2.5, 0.5):
        S_AB = tsallis_entropy(d, q)
        S_A = tsallis_entropy(d, q, [0])
        S_B = tsallis_entropy(d, q, [1])
        pa_prop = S_A + S_B + (1-q)*S_A*S_B
        yield assert_almost_equal, S_AB, pa_prop
