"""
Tests for dit.multivariate.lautum_information.
"""

from __future__ import division

import pytest

from dit import Distribution as D
from dit.multivariate import lautum_information as L


def test_lm1():
    """ Test L """
    outcomes = ['000', '001', '010', '011', '100', '101', '110', '111']
    pmf = [3/16, 1/16, 1/16, 3/16, 1/16, 3/16, 3/16, 1/16]
    d = D(outcomes, pmf)
    assert L(d) == pytest.approx(0.20751874963942196)
    assert L(d, [[0], [1]]) == pytest.approx(0)
    assert L(d, [[0], [1]], [2]) == pytest.approx(0.20751874963942196)
