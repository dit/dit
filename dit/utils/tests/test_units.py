"""
Tests for dit.utils.units.
"""

from __future__ import division

import pytest

import numpy as np

pint = pytest.importorskip('pint')

from dit import Distribution, ditParams
from dit.multivariate import entropy
from dit.utils.units import ureg


def test_bit():
    """
    Test known bit values.
    """
    d = Distribution(['0', '1'], [1/2, 1/2])
    ditParams['units'] = True
    h = entropy(d)
    ditParams['units'] = False
    assert h == 1*ureg.bit


def test_nat():
    """
    Test known bit values.
    """
    d = Distribution(['0', '1'], [1/2, 1/2])
    ditParams['units'] = True
    h = entropy(d)
    ditParams['units'] = False
    assert h == np.log(2)*ureg.nat


def test_dit():
    """
    Test known bit values.
    """
    d = Distribution(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], [1/10]*10)
    ditParams['units'] = True
    h = entropy(d)
    ditParams['units'] = False
    assert h == 1*ureg.dit
