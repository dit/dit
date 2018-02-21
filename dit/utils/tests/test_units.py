"""
Tests for dit.utils.units.
"""

from __future__ import division

import pytest

pint = pytest.importorskip('pint')

import numpy as np

from dit import Distribution, ditParams
from dit.algorithms.channelcapacity import channel_capacity_joint
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
    true = ureg.Quantity(1, ureg.bit)
    assert h == pytest.approx(true)


def test_nat():
    """
    Test known bit values.
    """
    d = Distribution(['0', '1'], [1/2, 1/2])
    ditParams['units'] = True
    h = entropy(d)
    ditParams['units'] = False
    true = ureg.Quantity(np.log(2), ureg.nat)
    assert h == pytest.approx(true)


def test_dit():
    """
    Test known bit values.
    """
    d = Distribution(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], [1/10]*10)
    ditParams['units'] = True
    h = entropy(d)
    ditParams['units'] = False
    true = ureg.Quantity(1, ureg.dit)
    assert h == pytest.approx(true)


def test_cc():
    """
    Test against a known value.
    """
    gm = Distribution(['00', '01', '10'], [1/3]*3)
    ditParams['units'] = True
    cc, marg = channel_capacity_joint(gm, [0], [1], marginal=True)
    ditParams['units'] = False
    true = ureg.Quantity(0.3219280796196524, ureg.bit)
    assert cc == pytest.approx(true)
