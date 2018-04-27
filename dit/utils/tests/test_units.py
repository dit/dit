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
from dit.params import reset_params
from dit.utils.units import ureg


def test_bit():
    """
    Test known bit values.
    """
    d = Distribution(['0', '1'], [1/2, 1/2])
    ditParams['units'] = True
    h = entropy(d)
    reset_params()
    true = ureg.Quantity(1, ureg.bit)
    assert float(h) == pytest.approx(true)


def test_nat():
    """
    Test known bit values.
    """
    d = Distribution(['0', '1'], [1/2, 1/2])
    ditParams['units'] = True
    h = entropy(d)
    reset_params()
    true = ureg.Quantity(np.log(2), ureg.nat)
    assert float(h) == pytest.approx(true)


def test_dit():
    """
    Test known bit values.
    """
    d = Distribution(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], [1/10]*10)
    ditParams['units'] = True
    h = entropy(d)
    reset_params()
    true = ureg.Quantity(1, ureg.dit)
    assert float(h) == pytest.approx(true)


def test_cc():
    """
    Test against a known value.
    """
    gm = Distribution(['00', '01', '10'], [1/3]*3)
    ditParams['units'] = True
    cc, _ = channel_capacity_joint(gm, [0], [1], marginal=True)
    reset_params()
    true = ureg.Quantity(0.3219280796196524, ureg.bit)
    assert float(cc) == pytest.approx(true)
