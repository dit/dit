"""
Tests for dit.utils.units.
"""

import numpy as np
import pytest

from dit import Distribution, ditParams
from dit.algorithms.channelcapacity import channel_capacity_joint
from dit.multivariate import entropy
from dit.params import reset_params
from dit.utils.units import ureg

pint = pytest.importorskip('pint')


def test_bit():
    """
    Test known bit values.
    """
    d = Distribution(['0', '1'], [1 / 2, 1 / 2])
    ditParams['units'] = True
    h = entropy(d)
    reset_params()
    true = ureg.Quantity(1, ureg.bit)
    assert h.m == pytest.approx(true.m)


def test_nat():
    """
    Test known bit values.
    """
    d = Distribution(['0', '1'], [1 / 2, 1 / 2])
    ditParams['units'] = True
    h = entropy(d)
    reset_params()
    true = ureg.Quantity(np.log(2), ureg.nat).to_base_units()
    assert h.m == pytest.approx(true.m)


def test_dit():
    """
    Test known bit values.
    """
    d = Distribution(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], [1 / 10] * 10)
    ditParams['units'] = True
    h = entropy(d)
    reset_params()
    true = ureg.Quantity(1, ureg.dit).to_base_units()
    assert h.m == pytest.approx(true.m)


def test_cc():
    """
    Test against a known value.
    """
    gm = Distribution(['00', '01', '10'], [1 / 3] * 3)
    ditParams['units'] = True
    cc, _ = channel_capacity_joint(gm, [0], [1], marginal=True)
    reset_params()
    true = ureg.Quantity(0.3219280796196524, ureg.bit)
    assert cc.m == pytest.approx(true.m)
